"""Fixed quality criteria using the d3-flash SystemOne choice API."""

import json
import os
import re
from time import perf_counter

import httpx

OCR_VISUAL = {"visible_noise": "Is the text free of garbled characters and abnormal line breaks?"}
OCR_FIDELITY = {
    "fidelity": "Does the OCR faithfully preserve the characters, numbers, and lines in the reference?"
}
TRANSLATION = {
    "facts": "Are key facts consistent with the source?",
    "number_and_unit": (
        "Are the source numbers and units transferred correctly? If neither text contains "
        "a number or unit, pass this check."
    ),
    "negation_and_terms": (
        "Are negation and terminology preserved? Evaluate negation and terminology only. "
        "A changed numerical value alone does not fail this check."
    ),
    "omission": (
        "Is all meaningful source content represented? Evaluate missing information only. "
        "A detail that is present but mistranslated belongs to another check."
    ),
    "fluency": (
        "Read the candidate on its own. Is it grammatical and natural in the target "
        "language? Ignore its factual agreement with the source."
    ),
}
PROMPT_VERSION = "quality-review-systemone-v1"
DEFAULT_BASE_URL = "http://192.168.0.213:18892/v1"
DEFAULT_MODEL = "vllm-sr/d3-flash"
DECISIONS = {
    "pass": "This specific criterion is satisfied, even if another criterion fails.",
    "minor": "The criterion fails, with a minor error that does not materially affect meaning or usability.",
    "major": "The criterion fails, with an error affecting meaning or usability.",
    "unknown": "The available text is insufficient to decide.",
}
VERDICTS = {
    "pass": ("pass", "none"),
    "minor": ("fail", "minor"),
    "major": ("fail", "major"),
    "unknown": ("insufficient_evidence", "none"),
}
# Split at sentence boundaries, including CJK punctuation, while keeping decimal points.
SENTENCE = re.compile(r"[^\n]+?(?:[。！？；;]+|[.!?](?=\s|$)|(?=\n|$))")


def review_input(sample, mode):
    if sample["task"] == "translation":
        return TRANSLATION, sample["source"], sample["candidate"]
    source = sample.get("source") if mode == "ocr_reference" else None
    return OCR_VISUAL | OCR_FIDELITY, source, sample["candidate"]


def choice_question(instructions, criteria):
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def evidence_options(text):
    options = {"none": "No evidence on this side of the comparison."}
    for sentence in SENTENCE.findall(text):
        sentence = sentence.strip()
        for offset in range(0, len(sentence), 96):
            excerpt = sentence[offset:offset + 96].strip()
            if excerpt and excerpt not in options.values():
                options[f"quote_{len(options)}"] = excerpt
    if len(options) > 24:
        raise ValueError("too many evidence excerpts; use a shorter passage")
    return options


def decode_choices(payload, questions):
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, dict) or answers.keys() != questions.keys():
        raise ValueError("SystemOne returned missing or unexpected answer keys")
    choices = {}
    for name, question in questions.items():
        answer = answers[name]
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError(f"{name}: expected a SystemOne choice answer")
        selected = answer.get("choice")
        if not isinstance(selected, str) or selected not in question["criteria"]:
            raise ValueError(f"{name}: unknown choice")
        choices[name] = selected
    return choices


class Judge:
    def __init__(self, client, timeout):
        self.client = client
        self.timeout = timeout
        base_url = os.environ.get("D3_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.url = base_url if base_url.endswith("/systemone") else base_url + "/systemone"
        api_key = os.environ.get("D3_API_KEY")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = os.environ.get("D3_MODEL", DEFAULT_MODEL)

    def choices(self, state, questions, result):
        result["request_count"] += 1
        response = self.client.post(
            self.url,
            headers=self.headers,
            json={"model": self.model, "state": state, "questions": questions},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        selected = decode_choices(payload, questions)
        reported_model = payload.get("model")
        if isinstance(reported_model, str) and reported_model:
            if result.get("model", reported_model) != reported_model:
                raise ValueError("SystemOne reported different models across review requests")
            result["model"] = reported_model
        usage = payload.get("usage")
        if isinstance(usage, dict):
            for key in ("input_tokens", "output_tokens"):
                if type(usage.get(key)) is int and usage[key] >= 0:
                    total = result.setdefault("usage", {})
                    total[key] = total.get(key, 0) + usage[key]
        return selected

    def add_evidence(self, state, criteria, checks, source, candidate, result):
        failed = [check for check in checks if check["verdict"] == "fail"]
        if not failed:
            return

        questions = {}
        targets = {}
        inputs = {"source_evidence": source, "candidate_evidence": candidate}
        options_by_field = {
            field: evidence_options(content) for field, content in inputs.items() if content
        }
        for check in failed:
            criterion = check["criterion"]
            for field, options in options_by_field.items():
                if field == "source_evidence" and criterion in {"visible_noise", "fluency"}:
                    continue
                if len(options) == 1:
                    continue
                name = f"{criterion}_{field}"
                side = "SOURCE" if field == "source_evidence" else "CANDIDATE"
                questions[name] = choice_question(
                    f"Criterion: {criteria[criterion]} It failed. Select the {side} excerpt "
                    "that best supports the failure when compared with the other text. "
                    "Choose none if this side contains no useful evidence.",
                    options,
                )
                targets[name] = (check, field)
        if questions:
            for name, selected in self.choices(state, questions, result).items():
                if selected != "none":
                    check, field = targets[name]
                    excerpt = questions[name]["criteria"][selected]
                    if excerpt not in inputs[field]:
                        raise ValueError(f"{name}: evidence does not occur in the input")
                    check[field] = excerpt
        for check in failed:
            if not check["source_evidence"] and not check["candidate_evidence"]:
                check.update(verdict="insufficient_evidence", severity="none",
                             evidence_status="no_excerpt_selected")

    def request(self, criteria, source, candidate, mode):
        started = perf_counter()
        result = {"request_count": 0}
        state = json.dumps(
            {"mode": mode, "source": source, "candidate": candidate}, ensure_ascii=False
        )
        questions = {
            name: choice_question(question, DECISIONS)
            for name, question in criteria.items()
            if name != "fidelity" or source
        }
        fluency = questions.pop("fluency", None)
        try:
            decisions = self.choices(state, questions, result)
            if fluency is not None:
                candidate_only = json.dumps({"candidate": candidate}, ensure_ascii=False)
                decisions.update(self.choices(candidate_only, {"fluency": fluency}, result))
            checks = []
            for name in criteria:
                verdict, severity = VERDICTS[decisions.get(name, "unknown")]
                checks.append({
                    "criterion": name, "verdict": verdict, "severity": severity,
                    "systemone_choice": decisions.get(name),
                    "source_evidence": "", "candidate_evidence": "",
                })
            self.add_evidence(state, criteria, checks, source, candidate, result)
        except httpx.HTTPStatusError as error:
            result.update(status="request_error", error=f"HTTP {error.response.status_code}")
        except httpx.RequestError as error:
            result.update(status="request_error", error=type(error).__name__)
        except ValueError as error:
            result.update(status="invalid_output", error=str(error).splitlines()[0])
        else:
            result.update(status="ok", checks=checks)
        result["seconds"] = perf_counter() - started
        return result

    def score(self, sample, repeat_count):
        modes = ("translation",) if sample["task"] == "translation" else ("ocr_text", "ocr_reference")
        results = {}
        for mode in modes:
            criteria, source, candidate = review_input(sample, mode)
            if not candidate.strip() or (mode != "ocr_text" and not (source or "").strip()):
                results[mode] = {"status": "missing_input", "error": "candidate or required source is empty"}
                continue
            primary = self.request(criteria, source, candidate, mode)
            if repeat_count:
                primary["repeats"] = [
                    self.request(criteria, source, candidate, mode) for _ in range(repeat_count)
                ]
            results[mode] = primary
        return results
