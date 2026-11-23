"""调用 d3-flash，按固定选项逐项评审候选译文。"""

import json
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen

URL = "http://192.168.0.213:18892/v1/systemone"
MODEL = "vllm-sr/d3-flash"
TIMEOUT_SECONDS = 30

QUESTIONS = {
    "facts": "Are the key facts consistent with the source?",
    "number_and_unit": "Are numbers and units correct? If neither text has any, this check passes.",
    "negation_and_terms": "Does the translation preserve negation and terminology?",
    "omission": "Is all meaningful source content present? A changed but present detail is not an omission.",
    "fluency": "Is the candidate natural and grammatical? Assess wording only.",
}
OPTIONS = {
    "pass": "This check passes.",
    "minor": "This check fails with an issue that does not materially affect meaning.",
    "major": "This check fails with an error that materially affects meaning or usability.",
    "unknown": "The supplied text is insufficient to make this judgment.",
}
# 数值用于复查排序。风险相同时，评估脚本沿用固定的平局规则。
CHOICES = {
    "pass": ("pass", "none", 0),
    "unknown": ("insufficient_evidence", "none", 1),
    "minor": ("fail", "minor", 2),
    "major": ("fail", "major", 3),
}


def configuration():
    return {"model": MODEL, "questions": QUESTIONS, "options": OPTIONS, "candidate_only": ["fluency"]}


def decide(sample):
    if not sample["source"].strip() or not sample["candidate"].strip():
        return {"status": "missing_input", "error": "source or candidate is empty"}

    state = {"source": sample["source"], "candidate": sample["candidate"]}
    bilingual = {name: text for name, text in QUESTIONS.items() if name != "fluency"}
    started = perf_counter()
    try:
        answers = ask(state, bilingual)
        # 通顺度只看译文，避免把事实错译当作语言生硬。
        answers.update(ask({"candidate": sample["candidate"]}, {"fluency": QUESTIONS["fluency"]}))
        result = interpret(answers)
    except HTTPError as error:
        result = {"status": "request_error", "error": f"HTTP {error.code}"}
    except OSError as error:
        result = {"status": "request_error", "error": type(error).__name__}
    except ValueError as error:
        result = {"status": "invalid_output", "error": str(error)}
    result["seconds"] = perf_counter() - started
    return result


def ask(state, criteria):
    questions = {
        name: {"type": "choice", "instructions": instruction, "criteria": OPTIONS}
        for name, instruction in criteria.items()
    }
    payload = {"model": MODEL, "state": json.dumps(state, ensure_ascii=False), "questions": questions}
    request = Request(
        URL, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or payload.get("model") != MODEL:
        raise ValueError("响应没有返回预期的模型标识")
    answers = payload.get("answers")
    if not isinstance(answers, dict) or answers.keys() != criteria.keys():
        raise ValueError("响应的判项与请求不一致")
    return answers


def interpret(answers):
    """把 SystemOne 的选择转换为判项和供排序使用的风险分。"""
    checks = []
    for criterion in QUESTIONS:
        answer = answers[criterion]
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError(f"{criterion}: 缺少 choice 类型的答案")
        choice = answer.get("choice")
        if not isinstance(choice, str) or choice not in CHOICES:
            raise ValueError(f"{criterion}: 模型返回了选项之外的答案")
        verdict, severity, risk = CHOICES[choice]
        checks.append({
            "criterion": criterion, "choice": choice, "verdict": verdict,
            "severity": severity, "risk": risk,
        })
    return {
        "status": "ok", "model": MODEL, "checks": checks,
        "risk": max(check["risk"] for check in checks),
    }
