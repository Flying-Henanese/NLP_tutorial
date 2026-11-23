"""Command-line workflow for scoring and evaluating the practicum."""

import argparse
from collections import Counter
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from .evaluate import compare
from .io import append_row, read_rows, read_samples, write_json
from .rules import score_ocr


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def score(args):
    if args.mismatches and not args.probability:
        raise ValueError("mismatch experiments require --probability")
    if args.repeat_first and not args.judge:
        raise ValueError("repeat scoring requires --judge")
    samples = read_samples(args.samples)
    alternates = {}
    if args.mismatches:
        for identifier, row in read_rows(args.mismatches).items():
            sample = samples.get(identifier)
            source = row.get("source")
            if (not sample or sample["task"] != "translation" or not isinstance(source, str)
                    or not source.strip() or source.strip() == sample["source"].strip()):
                raise ValueError(f"{identifier}: invalid alternate source")
            alternates[identifier] = source

    output = Path(args.out)
    metadata = output.with_name(output.name + ".meta.json")
    if metadata.exists() or output.exists():
        raise ValueError(f"{output} or its metadata already exists. Choose a new output path.")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    config = {}
    with ExitStack() as stack:
        if args.judge:
            import httpx

            from .judge import PROMPT_VERSION, Judge

            client = stack.enter_context(httpx.Client(follow_redirects=False))
            judge = Judge(client, args.timeout)
            config["judge"] = {"requested_model": judge.model, "prompt_version": PROMPT_VERSION}
        if args.probability:
            try:
                from .probability import ProbabilityScorer
            except ImportError as error:
                raise ValueError("Install the GPU dependencies with uv sync --locked --extra gpu") from error
            scorer = ProbabilityScorer(args.model, args.revision, args.max_tokens, args.device)
            config["probability"] = scorer.metadata | {"transformers": version("transformers")}
        setup_seconds = perf_counter() - started
        handle = stack.enter_context(output.open("x", encoding="utf-8"))
        reviewed_tasks = Counter()
        reported_models = set()
        for identifier, sample in samples.items():
            record = {"id": identifier}
            if sample["task"] == "ocr" and sample["candidate"].strip():
                start = perf_counter()
                record["rules"] = score_ocr(sample["candidate"])
                record["rules"]["seconds"] = perf_counter() - start
            if args.probability:
                record["probability"] = scorer.score(sample, alternates.get(identifier))
            if args.judge:
                task = sample["task"]
                repeats = args.repeat_count if reviewed_tasks[task] < args.repeat_first else 0
                reviewed_tasks[task] += 1
                record["judge"] = judge.score(sample, repeats)
                for first in record["judge"].values():
                    for review in (first, *first.get("repeats", [])):
                        if "model" in review:
                            reported_models.add(review["model"])
            append_row(handle, record)
    write_json(metadata, {
        "configuration": config, "reported_judge_models": sorted(reported_models),
        "candidates": len(samples), "setup_seconds": setup_seconds,
        "elapsed_seconds": perf_counter() - started, "repeat_first": args.repeat_first,
        "repeat_count": args.repeat_count if args.repeat_first else 0,
    })
    print(f"Wrote {len(samples)} rows to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    scoring = subparsers.add_parser("score", help="score JSONL without reading human labels")
    scoring.add_argument("--samples", required=True)
    scoring.add_argument("--out", required=True)
    scoring.add_argument("--probability", action="store_true")
    scoring.add_argument("--judge", action="store_true")
    scoring.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    scoring.add_argument("--revision")
    scoring.add_argument("--max-tokens", type=positive, default=512)
    scoring.add_argument("--device", default="cuda")
    scoring.add_argument("--timeout", type=positive, default=60)
    scoring.add_argument("--repeat-first", type=positive, default=0)
    scoring.add_argument("--repeat-count", type=positive, default=1)
    scoring.add_argument("--mismatches", help="JSONL of sample IDs and alternate source strings")
    scoring.set_defaults(run=score)

    evaluation = subparsers.add_parser("evaluate", help="freeze on dev, or evaluate the frozen settings")
    evaluation.add_argument("--samples", required=True)
    evaluation.add_argument("--labels", required=True)
    evaluation.add_argument("--scores", required=True)
    evaluation.add_argument("--split", choices=("dev", "test"), required=True)
    evaluation.add_argument("--settings", required=True)
    evaluation.add_argument("--out", required=True)
    evaluation.add_argument("--budget", type=positive, default=20, help="development setting")
    evaluation.add_argument("--seed", type=int, default=2026, help="development setting")
    evaluation.set_defaults(run=compare)

    args = parser.parse_args()
    try:
        args.run(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
