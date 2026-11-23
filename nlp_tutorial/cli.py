"""实训命令行入口。

score 读取译文样本，调用概率模型，把分数逐条写入 JSONL。
evaluate 把分数和人工标签交给评估模块，生成开发集或测试集的结果。
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from .evaluate import compare
from .io import append_row, read_rows, read_samples, write_json

DEFAULT_MODEL = "/nas_data/models/Qwen/Qwen3.5-4B-Base"


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def check_local_model(model_id):
    path = Path(model_id).expanduser()
    if not path.is_absolute():
        return
    if not (path / "config.json").is_file():
        raise ValueError(f"模型目录未就绪：{path}。请确认 NAS 上的模型已下载完成。")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        weights = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        missing = sorted({name for name in weights.values() if not (path / name).is_file()})
        if missing:
            raise ValueError(f"模型权重仍未下载完整：{path} 缺少 {', '.join(missing)}")


def score(args):
    if args.mismatches and not args.probability:
        raise ValueError("mismatch experiments require --probability")
    samples = read_samples(args.samples)
    alternates = {}
    if args.mismatches:
        for identifier, row in read_rows(args.mismatches).items():
            sample = samples.get(identifier)
            source = row.get("source")
            if (not sample or not isinstance(source, str)
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
    if args.probability:
        check_local_model(args.model)
        try:
            from .probability import ProbabilityScorer
        except ImportError as error:
            raise ValueError("Install the GPU dependencies with uv sync --locked --extra gpu") from error
        scorer = ProbabilityScorer(args.model, args.revision, args.max_tokens, args.device)
        config["probability"] = scorer.metadata | {"transformers": version("transformers")}
    setup_seconds = perf_counter() - started
    with output.open("x", encoding="utf-8") as handle:
        for identifier, sample in samples.items():
            record = {"id": identifier}
            if args.probability:
                record["probability"] = scorer.score(sample, alternates.get(identifier))
            append_row(handle, record)
    write_json(metadata, {
        "configuration": config,
        "candidates": len(samples), "setup_seconds": setup_seconds,
        "elapsed_seconds": perf_counter() - started,
    })
    print(f"Wrote {len(samples)} rows to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    scoring = subparsers.add_parser("score", help="score JSONL without reading human labels")
    scoring.add_argument("--samples", required=True)
    scoring.add_argument("--out", required=True)
    scoring.add_argument("--probability", action="store_true")
    scoring.add_argument("--model", default=DEFAULT_MODEL, help=f"默认读取 {DEFAULT_MODEL}")
    scoring.add_argument("--revision")
    scoring.add_argument("--max-tokens", type=positive, default=512)
    scoring.add_argument("--device", default="cuda")
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
