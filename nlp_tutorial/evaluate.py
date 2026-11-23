"""用人工标签比较各个打分方法的复查效果。

按 ID 对齐分数和标签，检查同一原始文档是否跨越开发集和测试集。
在开发集确定排序方向，在测试集按固定预算统计命中数、关键错误召回率
和译文配对胜率，并输出汇总 JSON 和逐条 CSV。
"""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from .io import read_rows, read_samples, write_json

METHODS = ("random", "ppl", "conditional", "ifd", "d3_flash")
TIE_ABS = 1e-9


def read_labels(samples, path):
    labels = read_rows(path)
    if labels.keys() != samples.keys():
        raise ValueError("samples and labels must contain identical IDs")
    document_splits = {}
    for identifier, label in labels.items():
        split = label.get("split")
        if split not in ("dev", "test") or label.get("origin") not in ("real", "synthetic"):
            raise ValueError(f"{identifier}: invalid split or origin")
        error, critical = label.get("error"), label.get("critical")
        if type(error) is not bool or type(critical) is not bool or critical and not error:
            raise ValueError(f"{identifier}: invalid error or critical label")
        if error and not all(
            isinstance(label.get(field), str) and label[field].strip()
            for field in ("error_type", "evidence")
        ):
            raise ValueError(f"{identifier}: error labels need error_type and evidence")
        doc_id = samples[identifier]["doc_id"]
        if document_splits.setdefault(doc_id, split) != split:
            raise ValueError(f"{doc_id}: document appears in both splits")
    return labels


def score_data(record, method):
    probability = record.get("probability", {})
    direct = probability.get("direct", {})
    conditional = probability.get("conditional", {})
    if method == "random":
        return 0, {}
    if method == "d3_flash":
        decision = record.get("d3_flash", {})
        return decision.get("risk") if decision.get("status") == "ok" else None, decision
    if method == "ppl":
        return probability.get("ppl"), direct
    if method == "conditional":
        return conditional.get("nll"), conditional
    if method == "ifd":
        timings = [x["seconds"] for x in (direct, conditional) if "seconds" in x]
        memory = [x["peak_allocated_bytes"] for x in (direct, conditional)
                  if x.get("peak_allocated_bytes") is not None]
        return conditional.get("ifd"), {
            "seconds": sum(timings) if timings else None,
            "peak_allocated_bytes": max(memory, default=None),
        }


def tie_key(identifier, seed):
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()


def pairs(ids, samples, labels, scores):
    groups = defaultdict(list)
    for identifier in ids:
        groups[samples[identifier]["group_id"]].append(identifier)
    counts = Counter()
    for group in groups.values():
        good = [i for i in group if not labels[i]["error"]]
        bad = [i for i in group if labels[i]["error"]]
        for correct, wrong in product(good, bad):
            a, b = scores[correct], scores[wrong]
            if a is None or b is None:
                counts["unscored"] += 1
                continue
            for left, right in zip(a, b, strict=True):
                if left is None or right is None:
                    counts["unscored"] += 1
                    break
                if not math.isclose(left, right, rel_tol=0, abs_tol=TIE_ABS):
                    counts["win" if left < right else "loss"] += 1
                    break
            else:
                counts["tie"] += 1
    scored = counts["win"] + counts["loss"] + counts["tie"]
    return {key: counts[key] for key in ("win", "loss", "tie", "unscored")} | {
        "win_rate": counts["win"] / scored if scored else None,
        "tie_rate": counts["tie"] / scored if scored else None,
    }


def review_counts(selected, eligible, labels):
    errors = sum(labels[i]["error"] for i in selected)
    critical = sum(labels[i]["critical"] for i in selected)
    available = sum(labels[i]["critical"] for i in eligible)
    return {
        "reviewed": len(selected),
        "true_errors": errors,
        "critical_errors": critical,
        "critical_recall": critical / available if available else None,
        "reviews_per_true_error": len(selected) / errors if errors else None,
    }


def summarize(ids, scores, details, samples, labels, budget, seed):
    # Missing scores always sort last. The same seeded tie order applies to every method.
    queue = sorted(ids, key=lambda i: (
        scores[i] is None,
        tuple(-value if value is not None else math.inf for value in (scores[i] or ())),
        tie_key(i, seed),
    ))
    selected = queue[:budget]
    report = review_counts(selected, ids, labels)
    report["by_origin"] = {
        origin: review_counts(
            [i for i in selected if labels[i]["origin"] == origin],
            [i for i in ids if labels[i]["origin"] == origin], labels,
        )
        for origin in ("real", "synthetic")
    }
    report["unscored"] = sum(scores[i] is None for i in ids)
    report["success_rate"] = 1 - report["unscored"] / len(ids)
    report["unscored_reviewed"] = sum(scores[i] is None for i in selected)
    report["pairs"] = pairs(ids, samples, labels, scores)
    timings = [d["seconds"] for d in details if d.get("seconds") is not None]
    memory = [d["peak_allocated_bytes"] for d in details if d.get("peak_allocated_bytes") is not None]
    if timings:
        report["measured_seconds"] = sum(timings)
        report["seconds_per_100_candidates"] = sum(timings) * 100 / len(ids)
    if memory:
        report["peak_allocated_bytes"] = max(memory)
    if any("status" in d for d in details):
        report["status_counts"] = dict(Counter(d.get("status", "missing_result") for d in details))
        report["insufficient_evidence_candidates"] = sum(
            any(check["verdict"] == "insufficient_evidence" for check in d.get("checks", []))
            for d in details
        )
    return report, queue


def method_scores(ids, records, method, direction, seed):
    scores = {}
    details = []
    for identifier in ids:
        record = records[identifier]
        value, detail = score_data(record, method)
        if method == "random":
            value = int(tie_key(identifier, seed)[:13], 16)
        details.append(detail)
        scores[identifier] = None if value is None else (value * direction,)
    return scores, details


def mismatch_check(ids, records):
    counts = Counter()
    for identifier in ids:
        record = records[identifier].get("probability", {})
        if "mismatch" not in record and "mismatch_error" not in record:
            continue
        counts["attempted"] += 1
        matched, alternate = record.get("conditional", {}), record.get("mismatch", {})
        if "ifd" not in matched or "ifd" not in alternate:
            counts["unscored"] += 1
            continue
        nll_order = (alternate["nll"] > matched["nll"]) - (alternate["nll"] < matched["nll"])
        ifd_order = (alternate["ifd"] > matched["ifd"]) - (alternate["ifd"] < matched["ifd"])
        counts["order_agrees"] += nll_order == ifd_order
        counts["matched_less_nll"] += nll_order > 0
        counts["ties"] += nll_order == 0
    return dict(counts)


def enabled_method(method, configuration):
    if method == "random":
        return True
    if method == "d3_flash":
        return "d3_flash" in configuration
    return "probability" in configuration


def compare(args):
    samples = read_samples(args.samples)
    labels = read_labels(samples, args.labels)
    records = read_rows(args.scores)
    if not records.keys() <= samples.keys():
        raise ValueError("score result contains unknown sample IDs")
    score_path = Path(args.scores)
    metadata_path = score_path.with_name(score_path.name + ".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    settings_path = Path(args.settings)
    if args.split == "dev":
        if settings_path.exists():
            raise ValueError("development settings already exist; choose a new path")
        settings = {
            "budget": args.budget, "seed": args.seed, "tasks": {},
            "configuration": metadata["configuration"],
        }
    else:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if settings["configuration"] != metadata["configuration"]:
            raise ValueError("test score configuration differs from frozen development configuration")

    result = {
        "split": args.split, "budget": settings["budget"], "tasks": {},
        "scoring_run": metadata,
    }
    rows = []
    task = "translation"
    ids = [i for i in samples if labels[i]["split"] == args.split]
    if not ids:
        raise ValueError(f"no samples in {args.split} split")
    if not set(ids) <= records.keys():
        raise ValueError(f"{task}: scores missing for {len(set(ids) - records.keys())} IDs")
    if args.split == "test" and task not in settings["tasks"]:
        raise ValueError(f"{task}: no frozen development settings")
    task_settings = settings["tasks"].get(task, {"directions": {}})
    reports = {}
    for method in METHODS:
        if not enabled_method(method, settings["configuration"]):
            continue
        directions = (1,)
        if args.split == "test":
            if method not in task_settings["directions"]:
                continue
            directions = (task_settings["directions"][method],)
        elif method in {"ppl", "conditional", "ifd"}:
            directions = (1, -1)
        outcomes = []
        for direction in directions:
            scores, details = method_scores(ids, records, method, direction, settings["seed"])
            stat, queue = summarize(ids, scores, details, samples, labels,
                                    settings["budget"], settings["seed"])
            outcomes.append((stat, queue, scores, direction))
        best = max(outcomes, key=lambda x: (
            x[0]["pairs"]["win_rate"] or 0,
            x[0]["critical_errors"], x[0]["true_errors"],
        ))
        reports[method] = best
        task_settings["directions"][method] = best[3]
    if args.split == "dev":
        task_settings["best_single"] = max(
            (name for name in reports if reports[name][0]["success_rate"]),
            key=lambda name: (
                reports[name][0]["critical_errors"], reports[name][0]["true_errors"],
            ),
        )
        settings["tasks"][task] = task_settings

    result["tasks"][task] = {
        "methods": {name: outcome[0] for name, outcome in reports.items()},
        "best_single_on_dev": task_settings["best_single"],
    }
    result["tasks"][task]["mismatch_check"] = mismatch_check(ids, records)
    for name, (_, queue, scores, _) in reports.items():
        for rank, identifier in enumerate(queue, 1):
            sample, label = samples[identifier], labels[identifier]
            reviewed = rank <= settings["budget"]
            rows.append({
                "id": identifier, "task": task, "method": name, "rank": rank,
                "scores": json.dumps(scores[identifier]), "reviewed": reviewed,
                "outcome": ("error_found" if label["error"] else "false_alarm") if reviewed
                else ("missed_error" if label["error"] else "clean_outside_budget"),
                "doc_id": sample["doc_id"], "location": json.dumps(sample.get("location"), ensure_ascii=False),
                "origin": label["origin"], "critical": label["critical"],
                "error_type": label.get("error_type", ""), "evidence": label.get("evidence", ""),
            })
    if not rows:
        raise ValueError(f"no samples in {args.split} split")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("summary.json", "per_sample.csv")):
        raise ValueError("evaluation outputs already exist; choose a new output directory")
    if args.split == "dev":
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(settings_path, settings)
    write_json(output / "summary.json", result)
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
