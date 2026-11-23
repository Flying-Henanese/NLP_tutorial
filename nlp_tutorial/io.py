"""读取、检查和保存实训数据。

负责读写 JSON 和 JSONL，检查样本 ID、必填字段和同组译文的原文是否一致。
打分和评估共用这里的读写函数，按样本 ID 关联输入、分数和标签。
"""

import json
from pathlib import Path


def read_rows(path):
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                identifier = row.get("id") if isinstance(row, dict) else None
                if not isinstance(identifier, str) or not identifier.strip():
                    raise ValueError("id must be a nonempty string")
                if identifier in rows:
                    raise ValueError(f"duplicate id {identifier!r}")
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            rows[identifier] = row
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def read_samples(path):
    rows = read_rows(path)
    groups = {}
    for identifier, row in rows.items():
        if row.get("task") != "translation":
            raise ValueError(f"{identifier}: task must be translation")
        if any(not isinstance(row.get(field), str)
               for field in ("doc_id", "group_id", "source", "candidate")):
            raise ValueError(f"{identifier}: missing or invalid translation fields")
        if not row["doc_id"].strip() or not row["group_id"].strip():
            raise ValueError(f"{identifier}: empty document or group ID")
        group = (
            row["doc_id"], row["source"], row.get("source_language", "Chinese"),
            row.get("target_language", "English"),
        )
        previous = groups.setdefault(row["group_id"], group)
        if previous != group:
            raise ValueError(f"{identifier}: translation group has inconsistent source or languages")
    return rows


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def append_row(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()
