from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_float(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    return parsed


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_score_rows(rows: list[dict[str, Any]], path: str | Path) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_event_id", "user_id", "item_id", "score"]
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    return {
        "score_path": str(target),
        "score_rows": len(rows),
        "score_sha256": sha256_file(target),
        "score_schema": fieldnames,
    }


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def audit_score_rows_against_candidates(
    *,
    candidate_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_keys = {
        (text(row.get("source_event_id")), text(row.get("user_id")), text(row.get("item_id")))
        for row in candidate_rows
        if text(row.get("source_event_id")) and text(row.get("user_id")) and text(row.get("item_id"))
    }
    score_keys: set[tuple[str, str, str]] = set()
    duplicate_score_keys = 0
    invalid_scores = 0
    blank_keys = 0
    for row in score_rows:
        key = (text(row.get("source_event_id")), text(row.get("user_id")), text(row.get("item_id")))
        if not all(key):
            blank_keys += 1
            continue
        score = finite_float(row.get("score"), default=float("nan"))
        if not math.isfinite(score):
            invalid_scores += 1
            continue
        duplicate_score_keys += int(key in score_keys)
        score_keys.add(key)
    missing = candidate_keys - score_keys
    extra = score_keys - candidate_keys
    return {
        "candidate_key_count": len(candidate_keys),
        "score_key_count": len(score_keys),
        "score_coverage_rate": len(candidate_keys & score_keys) / len(candidate_keys) if candidate_keys else 0.0,
        "missing_score_keys": len(missing),
        "extra_score_keys": len(extra),
        "duplicate_score_keys": duplicate_score_keys,
        "invalid_scores": invalid_scores,
        "blank_score_keys": blank_keys,
        "audit_ok": not (missing or extra or duplicate_score_keys or invalid_scores or blank_keys)
        and len(score_rows) == len(candidate_rows),
    }
