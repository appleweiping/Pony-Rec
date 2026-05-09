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


def audit_score_degeneracy(
    score_rows: list[dict[str, Any]],
    *,
    precision: int = 12,
    max_tie_pair_rate: float = 0.98,
) -> dict[str, Any]:
    by_event: dict[str, list[float]] = {}
    for row in score_rows:
        event_id = text(row.get("source_event_id"))
        if not event_id:
            continue
        score = finite_float(row.get("score"), default=float("nan"))
        if math.isfinite(score):
            by_event.setdefault(event_id, []).append(score)

    event_count = len(by_event)
    constant_event_count = 0
    tie_pair_count = 0
    total_pair_count = 0
    unique_counts: list[int] = []
    for scores in by_event.values():
        rounded = [round(score, precision) for score in scores]
        unique_values = set(rounded)
        unique_counts.append(len(unique_values))
        constant_event_count += int(len(scores) > 1 and len(unique_values) <= 1)
        counts: dict[float, int] = {}
        for value in rounded:
            counts[value] = counts.get(value, 0) + 1
        total_pair_count += max(0, len(scores) * (len(scores) - 1) // 2)
        tie_pair_count += sum(max(0, count * (count - 1) // 2) for count in counts.values())

    tie_pair_rate = tie_pair_count / total_pair_count if total_pair_count else 0.0
    degeneracy_audit_ok = event_count > 0 and constant_event_count == 0 and tie_pair_rate <= max_tie_pair_rate
    return {
        "score_degeneracy_event_count": event_count,
        "constant_score_event_count": constant_event_count,
        "constant_score_event_rate": constant_event_count / event_count if event_count else 0.0,
        "mean_unique_scores_per_event": sum(unique_counts) / event_count if event_count else 0.0,
        "min_unique_scores_per_event": min(unique_counts) if unique_counts else 0,
        "tie_pair_count": tie_pair_count,
        "total_score_pair_count": total_pair_count,
        "tie_pair_rate": tie_pair_rate,
        "max_tie_pair_rate": max_tie_pair_rate,
        "degeneracy_audit_ok": degeneracy_audit_ok,
    }
