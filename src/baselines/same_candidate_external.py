from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def load_score_rows(path: str | Path) -> list[dict[str, Any]]:
    score_path = Path(path)
    if not score_path.exists():
        raise FileNotFoundError(f"External score file not found: {score_path}")

    suffix = score_path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with score_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if suffix == ".csv":
        with score_path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    raise ValueError(f"Unsupported external score file suffix: {score_path.suffix}")


def build_score_lookup(
    score_rows: list[dict[str, Any]],
    *,
    user_col: str = "user_id",
    item_col: str = "item_id",
    score_col: str = "score",
    source_event_col: str = "source_event_id",
) -> tuple[dict[tuple[str, str, str], float], dict[tuple[str, str], float]]:
    event_lookup: dict[tuple[str, str, str], float] = {}
    user_item_lookup: dict[tuple[str, str], float] = {}

    for row in score_rows:
        user_id = _text(row.get(user_col))
        item_id = _text(row.get(item_col))
        source_event_id = _text(row.get(source_event_col))
        if not user_id or not item_id:
            continue
        try:
            score = float(row.get(score_col))
        except Exception:
            continue
        if not math.isfinite(score):
            continue
        user_item_lookup[(user_id, item_id)] = score
        if source_event_id:
            event_lookup[(source_event_id, user_id, item_id)] = score

    return event_lookup, user_item_lookup


def build_predictions_from_external_scores(
    ranking_samples: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    *,
    baseline_name: str,
    user_col: str = "user_id",
    item_col: str = "item_id",
    score_col: str = "score",
    source_event_col: str = "source_event_id",
    missing_score: float = -1.0e12,
    k: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_lookup, user_item_lookup = build_score_lookup(
        score_rows,
        user_col=user_col,
        item_col=item_col,
        score_col=score_col,
        source_event_col=source_event_col,
    )

    predictions: list[dict[str, Any]] = []
    total_candidates = 0
    matched_candidates = 0
    fully_scored_events = 0

    for sample in ranking_samples:
        user_id = _text(sample.get("user_id"))
        source_event_id = _text(sample.get("source_event_id"))
        candidate_ids = [_text(item_id) for item_id in sample.get("candidate_item_ids", [])]
        candidate_ids = [item_id for item_id in candidate_ids if item_id]
        popularity_groups = [str(group).strip().lower() for group in sample.get("candidate_popularity_groups", [])]

        scored_items: list[tuple[str, float, int, bool]] = []
        event_matched = 0
        for idx, item_id in enumerate(candidate_ids):
            event_key = (source_event_id, user_id, item_id)
            user_item_key = (user_id, item_id)
            if event_key in event_lookup:
                score = event_lookup[event_key]
                matched = True
            elif user_item_key in user_item_lookup:
                score = user_item_lookup[user_item_key]
                matched = True
            else:
                score = missing_score
                matched = False
            event_matched += int(matched)
            scored_items.append((item_id, score, idx, matched))

        total_candidates += len(candidate_ids)
        matched_candidates += event_matched
        if candidate_ids and event_matched == len(candidate_ids):
            fully_scored_events += 1

        ranked_ids = [item_id for item_id, _, _, _ in sorted(scored_items, key=lambda item: (-item[1], item[2]))]
        candidate_scores = {item_id: score for item_id, score, _, matched in scored_items if matched}

        predictions.append(
            {
                "user_id": sample.get("user_id"),
                "source_event_id": sample.get("source_event_id"),
                "split_name": sample.get("split_name"),
                "timestamp": sample.get("timestamp"),
                "positive_item_id": sample.get("positive_item_id"),
                "candidate_item_ids": candidate_ids,
                "candidate_titles": sample.get("candidate_titles", []),
                "candidate_texts": sample.get("candidate_texts", []),
                "candidate_popularity_groups": popularity_groups,
                "pred_ranked_item_ids": ranked_ids,
                "topk_item_ids": ranked_ids[:k],
                "parse_success": True,
                "latency": 0.0,
                "confidence": -1.0,
                "contains_out_of_candidate_item": False,
                "raw_response": baseline_name,
                "candidate_scores": candidate_scores,
                "external_score_coverage": float(event_matched / len(candidate_ids)) if candidate_ids else 0.0,
                "external_missing_score_count": int(len(candidate_ids) - event_matched),
            }
        )

    summary = {
        "baseline_name": baseline_name,
        "ranking_events": len(ranking_samples),
        "total_candidates": total_candidates,
        "matched_candidates": matched_candidates,
        "score_coverage_rate": float(matched_candidates / total_candidates) if total_candidates else 0.0,
        "fully_scored_event_rate": float(fully_scored_events / len(ranking_samples)) if ranking_samples else 0.0,
        "missing_score": missing_score,
    }
    return predictions, summary


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
