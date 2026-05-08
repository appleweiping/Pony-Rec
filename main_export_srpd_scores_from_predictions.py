from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.baselines.internal_scores import (
    audit_score_rows_against_candidates,
    finite_float,
    read_csv_rows,
    text,
    write_json,
    write_score_rows,
)
from src.utils.io import load_jsonl


def _rank_score(rank_index: int, count: int) -> float:
    if count <= 1:
        return 1.0
    return float((count - rank_index) / count)


def _score_from_prediction(record: dict[str, Any], item_id: str, rank_lookup: dict[str, int], count: int) -> float:
    candidate_scores = record.get("candidate_scores")
    if isinstance(candidate_scores, dict):
        value = candidate_scores.get(item_id)
        if isinstance(value, dict):
            for key in ("score", "decision_score", "final_score", "confidence"):
                if key in value:
                    return finite_float(value[key])
        if value is not None and not isinstance(value, (list, tuple, dict)):
            return finite_float(value)
    return _rank_score(rank_lookup.get(item_id, count), count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SRPD/framework ranking predictions as exact same-candidate score CSV.")
    parser.add_argument("--ranking_input_path", required=True)
    parser.add_argument("--candidate_items_path", required=True)
    parser.add_argument("--prediction_path", required=True)
    parser.add_argument("--output_scores_path", required=True)
    parser.add_argument("--provenance_output_path", required=True)
    parser.add_argument("--method_variant", default="SRPD")
    parser.add_argument("--status_label", default="same_schema_internal_ablation")
    parser.add_argument("--artifact_class", default="completed_result")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_rows = read_csv_rows(args.candidate_items_path)
    predictions = load_jsonl(args.prediction_path)
    by_event = {text(row.get("source_event_id")): row for row in predictions if text(row.get("source_event_id"))}
    score_rows: list[dict[str, Any]] = []
    used_rank_fallback = 0
    for candidate in candidate_rows:
        event_id = text(candidate.get("source_event_id"))
        user_id = text(candidate.get("user_id"))
        item_id = text(candidate.get("item_id"))
        prediction = by_event.get(event_id)
        if prediction is None:
            score = float("nan")
        else:
            ranked = [text(item) for item in prediction.get("pred_ranked_item_ids", []) if text(item)]
            rank_lookup = {item: idx for idx, item in enumerate(ranked)}
            count = len(ranked) or len(prediction.get("candidate_item_ids", []))
            score = _score_from_prediction(prediction, item_id, rank_lookup, count)
            used_rank_fallback += int(not isinstance(prediction.get("candidate_scores"), dict))
        score_rows.append({"source_event_id": event_id, "user_id": user_id, "item_id": item_id, "score": score})

    audit = audit_score_rows_against_candidates(candidate_rows=candidate_rows, score_rows=score_rows)
    if not audit["audit_ok"]:
        raise ValueError(f"SRPD score coverage audit failed: {audit}")
    score_summary = write_score_rows(score_rows, args.output_scores_path)
    provenance = {
        "method": "srpd",
        "method_variant": args.method_variant,
        "status_label": args.status_label,
        "artifact_class": args.artifact_class,
        "ranking_input_path": args.ranking_input_path,
        "candidate_items_path": args.candidate_items_path,
        "prediction_path": args.prediction_path,
        "score_source": "parsed_candidate_scores_or_rank_order_fallback",
        "rank_order_fallback_events": used_rank_fallback,
        **score_summary,
        **audit,
    }
    write_json(provenance, args.provenance_output_path)
    pd.DataFrame([provenance]).to_csv(Path(args.provenance_output_path).with_suffix(".csv"), index=False)
    print(f"Saved SRPD scores: {args.output_scores_path}")
    print(f"audit_ok={audit['audit_ok']} score_coverage_rate={audit['score_coverage_rate']:.6f}")


if __name__ == "__main__":
    main()
