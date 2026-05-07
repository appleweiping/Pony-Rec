from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd

from src.eval.ranking_task_metrics import build_ranking_eval_frame, compute_ranking_task_metrics
from src.shadow.decision_bridge import (
    build_shadow_v6_bridge_rows,
    build_shadow_v6_decision_predictions,
    rank_shadow_v6_bridge_rows,
    summarize_shadow_v6_bridge_rows,
)
from src.utils.io import load_jsonl, save_jsonl


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _metric_row(
    *,
    split_name: str,
    predictions: list[dict[str, Any]],
    bridge_rows: list[dict[str, Any]],
    k: int,
    gate_threshold: float,
    uncertainty_threshold: float,
    anchor_conflict_penalty: float,
    winner_signal_variant: str,
) -> dict[str, Any]:
    metrics = compute_ranking_task_metrics(build_ranking_eval_frame(pd.DataFrame(predictions)), k=k)
    summary = summarize_shadow_v6_bridge_rows(bridge_rows)
    row: dict[str, Any] = {
        "split_name": split_name,
        "winner_signal_variant": winner_signal_variant,
        "gate_threshold": gate_threshold,
        "uncertainty_threshold": uncertainty_threshold,
        "anchor_conflict_penalty": anchor_conflict_penalty,
    }
    row.update(metrics)
    row.update(summary)
    return row


def _run_bridge(
    *,
    ranking_records: list[dict[str, Any]],
    signal_records: list[dict[str, Any]],
    winner_signal_variant: str,
    signal_score_col: str,
    signal_uncertainty_col: str,
    gate_threshold: float,
    uncertainty_threshold: float,
    anchor_conflict_penalty: float,
    k: int,
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bridge_rows = build_shadow_v6_bridge_rows(
        ranking_records,
        signal_records,
        winner_signal_variant=winner_signal_variant,
        signal_score_col=signal_score_col,
        signal_uncertainty_col=signal_uncertainty_col,
        gate_threshold=gate_threshold,
        uncertainty_threshold=uncertainty_threshold,
        anchor_conflict_penalty=anchor_conflict_penalty,
        metadata=metadata,
    )
    ranked_rows = rank_shadow_v6_bridge_rows(bridge_rows)
    predictions = build_shadow_v6_decision_predictions(ranked_rows, ranking_records, topk=k)
    return ranked_rows, predictions


def _write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select shadow_v6 gate thresholds on valid and evaluate once on test."
    )
    parser.add_argument("--valid_rank_input_path", required=True)
    parser.add_argument("--test_rank_input_path", required=True)
    parser.add_argument("--valid_signal_input_path", required=True)
    parser.add_argument("--test_signal_input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--winner_signal_variant", default="shadow_v1")
    parser.add_argument("--signal_score_col", default="shadow_calibrated_score")
    parser.add_argument("--signal_uncertainty_col", default="shadow_uncertainty")
    parser.add_argument("--gate_thresholds", default="0.05,0.1,0.15,0.2,0.3")
    parser.add_argument("--uncertainty_thresholds", default="0.5,0.6,0.65,0.7,0.8")
    parser.add_argument("--anchor_conflict_penalties", default="0.25,0.5,0.75")
    parser.add_argument("--selection_metric", default="NDCG@10", choices=["NDCG@10", "MRR", "HR@10"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--artifact_class", default="diagnostic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_ranking = load_jsonl(args.valid_rank_input_path)
    test_ranking = load_jsonl(args.test_rank_input_path)
    valid_signal = load_jsonl(args.valid_signal_input_path)
    test_signal = load_jsonl(args.test_signal_input_path)

    valid_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_bridge_rows: list[dict[str, Any]] = []
    best_predictions: list[dict[str, Any]] = []

    metadata = {
        "domain": args.domain,
        "method": "shadow_v6_validation_selected_bridge",
        "artifact_class": args.artifact_class,
        "is_paper_result": args.artifact_class == "paper-result",
    }

    for gate_threshold in _parse_float_list(args.gate_thresholds):
        for uncertainty_threshold in _parse_float_list(args.uncertainty_thresholds):
            for penalty in _parse_float_list(args.anchor_conflict_penalties):
                bridge_rows, predictions = _run_bridge(
                    ranking_records=valid_ranking,
                    signal_records=valid_signal,
                    winner_signal_variant=args.winner_signal_variant,
                    signal_score_col=args.signal_score_col,
                    signal_uncertainty_col=args.signal_uncertainty_col,
                    gate_threshold=gate_threshold,
                    uncertainty_threshold=uncertainty_threshold,
                    anchor_conflict_penalty=penalty,
                    k=args.k,
                    metadata=metadata,
                )
                row = _metric_row(
                    split_name="valid",
                    predictions=predictions,
                    bridge_rows=bridge_rows,
                    k=args.k,
                    gate_threshold=gate_threshold,
                    uncertainty_threshold=uncertainty_threshold,
                    anchor_conflict_penalty=penalty,
                    winner_signal_variant=args.winner_signal_variant,
                )
                valid_rows.append(row)
                if best_row is None or float(row[args.selection_metric]) > float(best_row[args.selection_metric]):
                    best_row = row
                    best_bridge_rows = bridge_rows
                    best_predictions = predictions

    if best_row is None:
        raise RuntimeError("No valid sweep rows were produced.")

    _write_csv(valid_rows, output_dir / "valid_gate_sweep.csv")
    save_jsonl(best_bridge_rows, output_dir / "valid_selected_bridge_rows.jsonl")
    save_jsonl(best_predictions, output_dir / "valid_selected_predictions.jsonl")

    test_bridge_rows, test_predictions = _run_bridge(
        ranking_records=test_ranking,
        signal_records=test_signal,
        winner_signal_variant=args.winner_signal_variant,
        signal_score_col=args.signal_score_col,
        signal_uncertainty_col=args.signal_uncertainty_col,
        gate_threshold=float(best_row["gate_threshold"]),
        uncertainty_threshold=float(best_row["uncertainty_threshold"]),
        anchor_conflict_penalty=float(best_row["anchor_conflict_penalty"]),
        k=args.k,
        metadata=metadata,
    )
    test_row = _metric_row(
        split_name="test",
        predictions=test_predictions,
        bridge_rows=test_bridge_rows,
        k=args.k,
        gate_threshold=float(best_row["gate_threshold"]),
        uncertainty_threshold=float(best_row["uncertainty_threshold"]),
        anchor_conflict_penalty=float(best_row["anchor_conflict_penalty"]),
        winner_signal_variant=args.winner_signal_variant,
    )
    test_row["selected_on"] = "valid"
    test_row["selection_metric"] = args.selection_metric
    test_row[f"selected_valid_{args.selection_metric}"] = best_row[args.selection_metric]

    _write_csv([best_row], output_dir / "selected_valid_gate.csv")
    _write_csv([test_row], output_dir / "selected_gate_test_metrics.csv")
    save_jsonl(test_bridge_rows, output_dir / "test_selected_bridge_rows.jsonl")
    save_jsonl(test_predictions, output_dir / "test_selected_predictions.jsonl")

    print(f"Saved valid sweep: {output_dir / 'valid_gate_sweep.csv'}")
    print(f"Saved selected test metrics: {output_dir / 'selected_gate_test_metrics.csv'}")


if __name__ == "__main__":
    main()
