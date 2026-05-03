from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.eval.ranking_task_metrics import build_ranking_eval_frame, compute_ranking_task_metrics
from src.methods.uncertainty_ranker import summarize_rerank_effect
from src.shadow.decision_bridge import (
    build_shadow_v6_bridge_rows,
    build_shadow_v6_decision_predictions,
    rank_shadow_v6_bridge_rows,
    summarize_shadow_v6_bridge_rows,
)
from src.utils.io import load_jsonl, save_jsonl
from src.utils.paths import ensure_exp_dirs
from src.utils.reproducibility import set_global_seed


def _save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _result_row(
    *,
    method: str,
    predictions: list[dict[str, Any]],
    k: int,
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    df = pd.DataFrame(predictions)
    metrics = compute_ranking_task_metrics(build_ranking_eval_frame(df), k=k)
    row: dict[str, Any] = {"method": method}
    row.update(metrics)
    if extra_metrics:
        row.update(extra_metrics)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the shadow_v6 decision bridge from a winner shadow signal and an anchor ranking."
    )
    parser.add_argument("--exp_name", required=True, help="Output experiment name for the v6 bridge.")
    parser.add_argument("--rank_input_path", required=True, help="Anchor rank_predictions.jsonl path.")
    parser.add_argument("--signal_input_path", required=True, help="Winner-signal calibrated test jsonl path.")
    parser.add_argument("--output_root", default="outputs", help="Experiment output root.")
    parser.add_argument("--winner_signal_variant", default="shadow_v1", help="Winner signal feeding the bridge.")
    parser.add_argument("--signal_score_col", default="shadow_calibrated_score")
    parser.add_argument("--signal_uncertainty_col", default="shadow_uncertainty")
    parser.add_argument("--gate_threshold", type=float, default=0.15)
    parser.add_argument("--uncertainty_threshold", type=float, default=0.65)
    parser.add_argument("--anchor_conflict_penalty", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    paths = ensure_exp_dirs(args.exp_name, args.output_root)
    rank_input_path = Path(args.rank_input_path)
    signal_input_path = Path(args.signal_input_path)
    if not rank_input_path.exists():
        raise FileNotFoundError(f"Anchor ranking file not found: {rank_input_path}")
    if not signal_input_path.exists():
        raise FileNotFoundError(f"Winner-signal file not found: {signal_input_path}")

    print(f"[{args.exp_name}] Loading anchor ranking from: {rank_input_path}")
    ranking_records = load_jsonl(rank_input_path)
    print(f"[{args.exp_name}] Loaded {len(ranking_records)} ranking events.")

    print(f"[{args.exp_name}] Loading winner signal from: {signal_input_path}")
    signal_records = load_jsonl(signal_input_path)
    print(f"[{args.exp_name}] Loaded {len(signal_records)} signal rows.")

    bridge_rows = build_shadow_v6_bridge_rows(
        ranking_records,
        signal_records,
        winner_signal_variant=args.winner_signal_variant,
        signal_score_col=args.signal_score_col,
        signal_uncertainty_col=args.signal_uncertainty_col,
        gate_threshold=args.gate_threshold,
        uncertainty_threshold=args.uncertainty_threshold,
        anchor_conflict_penalty=args.anchor_conflict_penalty,
    )
    ranked_rows = rank_shadow_v6_bridge_rows(bridge_rows)
    bridge_predictions = build_shadow_v6_decision_predictions(
        ranked_rows,
        ranking_records,
        topk=args.k,
    )

    save_jsonl(ranked_rows, paths.reranked_dir / "shadow_v6_bridge_rows.jsonl")
    save_jsonl(bridge_predictions, paths.reranked_dir / "shadow_v6_decision_reranked.jsonl")
    _save_table(pd.DataFrame(ranked_rows), paths.tables_dir / "shadow_v6_bridge_rows.csv")

    bridge_summary: dict[str, Any] = summarize_shadow_v6_bridge_rows(ranked_rows)
    bridge_summary.update(
        {
            "winner_signal_variant": args.winner_signal_variant,
            "gate_threshold": float(args.gate_threshold),
            "uncertainty_threshold": float(args.uncertainty_threshold),
            "anchor_conflict_penalty": float(args.anchor_conflict_penalty),
        }
    )
    _save_table(pd.DataFrame([bridge_summary]), paths.tables_dir / "shadow_v6_bridge_summary.csv")

    baseline_row = _result_row(
        method="direct_candidate_ranking",
        predictions=ranking_records,
        k=args.k,
        extra_metrics={
            "changed_ranking_fraction": 0.0,
            "avg_position_shift": 0.0,
            "matched_signal_rate": float("nan"),
            "fallback_rate": float("nan"),
            "mean_correction_gate": float("nan"),
            "mean_pair_weight": float("nan"),
        },
    )
    effect_metrics = summarize_rerank_effect(pd.DataFrame(ranking_records), pd.DataFrame(bridge_predictions))
    bridge_row = _result_row(
        method="shadow_v6_decision_bridge",
        predictions=bridge_predictions,
        k=args.k,
        extra_metrics={**effect_metrics, **bridge_summary},
    )
    results_df = pd.DataFrame([baseline_row, bridge_row])
    _save_table(results_df, paths.tables_dir / "rerank_results.csv")

    print(f"[{args.exp_name}] Built {len(ranked_rows)} v6 bridge rows.")
    print(f"[{args.exp_name}] Saved bridge rows to: {paths.reranked_dir / 'shadow_v6_bridge_rows.jsonl'}")
    print(f"[{args.exp_name}] Saved decision predictions to: {paths.reranked_dir / 'shadow_v6_decision_reranked.jsonl'}")
    print(f"[{args.exp_name}] Saved summary tables to: {paths.tables_dir}")


if __name__ == "__main__":
    main()
