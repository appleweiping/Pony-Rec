from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.baselines.same_candidate_external import (
    build_predictions_from_external_scores,
    load_score_rows,
)
from src.eval.ranking_task_metrics import (
    build_ranking_eval_frame,
    compute_ranking_exposure_distribution,
    compute_ranking_task_metrics,
)
from src.utils.io import load_jsonl, save_jsonl
from src.utils.paths import ensure_exp_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import external baseline candidate scores and evaluate them under the local ranking schema."
    )
    parser.add_argument("--baseline_name", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--domain", default="")
    parser.add_argument("--ranking_input_path", required=True)
    parser.add_argument("--scores_path", required=True)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--user_col", default="user_id")
    parser.add_argument("--item_col", default="item_id")
    parser.add_argument("--score_col", default="score")
    parser.add_argument("--source_event_col", default="source_event_id")
    parser.add_argument("--missing_score", type=float, default=-1.0e12)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--artifact_class", default="completed_result")
    parser.add_argument("--status_label", default="same_schema_external_baseline")
    parser.add_argument("--min_score_coverage", type=float, default=1.0)
    parser.add_argument(
        "--allow_partial_scores",
        action="store_true",
        help="Allow same_schema_external_baseline imports with score coverage below --min_score_coverage.",
    )
    return parser.parse_args()


def _save_table(rows: list[dict[str, Any]] | pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
    else:
        pd.DataFrame(rows).to_csv(path, index=False)


def normalize_artifact_class(status_label: str, artifact_class: str) -> str:
    status = str(status_label or "").strip()
    artifact = str(artifact_class or "").strip() or "completed_result"
    if "scaffold" in status and artifact == "completed_result":
        return "adapter_scaffold_score"
    return artifact


def main() -> None:
    args = parse_args()
    artifact_class = normalize_artifact_class(args.status_label, args.artifact_class)
    if args.status_label == "same_schema_external_baseline" and "scaffold" in artifact_class:
        raise ValueError(
            "Refusing to import a scaffold artifact as same_schema_external_baseline. "
            "Use a non-main status_label such as llmesr_adapter_scaffold_score."
        )

    paths = ensure_exp_dirs(args.exp_name, args.output_root)
    ranking_samples = load_jsonl(args.ranking_input_path)
    score_rows = load_score_rows(args.scores_path)

    predictions, score_summary = build_predictions_from_external_scores(
        ranking_samples,
        score_rows,
        baseline_name=args.baseline_name,
        user_col=args.user_col,
        item_col=args.item_col,
        score_col=args.score_col,
        source_event_col=args.source_event_col,
        missing_score=args.missing_score,
        k=args.k,
    )
    score_summary["score_rows_loaded"] = len(score_rows)

    coverage = float(score_summary.get("score_coverage_rate", 0.0))
    if (
        args.status_label == "same_schema_external_baseline"
        and not args.allow_partial_scores
        and coverage + 1.0e-12 < args.min_score_coverage
    ):
        raise ValueError(
            "Refusing to write same_schema_external_baseline with partial score coverage: "
            f"score_coverage_rate={coverage:.6f}, required>={args.min_score_coverage:.6f}, "
            f"matched_candidates={score_summary.get('matched_candidates')}, "
            f"total_candidates={score_summary.get('total_candidates')}, "
            f"score_rows_loaded={len(score_rows)}. "
            "Fix the external score file or rerun with --allow_partial_scores and a non-main status label."
        )

    prediction_path = paths.predictions_dir / "rank_predictions.jsonl"
    save_jsonl(predictions, prediction_path)

    eval_df = build_ranking_eval_frame(pd.DataFrame(predictions))
    metrics = compute_ranking_task_metrics(eval_df, k=args.k)
    exposure_df = compute_ranking_exposure_distribution(eval_df, k=args.k)

    result_row = {
        "baseline_name": args.baseline_name,
        "domain": args.domain,
        "comparison_scope": "week8_same_candidate_external_baseline",
        "task": "candidate_ranking",
        "status_label": args.status_label,
        "artifact_class": artifact_class,
        "ranking_input_path": str(args.ranking_input_path),
        "scores_path": str(args.scores_path),
        "prediction_path": str(prediction_path),
        **score_summary,
        **metrics,
    }

    _save_table([metrics], paths.tables_dir / "ranking_metrics.csv")
    _save_table(exposure_df, paths.tables_dir / "ranking_exposure_distribution.csv")
    _save_table(eval_df, paths.tables_dir / "ranking_eval_records.csv")
    _save_table([score_summary], paths.tables_dir / "external_score_coverage.csv")
    _save_table([result_row], paths.tables_dir / "same_candidate_external_baseline_summary.csv")

    print(f"[{args.exp_name}] Saved predictions: {prediction_path}")
    print(f"[{args.exp_name}] Saved tables: {paths.tables_dir}")
    print(
        f"[{args.exp_name}] score_coverage_rate={score_summary['score_coverage_rate']:.6f} "
        f"NDCG@{args.k}={metrics.get(f'NDCG@{args.k}')}"
    )


if __name__ == "__main__":
    main()
