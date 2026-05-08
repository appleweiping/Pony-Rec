from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from src.baselines.same_candidate_external import (
    build_predictions_from_external_scores,
    load_score_rows,
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
    parser.add_argument("--fairness_policy_id", default="")
    parser.add_argument("--comparison_variant", default="")
    parser.add_argument("--implementation_status", default="")
    parser.add_argument("--provenance_path", default="")
    parser.add_argument(
        "--require_fairness_provenance",
        action="store_true",
        help="Require and validate official-baseline fairness provenance before import.",
    )
    parser.add_argument(
        "--allow_partial_scores",
        action="store_true",
        help="Allow same_schema_external_baseline imports with score coverage below --min_score_coverage.",
    )
    return parser.parse_args()


def _save_table(rows: Any, path: Path) -> None:
    import pandas as pd

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


def comparison_scope_for_status(status_label: str) -> str:
    status = str(status_label or "").strip()
    if status.startswith("same_schema_internal_"):
        return "week8_same_candidate_internal_method"
    return "week8_same_candidate_external_baseline"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in provenance file: {path}")
    return data


def _as_schema(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    return []


def _is_official_main_row(args: argparse.Namespace, artifact_class: str) -> bool:
    return (
        "_official_" in str(args.baseline_name)
        and args.status_label == "same_schema_external_baseline"
        and artifact_class == "completed_result"
    )


def _candidate_keys_from_ranking(ranking_samples: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    duplicate_count = 0
    for sample in ranking_samples:
        source_event_id = _text(sample.get("source_event_id"))
        user_id = _text(sample.get("user_id"))
        for item_id_value in sample.get("candidate_item_ids", []):
            item_id = _text(item_id_value)
            if not source_event_id or not user_id or not item_id:
                continue
            key = (source_event_id, user_id, item_id)
            duplicate_count += int(key in keys)
            keys.add(key)
    if duplicate_count:
        raise ValueError(f"Ranking input has duplicate candidate keys: duplicate_count={duplicate_count}")
    return keys


def _validate_exact_score_file(
    *,
    ranking_samples: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    source_event_col: str,
    user_col: str,
    item_col: str,
    score_col: str,
) -> dict[str, Any]:
    candidate_keys = _candidate_keys_from_ranking(ranking_samples)
    score_keys: set[tuple[str, str, str]] = set()
    duplicate_score_keys = 0
    invalid_score_rows = 0
    blank_key_rows = 0

    for row in score_rows:
        source_event_id = _text(row.get(source_event_col))
        user_id = _text(row.get(user_col))
        item_id = _text(row.get(item_col))
        if not source_event_id or not user_id or not item_id:
            blank_key_rows += 1
            continue
        try:
            score = float(row.get(score_col))
        except Exception:
            invalid_score_rows += 1
            continue
        if not math.isfinite(score):
            invalid_score_rows += 1
            continue
        key = (source_event_id, user_id, item_id)
        duplicate_score_keys += int(key in score_keys)
        score_keys.add(key)

    missing_keys = candidate_keys - score_keys
    extra_keys = score_keys - candidate_keys
    summary = {
        "candidate_key_count": len(candidate_keys),
        "score_key_count": len(score_keys),
        "duplicate_score_key_count": duplicate_score_keys,
        "blank_score_key_rows": blank_key_rows,
        "invalid_score_rows": invalid_score_rows,
        "missing_candidate_score_keys": len(missing_keys),
        "extra_score_keys": len(extra_keys),
    }
    if (
        duplicate_score_keys
        or blank_key_rows
        or invalid_score_rows
        or missing_keys
        or extra_keys
        or len(score_rows) != len(candidate_keys)
    ):
        raise ValueError(
            "Official main rows require an exact finite score for every candidate key. "
            f"diagnostics={summary}, score_rows_loaded={len(score_rows)}"
        )
    return summary


def _validate_fairness_provenance(args: argparse.Namespace) -> dict[str, Any]:
    if not args.provenance_path:
        raise ValueError("Official main rows require --provenance_path.")
    path = Path(args.provenance_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Fairness provenance file not found: {path}")
    provenance = _load_json(path)
    expected_schema = ["source_event_id", "user_id", "item_id", "score"]
    policy_id = args.fairness_policy_id or _text(provenance.get("fairness_policy_id"))
    if not policy_id:
        raise ValueError("Official main rows require fairness_policy_id in args or provenance.")
    if _text(provenance.get("fairness_policy_id")) and _text(provenance.get("fairness_policy_id")) != policy_id:
        raise ValueError(
            "Fairness provenance policy mismatch: "
            f"arg={policy_id}, provenance={provenance.get('fairness_policy_id')}"
        )
    if _as_schema(provenance.get("score_schema")) != expected_schema:
        raise ValueError(
            "Fairness provenance score_schema must be "
            f"{expected_schema}, got {provenance.get('score_schema')}"
        )
    if provenance.get("test_set_model_selection_allowed") is not False:
        raise ValueError("Fairness provenance must set test_set_model_selection_allowed=false.")
    if provenance.get("baseline_extra_tuning_allowed") is not False:
        raise ValueError("Primary official rows must set baseline_extra_tuning_allowed=false.")
    status = _text(args.implementation_status or provenance.get("implementation_status"))
    if status != "official_completed":
        raise ValueError(
            "Official main rows require implementation_status=official_completed; "
            f"got {status!r}."
        )
    variant = _text(args.comparison_variant or provenance.get("comparison_variant"))
    if variant != "official_code_qwen3base_default_hparams_declared_adaptation":
        raise ValueError(
            "Official main rows require comparison_variant="
            "official_code_qwen3base_default_hparams_declared_adaptation; "
            f"got {variant!r}."
        )
    adaptation_mode = _text(provenance.get("llm_adaptation_mode"))
    if not adaptation_mode:
        raise ValueError("Official main rows require llm_adaptation_mode in provenance.")
    if "full" in adaptation_mode.lower() and "finetune" in adaptation_mode.lower():
        raise ValueError("Full-finetune official rows must be imported as supplementary, not primary.")
    return {
        "fairness_policy_id": policy_id,
        "comparison_variant": variant,
        "implementation_status": status,
        "provenance_path": str(path),
        "official_fairness_checked": True,
    }


def main() -> None:
    args = parse_args()
    from src.eval.ranking_task_metrics import (
        build_ranking_eval_frame,
        compute_ranking_exposure_distribution,
        compute_ranking_task_metrics,
    )

    artifact_class = normalize_artifact_class(args.status_label, args.artifact_class)
    strict_same_schema_statuses = {
        "same_schema_external_baseline",
        "same_schema_internal_method",
        "same_schema_internal_ablation",
    }
    if args.status_label == "same_schema_external_baseline" and "scaffold" in artifact_class:
        raise ValueError(
            "Refusing to import a scaffold artifact as same_schema_external_baseline. "
            "Use a non-main status_label such as llmesr_adapter_scaffold_score."
        )

    paths = ensure_exp_dirs(args.exp_name, args.output_root)
    ranking_samples = load_jsonl(args.ranking_input_path)
    score_rows = load_score_rows(args.scores_path)
    official_main_row = _is_official_main_row(args, artifact_class)
    provenance_summary: dict[str, Any] = {
        "fairness_policy_id": args.fairness_policy_id,
        "comparison_variant": args.comparison_variant,
        "implementation_status": args.implementation_status,
        "provenance_path": args.provenance_path,
        "official_fairness_checked": False,
    }
    exact_score_audit: dict[str, Any] = {}

    if official_main_row or args.require_fairness_provenance:
        provenance_summary = _validate_fairness_provenance(args)
        exact_score_audit = _validate_exact_score_file(
            ranking_samples=ranking_samples,
            score_rows=score_rows,
            source_event_col=args.source_event_col,
            user_col=args.user_col,
            item_col=args.item_col,
            score_col=args.score_col,
        )

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
        args.status_label in strict_same_schema_statuses
        and not args.allow_partial_scores
        and coverage + 1.0e-12 < args.min_score_coverage
    ):
        raise ValueError(
            f"Refusing to write {args.status_label} with partial score coverage: "
            f"score_coverage_rate={coverage:.6f}, required>={args.min_score_coverage:.6f}, "
            f"matched_candidates={score_summary.get('matched_candidates')}, "
            f"total_candidates={score_summary.get('total_candidates')}, "
            f"score_rows_loaded={len(score_rows)}. "
            "Fix the external score file or rerun with --allow_partial_scores and a non-main status label."
        )

    prediction_path = paths.predictions_dir / "rank_predictions.jsonl"
    save_jsonl(predictions, prediction_path)

    import pandas as pd

    eval_df = build_ranking_eval_frame(pd.DataFrame(predictions))
    metrics = compute_ranking_task_metrics(eval_df, k=args.k)
    exposure_df = compute_ranking_exposure_distribution(eval_df, k=args.k)

    result_row = {
        "baseline_name": args.baseline_name,
        "domain": args.domain,
        "comparison_scope": comparison_scope_for_status(args.status_label),
        "task": "candidate_ranking",
        "status_label": args.status_label,
        "artifact_class": artifact_class,
        "ranking_input_path": str(args.ranking_input_path),
        "scores_path": str(args.scores_path),
        "prediction_path": str(prediction_path),
        **provenance_summary,
        **exact_score_audit,
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
