# main_calibrate.py
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.calibration_plotting import plot_before_after_reliability
from src.analysis.confidence_correctness import prepare_prediction_dataframe
from src.eval.calibration_metrics import (
    brier_score,
    compute_calibration_metrics,
    expected_calibration_error,
    get_reliability_dataframe,
)
from src.uncertainty.calibration import (
    apply_calibrator,
    build_split_metadata,
    fit_calibrator,
    fit_isotonic_calibrator,
    fit_platt_calibrator,
    user_level_split,
)
from src.uncertainty.verbalized_confidence import (
    add_uncertainty_from_confidence,
    normalize_confidence_column,
)
from src.utils.paths import ensure_exp_dirs


def load_jsonl(path: str | Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def save_jsonl(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_summary_dict(summary: dict, path: str | Path) -> None:
    save_table(pd.DataFrame([summary]), path)


def resolve_prediction_splits(
    *,
    exp_name: str,
    paths,
    input_path: str | Path | None,
    valid_path: str | Path | None,
    test_path: str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, str, Path | None, Path | None]:
    if valid_path is not None and test_path is not None:
        valid_source = Path(valid_path)
        test_source = Path(test_path)
        valid_df = load_jsonl(valid_source)
        test_df = load_jsonl(test_source)
        return valid_df, test_df, "explicit_valid_test_paths", valid_source, test_source

    default_valid_path = paths.predictions_dir / "valid_raw.jsonl"
    default_test_path = paths.predictions_dir / "test_raw.jsonl"

    if default_valid_path.exists() and default_test_path.exists():
        valid_df = load_jsonl(default_valid_path)
        test_df = load_jsonl(default_test_path)
        return valid_df, test_df, "predictions_valid_test_files", default_valid_path, default_test_path

    fallback_input_path = (
        Path(input_path)
        if input_path is not None
        else paths.predictions_dir / "test_raw.jsonl"
    )
    if not fallback_input_path.exists():
        raise FileNotFoundError(
            f"[{exp_name}] No usable prediction inputs found. "
            f"Checked valid/test files under {paths.predictions_dir} and fallback file {fallback_input_path}."
        )

    raw_df = load_jsonl(fallback_input_path)
    return raw_df, pd.DataFrame(), "single_file_fallback_split", fallback_input_path, None


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def assert_disjoint_users(valid_df: pd.DataFrame, test_df: pd.DataFrame, user_col: str = "user_id") -> None:
    if user_col not in valid_df.columns or user_col not in test_df.columns:
        return

    overlap = set(valid_df[user_col].dropna().unique()) & set(test_df[user_col].dropna().unique())
    if overlap:
        raise ValueError(f"User leakage detected between valid and test splits: {len(overlap)} overlapping users.")


def count_overlapping_users(valid_df: pd.DataFrame, test_df: pd.DataFrame, user_col: str = "user_id") -> int:
    if user_col not in valid_df.columns or user_col not in test_df.columns:
        return 0

    overlap = set(valid_df[user_col].dropna().unique()) & set(test_df[user_col].dropna().unique())
    return int(len(overlap))


def collect_item_ids(df: pd.DataFrame) -> set[str]:
    item_ids: set[str] = set()
    scalar_cols = [
        "item_id",
        "candidate_item_id",
        "positive_item_id",
        "target_item_id",
    ]
    list_cols = [
        "candidate_item_ids",
        "pred_ranked_item_ids",
        "topk_item_ids",
    ]
    for col in scalar_cols:
        if col in df.columns:
            item_ids.update(str(value) for value in df[col].dropna().tolist() if str(value).strip())
    for col in list_cols:
        if col not in df.columns:
            continue
        for value in df[col].dropna().tolist():
            if isinstance(value, list):
                item_ids.update(str(item) for item in value if str(item).strip())
            else:
                text = str(value).strip()
                if text.startswith("["):
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, list):
                            item_ids.update(str(item) for item in parsed if str(item).strip())
                            continue
                    except Exception:
                        pass
                if text:
                    item_ids.add(text)
    return item_ids


def count_overlapping_items(valid_df: pd.DataFrame, test_df: pd.DataFrame) -> int:
    return len(collect_item_ids(valid_df) & collect_item_ids(test_df))


def sha256_file(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_split_hash(valid_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    stable_cols = [
        "user_id",
        "source_event_id",
        "candidate_item_id",
        "positive_item_id",
        "timestamp",
        "label",
    ]
    payload: dict[str, Any] = {}
    for split_name, df in [("valid", valid_df), ("test", test_df)]:
        cols = [col for col in stable_cols if col in df.columns]
        if cols:
            normalized = (
                df[cols]
                .fillna("")
                .astype(str)
                .sort_values(cols)
                .to_dict(orient="records")
            )
        else:
            normalized = [{"row_count": int(len(df))}]
        payload[split_name] = normalized
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def nonempty_calibration_bin_count(
    df: pd.DataFrame,
    *,
    confidence_col: str,
    target_col: str = "is_correct",
    n_bins: int = 10,
) -> int:
    if df.empty or confidence_col not in df.columns or target_col not in df.columns:
        return 0
    _, _, bins = expected_calibration_error(
        df[target_col].astype(int).to_numpy(),
        df[confidence_col].astype(float).clip(0.0, 1.0).to_numpy(),
        n_bins=n_bins,
    )
    return int(sum(1 for item in bins if item.count > 0))


def calibration_bootstrap_ci(
    df: pd.DataFrame,
    *,
    split_name: str,
    stage: str,
    confidence_col: str,
    target_col: str = "is_correct",
    n_bins: int = 10,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    if df.empty or n_bootstrap <= 0:
        return pd.DataFrame(
            columns=["split", "stage", "metric", "estimate", "ci_low", "ci_high", "bootstrap_iters"]
        )

    y_true = df[target_col].astype(int).to_numpy()
    y_prob = df[confidence_col].astype(float).clip(0.0, 1.0).to_numpy()
    rng = np.random.default_rng(random_state)
    brier_values: list[float] = []
    ece_values: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(df), size=len(df))
        sample_y = y_true[idx]
        sample_p = y_prob[idx]
        brier_values.append(brier_score(sample_y, sample_p))
        ece, _, _ = expected_calibration_error(sample_y, sample_p, n_bins=n_bins)
        ece_values.append(ece)

    rows = []
    for metric, values, estimate in [
        ("brier_score", brier_values, brier_score(y_true, y_prob)),
        ("ece", ece_values, expected_calibration_error(y_true, y_prob, n_bins=n_bins)[0]),
    ]:
        rows.append(
            {
                "split": split_name,
                "stage": stage,
                "metric": metric,
                "estimate": float(estimate),
                "ci_low": float(np.percentile(values, 2.5)),
                "ci_high": float(np.percentile(values, 97.5)),
                "bootstrap_iters": int(n_bootstrap),
            }
        )
    return pd.DataFrame(rows)


def compare_metrics(
    raw_df: pd.DataFrame,
    calibrated_df: pd.DataFrame,
    split_name: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    before_metrics = compute_calibration_metrics(
        raw_df,
        confidence_col="confidence",
        target_col="is_correct",
        n_bins=n_bins,
    )
    after_metrics = compute_calibration_metrics(
        calibrated_df,
        confidence_col="calibrated_confidence",
        target_col="is_correct",
        n_bins=n_bins,
    )

    rows = []
    for metric_name in before_metrics.keys():
        rows.append(
            {
                "split": split_name,
                "metric": metric_name,
                "before": before_metrics[metric_name],
                "after": after_metrics[metric_name],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="clean",
        help="Experiment name."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Optional fallback path to a single raw prediction jsonl when valid/test prediction files are unavailable."
    )
    parser.add_argument(
        "--valid_path",
        type=str,
        default=None,
        help="Optional explicit path to valid raw predictions jsonl."
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
        help="Optional explicit path to test raw predictions jsonl."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs",
        help="Root directory for all experiment outputs."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="isotonic",
        choices=["isotonic", "platt"],
        help="Calibration method."
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.5,
        help="User-level validation split ratio."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for split."
    )
    parser.add_argument(
        "--strict_split_check",
        type=str_to_bool,
        default=True,
        help="When true, fail on split leakage instead of only reporting it."
    )
    parser.add_argument(
        "--allow_user_overlap",
        type=str_to_bool,
        default=False,
        help="Allow valid/test user overlap. Defaults to false for leakage-aware calibration."
    )
    parser.add_argument(
        "--allow_item_overlap",
        type=str_to_bool,
        default=True,
        help="Allow valid/test item overlap. Set false for strict cold-item calibration audits."
    )
    parser.add_argument(
        "--min_isotonic_valid_samples",
        type=int,
        default=200,
        help="Minimum validation rows required to fit isotonic calibration; below this, Platt is used."
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=10,
        help="Number of bins used for ECE and reliability summaries."
    )
    parser.add_argument(
        "--bootstrap_iters",
        type=int,
        default=500,
        help="Bootstrap iterations for ECE/Brier confidence intervals. Set 0 to skip."
    )
    args = parser.parse_args()

    paths = ensure_exp_dirs(args.exp_name, args.output_root)
    raw_valid_df, raw_test_df, split_mode, valid_source_path, test_source_path = resolve_prediction_splits(
        exp_name=args.exp_name,
        paths=paths,
        input_path=args.input_path,
        valid_path=args.valid_path,
        test_path=args.test_path,
    )

    if split_mode == "single_file_fallback_split":
        print(f"[{args.exp_name}] Loading fallback predictions from: {args.input_path or (paths.predictions_dir / 'test_raw.jsonl')}")
        df = prepare_prediction_dataframe(raw_valid_df)
        df = normalize_confidence_column(df, input_col="confidence", output_col="confidence")
        print(f"[{args.exp_name}] Loaded {len(df)} samples from fallback raw predictions.")

        split_result = user_level_split(
            df,
            user_col="user_id",
            valid_ratio=args.valid_ratio,
            random_state=args.random_state
        )
        valid_df = split_result.valid_df
        test_df = split_result.test_df
    else:
        valid_df = prepare_prediction_dataframe(raw_valid_df)
        valid_df = normalize_confidence_column(valid_df, input_col="confidence", output_col="confidence")
        test_df = prepare_prediction_dataframe(raw_test_df)
        test_df = normalize_confidence_column(test_df, input_col="confidence", output_col="confidence")
        print(f"[{args.exp_name}] Loaded {len(valid_df)} valid samples and {len(test_df)} test samples from prediction files.")

    overlapping_users = count_overlapping_users(valid_df, test_df, user_col="user_id")
    overlapping_items = count_overlapping_items(valid_df, test_df)
    if args.strict_split_check:
        if "user_id" not in valid_df.columns or "user_id" not in test_df.columns:
            raise ValueError("Strict split check requires `user_id` in both valid and test splits.")
        if overlapping_users > 0 and not args.allow_user_overlap:
            raise ValueError(
                "User leakage detected between valid and test splits: "
                f"{overlapping_users} overlapping users. "
                "Pass --allow_user_overlap true only for non-main diagnostic audits."
            )
        if overlapping_items > 0 and not args.allow_item_overlap:
            raise ValueError(
                "Item overlap detected between valid and test splits: "
                f"{overlapping_items} overlapping items. "
                "Pass --allow_item_overlap true when warm-item evaluation is intended."
            )
    elif split_mode == "single_file_fallback_split":
        assert_disjoint_users(valid_df, test_df, user_col="user_id")

    valid_user_count = int(valid_df["user_id"].nunique()) if "user_id" in valid_df.columns else -1
    test_user_count = int(test_df["user_id"].nunique()) if "user_id" in test_df.columns else -1
    valid_items = collect_item_ids(valid_df)
    test_items = collect_item_ids(test_df)
    reported_split_strategy = (
        "fallback_internal_validation_only"
        if split_mode == "single_file_fallback_split"
        else split_mode
    )
    requested_method = args.method
    effective_method = requested_method
    if requested_method == "isotonic" and len(valid_df) < int(args.min_isotonic_valid_samples):
        effective_method = "platt"
        print(
            f"[{args.exp_name}] Validation rows ({len(valid_df)}) below "
            f"--min_isotonic_valid_samples={args.min_isotonic_valid_samples}; fitting Platt instead of isotonic."
        )

    split_meta = build_split_metadata(valid_df, test_df)
    split_meta.update(
        {
            "split_strategy": reported_split_strategy,
            "raw_split_mode": split_mode,
            "main_table_eligible": bool(split_mode != "single_file_fallback_split"),
            "valid_ratio": float(args.valid_ratio),
            "random_state": int(args.random_state),
            "calibration_method_requested": requested_method,
            "calibration_method": effective_method,
            "strict_split_check": bool(args.strict_split_check),
            "allow_user_overlap": bool(args.allow_user_overlap),
            "allow_item_overlap": bool(args.allow_item_overlap),
            "num_overlapping_users": int(overlapping_users),
            "overlap_user_count": int(overlapping_users),
            "valid_user_count": int(valid_user_count),
            "test_user_count": int(test_user_count),
            "overlap_user_rate": float(
                overlapping_users / max(1, min(valid_user_count, test_user_count))
            )
            if valid_user_count >= 0 and test_user_count >= 0
            else float("nan"),
            "valid_item_count": int(len(valid_items)),
            "test_item_count": int(len(test_items)),
            "overlap_item_count": int(overlapping_items),
            "overlap_item_rate": float(overlapping_items / max(1, min(len(valid_items), len(test_items)))),
            "split_hash": dataframe_split_hash(valid_df, test_df),
            "valid_file_sha256": sha256_file(valid_source_path),
            "test_file_sha256": sha256_file(test_source_path),
            "fallback_input_sha256": sha256_file(valid_source_path) if split_mode == "single_file_fallback_split" else "",
            "n_bins": int(args.n_bins),
            "valid_nonempty_confidence_bins": nonempty_calibration_bin_count(
                valid_df,
                confidence_col="confidence",
                n_bins=args.n_bins,
            ),
            "test_nonempty_confidence_bins": nonempty_calibration_bin_count(
                test_df,
                confidence_col="confidence",
                n_bins=args.n_bins,
            ),
        }
    )
    save_summary_dict(split_meta, paths.tables_dir / "calibration_split_metadata.csv")

    if effective_method == "isotonic":
        calibrator = fit_isotonic_calibrator(
            valid_df=valid_df,
            confidence_col="confidence",
            target_col="is_correct",
        )
    elif effective_method == "platt":
        calibrator = fit_platt_calibrator(
            valid_df=valid_df,
            confidence_col="confidence",
            target_col="is_correct",
        )
    else:
        calibrator = fit_calibrator(
            valid_df=valid_df,
            method=effective_method,
            confidence_col="confidence",
            target_col="is_correct",
        )
    print(f"[{args.exp_name}] Fitted {effective_method} calibrator on valid split.")

    valid_calibrated = apply_calibrator(valid_df, calibrator, input_col="confidence", output_col="calibrated_confidence")
    valid_calibrated = add_uncertainty_from_confidence(valid_calibrated, confidence_col="calibrated_confidence", output_col="uncertainty")

    test_calibrated = apply_calibrator(test_df, calibrator, input_col="confidence", output_col="calibrated_confidence")
    test_calibrated = add_uncertainty_from_confidence(test_calibrated, confidence_col="calibrated_confidence", output_col="uncertainty")

    save_jsonl(test_calibrated, paths.calibrated_dir / "test_calibrated.jsonl")
    save_jsonl(valid_calibrated, paths.calibrated_dir / "valid_calibrated.jsonl")

    comparison_df = pd.concat(
        [
            compare_metrics(valid_df, valid_calibrated, split_name="valid", n_bins=args.n_bins),
            compare_metrics(test_df, test_calibrated, split_name="test", n_bins=args.n_bins),
        ],
        ignore_index=True,
    )
    save_table(comparison_df, paths.tables_dir / "calibration_comparison.csv")
    save_table(comparison_df, paths.calibrated_dir / "calibration_comparison.csv")

    ci_df = pd.concat(
        [
            calibration_bootstrap_ci(
                valid_df,
                split_name="valid",
                stage="before",
                confidence_col="confidence",
                n_bins=args.n_bins,
                n_bootstrap=args.bootstrap_iters,
                random_state=args.random_state,
            ),
            calibration_bootstrap_ci(
                valid_calibrated,
                split_name="valid",
                stage="after",
                confidence_col="calibrated_confidence",
                n_bins=args.n_bins,
                n_bootstrap=args.bootstrap_iters,
                random_state=args.random_state,
            ),
            calibration_bootstrap_ci(
                test_df,
                split_name="test",
                stage="before",
                confidence_col="confidence",
                n_bins=args.n_bins,
                n_bootstrap=args.bootstrap_iters,
                random_state=args.random_state,
            ),
            calibration_bootstrap_ci(
                test_calibrated,
                split_name="test",
                stage="after",
                confidence_col="calibrated_confidence",
                n_bins=args.n_bins,
                n_bootstrap=args.bootstrap_iters,
                random_state=args.random_state,
            ),
        ],
        ignore_index=True,
    )
    save_table(ci_df, paths.tables_dir / "calibration_metric_ci.csv")
    save_table(ci_df, paths.calibrated_dir / "calibration_metric_ci.csv")

    before_rel = get_reliability_dataframe(
        test_df["is_correct"].to_numpy(),
        test_df["confidence"].to_numpy(),
        n_bins=args.n_bins
    )
    after_rel = get_reliability_dataframe(
        test_calibrated["is_correct"].to_numpy(),
        test_calibrated["calibrated_confidence"].to_numpy(),
        n_bins=args.n_bins
    )

    save_table(before_rel, paths.tables_dir / "reliability_before_calibration.csv")
    save_table(after_rel, paths.tables_dir / "reliability_after_calibration.csv")

    
    plot_before_after_reliability(
        before_rel,
        after_rel,
        paths.figures_dir / "reliability_before_after_calibration.png"
    )

    print(f"[{args.exp_name}] Calibration done.")
    print(f"Calibrated files saved to: {paths.calibrated_dir}")
    print(f"Tables saved to:          {paths.tables_dir}")
    print(f"Figures saved to:         {paths.figures_dir}")


if __name__ == "__main__":
    main()
