from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    roc_auc_score_manual,
)
from src.eval.ranking_metrics import compute_ranking_metrics
from src.shadow.ccrp import apply_ccrp_scores, parse_weights


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/summary/shadow_ccrp")
    parser.add_argument("--ablation", type=str, default="full")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--weights", type=float, nargs=3, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--status_label",
        type=str,
        default="runnable_not_complete",
        choices=["completed_result", "runnable_not_complete", "design_only", "proxy_only", "future_extension"],
    )
    parser.add_argument(
        "--prompt_only",
        type=str_to_bool,
        default=False,
        help="When true, ranking main-table eligibility is forced off.",
    )
    return parser.parse_args()


def load_records(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_json(path, lines=True)


def compute_pointwise_metrics(df: pd.DataFrame, *, n_bins: int = 10) -> dict[str, float]:
    if "label" not in df.columns:
        return {}
    y_true = df["label"].astype(int).to_numpy()
    y_score = df["ccrp_calibrated_probability"].astype(float).clip(0.0, 1.0).to_numpy()
    ece, mce, _ = expected_calibration_error(y_true, y_score, n_bins=n_bins)
    pred = (y_score >= 0.5).astype(int)
    return {
        "num_samples": int(len(df)),
        "accuracy": float((pred == y_true).mean()),
        "auroc": roc_auc_score_manual(y_true, y_score),
        "ece": ece,
        "mce": mce,
        "brier_score": brier_score(y_true, y_score),
        "avg_uncertainty": float(df["ccrp_uncertainty"].mean()),
        "parse_success_rate": float(df["parse_success"].astype(float).mean()) if "parse_success" in df.columns else float("nan"),
        "out_of_candidate_rate": float(df["contains_out_of_candidate_item"].astype(float).mean())
        if "contains_out_of_candidate_item" in df.columns
        else float("nan"),
    }


def build_risk_coverage_curve(df: pd.DataFrame) -> pd.DataFrame:
    if "label" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["pred_label"] = (out["ccrp_calibrated_probability"].astype(float) >= 0.5).astype(int)
    out["is_correct"] = (out["pred_label"] == out["label"].astype(int)).astype(float)
    out = out.sort_values("ccrp_uncertainty", ascending=True).reset_index(drop=True)
    rows = []
    for coverage in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        n_keep = max(1, int(round(len(out) * coverage)))
        kept = out.head(n_keep)
        rows.append(
            {
                "coverage": float(coverage),
                "risk": float(1.0 - kept["is_correct"].mean()),
                "accuracy": float(kept["is_correct"].mean()),
                "kept_samples": int(len(kept)),
            }
        )
    return pd.DataFrame(rows)


def build_ranked_frame(df: pd.DataFrame, *, k: int) -> pd.DataFrame:
    if not {"candidate_item_id", "label", "ccrp_risk_adjusted_score"}.issubset(df.columns):
        return pd.DataFrame()
    id_col = "source_event_id" if "source_event_id" in df.columns else "user_id"
    rows = []
    for event_id, group in df.groupby(id_col, dropna=False):
        ranked = group.sort_values("ccrp_risk_adjusted_score", ascending=False).reset_index(drop=True)
        for idx, record in enumerate(ranked.to_dict(orient="records"), start=1):
            rows.append(
                {
                    "user_id": str(event_id),
                    "item_id": record.get("candidate_item_id"),
                    "label": int(record.get("label", 0)),
                    "rank": idx,
                }
            )
    return pd.DataFrame(rows)


def compute_coverage_at_k(ranked_df: pd.DataFrame, *, k: int) -> float:
    if ranked_df.empty or "item_id" not in ranked_df.columns:
        return float("nan")
    exposed = set()
    candidates = set(ranked_df["item_id"].astype(str).tolist())
    for _, group in ranked_df.groupby("user_id"):
        exposed.update(group.nsmallest(k, "rank")["item_id"].astype(str).tolist())
    return float(len(exposed) / len(candidates)) if candidates else float("nan")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    raw_df = load_records(input_path)
    scored_df = apply_ccrp_scores(
        raw_df,
        weights=parse_weights(args.weights),
        eta=args.eta,
        ablation=args.ablation,
    )

    pointwise_metrics = compute_pointwise_metrics(scored_df)
    ranked_df = build_ranked_frame(scored_df, k=args.k)
    ranking_metrics = compute_ranking_metrics(ranked_df, k=args.k) if not ranked_df.empty else {}
    if ranking_metrics:
        ranking_metrics[f"coverage@{args.k}"] = compute_coverage_at_k(ranked_df, k=args.k)

    main_table_eligible = (
        args.status_label == "completed_result"
        and not args.prompt_only
        and not ranked_df.empty
    )
    summary = {
        "method": "C-CRP" if args.ablation == "full" else f"C-CRP {args.ablation}",
        "ablation": args.ablation,
        "eta": float(args.eta),
        "status_label": args.status_label,
        "prompt_only": bool(args.prompt_only),
        "main_table_eligible": bool(main_table_eligible),
    }
    summary.update(pointwise_metrics)
    summary.update(ranking_metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / f"ccrp_{args.ablation}_scored.csv"
    summary_path = output_dir / f"ccrp_{args.ablation}_summary.csv"
    risk_curve_path = output_dir / f"ccrp_{args.ablation}_risk_coverage.csv"
    ranked_path = output_dir / f"ccrp_{args.ablation}_ranking_records.csv"

    scored_df.to_csv(scored_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    build_risk_coverage_curve(scored_df).to_csv(risk_curve_path, index=False)
    if not ranked_df.empty:
        ranked_df.to_csv(ranked_path, index=False)

    print(f"Saved C-CRP scored records to: {scored_path}")
    print(f"Saved C-CRP summary to: {summary_path}")
    print(f"Saved C-CRP risk-coverage curve to: {risk_curve_path}")


if __name__ == "__main__":
    main()
