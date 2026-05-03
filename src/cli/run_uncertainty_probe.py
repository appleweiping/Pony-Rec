"""CLI: extract CARE-oriented uncertainty features from rank_predictions.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.protocol import read_jsonl, write_jsonl
from src.uncertainty.features import extract_care_probe_features, summarize_probe_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract uncertainty / exposure probe features from ranking predictions.")
    p.add_argument("--predictions_path", required=True, help="Path to rank_predictions.jsonl")
    p.add_argument(
        "--output_path",
        default=None,
        help="Output JSONL (default: alongside predictions as uncertainty_features.jsonl)",
    )
    p.add_argument(
        "--summary_path",
        default=None,
        help="Output JSON summary (default: same dir as output_path, uncertainty_probe_summary.json)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = Path(args.predictions_path)
    rows = read_jsonl(str(pred_path))
    out_path = Path(args.output_path) if args.output_path else pred_path.parent / "uncertainty_features.jsonl"
    summary_path = Path(args.summary_path) if args.summary_path else out_path.parent / "uncertainty_probe_summary.json"

    keep_keys = (
        "user_id",
        "dataset",
        "domain",
        "split",
        "seed",
        "method",
        "backend",
        "backend_type",
        "target_item_id",
        "predicted_item_id",
        "correctness",
        "raw_confidence",
        "calibrated_confidence",
        "uncertainty_score",
        "uncertainty_estimator_name",
        "missing_confidence",
        "item_popularity_bucket",
    )
    enriched: list[dict] = []
    for row in rows:
        features = extract_care_probe_features(row)
        slim = {k: row.get(k) for k in keep_keys}
        slim["care_features"] = features
        enriched.append(slim)

    write_jsonl(enriched, str(out_path))
    summary = summarize_probe_rows(enriched)
    summary["predictions_path"] = str(pred_path)
    summary["output_path"] = str(out_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[uncertainty_probe] wrote {out_path} ({len(enriched)} rows)")
    print(f"[uncertainty_probe] summary {summary_path}")


if __name__ == "__main__":
    main()
