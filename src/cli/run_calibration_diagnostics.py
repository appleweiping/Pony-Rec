"""CLI: CARE-Rec calibration diagnostics from rank_predictions (+ optional uncertainty_features)."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.data.protocol import read_jsonl, write_jsonl
from src.uncertainty.calibration import (
    build_calibration_diagnostic_rows,
    group_calibration_by_popularity_bucket,
    high_confidence_error_by_bucket,
    reliability_bins,
    summarize_calibration_diagnostics,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibration diagnostics for CARE-Rec (pilot-scale JSONL).")
    p.add_argument("--predictions_path", default=None, help="rank_predictions.jsonl (single run).")
    p.add_argument("--features_path", default=None, help="Optional uncertainty_features.jsonl (merge by user_id).")
    p.add_argument("--output_dir", required=True, help="Directory for calibration outputs.")
    p.add_argument(
        "--confidence_field",
        default="auto",
        help="auto | verbalized | confidence | calibrated | score | margin | entropy | self_consistency | raw_confidence",
    )
    p.add_argument("--num_bins", type=int, default=10)
    p.add_argument(
        "--batch_glob",
        default=None,
        help="Recursive glob of rank_predictions.jsonl (e.g. outputs/pilots/deepseek_*/**/rank_predictions.jsonl).",
    )
    return p.parse_args()


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _load_features_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(str(path)):
        uid = str(row.get("user_id", ""))
        if uid:
            out[uid] = row
    return out


def _default_features_path(pred_path: Path) -> Path | None:
    cand = pred_path.parent / "uncertainty_features.jsonl"
    return cand if cand.is_file() else None


def _safe_subdir(output_dir: Path, pred_path: Path, cwd: Path) -> Path:
    try:
        rel = pred_path.resolve().relative_to(cwd.resolve())
    except ValueError:
        rel = pred_path
    key = "__".join(rel.parts)
    key = re.sub(r"[^\w.\-]+", "_", key)[:200]
    return output_dir / key


def _run_one(
    predictions_path: Path,
    features_path: Path | None,
    output_dir: Path,
    *,
    confidence_field: str,
    num_bins: int,
) -> None:
    preds = read_jsonl(str(predictions_path))
    feats = _load_features_map(features_path)
    rows, meta = build_calibration_diagnostic_rows(
        preds,
        feats if feats else None,
        confidence_field=confidence_field,
        num_bins=num_bins,
    )
    summary = summarize_calibration_diagnostics(rows, n_bins=num_bins)
    summary["predictions_path"] = str(predictions_path)
    summary["features_path"] = str(features_path) if features_path else None
    summary["build_meta"] = meta

    y = []
    c = []
    for r in rows:
        try:
            cv = float(r["confidence"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(cv):
            continue
        y.append(float(bool(r.get("is_correct_at_1"))))
        c.append(cv)
    rel_bins: list[dict[str, Any]] = []
    if y:
        rel_bins = reliability_bins(np.asarray(y, dtype=float), np.asarray(c, dtype=float), n_bins=num_bins)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, str(output_dir / "calibration_rows.jsonl"))
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(rel_bins, output_dir / "reliability_bins.csv")
    _write_csv(
        group_calibration_by_popularity_bucket(rows, n_bins=num_bins),
        output_dir / "group_calibration_by_popularity.csv",
    )
    _write_csv(
        high_confidence_error_by_bucket(rows),
        output_dir / "high_confidence_error_by_bucket.csv",
    )


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    out_root = Path(args.output_dir)

    if args.batch_glob:
        paths = sorted(Path(p) for p in glob.glob(args.batch_glob, recursive=True) if p.endswith("rank_predictions.jsonl"))
        if not paths:
            print(f"[calibration_diagnostics] no files matched: {args.batch_glob}")
            return
        for pred_path in paths:
            sub = _safe_subdir(out_root, pred_path, cwd)
            feat = Path(args.features_path) if args.features_path else _default_features_path(pred_path)
            _run_one(pred_path, feat, sub, confidence_field=args.confidence_field, num_bins=int(args.num_bins))
            print(f"[calibration_diagnostics] wrote {sub}")
        return

    if not args.predictions_path:
        raise SystemExit("Either --predictions_path or --batch_glob is required.")
    pred_path = Path(args.predictions_path)
    feat_path = Path(args.features_path) if args.features_path else _default_features_path(pred_path)
    _run_one(pred_path, feat_path, out_root, confidence_field=args.confidence_field, num_bins=int(args.num_bins))
    print(f"[calibration_diagnostics] wrote {out_root}")


if __name__ == "__main__":
    main()
