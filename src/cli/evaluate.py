from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.data.protocol import read_jsonl, write_json
from src.eval.paper_metrics import brier_score, ece_mce, exposure_metrics, ranking_metrics, risk_coverage
from src.utils.manifest import build_manifest, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ranking, uncertainty calibration, exposure, and risk-coverage.")
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--candidates_source_path",
        default=None,
        help="Optional reprocess/candidates JSONL path recorded in manifest processed_data_paths.",
    )
    return parser.parse_args()


def _write_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.predictions_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking = ranking_metrics(rows, ks=(1, 5, 10))
    y_true = np.asarray([int(bool(row.get("correctness", False))) for row in rows])
    y_prob = np.asarray(
        [float(row.get("calibrated_confidence", row.get("raw_confidence", 0.5)) or 0.5) for row in rows]
    )
    ece, mce = ece_mce(y_true, y_prob)
    calibration = {
        "ece": ece,
        "mce": mce,
        "brier": brier_score(y_true, y_prob),
        "avg_confidence": float(np.mean(y_prob)) if len(y_prob) else float("nan"),
        "accuracy": float(np.mean(y_true)) if len(y_true) else float("nan"),
    }
    exposure = exposure_metrics(rows)
    first = rows[0] if rows else {}
    metadata = {
        "dataset": first.get("dataset"),
        "domain": first.get("domain"),
        "seed": first.get("seed"),
        "method": first.get("method"),
        "config_hash": first.get("config_hash"),
        "backend": first.get("backend"),
        "backend_type": first.get("backend_type"),
        "run_type": first.get("run_type"),
        "is_paper_result": first.get("is_paper_result"),
    }
    metrics = {"metadata": metadata, "ranking": ranking, "calibration": calibration, "exposure": exposure}
    write_json(metrics, out_dir / "metrics.json")
    processed_paths: list[str] = [args.predictions_path]
    if args.candidates_source_path:
        processed_paths = [args.candidates_source_path, args.predictions_path]
    write_manifest(
        out_dir / "manifest.json",
        build_manifest(
            config={
                "predictions_path": args.predictions_path,
                "run_type": str(first.get("run_type", "pilot") or "pilot").lower(),
            },
            dataset=str(first.get("dataset", "unknown")),
            domain=str(first.get("domain", "unknown")),
            processed_data_paths=processed_paths,
            method="evaluate",
            backend=str(first.get("backend", "unknown")),
            model=str(first.get("model", "unknown")),
            prompt_template=str(first.get("prompt_template_id", "unknown")),
            seed=int(first.get("seed", 0) or 0),
            candidate_size=len(first.get("candidate_item_ids", [])) if first else None,
            calibration_source="prediction_file",
            mock_data_used=str(first.get("backend_type", "")) == "mock",
        ),
    )
    _write_csv(risk_coverage(rows), out_dir / "risk_coverage.csv")
    print(f"[evaluate] saved={out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
