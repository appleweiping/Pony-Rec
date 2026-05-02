from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.data.protocol import read_jsonl, write_json
from src.eval.paper_metrics import brier_score, ece_mce, exposure_metrics, ranking_metrics, risk_coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ranking, uncertainty calibration, exposure, and risk-coverage.")
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--output_dir", required=True)
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
    metrics = {"ranking": ranking, "calibration": calibration, "exposure": exposure}
    write_json(metrics, out_dir / "metrics.json")
    _write_csv(risk_coverage(rows), out_dir / "risk_coverage.csv")
    print(f"[evaluate] saved={out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
