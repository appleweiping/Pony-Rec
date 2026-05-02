from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.data.protocol import read_jsonl, write_json, write_jsonl
from src.eval.paper_metrics import brier_score, ece_mce
from src.utils.manifest import build_manifest, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit calibrators on validation predictions and apply to test.")
    parser.add_argument("--valid_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--metrics_path", default=None)
    parser.add_argument("--method", default="isotonic", choices=["isotonic", "platt"])
    parser.add_argument("--group_by", default=None, choices=["domain", "item_popularity_bucket", "history_length_bucket"])
    return parser.parse_args()


def _fit(x: np.ndarray, y: np.ndarray, method: str):
    if len(np.unique(y.astype(int))) < 2:
        return float(np.mean(y))
    if method == "platt":
        model = LogisticRegression()
        model.fit(x.reshape(-1, 1), y.astype(int))
        return model
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(x, y)
    return model


def _predict(model, x: np.ndarray) -> np.ndarray:
    if isinstance(model, float):
        return np.full(shape=x.shape, fill_value=model, dtype=float)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x.reshape(-1, 1))[:, 1]
    return model.predict(x)


def main() -> None:
    args = parse_args()
    valid_rows = read_jsonl(args.valid_path)
    test_rows = read_jsonl(args.test_path)
    valid = pd.DataFrame(valid_rows)
    test = pd.DataFrame(test_rows)
    if valid.empty or test.empty:
        raise ValueError("valid/test predictions must both be non-empty.")
    if set(valid.get("split", [])) - {"valid"}:
        raise ValueError("Calibration fit input must contain only validation predictions.")
    if set(test.get("split", [])) - {"test"}:
        raise ValueError("Calibration apply input must contain only test predictions.")
    valid["raw_confidence"] = valid["raw_confidence"].astype(float).clip(0.0, 1.0)
    test["raw_confidence"] = test["raw_confidence"].astype(float).clip(0.0, 1.0)
    valid["correctness_int"] = valid["correctness"].astype(int)
    if args.group_by:
        global_model = _fit(valid["raw_confidence"].to_numpy(), valid["correctness_int"].to_numpy(), args.method)
        calibrated = np.zeros(len(test), dtype=float)
        for group_value, group_test in test.groupby(args.group_by, dropna=False):
            group_valid = valid[valid[args.group_by] == group_value]
            model = (
                _fit(group_valid["raw_confidence"].to_numpy(), group_valid["correctness_int"].to_numpy(), args.method)
                if len(group_valid) >= 4
                else global_model
            )
            calibrated[group_test.index.to_numpy()] = _predict(model, group_test["raw_confidence"].to_numpy())
    else:
        model = _fit(valid["raw_confidence"].to_numpy(), valid["correctness_int"].to_numpy(), args.method)
        calibrated = _predict(model, test["raw_confidence"].to_numpy())
    test["calibrated_confidence"] = np.clip(calibrated, 0.0, 1.0)
    test["uncertainty_score"] = 1.0 - test["calibrated_confidence"]
    test["uncertainty_estimator_name"] = "calibrated_verbalized_confidence"
    out_rows = json.loads(test.to_json(orient="records"))
    write_jsonl(out_rows, args.output_path)
    y_true = test["correctness"].astype(int).to_numpy()
    y_prob = test["calibrated_confidence"].astype(float).to_numpy()
    ece, mce = ece_mce(y_true, y_prob)
    metrics = {
        "method": args.method,
        "group_by": args.group_by,
        "fit_split": "valid",
        "apply_split": "test",
        "num_valid": int(len(valid)),
        "num_test": int(len(test)),
        "ece": ece,
        "mce": mce,
        "brier": brier_score(y_true, y_prob),
    }
    metrics_path = Path(args.metrics_path) if args.metrics_path else Path(args.output_path).with_suffix(".metrics.json")
    write_json(metrics, metrics_path)
    first = out_rows[0] if out_rows else {}
    write_manifest(
        Path(args.output_path).parent.parent / "manifest.json",
        build_manifest(
            config={"method": args.method, "group_by": args.group_by},
            dataset=str(first.get("dataset", "unknown")),
            domain=str(first.get("domain", "unknown")),
            processed_data_paths=[args.valid_path, args.test_path, args.output_path],
            method="calibrate",
            backend=str(first.get("backend", "unknown")),
            model=str(first.get("model", "unknown")),
            prompt_template=str(first.get("prompt_template_id", "unknown")),
            seed=int(first.get("seed", 0) or 0),
            candidate_size=len(first.get("candidate_item_ids", [])) if first else None,
            calibration_source="valid",
            mock_data_used=str(first.get("backend_type", "")) == "mock",
        ),
    )
    print(f"[calibrate] saved={args.output_path}")
    print(f"[calibrate] metrics={metrics_path}")


if __name__ == "__main__":
    main()
