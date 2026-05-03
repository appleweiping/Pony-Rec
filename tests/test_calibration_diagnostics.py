from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from src.cli.run_calibration_diagnostics import main as diag_main
from src.data.protocol import read_jsonl, write_jsonl
from src.uncertainty.calibration import (
    adaptive_ece,
    brier_score,
    build_calibration_diagnostic_rows,
    compare_confidence_sources,
    expected_calibration_error,
    high_confidence_wrong_rate,
    low_confidence_correct_rate,
    summarize_calibration_diagnostics,
)


def test_well_matched_confidence_better_than_overconfident() -> None:
    p_ok = np.repeat([0.1, 0.9], 100)
    y_ok = np.repeat([0.0, 1.0], 100)
    ece_ok, _, _ = expected_calibration_error(y_ok, p_ok, n_bins=10)
    y_bad = np.zeros(100, dtype=float)
    p_bad = np.ones(100, dtype=float) * 0.95
    ece_bad, _, _ = expected_calibration_error(y_bad, p_bad, n_bins=10)
    assert ece_ok < ece_bad
    assert brier_score(y_ok, p_ok) < brier_score(y_bad, p_bad)


def test_overconfident_wrong_inflates_ece() -> None:
    y = np.zeros(20, dtype=float)
    p = np.ones(20, dtype=float) * 0.99
    ece, _, _ = expected_calibration_error(y, p, n_bins=10)
    assert ece > 0.4


def test_missing_confidence_auto_defaults() -> None:
    preds = [
        {
            "user_id": "u1",
            "target_item_id": "a",
            "predicted_item_id": "b",
            "predicted_ranking": ["b", "a"],
            "correctness": False,
            "item_popularity_bucket": "head",
        }
    ]
    rows, meta = build_calibration_diagnostic_rows(preds, None, confidence_field="auto", num_bins=10)
    assert rows[0]["confidence"] == 0.5
    assert "default" in rows[0]["confidence_source"] or rows[0]["confidence_source"].startswith("default")
    assert meta["n_predictions"] == 1


def test_popularity_bucket_grouping_and_high_conf_wrong() -> None:
    preds = [
        {
            "user_id": "a",
            "target_item_id": "t1",
            "predicted_item_id": "x",
            "predicted_ranking": ["x"],
            "correctness": False,
            "raw_confidence": 0.95,
            "item_popularity_bucket": "head",
        },
        {
            "user_id": "b",
            "target_item_id": "t2",
            "predicted_item_id": "t2",
            "predicted_ranking": ["t2"],
            "correctness": True,
            "raw_confidence": 0.2,
            "item_popularity_bucket": "tail",
        },
    ]
    rows, _ = build_calibration_diagnostic_rows(preds, None, confidence_field="verbalized", num_bins=10)
    assert rows[0]["high_confidence_wrong"] is True
    assert rows[1]["low_confidence_correct"] is True
    summ = summarize_calibration_diagnostics(rows, n_bins=5)
    assert summ["n_used"] == 2
    assert summ["high_confidence_wrong_rate"] == 1.0


def test_compare_confidence_sources_handles_nan_channel() -> None:
    y = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=float)
    src = compare_confidence_sources(
        y,
        {
            "good": np.asarray([0.9, 0.1, 0.85, 0.15], dtype=float),
            "bad": np.asarray([float("nan")] * 4, dtype=float),
        },
        n_bins=5,
    )
    assert src["bad"]["n_valid"] == 0
    assert np.isnan(src["bad"]["ece"])
    assert np.isfinite(src["good"]["ece"])


def test_adaptive_ece_finite() -> None:
    y = np.asarray([0, 1, 0, 1, 0, 1], dtype=float)
    p = np.asarray([0.1, 0.9, 0.15, 0.85, 0.2, 0.8], dtype=float)
    v = adaptive_ece(y, p, n_bins=3)
    assert np.isfinite(v)


def test_cli_smoke_tiny_fixture(tmp_path: Path) -> None:
    pred = tmp_path / "rank_predictions.jsonl"
    write_jsonl(
        [
            {
                "user_id": "u1",
                "dataset": "d",
                "domain": "d",
                "split": "valid",
                "target_item_id": "a",
                "predicted_item_id": "a",
                "predicted_ranking": ["a", "b"],
                "correctness": True,
                "raw_confidence": 0.8,
                "item_popularity_bucket": "mid",
            }
        ],
        pred,
    )
    out = tmp_path / "out"
    old = sys.argv
    sys.argv = [
        "run_calibration_diagnostics",
        "--predictions_path",
        str(pred),
        "--output_dir",
        str(out),
        "--confidence_field",
        "verbalized",
        "--num_bins",
        "5",
    ]
    try:
        diag_main()
    finally:
        sys.argv = old
    assert (out / "calibration_summary.json").is_file()
    summ = json.loads((out / "calibration_summary.json").read_text(encoding="utf-8"))
    assert summ["n_used"] == 1
    rows = read_jsonl(str(out / "calibration_rows.jsonl"))
    assert rows[0]["is_correct_at_1"] is True
