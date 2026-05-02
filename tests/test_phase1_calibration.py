from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.cli.calibrate import main as calibrate_main
from src.data.protocol import read_jsonl, write_jsonl


def test_calibration_fits_validation_and_applies_test_only(tmp_path: Path) -> None:
    valid = [
        {"split": "valid", "raw_confidence": 0.9, "correctness": True, "user_id": "u1", "dataset": "d", "domain": "D", "seed": 1},
        {"split": "valid", "raw_confidence": 0.1, "correctness": False, "user_id": "u2", "dataset": "d", "domain": "D", "seed": 1},
    ]
    test = [
        {"split": "test", "raw_confidence": 0.8, "correctness": True, "user_id": "u3", "dataset": "d", "domain": "D", "seed": 1},
        {"split": "test", "raw_confidence": 0.2, "correctness": False, "user_id": "u4", "dataset": "d", "domain": "D", "seed": 1},
    ]
    valid_path = tmp_path / "valid.jsonl"
    test_path = tmp_path / "test.jsonl"
    out_path = tmp_path / "calibrated.jsonl"
    write_jsonl(valid, valid_path)
    write_jsonl(test, test_path)
    old_argv = sys.argv
    sys.argv = ["calibrate", "--valid_path", str(valid_path), "--test_path", str(test_path), "--output_path", str(out_path), "--method", "isotonic"]
    try:
        calibrate_main()
    finally:
        sys.argv = old_argv
    rows = read_jsonl(out_path)
    assert {row["split"] for row in rows} == {"test"}
    assert all("calibrated_confidence" in row for row in rows)


def test_calibration_rejects_valid_file_containing_test_split(tmp_path: Path) -> None:
    path = tmp_path / "bad_valid.jsonl"
    test_path = tmp_path / "test.jsonl"
    write_jsonl([{"split": "test", "raw_confidence": 0.5, "correctness": True}], path)
    write_jsonl([{"split": "test", "raw_confidence": 0.5, "correctness": True}], test_path)
    old_argv = sys.argv
    sys.argv = ["calibrate", "--valid_path", str(path), "--test_path", str(test_path), "--output_path", str(tmp_path / "out.jsonl")]
    try:
        with pytest.raises(ValueError):
            calibrate_main()
    finally:
        sys.argv = old_argv
