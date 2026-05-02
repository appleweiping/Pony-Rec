from __future__ import annotations

import asyncio

import pytest

from src.backends.base import GenerationRequest
from src.backends.mock_backend import MockBackend
from src.cli.train_lora import _format_sft_rows


def test_mock_backend_returns_parseable_ranking() -> None:
    backend = MockBackend()
    response = asyncio.run(
        backend.agenerate(GenerationRequest(prompt="Candidates:\n1. item_id=I2\n2. item_id=I1", request_id="r1"))
    )
    assert response.backend == "mock"
    assert "ranked_item_ids" in response.raw_text


def test_lora_data_formatting_contains_prompt_and_target() -> None:
    rows = [
        {
            "history_item_ids": ["I1", "I2"],
            "candidate_item_ids": ["I3", "I4"],
            "target_item_id": "I4",
        }
    ]
    formatted = _format_sft_rows(rows)
    assert "Candidate item IDs" in formatted[0]["prompt"]
    assert "I4" in formatted[0]["response"]


def test_calibration_leakage_guard_raises_on_test_fit(tmp_path) -> None:
    from src.cli.calibrate import main
    from src.data.protocol import write_jsonl
    import sys

    test_rows = [
        {"split": "test", "raw_confidence": 0.8, "correctness": True, "user_id": "u1"},
        {"split": "test", "raw_confidence": 0.2, "correctness": False, "user_id": "u2"},
    ]
    valid_path = tmp_path / "valid.jsonl"
    test_path = tmp_path / "test.jsonl"
    out_path = tmp_path / "out.jsonl"
    write_jsonl(test_rows, valid_path)
    write_jsonl(test_rows, test_path)
    old_argv = sys.argv
    sys.argv = [
        "calibrate",
        "--valid_path",
        str(valid_path),
        "--test_path",
        str(test_path),
        "--output_path",
        str(out_path),
    ]
    try:
        with pytest.raises(ValueError):
            main()
    finally:
        sys.argv = old_argv
