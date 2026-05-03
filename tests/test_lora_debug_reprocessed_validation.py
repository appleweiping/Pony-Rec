from __future__ import annotations

import pytest

from src.cli.run_lora_debug_reprocessed import validate_reprocess_candidate_rows


def test_validate_candidates_ok() -> None:
    rows = [
        {
            "user_id": "u1",
            "history_item_ids": ["a"],
            "target_item_id": "t1",
            "candidate_item_ids": ["x", "t1", "y"],
        }
    ]
    validate_reprocess_candidate_rows(rows, split="valid")


def test_validate_candidates_target_missing() -> None:
    rows = [{"user_id": "u1", "history_item_ids": [], "target_item_id": "t1", "candidate_item_ids": ["x", "y"]}]
    with pytest.raises(ValueError, match="target not in candidate"):
        validate_reprocess_candidate_rows(rows, split="test")


def test_validate_candidates_target_in_history() -> None:
    rows = [{"user_id": "u1", "history_item_ids": ["t1"], "target_item_id": "t1", "candidate_item_ids": ["t1", "x"]}]
    with pytest.raises(ValueError, match="target in history"):
        validate_reprocess_candidate_rows(rows, split="train")
