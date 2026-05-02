from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.protocol import (
    build_candidates_from_processed,
    iterative_k_core_filter,
    load_yaml_config,
    preprocess_from_config,
    protocol_config_from_dict,
    read_jsonl,
)


def _smoke_config(tmp_path: Path):
    cfg = load_yaml_config("configs/datasets/smoke_amazon_tiny.yaml")
    cfg["processed_dir"] = str(tmp_path / "processed")
    return protocol_config_from_dict(cfg)


def test_iterative_k_core_stable_output() -> None:
    interactions = pd.DataFrame(
        [
            ("u1", "i1", 5, 1),
            ("u1", "i2", 5, 2),
            ("u2", "i1", 5, 1),
            ("u2", "i2", 5, 2),
            ("u3", "i3", 5, 1),
        ],
        columns=["user_id", "item_id", "rating", "timestamp"],
    )
    filtered, trace = iterative_k_core_filter(interactions, k_core=2)
    assert set(filtered["user_id"]) == {"u1", "u2"}
    assert set(filtered["item_id"]) == {"i1", "i2"}
    assert trace[-1]["interactions_after"] == 4


def test_temporal_splits_and_candidates_are_deterministic_and_leakage_free(tmp_path: Path) -> None:
    cfg = _smoke_config(tmp_path)
    preprocess_from_config(cfg)
    first = build_candidates_from_processed(cfg.processed_dir, seed=13, negative_count=3)
    rows_first = read_jsonl(cfg.processed_dir / "test_candidates.jsonl")
    second = build_candidates_from_processed(cfg.processed_dir, seed=13, negative_count=3)
    rows_second = read_jsonl(cfg.processed_dir / "test_candidates.jsonl")
    assert first == second
    assert rows_first == rows_second
    valid_rows = read_jsonl(cfg.processed_dir / "valid.jsonl")
    test_rows = read_jsonl(cfg.processed_dir / "test.jsonl")
    for valid, test in zip(valid_rows, test_rows):
        assert valid["target_item_id"] not in valid["history_item_ids"]
        assert test["target_item_id"] not in test["history_item_ids"]
        assert valid["target_item_id"] in test["history_item_ids"]
    for row in rows_first:
        candidates = set(row["candidate_item_ids"])
        assert row["target_item_id"] in candidates
        assert len(candidates) == len(row["candidate_item_ids"])
        negatives = candidates - {row["target_item_id"]}
        assert not negatives.intersection(set(row["history_item_ids"]))
