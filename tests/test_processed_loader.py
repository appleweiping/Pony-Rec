from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data.processed_loader import ProcessedDatasetLoader


def _clean_root(tmp_path: Path, domain: str = "amazon_beauty") -> Path:
    return tmp_path / "processed" / domain


def _write_processed(root: Path) -> None:
    root.mkdir(parents=True)
    pd.DataFrame(
        [
            ("u1", "i1", 5, 1),
            ("u1", "i2", 5, 2),
            ("u1", "i3", 5, 3),
            ("u1", "i4", 5, 4),
            ("u2", "i2", 4, 1),
            ("u2", "i3", 5, 2),
            ("u2", "i4", 5, 3),
            ("u2", "i5", 5, 4),
        ],
        columns=["user_id", "item_id", "rating", "timestamp"],
    ).to_csv(root / "interactions.csv", index=False)
    pd.DataFrame(
        [{"item_id": f"i{i}", "title": f"Item {i}", "categories": "Cat", "description": "Desc"} for i in range(1, 10)]
    ).to_csv(root / "items.csv", index=False)
    pd.DataFrame([{"user_id": "u1"}, {"user_id": "u2"}]).to_csv(root / "users.csv", index=False)
    pd.DataFrame(
        [
            {"item_id": "i1", "interaction_count": 1, "popularity_group": "tail"},
            {"item_id": "i2", "interaction_count": 2, "popularity_group": "head"},
            {"item_id": "i3", "interaction_count": 2, "popularity_group": "head"},
            {"item_id": "i4", "interaction_count": 2, "popularity_group": "head"},
            {"item_id": "i5", "interaction_count": 1, "popularity_group": "tail"},
        ]
    ).to_csv(root / "popularity_stats.csv", index=False)


def test_processed_files_load_and_normalize_schema(tmp_path: Path) -> None:
    root = _clean_root(tmp_path)
    _write_processed(root)
    ds = ProcessedDatasetLoader(root).load()
    assert list(ds.interactions.columns) == ["user_id", "item_id", "rating", "timestamp"]
    assert "candidate_text" in ds.items.columns
    assert ds.users["user_id"].is_unique


def test_required_column_detection(tmp_path: Path) -> None:
    root = _clean_root(tmp_path)
    _write_processed(root)
    pd.DataFrame([{"user_id": "u1", "rating": 5, "timestamp": 1}]).to_csv(root / "interactions.csv", index=False)
    with pytest.raises(ValueError, match="item_id"):
        ProcessedDatasetLoader(root).load()


def test_reject_srpd_directory_as_dataset_source(tmp_path: Path) -> None:
    root = tmp_path / "processed" / "amazon_beauty" / "srpd"
    _write_processed(root)
    with pytest.raises(ValueError, match="Rejected"):
        ProcessedDatasetLoader(root, allowed_domains=None).load()


@pytest.mark.parametrize("domain", ["amazon_books_small", "amazon_movies_noisy_nl10"])
def test_reject_small_and_noisy_dataset_sources(tmp_path: Path, domain: str) -> None:
    root = tmp_path / "processed" / domain
    _write_processed(root)
    with pytest.raises(ValueError, match="Rejected|Unsupported"):
        ProcessedDatasetLoader(root, allowed_domains=None).load()


def test_reject_old_split_or_prediction_artifacts(tmp_path: Path) -> None:
    root = _clean_root(tmp_path)
    _write_processed(root)
    (root / "train.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="legacy artifacts"):
        ProcessedDatasetLoader(root).load()


def test_reject_non_clean_processed_root(tmp_path: Path) -> None:
    root = tmp_path / "processed_old" / "amazon_beauty"
    _write_processed(root)
    with pytest.raises(ValueError, match="processed"):
        ProcessedDatasetLoader(root).load()


def test_item_user_consistency(tmp_path: Path) -> None:
    root = _clean_root(tmp_path)
    _write_processed(root)
    pd.DataFrame([{"item_id": "i1", "title": "Item 1"}]).to_csv(root / "items.csv", index=False)
    with pytest.raises(ValueError, match="items.csv missing"):
        ProcessedDatasetLoader(root).load()


def test_temporal_split_no_leakage(tmp_path: Path) -> None:
    root = _clean_root(tmp_path)
    _write_processed(root)
    ds = ProcessedDatasetLoader(root).load()
    splits = ProcessedDatasetLoader.temporal_leave_one_out_split(ds.interactions, min_sequence_length=3)
    assert {len(rows) for rows in splits.values()} == {2}
    for valid, test in zip(splits["valid"], splits["test"]):
        assert valid["target_item_id"] not in valid["history_item_ids"]
        assert test["target_item_id"] not in test["history_item_ids"]
        assert valid["timestamp"] < test["timestamp"]


def test_negative_candidates_no_target_or_history_leakage(tmp_path: Path) -> None:
    root = _clean_root(tmp_path)
    _write_processed(root)
    ds = ProcessedDatasetLoader(root).load()
    splits = ProcessedDatasetLoader.temporal_leave_one_out_split(ds.interactions, min_sequence_length=3)
    candidates = ProcessedDatasetLoader.build_candidates(splits, ds.items, seed=7, negative_count=2)
    for rows in candidates.values():
        for row in rows:
            candidate_set = set(row["candidate_item_ids"])
            assert row["target_item_id"] in candidate_set
            negatives = candidate_set - {row["target_item_id"]}
            assert row["target_item_id"] not in negatives
            assert not negatives.intersection(set(row["history_item_ids"]))
            assert not negatives.intersection(set(row["full_positive_item_ids"]))


def test_popularity_recomputation_matches_or_reports_mismatch(tmp_path: Path) -> None:
    root = _clean_root(tmp_path)
    _write_processed(root)
    ds = ProcessedDatasetLoader(root).load()
    recomputed = ProcessedDatasetLoader.recompute_popularity(ds.interactions)
    assert ProcessedDatasetLoader.compare_popularity(ds.popularity_stats, recomputed)["matches"] is True
    broken = ds.popularity_stats.copy()
    broken.loc[broken["item_id"] == "i1", "interaction_count"] = 99
    report = ProcessedDatasetLoader.compare_popularity(broken, recomputed)
    assert report["matches"] is False
    assert report["mismatched_items"] == 1


def test_official_dataset_configs_use_clean_processed_entry() -> None:
    config_paths = [
        Path("configs/datasets/amazon_beauty.yaml"),
        Path("configs/datasets/amazon_books.yaml"),
        Path("configs/datasets/amazon_electronics.yaml"),
        Path("configs/data/amazon_beauty.yaml"),
        Path("configs/data/amazon_books.yaml"),
        Path("configs/data/amazon_electronics.yaml"),
        Path("configs/data/amazon_movies.yaml"),
        Path("configs/lora/qwen_small_debug.yaml"),
        Path("configs/lora/qwen_server_rank.yaml"),
    ]
    for path in config_paths:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        processed_dir = str(cfg.get("processed_dir", ""))
        assert processed_dir.startswith("data/processed/"), path
        assert "_small" not in processed_dir
        assert "_noisy" not in processed_dir
