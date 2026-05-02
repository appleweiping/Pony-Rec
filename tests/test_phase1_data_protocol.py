from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.protocol import (
    DataProtocolConfig,
    build_candidates_from_processed,
    build_user_sequences,
    item_popularity_from_train,
    load_raw_interactions,
    load_raw_items,
    popularity_buckets,
    preprocess_from_config,
    read_jsonl,
    temporal_leave_one_out,
    write_jsonl,
)
from src.data.raw_validation import validate_raw_data_config


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_amazon_raw_review_parser_normalizes_fields(tmp_path: Path) -> None:
    review = tmp_path / "reviews.jsonl"
    meta = tmp_path / "meta.jsonl"
    _write_jsonl(review, [{"reviewerID": "u1", "asin": "i1", "overall": 4.0, "unixReviewTime": 7}])
    _write_jsonl(meta, [{"asin": "i1", "title": "Item", "categories": ["A"], "description": "Desc"}])
    cfg = DataProtocolConfig("d", "D", review, meta, tmp_path / "processed")
    df = load_raw_interactions(cfg)
    assert df.to_dict(orient="records")[0] == {"user_id": "u1", "item_id": "i1", "rating": 4.0, "timestamp": 7}


def test_amazon_metadata_parser_builds_candidate_text(tmp_path: Path) -> None:
    review = tmp_path / "reviews.jsonl"
    meta = tmp_path / "meta.jsonl"
    _write_jsonl(review, [{"user_id": "u1", "parent_asin": "i1", "rating": 5, "timestamp": 1}])
    _write_jsonl(meta, [{"parent_asin": "i1", "title": "Serum", "categories": ["Beauty"], "description": ["Hydrating"]}])
    cfg = DataProtocolConfig("d", "D", review, meta, tmp_path / "processed")
    items = load_raw_items(cfg, {"i1"})
    assert "Title: Serum" in items.iloc[0]["candidate_text"]
    assert "Categories: Beauty" in items.iloc[0]["candidate_text"]


def test_movie_domain_parser_reads_csv(tmp_path: Path) -> None:
    ratings = tmp_path / "ratings.csv"
    movies = tmp_path / "movies.csv"
    ratings.write_text("userId,movieId,rating,timestamp\n1,10,5,123\n", encoding="utf-8")
    movies.write_text("movieId,title,genres\n10,Space,Drama|Sci-Fi\n", encoding="utf-8")
    cfg = DataProtocolConfig("movie", "Movie", ratings, movies, tmp_path / "processed", raw_format="movie")
    assert load_raw_interactions(cfg).iloc[0]["item_id"] == "10"
    assert "Sci-Fi" in load_raw_items(cfg, {"10"}).iloc[0]["candidate_text"]


@pytest.mark.parametrize("k_core,expected_rows", [(3, 9), (5, 0), (10, 0)])
def test_iterative_k_core_3_5_10_behavior(k_core: int, expected_rows: int) -> None:
    rows = []
    for user in ["u1", "u2", "u3"]:
        for item in ["i1", "i2", "i3"]:
            rows.append({"user_id": user, "item_id": item, "rating": 5, "timestamp": len(rows)})
    from src.data.protocol import iterative_k_core_filter

    filtered, _ = iterative_k_core_filter(pd.DataFrame(rows), k_core=k_core)
    assert len(filtered) == expected_rows


def test_rating_positive_filter_keeps_rating_at_least_four(tmp_path: Path) -> None:
    review = tmp_path / "reviews.jsonl"
    meta = tmp_path / "meta.jsonl"
    _write_jsonl(
        review,
        [
            {"user_id": "u1", "parent_asin": "i1", "rating": 3, "timestamp": 1},
            {"user_id": "u1", "parent_asin": "i2", "rating": 4, "timestamp": 2},
        ],
    )
    _write_jsonl(meta, [{"parent_asin": "i2", "title": "Item"}])
    cfg = DataProtocolConfig("d", "D", review, meta, tmp_path / "processed", k_core=1)
    stats = preprocess_from_config(cfg)
    assert stats["positive_interactions"] == 1


def test_temporal_leave_one_out_targets_are_last_two() -> None:
    splits = temporal_leave_one_out({"u": [{"item_id": f"i{i}", "timestamp": i, "rating": 5} for i in range(5)]})
    assert splits["valid"][0]["target_item_id"] == "i3"
    assert splits["test"][0]["target_item_id"] == "i4"
    assert splits["test"][0]["history_item_ids"] == ["i0", "i1", "i2", "i3"]


def test_no_validation_or_test_target_in_own_history() -> None:
    splits = temporal_leave_one_out({"u": [{"item_id": f"i{i}", "timestamp": i, "rating": 5} for i in range(5)]})
    assert splits["valid"][0]["target_item_id"] not in splits["valid"][0]["history_item_ids"]
    assert splits["test"][0]["target_item_id"] not in splits["test"][0]["history_item_ids"]


def test_train_only_popularity_statistics_ignore_valid_test_targets() -> None:
    rows_by_split = {
        "train": [{"train_item_ids": ["i1", "i1", "i2"]}],
        "valid": [{"target_item_id": "valid_only"}],
        "test": [{"target_item_id": "test_only"}],
    }
    counts = item_popularity_from_train(rows_by_split)
    assert counts["i1"] == 2
    assert counts["valid_only"] == 0


def test_head_mid_tail_bucket_assignment() -> None:
    buckets = popularity_buckets(__import__("collections").Counter({"a": 10, "b": 5, "c": 1, "d": 1}), head_ratio=0.25, mid_ratio=0.5)
    assert buckets["a"] == "head"
    assert buckets["b"] == "mid"
    assert buckets["d"] == "tail"


@pytest.mark.parametrize("negative_count", [19, 99, 199])
def test_candidate_size_19_99_199(tmp_path: Path, negative_count: int) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame(
        [{"item_id": f"i{i}", "title": f"Item {i}", "candidate_text": f"Item {i}", "item_popularity_count": i, "item_popularity_bucket": "mid"} for i in range(250)]
    ).to_csv(processed / "items.csv", index=False)
    row = {"user_id": "u", "history_item_ids": ["i1"], "target_item_id": "i0", "full_positive_item_ids": ["i0", "i1"], "timestamp": 1}
    for split in ["train", "valid", "test"]:
        write_jsonl([row], processed / f"{split}.jsonl")
    build_candidates_from_processed(processed, seed=1, negative_count=negative_count)
    out = read_jsonl(processed / "test_candidates.jsonl")[0]
    assert len(out["candidate_item_ids"]) == negative_count + 1
    assert "i0" in out["candidate_item_ids"]
    assert "i1" not in set(out["candidate_item_ids"]) - {"i0"}


def test_raw_validator_reports_missing_paths_precisely() -> None:
    report = validate_raw_data_config(
        {
            "dataset": "missing",
            "domain": "Missing",
            "raw": {"format": "amazon", "review_path": "missing_reviews.jsonl", "meta_path": "missing_meta.jsonl"},
            "processed_dir": "unused",
        }
    )
    assert report.ok is False
    assert "missing_reviews.jsonl" in report.errors[0]


def test_raw_validator_reports_metadata_coverage(tmp_path: Path) -> None:
    review = tmp_path / "reviews.jsonl"
    meta = tmp_path / "meta.jsonl"
    _write_jsonl(review, [{"user_id": "u1", "parent_asin": "i1", "rating": 5, "timestamp": 1}])
    _write_jsonl(meta, [{"parent_asin": "i1", "title": "Item"}])
    report = validate_raw_data_config(
        {"dataset": "d", "domain": "D", "raw": {"review_path": str(review), "meta_path": str(meta)}, "processed_dir": str(tmp_path / "p"), "filter": {"k_core": 1}},
        min_post_filter_interactions=1,
    )
    assert report.ok is True
    assert report.metadata_coverage == pytest.approx(1.0)
