from __future__ import annotations

from src.uncertainty.features import extract_care_probe_features, summarize_probe_rows


def test_extract_care_probe_rank_target_and_buckets() -> None:
    row = {
        "candidate_item_ids": ["a", "b", "c"],
        "candidate_popularity_buckets": ["head", "tail", "tail"],
        "predicted_ranking": ["b", "a", "c"],
        "predicted_item_id": "b",
        "target_item_id": "c",
        "correctness": False,
        "raw_confidence": 0.85,
        "missing_confidence": False,
        "item_popularity_bucket": "tail",
    }
    f = extract_care_probe_features(row)
    assert f["rank_target_in_predicted_list"] == 3
    assert f["predicted_top_popularity_bucket"] == "tail"
    assert f["target_popularity_bucket"] == "tail"
    assert f["high_confidence_wrong"] is True
    assert f["head_top1_wrong"] is False


def test_extract_low_confidence_correct() -> None:
    row = {
        "candidate_item_ids": ["x", "y"],
        "candidate_popularity_buckets": ["head", "tail"],
        "predicted_ranking": ["x"],
        "predicted_item_id": "x",
        "target_item_id": "x",
        "correctness": True,
        "raw_confidence": 0.35,
        "missing_confidence": False,
    }
    f = extract_care_probe_features(row)
    assert f["low_confidence_correct"] is True


def test_summarize_probe_rows() -> None:
    rows = [
        {
            "correctness": True,
            "care_features": {"missing_confidence": False, "high_confidence_wrong": False, "verbalized_raw_confidence": 0.9},
        },
        {
            "correctness": False,
            "care_features": {"missing_confidence": True, "high_confidence_wrong": True, "verbalized_raw_confidence": 0.8},
        },
    ]
    s = summarize_probe_rows(rows)
    assert s["n_rows"] == 2
    assert s["accuracy"] == 0.5
    assert s["missing_confidence_rate"] == 0.5
    assert s["high_confidence_wrong_rate"] == 0.5
