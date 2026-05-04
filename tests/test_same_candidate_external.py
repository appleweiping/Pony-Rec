from __future__ import annotations

from src.baselines.same_candidate_external import build_predictions_from_external_scores


def test_external_scores_rank_exact_candidates_by_source_event() -> None:
    ranking_samples = [
        {
            "source_event_id": "e1",
            "user_id": "u1",
            "split_name": "test",
            "timestamp": 1,
            "positive_item_id": "i2",
            "candidate_item_ids": ["i1", "i2", "i3"],
            "candidate_titles": ["one", "two", "three"],
            "candidate_texts": ["one", "two", "three"],
            "candidate_popularity_groups": ["head", "tail", "mid"],
        }
    ]
    score_rows = [
        {"source_event_id": "e1", "user_id": "u1", "item_id": "i1", "score": "0.2"},
        {"source_event_id": "e1", "user_id": "u1", "item_id": "i2", "score": "0.9"},
        {"source_event_id": "e1", "user_id": "u1", "item_id": "i3", "score": "0.1"},
    ]

    predictions, summary = build_predictions_from_external_scores(
        ranking_samples,
        score_rows,
        baseline_name="sasrec",
        k=2,
    )

    assert predictions[0]["pred_ranked_item_ids"] == ["i2", "i1", "i3"]
    assert predictions[0]["topk_item_ids"] == ["i2", "i1"]
    assert predictions[0]["external_score_coverage"] == 1.0
    assert summary["score_coverage_rate"] == 1.0


def test_external_scores_keep_candidate_order_for_missing_ties() -> None:
    ranking_samples = [
        {
            "source_event_id": "e1",
            "user_id": "u1",
            "positive_item_id": "i3",
            "candidate_item_ids": ["i1", "i2", "i3"],
            "candidate_popularity_groups": ["head", "tail", "mid"],
        }
    ]
    score_rows = [
        {"user_id": "u1", "item_id": "i2", "score": "1.0"},
    ]

    predictions, summary = build_predictions_from_external_scores(
        ranking_samples,
        score_rows,
        baseline_name="external",
    )

    assert predictions[0]["pred_ranked_item_ids"] == ["i2", "i1", "i3"]
    assert predictions[0]["external_missing_score_count"] == 2
    assert summary["matched_candidates"] == 1
