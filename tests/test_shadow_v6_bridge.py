from __future__ import annotations

from src.shadow.decision_bridge import (
    build_shadow_v6_bridge_rows,
    build_shadow_v6_decision,
    build_shadow_v6_decision_predictions,
    rank_shadow_v6_bridge_rows,
)
from src.shadow.scoring import compute_shadow_scores


def test_shadow_v6_decision_accepts_reliable_signal() -> None:
    decision = build_shadow_v6_decision(
        signal_score=0.9,
        signal_uncertainty=0.1,
        anchor_score=0.5,
        matched_signal=True,
    )

    assert decision["fallback_flag"] is False
    assert decision["correction_gate"] > 0.0
    assert decision["decision_score"] > 0.5
    assert decision["pair_type"] == "shadow_preferred_over_anchor"
    assert decision["pair_weight"] > 0.0


def test_shadow_v6_decision_falls_back_without_signal() -> None:
    decision = build_shadow_v6_decision(
        signal_score=0.7,
        signal_uncertainty=0.2,
        anchor_score=0.4,
        matched_signal=False,
    )

    assert decision["fallback_flag"] is True
    assert decision["fallback_reason"] == "missing_signal"
    assert decision["correction_gate"] == 0.0
    assert decision["decision_score"] == 0.4
    assert decision["pair_weight"] == 0.0


def test_shadow_v6_bridge_rows_rank_by_decision_score() -> None:
    ranking_records = [
        {
            "user_id": "u1",
            "source_event_id": "u1::1",
            "positive_item_id": "b",
            "candidate_item_ids": ["a", "b", "c"],
            "candidate_titles": ["A", "B", "C"],
            "candidate_popularity_groups": ["head", "tail", "mid"],
            "pred_ranked_item_ids": ["a", "b", "c"],
            "parse_success": True,
        }
    ]
    signal_records = [
        {
            "user_id": "u1",
            "candidate_item_id": "a",
            "shadow_calibrated_score": 0.2,
            "shadow_uncertainty": 0.1,
        },
        {
            "user_id": "u1",
            "candidate_item_id": "b",
            "shadow_calibrated_score": 0.95,
            "shadow_uncertainty": 0.05,
        },
    ]

    rows = build_shadow_v6_bridge_rows(ranking_records, signal_records)
    ranked_rows = rank_shadow_v6_bridge_rows(rows)
    predictions = build_shadow_v6_decision_predictions(ranked_rows, ranking_records, topk=2)

    assert len(rows) == 3
    assert predictions[0]["pred_ranked_item_ids"][0] == "b"
    missing_row = next(row for row in rows if row["candidate_item_id"] == "c")
    assert missing_row["matched_signal"] is False
    assert missing_row["fallback_flag"] is True


def test_shadow_v6_scoring_uses_anchor_disagreement() -> None:
    low_disagreement = compute_shadow_scores(
        {
            "decision_score": 0.7,
            "signal_score": 0.7,
            "signal_uncertainty": 0.2,
            "correction_gate": 0.7,
            "anchor_score": 0.65,
            "anchor_disagreement": 0.05,
        },
        variant="shadow_v6",
    )
    high_disagreement = compute_shadow_scores(
        {
            "decision_score": 0.7,
            "signal_score": 0.7,
            "signal_uncertainty": 0.2,
            "correction_gate": 0.7,
            "anchor_score": 0.1,
            "anchor_disagreement": 0.6,
        },
        variant="shadow_v6",
    )

    assert high_disagreement["shadow_uncertainty"] > low_disagreement["shadow_uncertainty"]
