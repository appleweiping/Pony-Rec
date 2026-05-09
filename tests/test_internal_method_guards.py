import pytest

from main_export_srpd_scores_from_predictions import _score_from_prediction
from main_select_ccrp_variant_on_valid import _evaluate_candidate_scores
from src.baselines.internal_scores import audit_score_degeneracy
from src.utils.io import save_jsonl
from src.shadow.ccrp import compute_ccrp_record, parse_weights


def test_score_degeneracy_flags_constant_event_scores():
    rows = [
        {"source_event_id": "e1", "user_id": "u", "item_id": "a", "score": 0.5},
        {"source_event_id": "e1", "user_id": "u", "item_id": "b", "score": 0.5},
        {"source_event_id": "e2", "user_id": "u", "item_id": "c", "score": 0.7},
        {"source_event_id": "e2", "user_id": "u", "item_id": "d", "score": 0.2},
    ]

    audit = audit_score_degeneracy(rows)

    assert audit["constant_score_event_count"] == 1
    assert audit["degeneracy_audit_ok"] is False
    assert audit["mean_unique_scores_per_event"] == 1.5


def test_ccrp_weight_grid_changes_uncertainty_components():
    record = {
        "relevance_probability": 0.9,
        "calibrated_relevance_probability": 0.7,
        "evidence_support": 0.8,
        "counterevidence_strength": 0.3,
    }

    boundary_only = compute_ccrp_record(record, weights=parse_weights([1.0, 0.0, 0.0]))
    evidence_only = compute_ccrp_record(record, weights=parse_weights([0.0, 0.0, 1.0]))

    assert boundary_only["ccrp_weight_boundary"] == 1.0
    assert evidence_only["ccrp_weight_evidence"] == 1.0
    assert boundary_only["ccrp_uncertainty"] != evidence_only["ccrp_uncertainty"]


def test_ccrp_ablation_labels_zero_removed_weight_components():
    record = {
        "relevance_probability": 0.9,
        "calibrated_relevance_probability": 0.7,
        "evidence_support": 0.8,
        "counterevidence_strength": 0.3,
    }

    no_gap = compute_ccrp_record(
        record,
        weights=parse_weights([0.2, 0.6, 0.2]),
        ablation="without_calibration_gap",
    )
    no_evidence = compute_ccrp_record(
        record,
        weights=parse_weights([0.2, 0.2, 0.6]),
        ablation="without_evidence_support",
    )

    assert no_gap["ccrp_weight_calibration_gap"] == 0.0
    assert no_evidence["ccrp_weight_evidence"] == 0.0


def test_parse_weights_requires_three_values():
    with pytest.raises(ValueError):
        parse_weights([0.5, 0.5])


def test_srpd_rank_fallback_score_is_detectable_by_missing_candidate_scores():
    prediction = {"pred_ranked_item_ids": ["i1", "i2", "i3"]}
    rank_lookup = {"i1": 0, "i2": 1, "i3": 2}

    assert _score_from_prediction(prediction, "i1", rank_lookup, 3) > _score_from_prediction(
        prediction,
        "i3",
        rank_lookup,
        3,
    )
    assert not isinstance(prediction.get("candidate_scores"), dict)


def test_srpd_partial_candidate_scores_use_rank_fallback_for_missing_item():
    prediction = {"pred_ranked_item_ids": ["i1", "i2"], "candidate_scores": {"i1": 0.9}}
    rank_lookup = {"i1": 0, "i2": 1}

    assert _score_from_prediction(prediction, "i1", rank_lookup, 2) == 0.9
    assert _score_from_prediction(prediction, "i2", rank_lookup, 2) == 0.5


def test_ccrp_validation_can_report_degenerate_scores_without_raising(tmp_path):
    ranking_path = tmp_path / "ranking.jsonl"
    candidate_path = tmp_path / "candidates.csv"
    signal_path = tmp_path / "signal.jsonl"
    save_jsonl(
        [
            {
                "source_event_id": "e1",
                "user_id": "u1",
                "candidate_item_ids": ["i1", "i2"],
                "positive_item_id": "i1",
            }
        ],
        ranking_path,
    )
    candidate_path.write_text(
        "source_event_id,user_id,item_id\n"
        "e1,u1,i1\n"
        "e1,u1,i2\n",
        encoding="utf-8",
    )
    save_jsonl(
        [
            {
                "source_event_id": "e1",
                "user_id": "u1",
                "candidate_item_id": "i1",
                "calibrated_relevance_probability": 0.5,
            },
            {
                "source_event_id": "e1",
                "user_id": "u1",
                "candidate_item_id": "i2",
                "calibrated_relevance_probability": 0.5,
            },
        ],
        signal_path,
    )

    metrics, _, _ = _evaluate_candidate_scores(
        ranking_path=ranking_path,
        candidate_items_path=candidate_path,
        signal_path=signal_path,
        score_mode="confidence_only",
        ablation="full",
        eta=1.0,
        confidence_weight=0.5,
        weights=(0.5, 0.3, 0.2),
        k=1,
        fail_on_degeneracy=False,
    )

    assert metrics["audit_ok"] is True
    assert metrics["degeneracy_audit_ok"] is False
    assert metrics["constant_score_event_count"] == 1
