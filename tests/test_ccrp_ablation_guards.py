import pytest

from src.shadow.ccrp import compute_ccrp_record, parse_weights


def test_without_boundary_uncertainty_ablation_removes_boundary_weight():
    record = {
        "relevance_probability": 0.9,
        "calibrated_relevance_probability": 0.7,
        "evidence_support": 0.8,
        "counterevidence_strength": 0.3,
    }

    no_boundary = compute_ccrp_record(
        record,
        weights=parse_weights([0.6, 0.2, 0.2]),
        ablation="without_boundary_uncertainty",
    )
    only_boundary_removed = compute_ccrp_record(
        record,
        weights=parse_weights([1.0, 0.0, 0.0]),
        ablation="without_boundary_uncertainty",
    )

    assert no_boundary["ccrp_weight_boundary"] == 0.0
    assert only_boundary_removed["ccrp_weight_boundary"] == 0.0
    assert (
        only_boundary_removed["ccrp_weight_calibration_gap"]
        + only_boundary_removed["ccrp_weight_evidence"]
    ) == pytest.approx(1.0)
