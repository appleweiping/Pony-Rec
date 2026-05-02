from __future__ import annotations

import pytest

from src.eval.paper_metrics import brier_score, ece_mce, ranking_metrics
from src.methods.uncertainty_reranking import uncertainty_aware_score
from src.prompts.parsers import parse_pointwise_output, parse_ranking_output


def test_parser_flags_malformed_hallucinated_duplicate_outputs() -> None:
    parsed = parse_ranking_output(
        "{'ranked_item_ids':['I1','I1','BAD'], 'confidence':'80%',}",
        allowed_item_ids=["I1", "I2"],
    )
    assert parsed.repaired_json is True
    assert parsed.duplicate_item is True
    assert parsed.hallucinated_item is True
    assert parsed.output_not_in_candidate_set is True
    assert parsed.is_valid is False


def test_pointwise_parser_extracts_confidence_without_silent_missing_coercion() -> None:
    parsed = parse_pointwise_output('{"recommend":"yes","confidence":0.75}')
    assert parsed.is_valid is True
    assert parsed.confidence == pytest.approx(0.75)
    missing = parse_pointwise_output('{"recommend":"yes"}')
    assert missing.is_valid is False
    assert missing.missing_confidence is True


def test_ranking_metrics_known_values() -> None:
    metrics = ranking_metrics(
        [
            {"target_item_id": "A", "predicted_ranking": ["A", "B", "C"]},
            {"target_item_id": "B", "predicted_ranking": ["A", "B", "C"]},
        ],
        ks=(1, 2),
    )
    assert metrics["HR@1"] == pytest.approx(0.5)
    assert metrics["Recall@2"] == pytest.approx(1.0)
    assert metrics["MRR@2"] == pytest.approx(0.75)


def test_calibration_metrics_known_values() -> None:
    assert brier_score([1, 0], [0.75, 0.25]) == pytest.approx(0.0625)
    ece, mce = ece_mce([1, 0], [0.75, 0.25], n_bins=2)
    assert ece == pytest.approx(0.25)
    assert mce == pytest.approx(0.25)


def test_uncertainty_reranking_formula() -> None:
    assert uncertainty_aware_score(0.8, 0.3, 0.5) == pytest.approx(0.65)
