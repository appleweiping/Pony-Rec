from __future__ import annotations

import pytest

from src.analysis.echo_chamber import echo_chamber_report
from src.eval.paper_metrics import exposure_metrics, groupwise_ece
from src.methods.uncertainty_reranking import RerankConfig, rerank_candidate_record, uncertainty_aware_score
from src.prompts.parsers import parse_pointwise_output, parse_ranking_output
from src.prompts.templates import candidate_block, get_prompt_template, history_block


def test_prompt_template_rendering_listwise_contains_allowed_ids() -> None:
    template = get_prompt_template("listwise_ranking_v1")
    text = template.render(history_block="1. item_id=i1", candidate_block="1. item_id=i2", allowed_item_ids="i2", topk=1)
    assert "Allowed item IDs: i2" in text
    assert "ranked_item_ids" in text


def test_history_and_candidate_block_render_ids_and_text() -> None:
    assert "Face Wash" in history_block(["i1"], {"i1": "Face Wash"})
    assert "item_id=i2" in candidate_block(["i2"], {"i2": "Serum"})


@pytest.mark.parametrize("raw,expected", [('{"recommend":"yes","confidence":"88%"}', 0.88), ("recommend: no\nconfidence: 0.25", 0.25)])
def test_yes_no_confidence_parser(raw: str, expected: float) -> None:
    parsed = parse_pointwise_output(raw)
    assert parsed.confidence == pytest.approx(expected)
    assert parsed.recommend in {"yes", "no"}


def test_listwise_ranking_parser_json() -> None:
    parsed = parse_ranking_output('{"ranked_item_ids":["i2","i1"],"confidence":0.7}', allowed_item_ids=["i1", "i2"])
    assert parsed.ranked_item_ids == ["i2", "i1"]
    assert parsed.is_valid is True


def test_hallucinated_item_detection() -> None:
    parsed = parse_ranking_output('{"ranked_item_ids":["bad"],"confidence":0.7}', allowed_item_ids=["i1"])
    assert parsed.hallucinated_item is True
    assert parsed.is_valid is False


def test_duplicate_item_detection() -> None:
    parsed = parse_ranking_output('{"ranked_item_ids":["i1","i1"],"confidence":0.7}', allowed_item_ids=["i1"])
    assert parsed.duplicate_item is True


def test_malformed_json_recovery() -> None:
    parsed = parse_ranking_output("{'ranked_item_ids':['i1',], 'confidence':'70%',}", allowed_item_ids=["i1"])
    assert parsed.repaired_json is True
    assert parsed.confidence == pytest.approx(0.7)


def test_groupwise_ece_by_popularity_bucket() -> None:
    values = groupwise_ece(
        [
            {"item_popularity_bucket": "head", "correctness": True, "calibrated_confidence": 0.5},
            {"item_popularity_bucket": "tail", "correctness": False, "calibrated_confidence": 0.5},
        ],
        group_key="item_popularity_bucket",
        n_bins=2,
    )
    assert values["head"] == pytest.approx(0.5)
    assert values["tail"] == pytest.approx(0.5)


def test_exposure_metric_known_values() -> None:
    metrics = exposure_metrics(
        [
            {"predicted_ranking": ["i1"], "candidate_item_ids": ["i1"], "candidate_popularity_buckets": ["head"]},
            {"predicted_ranking": ["i2"], "candidate_item_ids": ["i2"], "candidate_popularity_buckets": ["tail"]},
        ]
    )
    assert metrics["head_exposure_share"] == pytest.approx(0.5)
    assert metrics["tail_exposure_share"] == pytest.approx(0.5)


def test_echo_chamber_metric_known_values() -> None:
    report = echo_chamber_report(
        [
            {"predicted_ranking": ["i1"], "candidate_item_ids": ["i1"], "candidate_popularity_buckets": ["head"], "item_popularity_bucket": "head", "item_popularity_count": 10, "raw_confidence": 0.9},
            {"predicted_ranking": ["i2"], "candidate_item_ids": ["i2"], "candidate_popularity_buckets": ["tail"], "item_popularity_bucket": "tail", "item_popularity_count": 1, "raw_confidence": 0.2},
        ]
    )
    assert report["head_item_exposure_share"] == pytest.approx(0.5)
    assert report["head_tail_confidence_gap"] == pytest.approx(0.7)


def test_uncertainty_reranking_formula_phase1() -> None:
    assert uncertainty_aware_score(1.0, 0.4, 0.25) == pytest.approx(0.9)


def test_uncertainty_reranker_abstains_above_threshold() -> None:
    out = rerank_candidate_record(
        {"candidate_item_ids": ["i1"], "ranked_item_ids": ["i1"], "uncertainty_score": 0.9, "target_item_id": "i1"},
        RerankConfig(abstention_threshold=0.8),
    )
    assert out["abstained"] is True
    assert out["predicted_item_id"] == ""
