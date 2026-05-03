"""Strict listwise JSON parse path for CARE-LoRA (no GPU)."""

from __future__ import annotations

import json

from src.prompts.parsers import (
    extract_first_json_object,
    parse_ranking_output,
    ranking_parse_strict_for_prompt,
)


def test_ranking_parse_strict_for_prompt_flag() -> None:
    assert ranking_parse_strict_for_prompt("listwise_ranking_json_lora")
    assert not ranking_parse_strict_for_prompt("listwise_ranking_v1")


def test_strict_accepts_canonical_permutation() -> None:
    allowed = ["a", "b", "c"]
    text = '{"ranking":["c","a","b"],"confidence":0.25}'
    out = parse_ranking_output(text, allowed_item_ids=allowed, strict_json_only=True)
    assert out.is_valid
    assert out.ranked_item_ids == ["c", "a", "b"]
    assert out.confidence == 0.25


def test_strict_rejects_ids_outside_candidate_set() -> None:
    allowed = ["a", "b"]
    text = '{"ranking":["a","b","ghost"],"confidence":0.5}'
    out = parse_ranking_output(text, allowed_item_ids=allowed, strict_json_only=True)
    assert not out.is_valid
    assert out.hallucinated_item
    assert out.ranked_item_ids == []


def test_strict_rejects_duplicates_even_if_subset_complete() -> None:
    allowed = ["a", "b", "c"]
    text = '{"ranking":["a","a","b"],"confidence":0.5}'
    out = parse_ranking_output(text, allowed_item_ids=allowed, strict_json_only=True)
    assert not out.is_valid
    assert out.duplicate_item


def test_extract_first_json_object_skips_leading_prose() -> None:
    s = 'preamble\n{"ranking":["y","x"],"confidence":0.9}\n'
    blob = extract_first_json_object(s)
    assert blob is not None
    assert json.loads(blob)["ranking"] == ["y", "x"]


def test_strict_uses_first_json_object_after_prose() -> None:
    allowed = ["x", "y"]
    text = 'Sure.\n{"ranking":["y","x"], "confidence": 0.9}\n'
    out = parse_ranking_output(text, allowed_item_ids=allowed, strict_json_only=True)
    assert out.is_valid


def test_strict_json_repair_trailing_comma() -> None:
    allowed = ["a", "b"]
    text = '{"ranking":["b","a"], "confidence": 0.5,}'
    out = parse_ranking_output(text, allowed_item_ids=allowed, strict_json_only=True)
    assert out.is_valid
    assert out.repaired_json


def test_strict_no_target_side_completion() -> None:
    """Partial ranking stays invalid; parser does not invent missing slots."""
    allowed = ["t", "u", "v"]
    text = '{"ranking":["t"],"confidence":0.1}'
    out = parse_ranking_output(text, allowed_item_ids=allowed, strict_json_only=True)
    assert not out.is_valid
    assert out.ranked_item_ids == []
