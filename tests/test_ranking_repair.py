from __future__ import annotations

import json
from pathlib import Path

from src.parsing.ranking_repair import (
    build_repair_summary,
    failure_taxonomy_row,
    safe_repair_ranking,
    write_failure_taxonomy_csv,
)


def test_no_target_leakage_candidate_completion() -> None:
    allowed = ["A", "B", "C"]
    out = safe_repair_ranking(
        raw_text="n/a",
        allowed_item_ids=allowed,
        strict_json_valid=False,
        strict_ranking=[],
        strict_confidence_available=False,
    )
    assert out.usable_ranking
    assert set(out.ranking) == set(allowed)
    assert out.ranking == allowed


def test_dedupe_preserves_candidate_set() -> None:
    allowed = ["A", "B", "C"]
    raw = '{"ranking":["B","B","A"],"confidence":0.2}'
    out = safe_repair_ranking(
        raw_text=raw,
        allowed_item_ids=allowed,
        strict_json_valid=False,
        strict_ranking=[],
        strict_confidence_available=False,
    )
    assert out.usable_ranking
    assert set(out.ranking) == set(allowed)
    assert len(out.ranking) == 3


def test_ood_ids_rejected_in_seed_then_completed_with_allowed_only() -> None:
    allowed = ["A", "B", "C"]
    raw = '{"ranking":["A","OOD","B"],"confidence":0.4}'
    out = safe_repair_ranking(
        raw_text=raw,
        allowed_item_ids=allowed,
        strict_json_valid=False,
        strict_ranking=[],
        strict_confidence_available=False,
    )
    assert set(out.ranking) == set(allowed)
    assert "OOD" not in out.ranking


def test_strict_valid_unchanged_and_no_repair() -> None:
    allowed = ["A", "B"]
    strict = ["B", "A"]
    out = safe_repair_ranking(
        raw_text='{"ranking":["B","A"],"confidence":0.7}',
        allowed_item_ids=allowed,
        strict_json_valid=True,
        strict_ranking=strict,
        strict_confidence_available=True,
    )
    assert out.strict_json_valid
    assert out.ranking == strict
    assert out.repaired_by is None
    assert out.confidence_available_after_repair is True


def test_confidence_not_hallucinated_by_repair() -> None:
    out = safe_repair_ranking(
        raw_text="no confidence",
        allowed_item_ids=["A", "B"],
        strict_json_valid=False,
        strict_ranking=[],
        strict_confidence_available=False,
    )
    assert out.confidence_available_after_repair is False


def test_failure_taxonomy_smoke() -> None:
    row = failure_taxonomy_row(
        user_id="u1",
        request_id="valid:u1",
        raw_text="hello",
        strict_json_valid=False,
        strict_ranking=[],
        allowed_item_ids=["A", "B"],
        missing_confidence=True,
        malformed_json=False,
        duplicate_item=False,
        output_not_in_candidate_set=False,
        max_new_tokens=64,
    )
    assert row["primary_failure"] in {"no_json_object", "other"}


def test_cli_postprocess_like_smoke(tmp_path: Path) -> None:
    taxonomy = [
        {
            "request_id": "r1",
            "user_id": "u1",
            "strict_json_valid": False,
            "primary_failure": "no_json_object",
            "no_json_object": 1,
            "prose_before_after_json": 0,
            "incomplete_ranking": 0,
            "duplicate_item_ids": 0,
            "ood_item_ids": 0,
            "malformed_quotes_brackets": 0,
            "confidence_missing": 1,
            "thinking_text_leaked": 0,
            "truncation_or_max_new_tokens_issue": 0,
            "other": 0,
        }
    ]
    out_csv = tmp_path / "format_failure_taxonomy.csv"
    write_failure_taxonomy_csv(taxonomy, out_csv)
    assert out_csv.is_file()
    assert "primary_failure" in out_csv.read_text(encoding="utf-8")

    summary = build_repair_summary(
        pred_rows=[
            {
                "strict_json_valid": False,
                "usable_ranking": True,
                "confidence_available_strict": False,
                "confidence_available": False,
                "repair_reason": "strict_json_invalid_then_candidate_completion",
            }
        ],
        taxonomy_rows=taxonomy,
    )
    assert summary["usable_ranking_rate_after_safe_repair"] == 1.0
    assert summary["strict_json_valid_rate"] == 0.0
    assert json.loads(json.dumps(summary))["repair_reason_counts"]["strict_json_invalid_then_candidate_completion"] == 1
