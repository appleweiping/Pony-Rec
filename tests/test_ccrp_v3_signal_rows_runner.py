import csv

from experiments.rsc.run_ccrp_v3_signal_rows import (
    SIGNAL_SCHEMA_VERSION,
    iter_task_prompts,
    parse_signal_response,
    write_signal_rows_csv,
)
from scripts.audit.main_audit_ccrp_uncertainty_sources import audit_source


def test_parse_signal_response_reads_structured_json():
    parsed = parse_signal_response(
        '{"relevance_probability": 0.8, "calibrated_relevance_probability": 0.7, '
        '"evidence_support": 0.6, "counterevidence_strength": 0.2, "reason": "fits history"}'
    )

    assert parsed["parse_success"] is True
    assert parsed["relevance_probability"] == 0.8
    assert parsed["calibrated_relevance_probability"] == 0.7
    assert parsed["evidence_support"] == 0.6
    assert parsed["counterevidence_strength"] == 0.2
    assert parsed["reason"] == "fits history"


def test_parse_signal_response_falls_back_conservatively():
    parsed = parse_signal_response("probability maybe 0.42")

    assert parsed["parse_success"] is False
    assert parsed["relevance_probability"] == 0.42
    assert parsed["calibrated_relevance_probability"] == 0.42
    assert parsed["evidence_support"] == 0.0
    assert parsed["counterevidence_strength"] == 1.0


def test_iter_task_prompts_preserves_same_candidate_identity():
    records = [
        {
            "source_event_id": "evt1",
            "user_id": "u1",
            "history": ["old item"],
            "candidate_titles": ["hammer", "saw"],
            "candidate_item_ids": ["i1", "i2"],
            "candidate_texts": ["steel hammer", "hand saw"],
        }
    ]

    prompts, meta = iter_task_prompts(records)

    assert len(prompts) == 2
    assert "Return ONLY JSON" in prompts[0]
    assert meta == [
        {
            "source_event_id": "evt1",
            "user_id": "u1",
            "candidate_item_id": "i1",
            "item_id": "i1",
            "candidate_idx": 0,
        },
        {
            "source_event_id": "evt1",
            "user_id": "u1",
            "candidate_item_id": "i2",
            "item_id": "i2",
            "candidate_idx": 1,
        },
    ]


def test_written_signal_rows_are_recomputable_for_auditor(tmp_path):
    signal_path = tmp_path / "test_ccrp_signal_rows.csv"
    candidate_path = tmp_path / "candidate_items.csv"
    rows = [
        {
            "source_event_id": "evt1",
            "user_id": "u1",
            "candidate_item_id": "i1",
            "item_id": "i1",
            "candidate_idx": 0,
            "relevance_probability": 0.8,
            "calibrated_relevance_probability": 0.75,
            "evidence_support": 0.7,
            "counterevidence_strength": 0.1,
            "reason": "matches",
            "parse_success": True,
            "signal_schema_version": SIGNAL_SCHEMA_VERSION,
        },
        {
            "source_event_id": "evt1",
            "user_id": "u1",
            "candidate_item_id": "i2",
            "item_id": "i2",
            "candidate_idx": 1,
            "relevance_probability": 0.2,
            "calibrated_relevance_probability": 0.25,
            "evidence_support": 0.2,
            "counterevidence_strength": 0.5,
            "reason": "weak",
            "parse_success": True,
            "signal_schema_version": SIGNAL_SCHEMA_VERSION,
        },
    ]
    write_signal_rows_csv(signal_path, rows)
    with candidate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_event_id", "user_id", "item_id"])
        writer.writeheader()
        writer.writerow({"source_event_id": "evt1", "user_id": "u1", "item_id": "i1"})
        writer.writerow({"source_event_id": "evt1", "user_id": "u1", "item_id": "i2"})

    candidate_keys = {("evt1", "u1", "i1"), ("evt1", "u1", "i2")}
    audit = audit_source(
        label="signal",
        path=signal_path,
        candidate_keys=candidate_keys,
        expected_events=1,
        expected_candidates_per_event=2,
    )

    assert audit["status"] == "recomputable_signal_rows"
    assert audit["recomputable_signal_rows"] is True
    assert audit["candidate_key_coverage_rate"] == 1.0
    assert audit["failures"] == ["missing_uncertainty_column"]
