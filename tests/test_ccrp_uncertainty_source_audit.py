from scripts.audit.main_audit_ccrp_uncertainty_sources import audit_sources


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_audit_marks_score_only_file_as_not_uncertainty_source(tmp_path):
    candidates = _write(
        tmp_path / "candidate_items.csv",
        "source_event_id,user_id,item_id\n"
        "e1,u1,i1\n"
        "e1,u1,i2\n",
    )
    score_only = _write(
        tmp_path / "scores.csv",
        "source_event_id,user_id,item_id,score\n"
        "e1,u1,i1,0.1\n"
        "e1,u1,i2,0.2\n",
    )

    payload = audit_sources(
        sources=[f"score_only={score_only}"],
        candidate_items_path=str(candidates),
        expected_events=1,
        expected_candidates_per_event=2,
    )

    row = payload["sources"][0]
    assert row["status"] == "score_only_not_uncertainty"
    assert row["paper_ready_uncertainty_rows"] is False
    assert "missing_uncertainty_column" in row["failures"]
    assert row["candidate_key_coverage_rate"] == 1.0


def test_audit_accepts_full_scored_rows_with_uncertainty(tmp_path):
    candidates = _write(
        tmp_path / "candidate_items.csv",
        "source_event_id,user_id,item_id\n"
        "e1,u1,i1\n"
        "e1,u1,i2\n",
    )
    scored = _write(
        tmp_path / "ccrp_selected_test_scored_rows.csv",
        "source_event_id,user_id,candidate_item_id,ccrp_uncertainty,ccrp_boundary_uncertainty,"
        "ccrp_calibration_gap,ccrp_evidence_uncertainty,ccrp_risk_adjusted_score\n"
        "e1,u1,i1,0.3,0.4,0.1,0.2,0.7\n"
        "e1,u1,i2,0.6,0.8,0.2,0.5,0.1\n",
    )

    payload = audit_sources(
        sources=[f"scored={scored}"],
        candidate_items_path=str(candidates),
        expected_events=1,
        expected_candidates_per_event=2,
    )

    row = payload["sources"][0]
    assert row["status"] == "paper_ready_uncertainty_rows"
    assert row["paper_ready_uncertainty_rows"] is True
    assert row["uncertainty_col"] == "ccrp_uncertainty"
    assert row["has_ccrp_components"] is True
    assert row["matched_candidate_keys"] == 2


def test_audit_classifies_raw_signal_rows_as_recomputable(tmp_path):
    candidates = _write(
        tmp_path / "candidate_items.csv",
        "source_event_id,user_id,item_id\n"
        "e1,u1,i1\n"
        "e1,u1,i2\n",
    )
    signal = _write(
        tmp_path / "test_calibrated.jsonl",
        '{"source_event_id":"e1","user_id":"u1","candidate_item_id":"i1",'
        '"relevance_probability":0.8,"calibrated_confidence":0.7,'
        '"evidence_support":0.9,"counterevidence_strength":0.1}\n'
        '{"source_event_id":"e1","user_id":"u1","candidate_item_id":"i2",'
        '"relevance_probability":0.2,"calibrated_confidence":0.3,'
        '"evidence_support":0.2,"counterevidence_strength":0.5}\n',
    )

    payload = audit_sources(
        sources=[f"signal={signal}"],
        candidate_items_path=str(candidates),
        expected_events=1,
        expected_candidates_per_event=2,
    )

    row = payload["sources"][0]
    assert row["status"] == "recomputable_signal_rows"
    assert row["recomputable_signal_rows"] is True
    assert row["paper_ready_uncertainty_rows"] is False
    assert row["has_raw_probability"] is True
    assert row["has_evidence"] is True
    assert row["has_counterevidence"] is True
