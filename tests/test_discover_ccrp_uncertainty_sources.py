from scripts.audit.main_discover_ccrp_uncertainty_sources import _path_matches, classify_header, discover_sources


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_classify_header_recomputable_signal_candidate():
    row = classify_header(
        [
            "source_event_id",
            "user_id",
            "candidate_item_id",
            "relevance_probability",
            "evidence_support",
            "counterevidence_strength",
        ]
    )

    assert row["header_status"] == "header_recomputable_signal_candidate"
    assert row["has_recomputable_signal"] is True


def test_discover_sources_filters_domains_and_detects_score_only(tmp_path):
    root = tmp_path / "outputs"
    _write(
        root / "sports_large10000_100neg_ccrp_v3" / "scores.csv",
        "source_event_id,user_id,item_id,score\n"
        "e1,u1,i1,0.1\n",
    )
    _write(
        root / "toys_large10000_100neg_qwen3_shadow_v1" / "test_calibrated.jsonl",
        '{"source_event_id":"e2","user_id":"u2","candidate_item_id":"i2",'
        '"relevance_probability":0.8,"evidence_support":0.9,"counterevidence_strength":0.1}\n',
    )

    payload = discover_sources(roots=[str(root)], domains=["sports"], full_audit=False)

    assert payload["candidate_count"] == 1
    assert payload["sources"][0]["header_status"] == "header_score_only_candidate"


def test_path_matches_domain_filter_uses_relative_path_not_absolute_home(tmp_path):
    root = tmp_path / "home" / "project" / "outputs"
    path = root / "beauty_large10000_100neg_ccrp_v3" / "scores.csv"

    assert _path_matches(path, root=root, domains=["home"], name_tokens=["ccrp"]) is False


def test_discover_sources_full_audit_reports_candidate_coverage(tmp_path):
    root = tmp_path / "outputs"
    candidates = _write(
        tmp_path / "candidate_items.csv",
        "source_event_id,user_id,item_id\n"
        "e1,u1,i1\n",
    )
    _write(
        root / "sports_large10000_100neg_ccrp_v3" / "scores.csv",
        "source_event_id,user_id,item_id,score\n"
        "e1,u1,i1,0.1\n",
    )

    payload = discover_sources(
        roots=[str(root)],
        domains=["sports"],
        candidate_items_path=str(candidates),
        expected_events=1,
        expected_candidates_per_event=1,
        full_audit=True,
    )

    row = payload["sources"][0]
    assert row["audit_status"] == "score_only_not_uncertainty"
    assert row["audit_candidate_key_coverage_rate"] == 1.0
    assert row["audit_failures"] == ["missing_uncertainty_column"]
