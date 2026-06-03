from scripts.audit.main_trace_ccrp_formal_signal_path import trace_ccrp_formal_signal_path


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_trace_marks_formal_score_only_runner_as_not_rebuildable(tmp_path):
    runner = _write(
        tmp_path / "run_ccrp_v3_domain.py",
        'prompt = "Return ONLY JSON: {\\"relevance_probability\\": 0.0}"\n'
        'writer.writerow(["source_event_id", "user_id", "item_id", "score"])\n'
        'open(out_dir / "scores.csv", "w")\n'
        'open(out_dir / "report.json", "w")\n'
        'open(out_dir / "user_ranks.jsonl", "w")\n',
    )
    selector = _write(
        tmp_path / "main_select_ccrp_variant_on_valid.py",
        'parser.add_argument("--valid_signal_path", required=True)\n'
        'parser.add_argument("--test_signal_path", required=True)\n'
        'pd.DataFrame(valid_rows).to_csv(output_dir / "valid_ccrp_sweep.csv")\n'
        'test_scored_df.to_csv(output_dir / "ccrp_selected_test_scored_rows.csv")\n'
        'write_score_rows(rows, output_dir / "ccrp_selected_test_scores.csv")\n'
        'write_json(prov, output_dir / "ccrp_internal_provenance.json")\n',
    )
    config = _write(
        tmp_path / "week8.yaml",
        "shadow:\n  winner_signal_variant: shadow_v1\n  pointwise_output_dir: outputs/signals\n  ccrp_formal_output_dir: outputs/ccrp\n",
    )

    payload = trace_ccrp_formal_signal_path(
        formal_runner_path=runner,
        selector_path=selector,
        config_path=config,
    )

    assert payload["can_rebuild_paper_ready_uncertainty_rows_from_formal_scores_only"] is False
    assert "formal_scores_schema_has_no_uncertainty_column" in payload["blockers"]
    assert "formal_runner_prompt_does_not_request_evidence_or_counterevidence_fields" in payload["blockers"]
    assert payload["selector_contract_detected"]["requires_valid_signal_path"] is True
    assert payload["selector_contract_detected"]["writes_selected_scored_rows"] is True


def test_trace_detects_recomputable_signal_prompt(tmp_path):
    runner = _write(
        tmp_path / "runner.py",
        '"relevance_probability" "evidence_support" "counterevidence"\n'
        '"source_event_id" "user_id" "item_id" "score"\n',
    )
    selector = _write(tmp_path / "selector.py", "--valid_signal_path --test_signal_path\n")
    config = _write(tmp_path / "config.yaml", "")

    payload = trace_ccrp_formal_signal_path(
        formal_runner_path=runner,
        selector_path=selector,
        config_path=config,
    )

    assert payload["formal_prompt_could_generate_recomputable_signal_if_preserved"] is True
    assert payload["can_rebuild_paper_ready_uncertainty_rows_from_formal_scores_only"] is False
    assert "formal_runner_prompt_does_not_request_evidence_or_counterevidence_fields" not in payload["blockers"]
