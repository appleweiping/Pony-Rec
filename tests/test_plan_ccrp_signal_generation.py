from scripts.audit.main_plan_ccrp_signal_generation import build_plan, guarded_shell_script


def test_plan_defaults_to_non_executing_sports_toys_flow():
    plan = build_plan(
        domains=[],
        remote_project="/repo",
        remote_root=".",
        output_dir="outputs/summary/paper_critical/ccrp_signal_generation_plan",
        expected_events=10000,
        expected_candidates_per_event=101,
        selection_metric="NDCG@10",
        plan_id="test_plan",
    )

    assert plan["will_start_experiment"] is False
    assert plan["status_label"] == "planning_only_not_executed"
    assert plan["domains"] == ["sports", "toys"]
    assert "score-only" in plan["current_blocker"]
    assert "main_remote_discover_ccrp_uncertainty_sources.py" in plan["global_commands"]["remote_header_discovery"]
    assert "--domain sports" in plan["global_commands"]["remote_header_discovery"]
    assert "official baseline row" in plan["required_status_before_execution"][0]
    assert "Home rlmrec_graphcl" not in " ".join(plan["required_status_before_execution"])


def test_domain_plan_uses_placeholders_and_full_scale_counts():
    plan = build_plan(
        domains=["sports"],
        remote_project="/repo",
        remote_root=".",
        output_dir="outputs/summary/paper_critical/ccrp_signal_generation_plan",
        expected_events=10000,
        expected_candidates_per_event=101,
        selection_metric="NDCG@10",
        plan_id="test_plan",
    )
    domain_plan = plan["domain_plans"][0]

    assert domain_plan["expected_score_rows"] == 1010000
    assert domain_plan["paths"]["valid_signal_placeholder"] == "TODO_VALID_SPORTS_CCRP_SIGNAL_JSONL_OR_CSV"
    assert domain_plan["paths"]["test_signal_placeholder"] == "TODO_TEST_SPORTS_CCRP_SIGNAL_JSONL_OR_CSV"
    assert domain_plan["paths"]["component_ablation_output_dir"] == domain_plan["paths"]["selector_output_dir"]
    selector = domain_plan["commands"]["select_ccrp_ablation_and_scores_template"]
    assert "main_select_ccrp_variant_on_valid.py" in selector
    assert "--valid_signal_path TODO_VALID_SPORTS_CCRP_SIGNAL_JSONL_OR_CSV" in selector
    assert "--test_signal_path TODO_TEST_SPORTS_CCRP_SIGNAL_JSONL_OR_CSV" in selector
    assert "--import_scores" in selector
    component = domain_plan["commands"]["build_component_ablation_summary_template"]
    assert "main_build_ccrp_component_ablation_summary.py" in component
    assert "--selector_dir outputs/summary/paper_critical/ccrp_signal_generation_plan/ccrp_ablation_sports" in component
    assert "--ablations full,without_boundary_uncertainty,without_calibration_gap" in component
    component_audit = domain_plan["commands"]["audit_component_ablation_package_template"]
    assert "main_audit_phase2_5_module_package.py" in component_audit
    assert "--module component_ablation" in component_audit
    assert "--expected_ablation full" in component_audit
    assert "--expected_ablation without_risk_penalty" in component_audit
    observation = domain_plan["commands"]["build_observation_study_template"]
    assert "ccrp_selected_test_scored_rows.csv" in observation
    assert "main_build_uncertainty_observation_study.py" in observation
    assert "--module observation_motivation" in domain_plan["commands"]["audit_observation_package_template"]
    assert "--module hyperparameter_analysis" in domain_plan["commands"]["audit_hyperparameter_package_template"]
    assert "--expected_control eta" in domain_plan["commands"]["audit_hyperparameter_package_template"]
    assert "matching baseline Python process" in domain_plan["execution_gates"][0]
    assert "Home rlmrec_graphcl" not in " ".join(domain_plan["execution_gates"])


def test_guarded_shell_exits_before_any_command():
    plan = build_plan(
        domains=["sports"],
        remote_project="/repo",
        remote_root=".",
        output_dir="outputs/summary/paper_critical/ccrp_signal_generation_plan",
        expected_events=10000,
        expected_candidates_per_event=101,
        selection_metric="NDCG@10",
        plan_id="test_plan",
    )

    shell = guarded_shell_script(plan)

    assert "exit 2" in shell
    assert shell.index("exit 2") < shell.index("cd /repo")
    assert "nohup" not in shell
    assert "TODO_TEST_SPORTS_CCRP_SIGNAL_JSONL_OR_CSV" in shell
    assert "main_build_ccrp_component_ablation_summary.py" in shell
    assert "main_audit_phase2_5_module_package.py" in shell
