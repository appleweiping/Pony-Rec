from scripts.audit.main_plan_official_completion_gates import build_plan, guarded_powershell


def test_completion_gate_plan_defaults_to_home_rlmrec_paths():
    plan = build_plan(
        domain="home",
        method="rlmrec_graphcl",
        output_dir="outputs/summary/official_completion_gate_plan",
        plan_id="home_rlmrec_graphcl_completion_gates_20260604",
    )

    assert plan["will_start_experiment"] is False
    assert plan["will_mark_official"] is False
    assert plan["status_label"] == "planning_only_not_executed"
    assert plan["remote_evidence_dir"] == (
        "outputs/home_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate"
    )
    assert plan["local_evidence_dir"] == (
        "outputs\\baselines\\official_adapters\\home_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate"
    )
    assert plan["expected_score_rows"] == 1010000


def test_completion_gate_plan_preserves_required_order_and_commands():
    plan = build_plan(
        domain="home",
        method="rlmrec_graphcl",
        output_dir="outputs/summary/official_completion_gate_plan",
        plan_id="home_rlmrec_graphcl_completion_gates_20260604",
    )

    assert plan["gate_order"] == [
        "server_final_audit",
        "server_large_artifact_manifest",
        "local_light_sync",
        "local_light_audit",
    ]
    assert "main_remote_official_evidence_audit.py" in plan["commands"]["server_final_audit"]
    assert "--mode server_final" in plan["commands"]["server_final_audit"]
    assert "main_remote_server_large_artifact_manifest.py" in plan["commands"]["server_large_artifact_manifest"]
    assert "main_sync_official_evidence_package.py" in plan["commands"]["local_light_sync"]
    assert "--copy" in plan["commands"]["local_light_sync"]
    assert "main_audit_official_evidence_package.py" in plan["commands"]["local_light_audit"]
    assert "--mode local_light" in plan["commands"]["local_light_audit"]


def test_guarded_powershell_blocks_before_gate_commands():
    plan = build_plan(
        domain="home",
        method="rlmrec_graphcl",
        output_dir="outputs/summary/official_completion_gate_plan",
        plan_id="home_rlmrec_graphcl_completion_gates_20260604",
    )
    script = guarded_powershell(plan)

    assert "throw 'This completion gate plan is intentionally non-runnable as generated.'" in script
    assert script.index("throw 'This completion gate plan") < script.index("main_remote_official_evidence_audit.py")
    assert "nohup" not in script
    assert "run_baselines_new_domains.sh" not in script
