import json

from scripts.audit.main_plan_phase2_5_retention_cleanup import (
    build_plan,
    decision_markdown,
    guarded_shell_script,
)


def _write_retention_audit(path, *, target_path=None):
    target = target_path or (
        "/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/"
        "pony_qwen3_8b_title_item_embs.npy"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "phase2_5_disk_gate": {
                    "current_free_bytes": 12_406_644_736,
                    "required_free_bytes_min": 16_106_127_360,
                    "experiment_launch_allowed": False,
                },
                "recommended_approval_candidate": {
                    "path": target,
                    "size_bytes": 5_662_687_360,
                    "retention_risk_tier": "approval_required_external_embedding_cache",
                    "retention_risk_rank": 20,
                    "expected_free_bytes_after_delete": 18_069_332_096,
                    "would_clear_min_free_gate": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_retention_plan_defaults_to_non_executing_tools_embedding():
    plan = build_plan(
        candidate="tools_llm2rec_upstream_embedding",
        current_free_bytes=12_407_414_784,
        min_free_gib=15,
        output_dir="outputs/summary/paper_critical/retention_cleanup_plan",
        plan_id="test_plan",
    )

    assert plan["status_label"] == "planning_only_not_executed"
    assert plan["will_delete"] is False
    assert plan["will_delete_files"] is False
    assert plan["will_execute_cleanup"] is False
    assert plan["will_start_experiment"] is False
    assert plan["phase2_5_experiment_launch_allowed_after_cleanup"] is False
    assert plan["requires_explicit_approval"] is True
    assert plan["retention_decision_state"] == "preserve_by_default"
    assert plan["ranked_retention_audit_source"].endswith(
        "server_storage_phase2_5_retention_audit_ranked_20260606.json"
    )
    assert plan["recommended_by_ranked_audit"] is True
    assert plan["retention_risk_tier"] == "approval_required_external_embedding_cache"
    assert plan["retention_risk_rank"] == 20
    assert plan["ranked_audit_current_free_bytes"] == 12_407_414_784
    assert plan["ranked_audit_expected_free_bytes_after_delete"] == 18_070_102_144
    assert plan["ranked_audit_would_clear_min_free_gate"] is True
    assert plan["candidate"]["target_path"].endswith("pony_qwen3_8b_title_item_embs.npy")
    assert plan["candidate"]["expected_size_bytes"] == 5_662_687_360
    assert plan["candidate"]["expected_sha256"] == (
        "306618d974eb4133d9cda87bae3251e17d793aa6f5a8cb38d558b549ed31d56e"
    )
    assert plan["candidate"]["classification"] == "NEEDS_APPROVAL_OR_ARCHIVE_DECISION"
    assert "fairness_provenance.json" in plan["candidate"]["provenance_sha256_source"]
    assert plan["candidate"]["retention_audit_source"].endswith(
        "server_storage_phase2_5_retention_audit_ranked_20260606.json"
    )
    assert plan["candidate"]["prior_retention_audit_source"].endswith(
        "server_storage_phase2_5_retention_audit_20260605.json"
    )
    assert plan["expected_free_bytes_after_delete"] == 18_070_102_144
    assert plan["expected_to_clear_min_free_gate"] is True
    assert "APPROVE_DELETE_COMPLETED_TOOLS_LLM2REC_UPSTREAM_EMBEDDING_20260605" == plan["approval_token_required"]


def test_retention_plan_records_required_guards_and_postconditions():
    plan = build_plan(
        candidate="tools_llm2rec_upstream_embedding",
        current_free_bytes=12_407_414_784,
        min_free_gib=15,
        output_dir="outputs/summary/paper_critical/retention_cleanup_plan",
        plan_id="test_plan",
    )

    assert "main_audit_domain_official_gate.py" in plan["commands"]["domain_gate_check"]
    assert "main_build_domain_official_comparison.py" in plan["commands"]["post_delete_comparison_gate_check"]
    assert "sha256sum" in plan["commands"]["manifest_before_delete"]
    assert "sha256sum" in plan["commands"]["target_sha256_read_only"]
    assert "realpath" in plan["commands"]["target_realpath_check"]
    assert "stat -c" in plan["commands"]["target_stat_read_only"]
    assert "rm --" in plan["commands"]["delete_target_after_approval"]
    assert "User records that no cheap rerun/resume need depends on this embedding." in plan[
        "preconditions_before_removing_script_guard"
    ]
    assert "Target sha256 matches expected_sha256 from provenance before deletion." in plan[
        "preconditions_before_removing_script_guard"
    ]
    assert "The ranked retention audit recommendation is accepted for this exact target." in plan[
        "preconditions_before_removing_script_guard"
    ]
    assert "score_coverage_rate=1.0" in " ".join(plan["preconditions_before_removing_script_guard"])
    assert "Deletion remains prohibited" in plan["approval_required_reminder"]
    assert "outputs/baselines/external_tasks/" in plan["protected_paths"]
    assert "/home/ajifang/models/Qwen/" in plan["protected_paths"]
    assert "outputs/*_same_candidate/*.pt" in plan["protected_paths"]
    assert "post_delete_domain_gate_json" in plan["manifest_outputs_if_executed"]
    assert "post_delete_comparison_dir" in plan["manifest_outputs_if_executed"]


def test_retention_plan_can_consume_current_storage_audit(tmp_path):
    audit_path = _write_retention_audit(tmp_path / "storage.json")
    plan = build_plan(
        candidate="tools_llm2rec_upstream_embedding",
        min_free_gib=15,
        output_dir="outputs/summary/paper_critical/retention_cleanup_plan",
        plan_id="test_plan",
        retention_audit_json=audit_path,
    )

    assert plan["ranked_retention_audit_source"].endswith("storage.json")
    assert plan["ranked_audit_current_free_bytes"] == 12_406_644_736
    assert plan["ranked_audit_expected_free_bytes_after_delete"] == 18_069_332_096
    assert plan["expected_to_clear_min_free_gate"] is True
    assert plan["will_delete"] is False
    assert plan["requires_explicit_approval"] is True


def test_retention_plan_rejects_audit_with_different_recommended_target(tmp_path):
    audit_path = _write_retention_audit(tmp_path / "storage.json", target_path="/tmp/other.npy")

    try:
        build_plan(
            candidate="tools_llm2rec_upstream_embedding",
            retention_audit_json=audit_path,
        )
    except ValueError as exc:
        assert "different target" in str(exc)
    else:
        raise AssertionError("expected target mismatch to fail")


def test_decision_markdown_is_non_destructive():
    plan = build_plan(
        candidate="tools_llm2rec_upstream_embedding",
        current_free_bytes=12_407_414_784,
        min_free_gib=15,
        output_dir="outputs/summary/paper_critical/retention_cleanup_plan",
        plan_id="test_plan",
    )
    text = decision_markdown(plan)

    assert "Will delete now: `False`" in text
    assert "Requires explicit approval: `True`" in text
    assert "rm --" not in text
    assert "Deletion remains prohibited" in text


def test_guarded_shell_exits_before_any_delete_or_manifest_command():
    plan = build_plan(
        candidate="tools_llm2rec_upstream_embedding",
        current_free_bytes=12_407_414_784,
        min_free_gib=15,
        output_dir="outputs/summary/paper_critical/retention_cleanup_plan",
        plan_id="test_plan",
    )
    script = guarded_shell_script(plan)

    assert "exit 2" in script
    assert script.index("exit 2") < script.index("sha256sum")
    assert script.index("exit 2") < script.index("rm --")
    assert "realpath" in script
    assert "APPROVAL_TOKEN_REQUIRED" in script
    assert "Remove-Item" not in script
    assert "find -delete" not in script
    assert "rsync --delete" not in script
    assert "nohup" not in script
    assert "run_baselines_new_domains.sh" not in script
