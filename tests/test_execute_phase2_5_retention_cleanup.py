import json
from pathlib import Path

from scripts.audit.main_execute_phase2_5_retention_cleanup import (
    EXPECTED_SHA256,
    EXPECTED_SIZE_BYTES,
    EXPECTED_TARGET,
    build_cleanup_action,
    run_cleanup_action,
)
from scripts.audit.main_plan_phase2_5_retention_cleanup import build_plan


TOKEN = "APPROVE_DELETE_COMPLETED_TOOLS_LLM2REC_UPSTREAM_EMBEDDING_20260605"


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _packet_audit_payload() -> dict:
    return {
        "ok": True,
        "read_only": True,
        "will_delete": False,
        "will_start_experiment": False,
        "plan_summary": {
            "target_path": EXPECTED_TARGET,
            "approval_token_required": TOKEN,
        },
    }


def _preapproval_payload(*, active_process_count: int = 0, failures: list[str] | None = None) -> dict:
    return {
        "preapproval_checks_ready_except_disk": True,
        "read_only": True,
        "will_delete": False,
        "will_start_experiment": False,
        "target_path": EXPECTED_TARGET,
        "expected_size_bytes": EXPECTED_SIZE_BYTES,
        "actual_size_bytes": EXPECTED_SIZE_BYTES,
        "expected_sha256": EXPECTED_SHA256,
        "actual_sha256": EXPECTED_SHA256,
        "active_process_count": active_process_count,
        "failures": failures if failures is not None else ["disk_below_min_free_before_cleanup"],
    }


def _write_inputs(tmp_path: Path, *, packet: dict | None = None, preapproval: dict | None = None) -> tuple[Path, Path, Path]:
    plan = build_plan(
        candidate="tools_llm2rec_upstream_embedding",
        current_free_bytes=12_406_620_160,
        min_free_gib=15,
        output_dir="outputs/summary/paper_critical/retention_cleanup_plan",
        plan_id="test_plan",
    )
    plan_json = _write_json(tmp_path / "plan.json", plan)
    packet_json = _write_json(tmp_path / "packet.json", packet or _packet_audit_payload())
    preapproval_json = _write_json(tmp_path / "preapproval.json", preapproval or _preapproval_payload())
    return plan_json, packet_json, preapproval_json


def test_dry_run_validates_without_remote_commands(tmp_path):
    plan_json, packet_json, preapproval_json = _write_inputs(tmp_path)
    action = build_cleanup_action(
        plan_json=plan_json,
        packet_audit_json=packet_json,
        preapproval_audit_json=preapproval_json,
    )

    executed = run_cleanup_action(action, command_runner=lambda _remote, _command: "should not run")

    assert executed["validation"]["ok"] is True
    assert executed["mode"] == "dry_run"
    assert executed["read_only"] is True
    assert executed["will_delete"] is False
    assert executed["execution"]["status"] == "dry_run_no_remote_commands"
    assert executed["execution"]["commands_executed"] == []


def test_execute_requires_exact_approval_token(tmp_path):
    plan_json, packet_json, preapproval_json = _write_inputs(tmp_path)

    action = build_cleanup_action(
        plan_json=plan_json,
        packet_audit_json=packet_json,
        preapproval_audit_json=preapproval_json,
        execute=True,
        approval_token="wrong",
    )
    executed = run_cleanup_action(action, command_runner=lambda _remote, _command: "should not run")

    assert executed["validation"]["ok"] is False
    assert "approval_token_mismatch" in executed["validation"]["failures"]
    assert executed["will_delete"] is False
    assert executed["execution"]["status"] == "validation_failed_no_remote_commands"


def test_execute_rejects_active_process_preapproval(tmp_path):
    plan_json, packet_json, preapproval_json = _write_inputs(
        tmp_path,
        preapproval=_preapproval_payload(active_process_count=1),
    )

    action = build_cleanup_action(
        plan_json=plan_json,
        packet_audit_json=packet_json,
        preapproval_audit_json=preapproval_json,
        execute=True,
        approval_token=TOKEN,
    )

    assert action["validation"]["ok"] is False
    assert "preapproval_active_processes_present" in action["validation"]["failures"]
    assert action["will_delete"] is False


def test_execute_sequence_manifests_before_delete(tmp_path):
    plan_json, packet_json, preapproval_json = _write_inputs(tmp_path)
    commands: list[str] = []

    def runner(_remote: str, command: str) -> str:
        commands.append(command)
        return "ok\n"

    action = build_cleanup_action(
        plan_json=plan_json,
        packet_audit_json=packet_json,
        preapproval_audit_json=preapproval_json,
        execute=True,
        approval_token=TOKEN,
    )
    executed = run_cleanup_action(action, command_runner=runner)

    assert executed["validation"]["ok"] is True
    assert executed["will_delete"] is True
    assert executed["execution"]["status"] == "executed"
    manifest_index = next(i for i, command in enumerate(commands) if "sha256sum" in command and ".sha256" in command)
    delete_index = next(i for i, command in enumerate(commands) if f"rm -- {EXPECTED_TARGET}" in command)
    assert manifest_index < delete_index
    assert all("run_baselines_new_domains.sh" not in command for command in commands)


def test_packet_audit_failure_blocks_even_with_token(tmp_path):
    packet = _packet_audit_payload()
    packet["ok"] = False
    plan_json, packet_json, preapproval_json = _write_inputs(tmp_path, packet=packet)

    action = build_cleanup_action(
        plan_json=plan_json,
        packet_audit_json=packet_json,
        preapproval_audit_json=preapproval_json,
        execute=True,
        approval_token=TOKEN,
    )

    assert action["validation"]["ok"] is False
    assert "packet_audit_not_ok" in action["validation"]["failures"]
    assert action["will_delete"] is False
