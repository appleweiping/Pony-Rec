from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


DEFAULT_REMOTE = "pony-rec-gpu"
ALLOWED_PREAPPROVAL_FAILURES = {"disk_below_min_free_before_cleanup"}
EXPECTED_CANDIDATE = "tools_llm2rec_upstream_embedding"
EXPECTED_TARGET = (
    "/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/"
    "pony_qwen3_8b_title_item_embs.npy"
)
EXPECTED_SHA256 = "306618d974eb4133d9cda87bae3251e17d793aa6f5a8cb38d558b549ed31d56e"
EXPECTED_SIZE_BYTES = 5_662_687_360


CommandRunner = Callable[[str, str], str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and render a guarded Phase 2.5 retention cleanup action. "
            "Default mode is dry-run and executes no remote commands."
        )
    )
    parser.add_argument("--plan_json", required=True)
    parser.add_argument("--packet_audit_json", required=True)
    parser.add_argument("--preapproval_audit_json", required=True)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--approval_token", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_md", default="")
    return parser.parse_args()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _run_remote(remote: str, command: str) -> str:
    completed = subprocess.run(
        ["ssh", remote, command],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout


def _bool_field(payload: dict[str, Any], field: str, expected: bool, failures: list[str], prefix: str) -> None:
    if payload.get(field) is not expected:
        failures.append(f"{prefix}_field_not_{str(expected).lower()}:{field}")


def _validate_plan(plan: dict[str, Any], failures: list[str]) -> None:
    candidate = plan.get("candidate") or {}
    commands = plan.get("commands") or {}
    target = str(candidate.get("target_path", ""))
    approval_token = str(plan.get("approval_token_required", ""))

    if plan.get("status_label") != "planning_only_not_executed":
        failures.append("plan_status_not_planning_only_not_executed")
    for field in (
        "will_delete",
        "will_delete_files",
        "will_execute_cleanup",
        "will_start_experiment",
        "phase2_5_experiment_launch_allowed_after_cleanup",
    ):
        _bool_field(plan, field, False, failures, "plan")
    for field in ("requires_explicit_approval", "expected_to_clear_min_free_gate"):
        _bool_field(plan, field, True, failures, "plan")
    if candidate.get("candidate") != EXPECTED_CANDIDATE:
        failures.append("candidate_id_mismatch")
    if target != EXPECTED_TARGET:
        failures.append("target_path_mismatch")
    if int(candidate.get("expected_size_bytes") or 0) != EXPECTED_SIZE_BYTES:
        failures.append("target_expected_size_mismatch")
    if str(candidate.get("expected_sha256", "")) != EXPECTED_SHA256:
        failures.append("target_expected_sha_mismatch")
    if not approval_token:
        failures.append("missing_approval_token_required")

    expected_delete_command = f"rm -- {shlex.quote(EXPECTED_TARGET)}"
    if commands.get("delete_target_after_approval") != expected_delete_command:
        failures.append("delete_command_not_exact_target")
    manifest_command = str(commands.get("manifest_before_delete", ""))
    if "sha256sum" not in manifest_command or "stat -c" not in manifest_command:
        failures.append("manifest_command_missing_sha_or_size")
    if EXPECTED_TARGET not in manifest_command:
        failures.append("manifest_command_missing_exact_target")
    if "main_audit_domain_official_gate.py" not in str(commands.get("domain_gate_check", "")):
        failures.append("missing_predelete_domain_gate_command")
    if "main_build_domain_official_comparison.py" not in str(commands.get("post_delete_comparison_gate_check", "")):
        failures.append("missing_postdelete_comparison_gate_command")


def _validate_packet_audit(packet: dict[str, Any], plan: dict[str, Any], failures: list[str]) -> None:
    if packet.get("ok") is not True:
        failures.append("packet_audit_not_ok")
    for field in ("read_only",):
        _bool_field(packet, field, True, failures, "packet_audit")
    for field in ("will_delete", "will_start_experiment"):
        _bool_field(packet, field, False, failures, "packet_audit")
    target = ((packet.get("plan_summary") or {}).get("target_path")) or ""
    if str(target) != str((plan.get("candidate") or {}).get("target_path")):
        failures.append("packet_target_mismatch")
    if str((packet.get("plan_summary") or {}).get("approval_token_required", "")) != str(
        plan.get("approval_token_required", "")
    ):
        failures.append("packet_approval_token_mismatch")


def _validate_preapproval(preapproval: dict[str, Any], plan: dict[str, Any], failures: list[str]) -> None:
    if preapproval.get("preapproval_checks_ready_except_disk") is not True:
        failures.append("preapproval_not_ready_except_disk")
    for field in ("read_only",):
        _bool_field(preapproval, field, True, failures, "preapproval")
    for field in ("will_delete", "will_start_experiment"):
        _bool_field(preapproval, field, False, failures, "preapproval")
    failure_set = set(str(item) for item in preapproval.get("failures", []))
    disallowed = sorted(failure_set - ALLOWED_PREAPPROVAL_FAILURES)
    for failure in disallowed:
        failures.append(f"preapproval_disallowed_failure:{failure}")
    if str(preapproval.get("target_path", "")) != str((plan.get("candidate") or {}).get("target_path", "")):
        failures.append("preapproval_target_mismatch")
    if int(preapproval.get("expected_size_bytes") or 0) != EXPECTED_SIZE_BYTES:
        failures.append("preapproval_expected_size_mismatch")
    if int(preapproval.get("actual_size_bytes") or 0) != EXPECTED_SIZE_BYTES:
        failures.append("preapproval_actual_size_mismatch")
    if str(preapproval.get("expected_sha256", "")) != EXPECTED_SHA256:
        failures.append("preapproval_expected_sha_mismatch")
    if str(preapproval.get("actual_sha256", "")) != EXPECTED_SHA256:
        failures.append("preapproval_actual_sha_mismatch")
    if int(preapproval.get("active_process_count") or 0) != 0:
        failures.append("preapproval_active_processes_present")


def validate_inputs(
    *,
    plan: dict[str, Any],
    packet_audit: dict[str, Any],
    preapproval_audit: dict[str, Any],
    execute: bool,
    approval_token: str = "",
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    _validate_plan(plan, failures)
    _validate_packet_audit(packet_audit, plan, failures)
    _validate_preapproval(preapproval_audit, plan, failures)

    required_token = str(plan.get("approval_token_required", ""))
    if execute and approval_token != required_token:
        failures.append("approval_token_mismatch")
    if execute and "disk_below_min_free_before_cleanup" not in set(preapproval_audit.get("failures", [])):
        failures.append("disk_not_below_floor_cleanup_unnecessary")
    if not execute and approval_token and approval_token == required_token:
        warnings.append("approval_token_supplied_in_dry_run")

    return {
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "approval_token_valid": bool(approval_token and approval_token == required_token),
        "allowed_preapproval_failures": sorted(ALLOWED_PREAPPROVAL_FAILURES),
    }


def build_ordered_steps(plan: dict[str, Any], *, execute: bool, approval_token: str = "") -> list[dict[str, Any]]:
    commands = plan["commands"]
    token = approval_token if execute else "<approval-token-not-used-in-dry-run>"
    approval_guard = f"APPROVAL_TOKEN={shlex.quote(token)}; {commands['approval_guard']}"
    step_specs = [
        ("server_preflight_read_only", True, False, commands["server_preflight_read_only"]),
        ("evidence_precondition_check", True, False, commands["evidence_precondition_check"]),
        ("target_realpath_check", True, False, commands["target_realpath_check"]),
        ("target_stat_read_only", True, False, commands["target_stat_read_only"]),
        ("target_sha256_read_only", True, False, commands["target_sha256_read_only"]),
        ("domain_gate_check", True, False, commands["domain_gate_check"]),
        ("approval_guard", True, False, approval_guard),
        ("manifest_before_delete", False, False, commands["manifest_before_delete"]),
        ("delete_target_after_approval", False, True, commands["delete_target_after_approval"]),
        ("post_delete_disk_check", True, False, commands["post_delete_disk_check"]),
        ("post_delete_domain_gate_check", False, False, commands["post_delete_domain_gate_check"]),
        ("post_delete_comparison_gate_check", False, False, commands["post_delete_comparison_gate_check"]),
    ]
    return [
        {
            "index": index,
            "name": name,
            "read_only": read_only,
            "destructive": destructive,
            "command": command,
        }
        for index, (name, read_only, destructive, command) in enumerate(step_specs, start=1)
    ]


def build_cleanup_action(
    *,
    plan_json: str | Path,
    packet_audit_json: str | Path,
    preapproval_audit_json: str | Path,
    remote: str = DEFAULT_REMOTE,
    execute: bool = False,
    approval_token: str = "",
) -> dict[str, Any]:
    plan = _read_json(plan_json)
    packet_audit = _read_json(packet_audit_json)
    preapproval_audit = _read_json(preapproval_audit_json)
    validation = validate_inputs(
        plan=plan,
        packet_audit=packet_audit,
        preapproval_audit=preapproval_audit,
        execute=execute,
        approval_token=approval_token,
    )
    steps = build_ordered_steps(plan, execute=execute, approval_token=approval_token)
    return {
        "schema_version": "2026-06-06.retention_cleanup_action.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "execute" if execute else "dry_run",
        "remote": remote,
        "read_only": not execute,
        "will_delete": bool(execute and validation["ok"]),
        "will_start_experiment": False,
        "plan_json": str(plan_json),
        "packet_audit_json": str(packet_audit_json),
        "preapproval_audit_json": str(preapproval_audit_json),
        "target_path": EXPECTED_TARGET,
        "expected_sha256": EXPECTED_SHA256,
        "expected_size_bytes": EXPECTED_SIZE_BYTES,
        "remote_project": plan.get("remote_project"),
        "approval_token_required": plan.get("approval_token_required"),
        "approval_token_valid": validation["approval_token_valid"],
        "validation": validation,
        "ordered_steps": steps,
        "execution": {
            "status": "not_started",
            "commands_executed": [],
            "outputs": {},
        },
    }


def _with_remote_project(project: str, command: str) -> str:
    return f"cd {shlex.quote(project)} && {command}"


def run_cleanup_action(action: dict[str, Any], *, command_runner: CommandRunner = _run_remote) -> dict[str, Any]:
    if action["mode"] != "execute":
        action["execution"]["status"] = "dry_run_no_remote_commands"
        return action
    if not action["validation"]["ok"]:
        action["will_delete"] = False
        action["execution"]["status"] = "validation_failed_no_remote_commands"
        return action

    project = str(action["remote_project"] or "/home/ajifang/projects/pony-rec-rescue-shadow-v6")
    executed: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}
    for step in action["ordered_steps"]:
        remote_command = _with_remote_project(project, step["command"])
        output = command_runner(action["remote"], remote_command)
        executed.append(
            {
                "index": step["index"],
                "name": step["name"],
                "read_only": step["read_only"],
                "destructive": step["destructive"],
            }
        )
        outputs[step["name"]] = output
    action["execution"] = {
        "status": "executed",
        "commands_executed": executed,
        "outputs": outputs,
    }
    return action


def write_markdown(path: str | Path, action: dict[str, Any]) -> None:
    lines = [
        "# Phase 2.5 Retention Cleanup Action",
        "",
        f"- Generated UTC: `{action['created_at_utc']}`",
        f"- Mode: `{action['mode']}`",
        f"- Read only: `{action['read_only']}`",
        f"- Will delete: `{action['will_delete']}`",
        f"- Will start experiment: `{action['will_start_experiment']}`",
        f"- Target: `{action['target_path']}`",
        f"- Validation OK: `{action['validation']['ok']}`",
        f"- Execution status: `{action['execution']['status']}`",
        "",
        "## Validation Failures",
        "",
    ]
    failures = action["validation"]["failures"]
    lines.extend(f"- {failure}" for failure in failures)
    if not failures:
        lines.append("- none")
    lines.extend(["", "## Ordered Steps", ""])
    for step in action["ordered_steps"]:
        lines.append(
            f"- {step['index']}. `{step['name']}` "
            f"read_only=`{step['read_only']}` destructive=`{step['destructive']}`"
        )
    if action["mode"] == "dry_run":
        lines.extend(
            [
                "",
                "## Verdict",
                "",
                "Dry-run only. No remote command was executed and no artifact was deleted.",
            ]
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    action = build_cleanup_action(
        plan_json=args.plan_json,
        packet_audit_json=args.packet_audit_json,
        preapproval_audit_json=args.preapproval_audit_json,
        remote=args.remote,
        execute=args.execute,
        approval_token=args.approval_token,
    )
    action = run_cleanup_action(action)
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(action, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(args.output_md, action)
    print(
        json.dumps(
            {
                "ok": action["validation"]["ok"],
                "mode": action["mode"],
                "will_delete": action["will_delete"],
                "execution_status": action["execution"]["status"],
                "failures": action["validation"]["failures"],
            },
            indent=2,
        )
    )
    if not action["validation"]["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
