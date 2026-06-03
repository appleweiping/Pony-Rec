from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = "outputs/summary/official_completion_gate_plan"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a guarded local plan for official-baseline completion gates. "
            "The generated PowerShell script is intentionally blocked by default "
            "and only documents the required gate sequence."
        )
    )
    parser.add_argument("--domain", default="home")
    parser.add_argument("--method", default="rlmrec_graphcl")
    parser.add_argument("--exp_prefix", default="", help="Defaults to <domain>_large10000_100neg.")
    parser.add_argument("--remote_evidence_dir", default="")
    parser.add_argument("--local_evidence_dir", default="")
    parser.add_argument("--expected_users", type=int, default=10000)
    parser.add_argument("--expected_candidates_per_user", type=int, default=101)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plan_id", default="")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_ps1", default="")
    return parser.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _git_status_short(paths: list[str] | None = None) -> str:
    try:
        cmd = ["git", "status", "--short"]
        if paths:
            cmd.extend(["--", *paths])
        return subprocess.check_output(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _exp_prefix(domain: str, exp_prefix: str) -> str:
    value = str(exp_prefix).strip()
    return value or f"{domain}_large10000_100neg"


def _remote_evidence_dir(domain: str, method: str, exp_prefix: str, requested: str) -> str:
    value = str(requested).strip()
    if value:
        return value
    return f"outputs/{_exp_prefix(domain, exp_prefix)}_{method}_official_qwen3base_same_candidate"


def _local_evidence_dir(domain: str, method: str, exp_prefix: str, requested: str) -> str:
    value = str(requested).strip()
    if value:
        return value.replace("/", "\\")
    return f"outputs\\baselines\\official_adapters\\{_exp_prefix(domain, exp_prefix)}_{method}_official_qwen3base_same_candidate"


def _ps_command(parts: list[str]) -> str:
    if len(parts) <= 4:
        return " ".join(parts)
    lines = [f"{parts[0]} {parts[1]} `"]
    idx = 2
    tokens: list[str] = []
    while idx < len(parts):
        token = parts[idx]
        if token.startswith("--") and idx + 1 < len(parts) and not parts[idx + 1].startswith("--"):
            tokens.append(f"{token} {parts[idx + 1]}")
            idx += 2
        else:
            tokens.append(token)
            idx += 1
    for pos, token in enumerate(tokens):
        suffix = " `" if pos < len(tokens) - 1 else ""
        lines.append(f"  {token}{suffix}")
    return "\n".join(lines)


def build_plan(
    *,
    domain: str,
    method: str,
    exp_prefix: str = "",
    remote_evidence_dir: str = "",
    local_evidence_dir: str = "",
    expected_users: int = 10000,
    expected_candidates_per_user: int = 101,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    plan_id: str = "",
) -> dict[str, Any]:
    domain = str(domain).strip().lower()
    method = str(method).strip()
    if not domain:
        raise ValueError("domain must not be empty")
    if not method:
        raise ValueError("method must not be empty")
    plan_id = plan_id or f"{domain}_{method}_completion_gates"
    remote_dir = _remote_evidence_dir(domain, method, exp_prefix, remote_evidence_dir)
    local_dir = _local_evidence_dir(domain, method, exp_prefix, local_evidence_dir)
    local_audit_json = f"{local_dir}\\local_light_evidence_audit.json"
    commands = {
        "server_final_audit": _ps_command(
            [
                "python",
                "scripts\\audit\\main_remote_official_evidence_audit.py",
                "--remote_evidence_dir",
                remote_dir,
                "--mode",
                "server_final",
                "--expected_users",
                str(expected_users),
                "--expected_candidates_per_user",
                str(expected_candidates_per_user),
                "--quiet",
            ]
        ),
        "server_large_artifact_manifest": _ps_command(
            [
                "python",
                "scripts\\audit\\main_remote_server_large_artifact_manifest.py",
                "--remote_evidence_dir",
                remote_dir,
                "--quiet",
            ]
        ),
        "local_light_sync": _ps_command(
            [
                "python",
                "scripts\\audit\\main_sync_official_evidence_package.py",
                "--remote_evidence_dir",
                remote_dir,
                "--local_evidence_dir",
                local_dir,
                "--copy",
                "--quiet",
            ]
        ),
        "local_light_audit": _ps_command(
            [
                "python",
                "scripts\\audit\\main_audit_official_evidence_package.py",
                "--evidence_dir",
                local_dir,
                "--mode",
                "local_light",
                "--expected_users",
                str(expected_users),
                "--expected_candidates_per_user",
                str(expected_candidates_per_user),
                "--output_json",
                local_audit_json,
                "--quiet",
            ]
        ),
    }
    return {
        "plan_id": plan_id,
        "artifact_class": "official_completion_gate_plan",
        "status_label": "planning_only_not_executed",
        "will_start_experiment": False,
        "will_mark_official": False,
        "domain": domain,
        "method": method,
        "exp_prefix": _exp_prefix(domain, exp_prefix),
        "remote_evidence_dir": remote_dir,
        "local_evidence_dir": local_dir,
        "expected_users": expected_users,
        "expected_candidates_per_user": expected_candidates_per_user,
        "expected_score_rows": expected_users * expected_candidates_per_user,
        "gate_order": [
            "server_final_audit",
            "server_large_artifact_manifest",
            "local_light_sync",
            "local_light_audit",
        ],
        "commands": commands,
        "preconditions": [
            "The active method-domain runner has exited normally.",
            "No duplicate matching Python process is active.",
            "The remote evidence directory contains final scores, provenance, imported tables, and predictions.",
            "Do not start the next baseline until every gate in gate_order passes.",
        ],
        "postconditions_before_claim": [
            "server_final_evidence_audit.json has ok=true.",
            "server_large_artifact_manifest.sha256/json exist and cover server-only large artifacts.",
            "light_evidence_sync_manifest.json has ok=true in the local evidence package.",
            "local_light_evidence_audit.json has ok=true.",
            "Canonical docs and shared memory are updated with full metrics, row counts, paths, and blockers.",
        ],
        "git_commit": _git_commit(),
        "git_worktree_dirty_at_generation": bool(_git_status_short()),
        "git_status_short_relevant_at_generation": _git_status_short(
            [
                "scripts/audit/main_plan_official_completion_gates.py",
                "tests/test_plan_official_completion_gates.py",
                "docs/active_todo_pony_uncertainty.md",
                "docs/server_runbook.md",
                output_dir,
            ]
        ),
    }


def guarded_powershell(plan: dict[str, Any]) -> str:
    lines = [
        "$ErrorActionPreference = 'Stop'",
        "Write-Host 'GUARDED PLAN ONLY: confirm the runner completed and remove this throw after all preconditions pass.'",
        "throw 'This completion gate plan is intentionally non-runnable as generated.'",
        "",
        "# Run from the local repository root: D:\\Research\\Uncertainty",
    ]
    for gate in plan["gate_order"]:
        lines.append("")
        lines.append(f"# {gate}")
        lines.append(plan["commands"][gate])
    return "\n".join(lines).rstrip() + "\n"


def write_plan_files(plan: dict[str, Any], *, output_json: str | Path, output_ps1: str | Path) -> dict[str, str]:
    json_path = Path(output_json)
    ps1_path = Path(output_ps1)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    ps1_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    ps1_path.write_text(guarded_powershell(plan), encoding="utf-8")
    return {"json": str(json_path), "powershell": str(ps1_path)}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.rstrip("/\\")
    plan_id = args.plan_id or f"{args.domain}_{args.method}_completion_gates"
    output_json = args.output_json or f"{output_dir}/{plan_id}.json"
    output_ps1 = args.output_ps1 or f"{output_dir}/{plan_id}.ps1"
    plan = build_plan(
        domain=args.domain,
        method=args.method,
        exp_prefix=args.exp_prefix,
        remote_evidence_dir=args.remote_evidence_dir,
        local_evidence_dir=args.local_evidence_dir,
        expected_users=args.expected_users,
        expected_candidates_per_user=args.expected_candidates_per_user,
        output_dir=output_dir,
        plan_id=plan_id,
    )
    outputs = write_plan_files(plan, output_json=output_json, output_ps1=output_ps1)
    print(json.dumps({"ok": True, "outputs": outputs, "will_start_experiment": False}, indent=2))


if __name__ == "__main__":
    main()
