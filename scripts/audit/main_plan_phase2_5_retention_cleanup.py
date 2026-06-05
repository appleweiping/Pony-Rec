from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = "outputs/summary/paper_critical/retention_cleanup_plan"
DEFAULT_REMOTE_PROJECT = "/home/ajifang/projects/pony-rec-rescue-shadow-v6"
DEFAULT_MIN_FREE_GIB = 15
BYTES_PER_GIB = 1024**3
RANKED_RETENTION_AUDIT_SOURCE = (
    "outputs/summary/paper_critical/"
    "server_storage_phase2_5_retention_audit_ranked_20260606.json"
)
PRIOR_RETENTION_AUDIT_SOURCE = (
    "outputs/summary/paper_critical/"
    "server_storage_phase2_5_retention_audit_20260605.json"
)
RANKED_AUDIT_CURRENT_FREE_BYTES = 12_407_414_784
RANKED_AUDIT_EXPECTED_FREE_AFTER_DELETE = 18_070_102_144


RETENTION_CANDIDATES: dict[str, dict[str, Any]] = {
    "tools_llm2rec_upstream_embedding": {
        "description": "Completed Tools LLM2Rec upstream Qwen3 item embedding",
        "target_path": "/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy",
        "expected_size_bytes": 5662687360,
        "expected_sha256": "306618d974eb4133d9cda87bae3251e17d793aa6f5a8cb38d558b549ed31d56e",
        "domain": "tools",
        "method": "llm2rec_sasrec",
        "classification": "NEEDS_APPROVAL_OR_ARCHIVE_DECISION",
        "retention_decision_state": "preserve_by_default",
        "retention_audit_source": RANKED_RETENTION_AUDIT_SOURCE,
        "ranked_retention_audit_source": RANKED_RETENTION_AUDIT_SOURCE,
        "prior_retention_audit_source": PRIOR_RETENTION_AUDIT_SOURCE,
        "recommended_by_ranked_audit": True,
        "retention_risk_tier": "approval_required_external_embedding_cache",
        "retention_risk_rank": 20,
        "ranked_audit_current_free_bytes": RANKED_AUDIT_CURRENT_FREE_BYTES,
        "ranked_audit_expected_free_bytes_after_delete": RANKED_AUDIT_EXPECTED_FREE_AFTER_DELETE,
        "ranked_audit_would_clear_min_free_gate": True,
        "provenance_sha256_source": (
            "outputs/baselines/official_adapters/"
            "tools_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate/"
            "fairness_provenance.json:qwen3_item_embedding_sha256"
        ),
        "approval_token": "APPROVE_DELETE_COMPLETED_TOOLS_LLM2REC_UPSTREAM_EMBEDDING_20260605",
        "decision_rationale": (
            "This artifact is outside the final evidence directory and is referenced by the completed "
            "Tools LLM2Rec run summary. It can recover enough disk for Phase 2.5, but deletion must be "
            "an explicit retention decision with sha256/size manifesting and post-delete gate checks. "
            "The ranked read-only retention audit identifies it as the lowest-risk high-yield "
            "approval-required candidate under the current policy."
        ),
        "protected_evidence_dir": "outputs/tools_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate",
        "local_evidence_dir": (
            "outputs/baselines/official_adapters/"
            "tools_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate"
        ),
        "gate_json": "outputs/summary/tools_official_ccrp_gate_post_cleanup_20260605.json",
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a guarded, non-executing plan for a Phase 2.5 storage retention "
            "decision. The generated shell script exits before any command that could "
            "delete a file."
        )
    )
    parser.add_argument("--candidate", default="tools_llm2rec_upstream_embedding")
    parser.add_argument("--remote_project", default=DEFAULT_REMOTE_PROJECT)
    parser.add_argument("--current_free_bytes", type=int, default=0)
    parser.add_argument("--min_free_gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plan_id", default="")
    parser.add_argument("--retention_audit_json", default="")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_sh", default="")
    parser.add_argument("--output_md", default="")
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


def _cmd(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part))


def _shell_var(value: str) -> str:
    return shlex.quote(value)


def _candidate(candidate_id: str) -> dict[str, Any]:
    value = str(candidate_id).strip()
    try:
        return {**RETENTION_CANDIDATES[value], "candidate": value}
    except KeyError as exc:
        known = ", ".join(sorted(RETENTION_CANDIDATES))
        raise ValueError(f"unknown candidate {value!r}; known candidates: {known}") from exc


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _apply_retention_audit(item: dict[str, Any], retention_audit_json: str | Path | None) -> dict[str, Any]:
    if not retention_audit_json:
        return item
    audit_path = str(retention_audit_json).replace("\\", "/")
    audit = _read_json(retention_audit_json)
    recommended = audit.get("recommended_approval_candidate") or {}
    if not recommended:
        raise ValueError(f"retention audit has no recommended_approval_candidate: {retention_audit_json}")
    expected_path = str(item["target_path"])
    recommended_path = str(recommended.get("path", ""))
    if recommended_path != expected_path:
        raise ValueError(
            "retention audit recommended a different target: "
            f"{recommended_path!r} != configured {expected_path!r}"
        )
    phase_gate = audit.get("phase2_5_disk_gate") or {}
    current_free = int(phase_gate.get("current_free_bytes") or 0)
    expected_after = int(recommended.get("expected_free_bytes_after_delete") or current_free + int(item["expected_size_bytes"]))
    updated = dict(item)
    updated.update(
        {
            "retention_audit_source": audit_path,
            "ranked_retention_audit_source": audit_path,
            "recommended_by_ranked_audit": True,
            "retention_risk_tier": recommended.get("retention_risk_tier", item["retention_risk_tier"]),
            "retention_risk_rank": int(recommended.get("retention_risk_rank", item["retention_risk_rank"])),
            "ranked_audit_current_free_bytes": current_free,
            "ranked_audit_expected_free_bytes_after_delete": expected_after,
            "ranked_audit_would_clear_min_free_gate": bool(recommended.get("would_clear_min_free_gate")),
        }
    )
    return updated


def _min_free_bytes(min_free_gib: float) -> int:
    return int(float(min_free_gib) * BYTES_PER_GIB)


def _manifest_base(candidate_id: str) -> str:
    return f"outputs/summary/{candidate_id}_retention_cleanup_APPROVAL_REQUIRED_20260605"


def build_plan(
    *,
    candidate: str,
    remote_project: str = DEFAULT_REMOTE_PROJECT,
    current_free_bytes: int = 0,
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    plan_id: str = "",
    retention_audit_json: str | Path | None = None,
) -> dict[str, Any]:
    item = _apply_retention_audit(_candidate(candidate), retention_audit_json)
    min_free_bytes = _min_free_bytes(min_free_gib)
    effective_current_free_bytes = int(current_free_bytes) or int(item.get("ranked_audit_current_free_bytes") or 0)
    expected_after_delete = (
        effective_current_free_bytes + int(item["expected_size_bytes"]) if effective_current_free_bytes else None
    )
    plan_id = plan_id or f"{item['candidate']}_retention_cleanup_plan"
    manifest_base = _manifest_base(item["candidate"])
    expected_size = int(item["expected_size_bytes"])
    target_path = str(item["target_path"])
    approval_token = str(item["approval_token"])

    commands = {
        "server_preflight_read_only": (
            "date '+%F %T %Z'; "
            "ps aux | grep python | grep -v grep | grep -i 'pony-rec\\|ccrp\\|baseline\\|uncertainty' || true; "
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; "
            "df -h /home/ajifang; df -B1 /home/ajifang"
        ),
        "evidence_precondition_check": (
            "test -f "
            + _shell_var(f"{remote_project}/{item['protected_evidence_dir']}/fairness_provenance.json")
            + " && test -f "
            + _shell_var(f"{remote_project}/{item['protected_evidence_dir']}/server_final_evidence_audit.json")
            + " && test -f "
            + _shell_var(f"{remote_project}/{item['protected_evidence_dir']}/tables/ranking_eval_records.csv")
            + " && test -f "
            + _shell_var(f"{remote_project}/{item['protected_evidence_dir']}/scores.csv")
        ),
        "domain_gate_check": _cmd(
            [
                "/home/ajifang/miniconda3/bin/python",
                "scripts/audit/main_audit_domain_official_gate.py",
                "--root",
                ".",
                "--domain",
                str(item["domain"]),
                "--output_json",
                "outputs/summary/phase2_5_retention_cleanup_domain_gate_recheck.json",
                "--output_csv",
                "outputs/summary/phase2_5_retention_cleanup_domain_gate_recheck.csv",
                "--quiet",
            ]
        ),
        "manifest_before_delete": (
            "mkdir -p outputs/summary && "
            f"sha256sum {_shell_var(target_path)} > {_shell_var(manifest_base + '.sha256')} && "
            f"stat -c '%s  %n' {_shell_var(target_path)} > {_shell_var(manifest_base + '.size.txt')}"
        ),
        "target_stat_read_only": f"stat -c '%s  %n' {_shell_var(target_path)}",
        "target_sha256_read_only": f"sha256sum {_shell_var(target_path)}",
        "target_realpath_check": f"test \"$(realpath {_shell_var(target_path)})\" = {_shell_var(target_path)}",
        "approval_guard": f"test \"${{APPROVAL_TOKEN:-}}\" = {_shell_var(approval_token)}",
        "delete_target_after_approval": f"rm -- {_shell_var(target_path)}",
        "post_delete_disk_check": "df -h /home/ajifang; df -B1 /home/ajifang",
        "post_delete_domain_gate_check": _cmd(
            [
                "/home/ajifang/miniconda3/bin/python",
                "scripts/audit/main_audit_domain_official_gate.py",
                "--root",
                ".",
                "--domain",
                str(item["domain"]),
                "--output_json",
                "outputs/summary/phase2_5_retention_cleanup_post_delete_domain_gate.json",
                "--output_csv",
                "outputs/summary/phase2_5_retention_cleanup_post_delete_domain_gate.csv",
                "--quiet",
            ]
        ),
        "post_delete_comparison_gate_check": _cmd(
            [
                "/home/ajifang/miniconda3/bin/python",
                "scripts/experiments/main_build_domain_official_comparison.py",
                "--root",
                ".",
                "--domain",
                str(item["domain"]),
                "--gate_json",
                "outputs/summary/phase2_5_retention_cleanup_post_delete_domain_gate.json",
                "--output_dir",
                "outputs/summary/phase2_5_retention_cleanup_comparison_recheck",
                "--stamp",
                "tools_retention_cleanup_post_delete",
                "--n_bootstrap",
                "0",
                "--quiet",
            ]
        ),
    }

    return {
        "plan_id": plan_id,
        "artifact_class": "phase2_5_retention_cleanup_plan",
        "status_label": "planning_only_not_executed",
        "will_delete": False,
        "will_delete_files": False,
        "will_execute_cleanup": False,
        "will_start_experiment": False,
        "phase2_5_experiment_launch_allowed_after_cleanup": False,
        "requires_explicit_approval": True,
        "approval_token_required": approval_token,
        "retention_decision_state": item["retention_decision_state"],
        "ranked_retention_audit_source": item["ranked_retention_audit_source"],
        "recommended_by_ranked_audit": bool(item["recommended_by_ranked_audit"]),
        "retention_risk_tier": item["retention_risk_tier"],
        "retention_risk_rank": int(item["retention_risk_rank"]),
        "ranked_audit_current_free_bytes": int(item["ranked_audit_current_free_bytes"]),
        "ranked_audit_expected_free_bytes_after_delete": int(
            item["ranked_audit_expected_free_bytes_after_delete"]
        ),
        "ranked_audit_would_clear_min_free_gate": bool(item["ranked_audit_would_clear_min_free_gate"]),
        "approval_required_reminder": (
            "This plan is a ranked decision surface only. Deletion remains prohibited until the user "
            "approves this exact target, the approval token is set, pre-delete sha256/size manifests "
            "are written, and post-delete domain/comparison gates pass."
        ),
        "candidate": item,
        "remote_project": remote_project,
        "output_dir": output_dir,
        "current_free_bytes": effective_current_free_bytes,
        "min_free_gib": float(min_free_gib),
        "min_free_bytes": min_free_bytes,
        "expected_free_bytes_after_delete": expected_after_delete,
        "expected_to_clear_min_free_gate": bool(expected_after_delete and expected_after_delete >= min_free_bytes),
        "manifest_outputs_if_executed": {
            "sha256": manifest_base + ".sha256",
            "size": manifest_base + ".size.txt",
            "domain_gate_recheck_json": "outputs/summary/phase2_5_retention_cleanup_domain_gate_recheck.json",
            "post_delete_domain_gate_json": "outputs/summary/phase2_5_retention_cleanup_post_delete_domain_gate.json",
            "post_delete_comparison_dir": "outputs/summary/phase2_5_retention_cleanup_comparison_recheck",
        },
        "commands": commands,
        "preconditions_before_removing_script_guard": [
            "User explicitly approves this exact target and records the retention/archive decision.",
            "The ranked retention audit recommendation is accepted for this exact target.",
            "User records that no cheap rerun/resume need depends on this embedding.",
            "No relevant Python experiment or matching baseline process is active.",
            "Target realpath exactly matches the plan target_path and size is at least expected_size_bytes.",
            "Target sha256 matches expected_sha256 from provenance before deletion.",
            "Completed row evidence remains official_completed with blockers=[] and score_coverage_rate=1.0.",
            "Domain gate passes before deletion.",
            "SHA256 and byte-size manifests are written before deletion.",
        ],
        "postconditions_after_delete": [
            "The target path is absent.",
            "Disk is above min_free_bytes.",
            "The domain gate still passes.",
            "The comparison/paired-test gate still passes without overwriting prior server-final evidence.",
            "Protected scores, provenance, audits, run summaries, and imported tables remain present.",
            "Canonical docs and shared memory record the deletion manifest, freed bytes, and preserved evidence.",
        ],
        "protected_paths": [
            "outputs/baselines/external_tasks/",
            "outputs/*_same_candidate/scores.csv",
            "outputs/*_same_candidate/fairness_provenance.json",
            "outputs/*_same_candidate/*score_audit*",
            "outputs/*_same_candidate/*run_summary*",
            "outputs/*_same_candidate/tables/",
            "outputs/*_same_candidate/*.pt",
            "outputs/*_same_candidate/*.pth",
            "/home/ajifang/models/Qwen/",
            "source code and canonical configs",
            "other projects and installed runtime packages",
        ],
        "git_commit": _git_commit(),
        "git_worktree_dirty_at_generation": bool(_git_status_short()),
        "git_status_short_relevant_at_generation": _git_status_short(
            [
                "scripts/audit/main_plan_phase2_5_retention_cleanup.py",
                "tests/test_plan_phase2_5_retention_cleanup.py",
                "docs/active_todo_pony_uncertainty.md",
                "docs/paper_claims_and_status.md",
                output_dir,
            ]
        ),
    }


def guarded_shell_script(plan: dict[str, Any]) -> str:
    item = plan["candidate"]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "echo 'GUARDED PLAN ONLY: this file documents an explicit retention decision path.'",
        "echo 'It is intentionally non-runnable as generated and exits before any deletion command.'",
        "exit 2",
        "",
        f"cd {_shell_var(plan['remote_project'])}",
        f"TARGET={_shell_var(item['target_path'])}",
        f"EXPECTED_SIZE={int(item['expected_size_bytes'])}",
        f"APPROVAL_TOKEN_REQUIRED={_shell_var(plan['approval_token_required'])}",
        "",
        "# 1. Read-only server preflight.",
        plan["commands"]["server_preflight_read_only"],
        "",
        "# 2. Evidence precondition check for the completed row.",
        plan["commands"]["evidence_precondition_check"],
        plan["commands"]["domain_gate_check"],
        "",
        "# 3. Exact target guard.",
        "test -f \"$TARGET\"",
        "RESOLVED_TARGET=$(realpath \"$TARGET\")",
        "test \"$RESOLVED_TARGET\" = \"$TARGET\"",
        "ACTUAL_SIZE=$(stat -c '%s' \"$TARGET\")",
        "test \"$ACTUAL_SIZE\" -ge \"$EXPECTED_SIZE\"",
        f"test \"$(sha256sum \"$TARGET\" | awk '{{print $1}}')\" = {_shell_var(item['expected_sha256'])}",
        "",
        "# 4. Explicit approval guard. Set APPROVAL_TOKEN only after the retention decision is recorded.",
        plan["commands"]["approval_guard"],
        "",
        "# 5. Manifest before delete.",
        plan["commands"]["manifest_before_delete"],
        "",
        "# 6. Delete only the approved target.",
        plan["commands"]["delete_target_after_approval"],
        "test ! -e \"$TARGET\"",
        "",
        "# 7. Recheck disk and domain evidence.",
        plan["commands"]["post_delete_disk_check"],
        plan["commands"]["post_delete_domain_gate_check"],
        plan["commands"]["post_delete_comparison_gate_check"],
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_plan_files(plan: dict[str, Any], *, output_json: str | Path, output_sh: str | Path) -> dict[str, str]:
    json_path = Path(output_json)
    sh_path = Path(output_sh)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    sh_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sh_path.write_text(guarded_shell_script(plan), encoding="utf-8")
    return {"json": str(json_path), "shell": str(sh_path)}


def decision_markdown(plan: dict[str, Any]) -> str:
    item = plan["candidate"]
    lines = [
        "# Phase 2.5 Retention Decision Packet",
        "",
        f"- Plan ID: `{plan['plan_id']}`",
        f"- Status: `{plan['status_label']}`",
        f"- Will delete now: `{plan['will_delete']}`",
        f"- Will start experiment: `{plan['will_start_experiment']}`",
        f"- Requires explicit approval: `{plan['requires_explicit_approval']}`",
        f"- Approval token required: `{plan['approval_token_required']}`",
        "",
        "## Candidate",
        "",
        f"- Description: {item['description']}",
        f"- Target path: `{item['target_path']}`",
        f"- Expected size bytes: `{item['expected_size_bytes']}`",
        f"- Expected sha256: `{item['expected_sha256']}`",
        f"- Risk tier: `{plan['retention_risk_tier']}`",
        f"- Risk rank: `{plan['retention_risk_rank']}`",
        f"- Retention audit source: `{plan['ranked_retention_audit_source']}`",
        f"- Current free bytes: `{plan['current_free_bytes']}`",
        f"- Expected free bytes after delete: `{plan['expected_free_bytes_after_delete']}`",
        f"- Clears 15GiB floor: `{plan['expected_to_clear_min_free_gate']}`",
        "",
        "## Required Preconditions",
        "",
    ]
    lines.extend(f"- {item}" for item in plan["preconditions_before_removing_script_guard"])
    lines.extend(
        [
            "",
            "## Required Postconditions",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in plan["postconditions_after_delete"])
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "This packet is non-destructive. Deletion remains prohibited until the exact target is approved, "
            "the approval token is set, the generated shell guard is deliberately removed, and all manifest/gate "
            "checks in the packet are run successfully.",
            "",
        ]
    )
    return "\n".join(lines)


def write_decision_markdown(plan: dict[str, Any], *, output_md: str | Path) -> str:
    md_path = Path(output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(decision_markdown(plan), encoding="utf-8")
    return str(md_path)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.rstrip("/\\")
    plan_id = args.plan_id or f"{args.candidate}_retention_cleanup_plan"
    output_json = args.output_json or f"{output_dir}/{plan_id}.json"
    output_sh = args.output_sh or f"{output_dir}/{plan_id}.sh"
    plan = build_plan(
        candidate=args.candidate,
        remote_project=args.remote_project,
        current_free_bytes=args.current_free_bytes,
        min_free_gib=args.min_free_gib,
        output_dir=output_dir,
        plan_id=plan_id,
        retention_audit_json=args.retention_audit_json or None,
    )
    outputs = write_plan_files(plan, output_json=output_json, output_sh=output_sh)
    if args.output_md:
        outputs["markdown"] = write_decision_markdown(plan, output_md=args.output_md)
    print(
        json.dumps(
            {
                "ok": True,
                "outputs": outputs,
                "will_delete": False,
                "requires_explicit_approval": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
