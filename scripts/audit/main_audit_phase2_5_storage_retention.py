from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REMOTE = "pony-rec-gpu"
DEFAULT_PROJECT = "~/projects/pony-rec-rescue-shadow-v6"
BYTES_PER_GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a read-only Phase 2.5 server storage audit. The script only "
            "uses df, ps, nvidia-smi, find, du, ls, and stat-style reads."
        )
    )
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--min_free_gib", type=float, default=15.0)
    parser.add_argument("--preferred_free_gib", type=float, default=25.0)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_md", default="")
    return parser.parse_args()


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


def _read_size_path_lines(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            size = int(parts[0])
        except ValueError:
            continue
        rows.append({"size_bytes": size, "path": parts[1]})
    return rows


def _parse_df_bytes(text: str) -> dict[str, Any]:
    for line in reversed([value for value in text.splitlines() if value.strip()]):
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            return {
                "filesystem": parts[0],
                "blocks_1b": int(parts[1]),
                "used_bytes": int(parts[2]),
                "free_bytes": int(parts[3]),
                "used_pct": int(parts[4].rstrip("%")),
                "mount": parts[5],
                "raw": line,
            }
        except ValueError:
            continue
    return {"free_bytes": 0, "used_pct": None, "raw": text.strip()}


def classify_large_file(path: str, size_bytes: int) -> dict[str, Any]:
    normalized = path.replace("\\", "/")
    lower = normalized.lower()
    if "/projects/llm2rec/item_info/" in lower and lower.endswith((".npy", ".pkl", ".npz")):
        return {
            "classification": "NEEDS_APPROVAL_OR_ARCHIVE_DECISION",
            "retention_risk_tier": "approval_required_external_embedding_cache",
            "retention_risk_rank": 20,
            "reason": (
                "Completed upstream embedding/cache outside the final evidence directory. "
                "High-yield, but protected unless an explicit retention/archive decision is recorded."
            ),
        }
    if normalized.startswith("outputs/baselines/external_tasks/"):
        return {
            "classification": "PROTECTED_TASK_SPLIT",
            "retention_risk_tier": "protected_task_split",
            "retention_risk_rank": 90,
            "reason": "Canonical same-candidate train/valid/test task split or metadata.",
        }
    if "ccrp_selected_test_scored_rows" in lower or "ccrp_selected_valid_scored_rows" in lower:
        return {
            "classification": "PROTECTED_SIGNAL_EVIDENCE",
            "retention_risk_tier": "protected_signal_evidence",
            "retention_risk_rank": 95,
            "reason": "Event-level C-CRP/scored-row evidence needed for paper-critical signal analysis.",
        }
    if normalized.endswith("/predictions/rank_predictions.jsonl"):
        return {
            "classification": "NEEDS_SERVER_FINAL_LOCAL_LIGHT_AUDIT_BEFORE_DELETE",
            "retention_risk_tier": "eligible_only_after_server_final_local_light",
            "retention_risk_rank": 40,
            "reason": "Large prediction file can be deleted only under the documented post-gate exception.",
        }
    final_evidence_tokens = (
        "/scores.csv",
        "/fairness_provenance.json",
        "/server_final_evidence_audit.json",
        "/local_light_evidence_audit.json",
        "/ranking_eval_records.csv",
    )
    if any(lower.endswith(token) for token in final_evidence_tokens):
        return {
            "classification": "PROTECTED_FINAL_EVIDENCE",
            "retention_risk_tier": "protected_final_evidence",
            "retention_risk_rank": 100,
            "reason": "Final scores/provenance/audit/imported records are required by gates and comparisons.",
        }
    if lower.endswith((".pt", ".pth", ".ckpt", ".safetensors", ".bin")):
        return {
            "classification": "NEEDS_APPROVAL_OR_ARCHIVE_DECISION",
            "retention_risk_tier": "approval_required_final_model_checkpoint",
            "retention_risk_rank": 30,
            "reason": "Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.",
        }
    if normalized.startswith("outputs/baselines/official_adapters/") and lower.endswith(".csv"):
        return {
            "classification": "PROTECTED_LEGACY_SCORE_EVIDENCE",
            "retention_risk_tier": "protected_legacy_score_evidence",
            "retention_risk_rank": 90,
            "reason": "Legacy/local-light official score evidence; not a safe disk cleanup target.",
        }
    return {
        "classification": "REVIEW_BEFORE_DELETE",
        "retention_risk_tier": "manual_review_required",
        "retention_risk_rank": 70,
        "reason": "Large file does not match a safe-now rule; manual retention review required.",
    }


def classify_safe_candidate(path: str, size_bytes: int) -> dict[str, Any]:
    normalized = path.replace("\\", "/")
    if normalized.startswith("outputs/baselines/paper_adapters/tools_large10000_100neg_"):
        classification = "SAFE_NOW_LOW_YIELD"
        reason = "Completed Tools paper-adapter staging; low yield and insufficient by itself for Phase 2.5."
    elif normalized.startswith("tmp_") or "/tmp/" in normalized:
        classification = "SAFE_NOW_LOW_YIELD"
        reason = "Temporary directory; low yield in current scan."
    else:
        classification = "REVIEW_BEFORE_DELETE"
        reason = "No safe-now rule matched."
    return {"path": path, "size_bytes": size_bytes, "classification": classification, "reason": reason}


def _approval_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    try:
        risk_rank = int(row.get("retention_risk_rank", 70))
    except (TypeError, ValueError):
        risk_rank = 70
    return risk_rank, -int(row["size_bytes"]), str(row["path"])


def _with_retention_decision_fields(row: dict[str, Any], *, current_free_bytes: int, min_free_bytes: int) -> dict[str, Any]:
    size = int(row["size_bytes"])
    expected_free = current_free_bytes + size
    value = dict(row)
    value["expected_free_bytes_after_delete"] = expected_free
    value["would_clear_min_free_gate"] = expected_free >= min_free_bytes
    value["approval_decision_required"] = True
    return value


def _remote_commands(project: str) -> dict[str, str]:
    cd = f"cd {project}"
    return {
        "processes": (
            f"{cd} && ps aux | grep python | grep -v grep | "
            "grep -i 'pony-rec\\|ccrp\\|baseline\\|uncertainty' || true"
        ),
        "gpu": "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader",
        "disk": "df -B1 /home/ajifang",
        "large_outputs": (
            f"{cd} && find outputs -type f -size +50M -printf '%s %p\\n' "
            "2>/dev/null | sort -nr | head -80"
        ),
        "llm2rec_item_info": (
            "find /home/ajifang/projects/LLM2Rec/item_info -type f -size +50M "
            "-printf '%s %p\\n' 2>/dev/null | sort -nr | head -40"
        ),
        "safe_candidate_dirs": (
            f"{cd} && du -sb "
            "outputs/baselines/paper_adapters/tools_large10000_100neg_llm2rec_official_adapter "
            "outputs/baselines/paper_adapters/tools_large10000_100neg_llmesr_official_adapter "
            "tmp_llm2rec_sync 2>/dev/null || true"
        ),
        "logs_and_pids": (
            f"{cd} && find . -maxdepth 1 -type f \\( -name '*.log' -o -name '*.pid' \\) "
            "-printf '%s %p\\n' 2>/dev/null | sort -nr | head -80"
        ),
    }


def build_audit(
    *,
    remote: str = DEFAULT_REMOTE,
    project: str = DEFAULT_PROJECT,
    min_free_gib: float = 15.0,
    preferred_free_gib: float = 25.0,
    command_runner: Any = _run_remote,
) -> dict[str, Any]:
    commands = _remote_commands(project)
    raw = {name: command_runner(remote, command) for name, command in commands.items()}
    disk = _parse_df_bytes(raw["disk"])
    min_free_bytes = int(min_free_gib * BYTES_PER_GIB)
    preferred_free_bytes = int(preferred_free_gib * BYTES_PER_GIB)
    current_free_bytes = int(disk.get("free_bytes") or 0)

    large_rows = _read_size_path_lines(raw["large_outputs"]) + _read_size_path_lines(raw["llm2rec_item_info"])
    seen_large: set[str] = set()
    large_files: list[dict[str, Any]] = []
    for row in large_rows:
        path = str(row["path"])
        if path in seen_large:
            continue
        seen_large.add(path)
        size = int(row["size_bytes"])
        large_files.append({"path": path, "size_bytes": size, **classify_large_file(path, size)})

    safe_candidates = [
        classify_safe_candidate(str(row["path"]), int(row["size_bytes"]))
        for row in _read_size_path_lines(raw["safe_candidate_dirs"])
    ]
    safe_now = [row for row in safe_candidates if row["classification"] == "SAFE_NOW_LOW_YIELD"]
    safe_now_total = sum(int(row["size_bytes"]) for row in safe_now)
    high_yield = [
        _with_retention_decision_fields(row, current_free_bytes=current_free_bytes, min_free_bytes=min_free_bytes)
        for row in large_files
        if row["classification"] == "NEEDS_APPROVAL_OR_ARCHIVE_DECISION"
        and int(row["size_bytes"]) + current_free_bytes >= min_free_bytes
    ]
    high_yield = sorted(high_yield, key=_approval_sort_key)
    recommended_candidate = high_yield[0] if high_yield else None

    active_processes = [line for line in raw["processes"].splitlines() if line.strip()]
    experiment_launch_allowed = (
        not active_processes and current_free_bytes >= min_free_bytes and disk.get("used_pct", 100) < 97
    )
    return {
        "schema_version": "2026-06-05.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "read_only_phase2_5_storage_retention_audit",
        "remote": remote,
        "project": project,
        "commands_executed_read_only": commands,
        "server": {
            "relevant_python_processes": active_processes,
            "gpu": raw["gpu"].strip(),
            "disk": disk,
        },
        "phase2_5_disk_gate": {
            "required_free_bytes_min": min_free_bytes,
            "preferred_free_bytes": preferred_free_bytes,
            "current_free_bytes": current_free_bytes,
            "deficit_to_min_free_bytes": max(0, min_free_bytes - current_free_bytes),
            "used_pct_limit": 97,
            "experiment_launch_allowed": experiment_launch_allowed,
        },
        "safe_now_candidates": safe_candidates,
        "safe_now_total_recoverable_bytes": safe_now_total,
        "safe_now_sufficient_for_min_free": current_free_bytes + safe_now_total >= min_free_bytes,
        "high_yield_candidates_requiring_approval": high_yield,
        "recommended_approval_candidate": recommended_candidate,
        "retention_recommendation": {
            "status": "approval_required" if recommended_candidate else "no_single_approval_candidate_found",
            "candidate_path": recommended_candidate.get("path") if recommended_candidate else "",
            "retention_risk_tier": recommended_candidate.get("retention_risk_tier") if recommended_candidate else "",
            "expected_free_bytes_after_delete": recommended_candidate.get("expected_free_bytes_after_delete")
            if recommended_candidate
            else None,
            "would_clear_min_free_gate": recommended_candidate.get("would_clear_min_free_gate") if recommended_candidate else False,
            "reason": (
                "This is the lowest-risk high-yield candidate under the audit's current policy. "
                "It still must not be deleted without an explicit archive/retention approval and post-delete gate checks."
            )
            if recommended_candidate
            else "No approval-required candidate individually clears the configured minimum free-space gate.",
        },
        "large_file_classification": large_files,
        "log_pid_files": _read_size_path_lines(raw["logs_and_pids"]),
        "audit_verdict": {
            "delete_performed": False,
            "safe_cleanup_enough_to_start_phase2_5": current_free_bytes + safe_now_total >= min_free_bytes,
            "minimum_next_action": (
                "Do not launch Phase 2.5 signal-row regeneration until disk is expanded "
                "or one high-yield completed artifact receives an explicit archive/retention decision."
            )
            if current_free_bytes + safe_now_total < min_free_bytes
            else "Safe-now cleanup could clear the minimum disk gate after a no-process recheck.",
        },
    }


def write_markdown(path: str | Path, audit: dict[str, Any]) -> None:
    gate = audit["phase2_5_disk_gate"]
    lines = [
        "# Phase 2.5 Storage Retention Audit",
        "",
        f"- Generated UTC: `{audit['created_at_utc']}`",
        f"- Remote: `{audit['remote']}`",
        f"- Project: `{audit['project']}`",
        f"- Active project Python processes: `{len(audit['server']['relevant_python_processes'])}`",
        f"- GPU: `{audit['server']['gpu']}`",
        f"- Free bytes: `{gate['current_free_bytes']}`",
        f"- Deficit to minimum: `{gate['deficit_to_min_free_bytes']}`",
        f"- Experiment launch allowed: `{gate['experiment_launch_allowed']}`",
        f"- Safe-now recoverable bytes: `{audit['safe_now_total_recoverable_bytes']}`",
        f"- Safe-now sufficient: `{audit['safe_now_sufficient_for_min_free']}`",
        "",
    ]
    recommendation = audit.get("retention_recommendation", {})
    if recommendation.get("candidate_path"):
        lines.extend(
            [
                "## Recommended Approval Candidate",
                "",
                f"- Path: `{recommendation['candidate_path']}`",
                f"- Risk tier: `{recommendation['retention_risk_tier']}`",
                f"- Expected free bytes after delete: `{recommendation['expected_free_bytes_after_delete']}`",
                f"- Clears minimum gate: `{recommendation['would_clear_min_free_gate']}`",
                f"- Note: {recommendation['reason']}",
                "",
                "## High-Yield Approval-Required Candidates",
                "",
            ]
        )
    for row in audit["high_yield_candidates_requiring_approval"][:12]:
        lines.append(
            f"- `{row['path']}`: `{row['size_bytes']}` bytes; "
            f"`{row.get('retention_risk_tier', '')}`; expected free "
            f"`{row.get('expected_free_bytes_after_delete')}`; {row['reason']}"
        )
    if not audit["high_yield_candidates_requiring_approval"]:
        lines.append("- None.")
    lines.extend(["", "## Safe-Now Low-Yield Candidates", ""])
    for row in audit["safe_now_candidates"]:
        lines.append(f"- `{row['path']}`: `{row['size_bytes']}` bytes; `{row['classification']}`")
    lines.extend(["", "## Verdict", "", audit["audit_verdict"]["minimum_next_action"], ""])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit = build_audit(
        remote=args.remote,
        project=args.project,
        min_free_gib=args.min_free_gib,
        preferred_free_gib=args.preferred_free_gib,
    )
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(args.output_md, audit)
    print(
        json.dumps(
            {
                "ok": True,
                "experiment_launch_allowed": audit["phase2_5_disk_gate"]["experiment_launch_allowed"],
                "safe_now_sufficient": audit["safe_now_sufficient_for_min_free"],
                "high_yield_approval_required_count": len(audit["high_yield_candidates_requiring_approval"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
