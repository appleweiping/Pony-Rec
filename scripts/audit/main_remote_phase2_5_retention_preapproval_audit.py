from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REMOTE = "pony-rec-gpu"
PROCESS_PATTERN = "pony-rec\\|ccrp\\|baseline\\|uncertainty"
REQUIRED_EVIDENCE_FILES = (
    "fairness_provenance.json",
    "server_final_evidence_audit.json",
    "tables/ranking_eval_records.csv",
    "scores.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run read-only remote pre-approval checks for a Phase 2.5 retention "
            "decision packet. This never deletes, manifests, or launches experiments."
        )
    )
    parser.add_argument("--plan_json", required=True)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
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


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _parse_df(text: str) -> dict[str, Any]:
    for line in reversed([row for row in text.splitlines() if row.strip()]):
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


def _parse_stat_size(text: str) -> int:
    parts = text.strip().split(maxsplit=1)
    if not parts:
        return 0
    try:
        return int(parts[0])
    except ValueError:
        return 0


def _parse_sha(text: str) -> str:
    return text.strip().split()[0] if text.strip().split() else ""


def _remote_python_json_command(script: str, *args: str) -> str:
    return " ".join(
        [
            "/home/ajifang/miniconda3/bin/python",
            "-c",
            shlex.quote(script),
            *[shlex.quote(arg) for arg in args],
        ]
    )


def _provenance_command(path: str) -> str:
    script = (
        "import json,sys;"
        "p=sys.argv[1];"
        "d=json.load(open(p,encoding='utf-8'));"
        "print(json.dumps({"
        "'implementation_status':d.get('implementation_status'),"
        "'blockers':d.get('blockers'),"
        "'score_coverage_rate':d.get('score_coverage_rate'),"
        "'qwen3_item_embedding_sha256':d.get('qwen3_item_embedding_sha256')"
        "},sort_keys=True))"
    )
    return _remote_python_json_command(script, path)


def _server_final_command(path: str) -> str:
    script = (
        "import json,sys;"
        "p=sys.argv[1];"
        "d=json.load(open(p,encoding='utf-8'));"
        "files=d.get('files') or {};"
        "print(json.dumps({"
        "'ok':d.get('ok'),"
        "'scores_present':(files.get('scores.csv') or {}).get('present'),"
        "'prediction_present':(files.get('predictions/rank_predictions.jsonl') or {}).get('present'),"
        "'ranking_eval_records_present':(files.get('tables/ranking_eval_records.csv') or {}).get('present')"
        "},sort_keys=True))"
    )
    return _remote_python_json_command(script, path)


def _evidence_file_command(project: str, protected_dir: str) -> str:
    checks = []
    for rel_path in REQUIRED_EVIDENCE_FILES:
        path = f"{project.rstrip('/')}/{protected_dir.strip('/')}/{rel_path}"
        checks.append(f"if test -f {shlex.quote(path)}; then stat -c '%s %n' {shlex.quote(path)}; else echo MISSING {shlex.quote(path)}; fi")
    return " ; ".join(checks)


def build_audit(
    *,
    plan_json: str | Path,
    remote: str = DEFAULT_REMOTE,
    command_runner: Any = _run_remote,
) -> dict[str, Any]:
    plan = _read_json(plan_json)
    candidate = plan.get("candidate") or {}
    target_path = str(candidate.get("target_path", ""))
    expected_size = int(candidate.get("expected_size_bytes") or 0)
    expected_sha = str(candidate.get("expected_sha256", ""))
    project = str(plan.get("remote_project") or "/home/ajifang/projects/pony-rec-rescue-shadow-v6")
    protected_dir = str(candidate.get("protected_evidence_dir", ""))
    min_free_bytes = int(plan.get("min_free_bytes") or 0)

    commands = {
        "processes": (
            f"cd {shlex.quote(project)} && ps aux | grep python | grep -v grep | "
            f"grep -i {shlex.quote(PROCESS_PATTERN)} || true"
        ),
        "gpu": "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader",
        "disk": "df -B1 /",
        "target_realpath": f"realpath {shlex.quote(target_path)}",
        "target_stat": f"stat -c '%s %n' {shlex.quote(target_path)}",
        "target_sha256": f"sha256sum {shlex.quote(target_path)}",
        "evidence_files": _evidence_file_command(project, protected_dir),
        "fairness_provenance": _provenance_command(
            f"{project.rstrip('/')}/{protected_dir.strip('/')}/fairness_provenance.json"
        ),
        "server_final_audit": _server_final_command(
            f"{project.rstrip('/')}/{protected_dir.strip('/')}/server_final_evidence_audit.json"
        ),
    }
    raw: dict[str, str] = {}
    failures: list[str] = []
    for name, command in commands.items():
        try:
            raw[name] = command_runner(remote, command)
        except subprocess.CalledProcessError as exc:
            raw[name] = exc.stdout or str(exc)
            failures.append(f"remote_command_failed:{name}")

    active_processes = [line for line in raw.get("processes", "").splitlines() if line.strip()]
    if active_processes:
        failures.append("active_project_python_processes_present")
    disk = _parse_df(raw.get("disk", ""))
    if int(disk.get("free_bytes") or 0) < min_free_bytes:
        failures.append("disk_below_min_free_before_cleanup")
    if disk.get("used_pct") is not None and int(disk["used_pct"]) >= 97:
        failures.append("disk_used_pct_at_or_above_97")
    realpath = raw.get("target_realpath", "").strip()
    if realpath != target_path:
        failures.append("target_realpath_mismatch")
    actual_size = _parse_stat_size(raw.get("target_stat", ""))
    if actual_size != expected_size:
        failures.append("target_size_mismatch")
    actual_sha = _parse_sha(raw.get("target_sha256", ""))
    if actual_sha != expected_sha:
        failures.append("target_sha256_mismatch")
    evidence_lines = [line for line in raw.get("evidence_files", "").splitlines() if line.strip()]
    if any(line.startswith("MISSING ") for line in evidence_lines):
        failures.append("required_evidence_file_missing")

    provenance: dict[str, Any] = {}
    try:
        provenance = json.loads(raw.get("fairness_provenance", "{}"))
    except json.JSONDecodeError:
        failures.append("fairness_provenance_parse_failed")
    if provenance.get("implementation_status") != "official_completed":
        failures.append("implementation_status_not_official_completed")
    if provenance.get("blockers") not in ([], None):
        failures.append("fairness_provenance_has_blockers")
    if provenance.get("score_coverage_rate") != 1.0:
        failures.append("score_coverage_rate_not_1")
    if provenance.get("qwen3_item_embedding_sha256") and provenance.get("qwen3_item_embedding_sha256") != expected_sha:
        failures.append("provenance_embedding_sha_mismatch")

    server_final: dict[str, Any] = {}
    try:
        server_final = json.loads(raw.get("server_final_audit", "{}"))
    except json.JSONDecodeError:
        failures.append("server_final_audit_parse_failed")
    if server_final.get("ok") is not True:
        failures.append("server_final_audit_not_ok")
    if server_final.get("scores_present") is not True:
        failures.append("server_final_scores_not_present")
    if server_final.get("ranking_eval_records_present") is not True:
        failures.append("server_final_ranking_eval_records_not_present")

    preapproval_ready = not [
        failure
        for failure in failures
        if failure not in {"disk_below_min_free_before_cleanup"}
    ]
    return {
        "schema_version": "2026-06-06.retention_preapproval.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "read_only_remote_phase2_5_retention_preapproval_audit",
        "read_only": True,
        "will_delete": False,
        "will_start_experiment": False,
        "remote": remote,
        "plan_json": str(plan_json),
        "target_path": target_path,
        "expected_size_bytes": expected_size,
        "actual_size_bytes": actual_size,
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "disk": disk,
        "min_free_bytes": min_free_bytes,
        "active_process_count": len(active_processes),
        "evidence_file_lines": evidence_lines,
        "fairness_provenance": provenance,
        "server_final_audit": server_final,
        "preapproval_checks_ready_except_disk": preapproval_ready,
        "ok": not failures,
        "failures": failures,
        "commands_executed_read_only": commands,
    }


def write_markdown(path: str | Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Phase 2.5 Retention Pre-Approval Audit",
        "",
        f"- Generated UTC: `{audit['created_at_utc']}`",
        f"- OK: `{audit['ok']}`",
        f"- Read only: `{audit['read_only']}`",
        f"- Will delete: `{audit['will_delete']}`",
        f"- Will start experiment: `{audit['will_start_experiment']}`",
        f"- Preapproval checks ready except disk: `{audit['preapproval_checks_ready_except_disk']}`",
        f"- Active process count: `{audit['active_process_count']}`",
        f"- Free bytes: `{audit['disk'].get('free_bytes')}`",
        f"- Target size bytes: `{audit['actual_size_bytes']}`",
        f"- Target sha256 matches expected: `{audit['actual_sha256'] == audit['expected_sha256']}`",
        "",
        "## Failures",
        "",
    ]
    lines.extend(f"- {failure}" for failure in audit["failures"])
    if not audit["failures"]:
        lines.append("- none")
    lines.append("")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit = build_audit(plan_json=args.plan_json, remote=args.remote)
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(args.output_md, audit)
    print(
        json.dumps(
            {
                "ok": audit["ok"],
                "preapproval_checks_ready_except_disk": audit["preapproval_checks_ready_except_disk"],
                "failures": audit["failures"],
            },
            indent=2,
        )
    )
    if not audit["preapproval_checks_ready_except_disk"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
