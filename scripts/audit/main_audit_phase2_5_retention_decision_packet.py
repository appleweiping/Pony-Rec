from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_TRUE_FIELDS = (
    "requires_explicit_approval",
    "expected_to_clear_min_free_gate",
)
REQUIRED_FALSE_FIELDS = (
    "will_delete",
    "will_delete_files",
    "will_execute_cleanup",
    "will_start_experiment",
)
FORBIDDEN_SHELL_SNIPPETS = (
    "nohup",
    "run_baselines_new_domains.sh",
    "rsync --delete",
    "find -delete",
    "Remove-Item",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a Phase 2.5 retention decision packet without SSH, deletion, "
            "manifesting, or experiment launch."
        )
    )
    parser.add_argument("--plan_json", required=True)
    parser.add_argument("--plan_sh", default="")
    parser.add_argument("--plan_md", default="")
    parser.add_argument("--packet_sha256", default="")
    parser.add_argument("--retention_audit_json", default="")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_md", default="")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sha_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            rows[parts[-1]] = parts[0]
    return rows


def _infer_peer(plan_json: Path, suffix: str) -> Path:
    return plan_json.with_suffix(suffix)


def build_audit(
    *,
    plan_json: str | Path,
    plan_sh: str | Path | None = None,
    plan_md: str | Path | None = None,
    packet_sha256: str | Path | None = None,
    retention_audit_json: str | Path | None = None,
) -> dict[str, Any]:
    json_path = Path(plan_json)
    shell_path = Path(plan_sh) if plan_sh else _infer_peer(json_path, ".sh")
    markdown_path = Path(plan_md) if plan_md else _infer_peer(json_path, ".md")
    sha_path = Path(packet_sha256) if packet_sha256 else _infer_peer(json_path, ".sha256")
    failures: list[str] = []
    warnings: list[str] = []

    for label, path in (("plan_json", json_path), ("plan_sh", shell_path), ("plan_md", markdown_path)):
        if not path.exists():
            failures.append(f"missing_{label}:{path}")
        elif path.stat().st_size <= 0:
            failures.append(f"empty_{label}:{path}")

    plan: dict[str, Any] = {}
    if json_path.exists() and json_path.stat().st_size > 0:
        plan = _read_json(json_path)

    for field in REQUIRED_TRUE_FIELDS:
        if plan.get(field) is not True:
            failures.append(f"plan_field_not_true:{field}")
    for field in REQUIRED_FALSE_FIELDS:
        if plan.get(field) is not False:
            failures.append(f"plan_field_not_false:{field}")
    if plan.get("status_label") != "planning_only_not_executed":
        failures.append("plan_status_not_planning_only")
    if not plan.get("approval_token_required"):
        failures.append("missing_approval_token_required")
    if plan.get("current_free_bytes") is None or int(plan.get("current_free_bytes") or 0) <= 0:
        failures.append("missing_current_free_bytes")
    if int(plan.get("expected_free_bytes_after_delete") or 0) <= int(plan.get("current_free_bytes") or 0):
        failures.append("expected_free_not_greater_than_current_free")

    shell_text = shell_path.read_text(encoding="utf-8", errors="replace") if shell_path.exists() else ""
    if "exit 2" not in shell_text:
        failures.append("shell_missing_exit_2_guard")
    else:
        exit_pos = shell_text.index("exit 2")
        for snippet in ("sha256sum", "rm --"):
            if snippet not in shell_text:
                failures.append(f"shell_missing_required_snippet:{snippet}")
            elif exit_pos > shell_text.index(snippet):
                failures.append(f"shell_exit_guard_after:{snippet}")
    for snippet in FORBIDDEN_SHELL_SNIPPETS:
        if snippet in shell_text:
            failures.append(f"shell_forbidden_snippet:{snippet}")

    markdown_text = markdown_path.read_text(encoding="utf-8", errors="replace") if markdown_path.exists() else ""
    for snippet in ("Will delete now: `False`", "Requires explicit approval: `True`", "Deletion remains prohibited"):
        if snippet not in markdown_text:
            failures.append(f"markdown_missing_safety_text:{snippet}")
    if "rm --" in markdown_text:
        failures.append("markdown_contains_delete_command")

    manifest = _load_sha_manifest(sha_path)
    manifest_checks: dict[str, Any] = {}
    if not manifest:
        failures.append(f"missing_or_empty_packet_sha256:{sha_path}")
    for path in (json_path, shell_path, markdown_path):
        expected = manifest.get(path.name)
        actual = _sha256(path) if path.exists() and path.is_file() else ""
        manifest_checks[path.name] = {"expected": expected, "actual": actual, "ok": bool(expected and expected == actual)}
        if manifest and not manifest_checks[path.name]["ok"]:
            failures.append(f"packet_sha256_mismatch:{path.name}")

    source_path = Path(retention_audit_json) if retention_audit_json else None
    if not source_path and plan.get("ranked_retention_audit_source"):
        source_path = Path(str(plan["ranked_retention_audit_source"]))
    retention_summary: dict[str, Any] = {}
    if source_path:
        if not source_path.exists():
            failures.append(f"missing_retention_audit_json:{source_path}")
        else:
            retention = _read_json(source_path)
            recommendation = retention.get("recommended_approval_candidate") or {}
            candidate = plan.get("candidate") or {}
            target = str(candidate.get("target_path", ""))
            if str(recommendation.get("path", "")) != target:
                failures.append("retention_audit_recommends_different_target")
            retention_summary = {
                "path": str(source_path),
                "experiment_launch_allowed": (retention.get("phase2_5_disk_gate") or {}).get(
                    "experiment_launch_allowed"
                ),
                "recommended_path": recommendation.get("path"),
                "recommended_expected_free_bytes_after_delete": recommendation.get(
                    "expected_free_bytes_after_delete"
                ),
            }
    else:
        warnings.append("no_retention_audit_source_provided")

    return {
        "schema_version": "2026-06-06.retention_packet_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "local_phase2_5_retention_decision_packet_audit",
        "read_only": True,
        "will_ssh": False,
        "will_delete": False,
        "will_start_experiment": False,
        "ok": not failures,
        "plan_json": str(json_path),
        "plan_sh": str(shell_path),
        "plan_md": str(markdown_path),
        "packet_sha256": str(sha_path),
        "plan_summary": {
            "plan_id": plan.get("plan_id"),
            "candidate": (plan.get("candidate") or {}).get("candidate"),
            "target_path": (plan.get("candidate") or {}).get("target_path"),
            "current_free_bytes": plan.get("current_free_bytes"),
            "expected_free_bytes_after_delete": plan.get("expected_free_bytes_after_delete"),
            "requires_explicit_approval": plan.get("requires_explicit_approval"),
            "approval_token_required": plan.get("approval_token_required"),
        },
        "retention_audit_summary": retention_summary,
        "manifest_checks": manifest_checks,
        "failures": failures,
        "warnings": warnings,
    }


def write_markdown(path: str | Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Phase 2.5 Retention Packet Audit",
        "",
        f"- Generated UTC: `{audit['created_at_utc']}`",
        f"- OK: `{audit['ok']}`",
        f"- Read only: `{audit['read_only']}`",
        f"- Will delete: `{audit['will_delete']}`",
        f"- Will start experiment: `{audit['will_start_experiment']}`",
        f"- Plan: `{audit['plan_json']}`",
        f"- Target: `{audit['plan_summary']['target_path']}`",
        f"- Current free bytes: `{audit['plan_summary']['current_free_bytes']}`",
        f"- Expected free bytes after delete: `{audit['plan_summary']['expected_free_bytes_after_delete']}`",
        "",
        "## Failures",
        "",
    ]
    lines.extend(f"- {failure}" for failure in audit["failures"])
    if not audit["failures"]:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in audit["warnings"])
    if not audit["warnings"]:
        lines.append("- none")
    lines.append("")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit = build_audit(
        plan_json=args.plan_json,
        plan_sh=args.plan_sh or None,
        plan_md=args.plan_md or None,
        packet_sha256=args.packet_sha256 or None,
        retention_audit_json=args.retention_audit_json or None,
    )
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(args.output_md, audit)
    print(json.dumps({"ok": audit["ok"], "failures": audit["failures"]}, indent=2))
    if not audit["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
