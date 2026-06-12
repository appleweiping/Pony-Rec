from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.audit.main_audit_external_proceedings_metadata import (
    DEFAULT_CONFIG as DEFAULT_EXTERNAL_CONFIG,
    build_external_proceedings_metadata_audit,
)
from scripts.audit.main_audit_external_proceedings_metadata import _write_md as write_external_md
from scripts.audit.main_audit_submission_package import (
    DEFAULT_CLAIM_AUDIT_JSON,
    DEFAULT_METADATA_FOLLOWUP_JSON,
    DEFAULT_PANEL_REVIEW_JSON,
    DEFAULT_TARGET_PROFILE_JSON,
    build_submission_package_audit,
)
from scripts.audit.main_audit_submission_package import _write_md as write_package_md
from scripts.audit.main_build_final_submission_gate import build_final_submission_gate
from scripts.audit.main_build_final_submission_gate import _write_md as write_final_gate_md
from scripts.audit.main_build_manual_submission_checklist import (
    DEFAULT_CONFIG as DEFAULT_MANUAL_CONFIG,
    build_manual_submission_checklist,
)
from scripts.audit.main_build_manual_submission_checklist import _write_md as write_manual_md
from scripts.audit.main_build_submission_metadata_packet import (
    DEFAULT_METADATA_CONFIG,
    build_submission_metadata_packet,
)
from scripts.audit.main_build_submission_metadata_packet import _write_md as write_metadata_md


DEFAULT_OUTPUT_DIR = Path("outputs/summary/paper_critical")
DEFAULT_STAMP = "20260612"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_state(path: Path, root: Path) -> dict[str, Any]:
    try:
        display_path = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        display_path = str(path)
    return {
        "path": display_path,
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        "sha256": _sha256_file(path) if path.exists() and path.is_file() else "",
    }


def _run_git(root: Path, args: list[str]) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _git_state(root: Path) -> dict[str, Any]:
    code, head, err = _run_git(root, ["rev-parse", "HEAD"])
    if code != 0:
        return {
            "available": False,
            "head": "",
            "tracked_dirty": None,
            "tracked_dirty_paths": [],
            "error": err or "git_rev_parse_failed",
        }
    status_code, status, status_err = _run_git(root, ["status", "--short", "--untracked-files=no"])
    tracked_paths = [line.strip() for line in status.splitlines() if line.strip()] if status_code == 0 else []
    return {
        "available": True,
        "head": head,
        "tracked_dirty": bool(tracked_paths),
        "tracked_dirty_paths": tracked_paths,
        "error": "" if status_code == 0 else status_err,
    }


def _input_fingerprints(root: Path, paths: list[str | Path]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = root / raw_path
        state = _file_state(path, root)
        key = state["path"]
        if key in seen:
            continue
        seen.add(key)
        result.append(state)
    return result


def _step_record(
    *,
    step_id: str,
    json_path: Path,
    md_path: Path,
    root: Path,
    payload: dict[str, Any],
    ready_field: str,
) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "json": _file_state(json_path, root),
        "md": _file_state(md_path, root),
        "ok": payload.get("ok") is True,
        "ready_field": ready_field,
        "ready": payload.get(ready_field) is True,
        "final_submission_ready": payload.get("final_submission_ready") is True,
        "failures": list(payload.get("failures") or []),
        "warnings": list(payload.get("warnings") or []),
        "remaining_blockers": list(payload.get("remaining_blockers") or []),
        "created_at_utc": payload.get("created_at_utc"),
        "schema_version": payload.get("schema_version"),
    }


def refresh_pre_submission_gates(
    *,
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    stamp: str = DEFAULT_STAMP,
    paper_dir: str | Path = "Paper",
    external_config_path: str | Path = DEFAULT_EXTERNAL_CONFIG,
    external_network_mode: str = "live",
    external_fixture_json: str | Path | None = None,
    external_timeout_seconds: int = 20,
    claim_audit_json: str | Path = DEFAULT_CLAIM_AUDIT_JSON,
    panel_review_json: str | Path = DEFAULT_PANEL_REVIEW_JSON,
    metadata_followup_json: str | Path = DEFAULT_METADATA_FOLLOWUP_JSON,
    target_profile_json: str | Path = DEFAULT_TARGET_PROFILE_JSON,
    target_profile_id: str | None = None,
    metadata_config: str | Path = DEFAULT_METADATA_CONFIG,
    manual_config: str | Path = DEFAULT_MANUAL_CONFIG,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    out_dir = repo / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    git_state_before_refresh = _git_state(repo)

    external_json = out_dir / f"external_proceedings_metadata_recheck_{stamp}.json"
    external_md = out_dir / f"external_proceedings_metadata_recheck_{stamp}.md"
    package_json = out_dir / f"submission_package_audit_{stamp}.json"
    package_md = out_dir / f"submission_package_audit_{stamp}.md"
    metadata_json = out_dir / f"submission_metadata_packet_{stamp}.json"
    metadata_md = out_dir / f"submission_metadata_packet_{stamp}.md"
    manual_json = out_dir / f"manual_submission_checklist_{stamp}.json"
    manual_md = out_dir / f"manual_submission_checklist_{stamp}.md"
    final_json = out_dir / f"final_submission_gate_{stamp}.json"
    final_md = out_dir / f"final_submission_gate_{stamp}.md"
    input_fingerprints = _input_fingerprints(
        repo,
        [
            Path(paper_dir) / "main.tex",
            Path(paper_dir) / "references.bib",
            Path(paper_dir) / "main.pdf",
            Path(paper_dir) / "main.log",
            Path(paper_dir) / "main.blg",
            external_config_path,
            claim_audit_json,
            panel_review_json,
            metadata_followup_json,
            target_profile_json,
            metadata_config,
            manual_config,
            "scripts/audit/main_refresh_pre_submission_gates.py",
            "scripts/audit/main_audit_external_proceedings_metadata.py",
            "scripts/audit/main_audit_submission_package.py",
            "scripts/audit/main_build_submission_metadata_packet.py",
            "scripts/audit/main_build_manual_submission_checklist.py",
            "scripts/audit/main_build_final_submission_gate.py",
            "scripts/audit/main_audit_pre_submission_refresh_freshness.py",
        ],
    )

    external = build_external_proceedings_metadata_audit(
        root=repo,
        config_path=external_config_path,
        network_mode=external_network_mode,
        fixture_json=external_fixture_json,
        timeout_seconds=external_timeout_seconds,
    )
    _write_json(external_json, external)
    write_external_md(external_md, external)

    package = build_submission_package_audit(
        root=repo,
        paper_dir=paper_dir,
        claim_audit_json=claim_audit_json,
        panel_review_json=panel_review_json,
        metadata_followup_json=metadata_followup_json,
        external_metadata_audit_json=external_json.relative_to(repo),
        target_profile_json=target_profile_json,
        target_profile_id=target_profile_id,
    )
    _write_json(package_json, package)
    write_package_md(package_md, package)

    metadata = build_submission_metadata_packet(
        root=repo,
        paper_dir=paper_dir,
        metadata_config=metadata_config,
        submission_audit_json=package_json.relative_to(repo),
    )
    _write_json(metadata_json, metadata)
    write_metadata_md(metadata_md, metadata)

    manual = build_manual_submission_checklist(
        root=repo,
        config_path=manual_config,
        metadata_packet_json=metadata_json.relative_to(repo),
        submission_package_audit_json=package_json.relative_to(repo),
        external_metadata_audit_json=external_json.relative_to(repo),
    )
    _write_json(manual_json, manual)
    write_manual_md(manual_md, manual)

    final_gate = build_final_submission_gate(
        root=repo,
        submission_package_audit_json=package_json.relative_to(repo),
        submission_metadata_packet_json=metadata_json.relative_to(repo),
        external_metadata_audit_json=external_json.relative_to(repo),
        manual_checklist_json=manual_json.relative_to(repo),
    )
    _write_json(final_json, final_gate)
    write_final_gate_md(final_md, final_gate)

    steps = [
        _step_record(
            step_id="external_proceedings_metadata",
            json_path=external_json,
            md_path=external_md,
            root=repo,
            payload=external,
            ready_field="external_proceedings_metadata_ready",
        ),
        _step_record(
            step_id="submission_package",
            json_path=package_json,
            md_path=package_md,
            root=repo,
            payload=package,
            ready_field="submission_package_ready_for_target_formatting",
        ),
        _step_record(
            step_id="submission_metadata_packet",
            json_path=metadata_json,
            md_path=metadata_md,
            root=repo,
            payload=metadata,
            ready_field="submission_metadata_packet_ready",
        ),
        _step_record(
            step_id="manual_submission_checklist",
            json_path=manual_json,
            md_path=manual_md,
            root=repo,
            payload=manual,
            ready_field="manual_submission_checklist_ready",
        ),
        _step_record(
            step_id="final_submission_gate",
            json_path=final_json,
            md_path=final_md,
            root=repo,
            payload=final_gate,
            ready_field="final_submission_ready",
        ),
    ]
    failures = [f"{step['step_id']}:{failure}" for step in steps for failure in step["failures"]]
    blockers = []
    for step in steps:
        for blocker in step["remaining_blockers"]:
            text = str(blocker)
            if text not in blockers:
                blockers.append(text)
    warnings = [f"{step['step_id']}:{warning}" for step in steps for warning in step["warnings"]]

    return {
        "schema_version": "2026-06-12.pre_submission_gate_refresh.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_read_only_pre_submission_gate_refresh",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "stamp": stamp,
        "output_dir": str(Path(output_dir)),
        "external_network_mode": external_network_mode,
        "git_state_before_refresh": git_state_before_refresh,
        "input_fingerprints": input_fingerprints,
        "ok": not failures,
        "final_submission_ready": final_gate.get("final_submission_ready") is True,
        "final_verdict": final_gate.get("verdict"),
        "steps": steps,
        "failures": failures,
        "warnings": warnings,
        "remaining_blockers": blockers,
        "next_actions": [
            "Use final_submission_gate as the first-read final submission status.",
            "Rerun this refresh after any Paper, BibTeX, target profile, or submission metadata change.",
            "Keep private author/COI/reviewer/declaration fields outside the repository.",
        ],
    }


def _write_md(path: Path, refresh: dict[str, Any]) -> None:
    lines = [
        "# Pre-Submission Gate Refresh",
        "",
        f"Generated: {refresh['created_at_utc']}",
        "",
        f"- OK: `{str(refresh['ok']).lower()}`",
        f"- Final submission ready: `{str(refresh['final_submission_ready']).lower()}`",
        f"- Final verdict: `{refresh['final_verdict']}`",
        f"- External network mode: `{refresh['external_network_mode']}`",
        f"- Stamp: `{refresh['stamp']}`",
        f"- Git HEAD before refresh: `{(refresh.get('git_state_before_refresh') or {}).get('head', '')}`",
        "- Tracked dirty before refresh: "
        f"`{str((refresh.get('git_state_before_refresh') or {}).get('tracked_dirty')).lower()}`",
        "",
        "## Steps",
        "",
    ]
    for step in refresh["steps"]:
        lines.append(
            f"- `{step['step_id']}`: ok=`{str(step['ok']).lower()}`, "
            f"ready=`{str(step['ready']).lower()}`, json=`{step['json']['path']}`"
        )
    lines.extend(["", "## Input Fingerprints", ""])
    for item in refresh.get("input_fingerprints", []):
        if item.get("exists"):
            lines.append(
                f"- `{item['path']}`: `{item['sha256']}` ({item['size_bytes']} bytes)"
            )
        else:
            lines.append(f"- `{item['path']}`: MISSING")
    lines.extend(["", "## Remaining Blockers", ""])
    blockers = refresh.get("remaining_blockers") or []
    lines.extend(f"- {blocker}" for blocker in blockers) if blockers else lines.append("- None")
    lines.extend(["", "## Failures", ""])
    failures = refresh.get("failures") or []
    lines.extend(f"- `{failure}`" for failure in failures) if failures else lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = refresh.get("warnings") or []
    lines.extend(f"- `{warning}`" for warning in warnings) if warnings else lines.append("- None")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {action}" for action in refresh.get("next_actions", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stamp", default=DEFAULT_STAMP)
    parser.add_argument("--paper-dir", default="Paper")
    parser.add_argument("--external-config-path", default=str(DEFAULT_EXTERNAL_CONFIG))
    parser.add_argument("--external-network-mode", choices=["live", "disabled", "fixture"], default="live")
    parser.add_argument("--external-fixture-json")
    parser.add_argument("--external-timeout-seconds", type=int, default=20)
    parser.add_argument("--claim-audit-json", default=str(DEFAULT_CLAIM_AUDIT_JSON))
    parser.add_argument("--panel-review-json", default=str(DEFAULT_PANEL_REVIEW_JSON))
    parser.add_argument("--metadata-followup-json", default=str(DEFAULT_METADATA_FOLLOWUP_JSON))
    parser.add_argument("--target-profile-json", default=str(DEFAULT_TARGET_PROFILE_JSON))
    parser.add_argument("--target-profile-id")
    parser.add_argument("--metadata-config", default=str(DEFAULT_METADATA_CONFIG))
    parser.add_argument("--manual-config", default=str(DEFAULT_MANUAL_CONFIG))
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refresh = refresh_pre_submission_gates(
        root=args.root,
        output_dir=args.output_dir,
        stamp=args.stamp,
        paper_dir=args.paper_dir,
        external_config_path=args.external_config_path,
        external_network_mode=args.external_network_mode,
        external_fixture_json=args.external_fixture_json,
        external_timeout_seconds=args.external_timeout_seconds,
        claim_audit_json=args.claim_audit_json,
        panel_review_json=args.panel_review_json,
        metadata_followup_json=args.metadata_followup_json,
        target_profile_json=args.target_profile_json,
        target_profile_id=args.target_profile_id,
        metadata_config=args.metadata_config,
        manual_config=args.manual_config,
    )
    if args.output_json:
        output = Path(args.output_json)
        _write_json(output, refresh)
    if args.output_md:
        _write_md(Path(args.output_md), refresh)
    if not args.output_json:
        print(json.dumps(refresh, indent=2, sort_keys=True))
    return 0 if refresh["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
