from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REFRESH_JSON = Path("outputs/summary/paper_critical/pre_submission_gate_refresh_20260612.json")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _state_for_record(root: Path, path_text: str) -> dict[str, Any]:
    path = root / Path(path_text.replace("\\", "/"))
    if not path.exists():
        return {
            "path": path_text,
            "exists": False,
            "size_bytes": 0,
            "sha256": "",
            "is_file": False,
        }
    if not path.is_file():
        return {
            "path": _repo_relative(path, root),
            "exists": True,
            "size_bytes": 0,
            "sha256": "",
            "is_file": False,
        }
    return {
        "path": _repo_relative(path, root),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "is_file": True,
    }


def _run_git(root: Path, args: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        return 127, "", str(exc)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


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


def _compare_record(root: Path, expected: dict[str, Any], *, record_type: str, owner: str) -> dict[str, Any]:
    path_text = str(expected.get("path") or "")
    actual = _state_for_record(root, path_text)
    expected_exists = bool(expected.get("exists"))
    mismatches: list[str] = []
    if actual["exists"] != expected_exists:
        mismatches.append(f"exists:{actual['exists']} != {expected_exists}")
    if actual["exists"] and not actual.get("is_file"):
        mismatches.append("not_a_regular_file")
    if expected_exists and actual["exists"]:
        if int(actual["size_bytes"]) != int(expected.get("size_bytes") or 0):
            mismatches.append(f"size_bytes:{actual['size_bytes']} != {expected.get('size_bytes')}")
        if str(actual["sha256"]) != str(expected.get("sha256") or ""):
            mismatches.append("sha256_mismatch")
    return {
        "record_type": record_type,
        "owner": owner,
        "path": path_text,
        "matches": not mismatches,
        "mismatches": mismatches,
        "expected": {
            "exists": expected_exists,
            "size_bytes": int(expected.get("size_bytes") or 0),
            "sha256": str(expected.get("sha256") or ""),
        },
        "actual": actual,
    }


def _step_file_records(refresh: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for step in refresh.get("steps") or []:
        step_id = str(step.get("step_id") or "unknown_step")
        for field in ("json", "md"):
            file_record = step.get(field)
            if isinstance(file_record, dict):
                records.append({"record": file_record, "step_id": step_id, "field": field})
    return records


def build_pre_submission_refresh_freshness_audit(
    *,
    root: str | Path = ".",
    refresh_json: str | Path = DEFAULT_REFRESH_JSON,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    refresh_path = repo / refresh_json
    failures: list[str] = []
    warnings: list[str] = []

    if not refresh_path.exists():
        return {
            "schema_version": "2026-06-12.pre_submission_refresh_freshness_audit.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "local_read_only_pre_submission_refresh_freshness_audit",
            "aris_skill": "aris-citation-audit",
            "read_only": True,
            "will_ssh": False,
            "will_copy": False,
            "will_delete": False,
            "will_start_experiment": False,
            "ok": False,
            "refresh_artifact_fresh": False,
            "final_submission_ready": False,
            "refresh_json": str(Path(refresh_json)),
            "failures": [f"refresh_json_missing:{refresh_json}"],
            "warnings": [],
            "remaining_blockers": [],
            "next_actions": ["Run main_refresh_pre_submission_gates before auditing freshness."],
        }

    refresh = _read_json(refresh_path)
    input_checks = [
        _compare_record(repo, item, record_type="input_fingerprint", owner="refresh")
        for item in refresh.get("input_fingerprints") or []
    ]
    step_checks = [
        _compare_record(
            repo,
            item["record"],
            record_type=f"step_{item['field']}",
            owner=item["step_id"],
        )
        for item in _step_file_records(refresh)
    ]

    input_mismatches = [item for item in input_checks if not item["matches"]]
    step_mismatches = [item for item in step_checks if not item["matches"]]
    for item in input_mismatches:
        failures.append(f"input_fingerprint_mismatch:{item['path']}")
    for item in step_mismatches:
        failures.append(f"generated_step_file_mismatch:{item['owner']}:{item['path']}")

    refresh_git_state = refresh.get("git_state_before_refresh") or {}
    if refresh_git_state.get("tracked_dirty") is True:
        warnings.append("refresh_recorded_tracked_dirty_inputs_before_generation")
    if not refresh_git_state:
        warnings.append("refresh_missing_git_state_before_refresh")

    current_git_state = _git_state(repo)
    git_head_changed = (
        bool(refresh_git_state.get("head"))
        and bool(current_git_state.get("head"))
        and refresh_git_state.get("head") != current_git_state.get("head")
    )

    refresh_artifact_fresh = not failures
    final_ready_from_refresh = refresh.get("final_submission_ready") is True
    remaining_blockers = list(refresh.get("remaining_blockers") or [])
    if final_ready_from_refresh:
        warnings.append("refresh_claims_final_submission_ready_true_verify_external_manual_gates")

    return {
        "schema_version": "2026-06-12.pre_submission_refresh_freshness_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_read_only_pre_submission_refresh_freshness_audit",
        "aris_skill": "aris-citation-audit",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "ok": refresh_artifact_fresh,
        "refresh_artifact_fresh": refresh_artifact_fresh,
        "final_submission_ready": False,
        "refresh_json": str(Path(refresh_json)),
        "refresh_schema_version": refresh.get("schema_version"),
        "refresh_created_at_utc": refresh.get("created_at_utc"),
        "refresh_ok": refresh.get("ok") is True,
        "refresh_final_submission_ready": final_ready_from_refresh,
        "refresh_final_verdict": refresh.get("final_verdict") or "",
        "refresh_git_state_before_refresh": refresh_git_state,
        "current_git_state": current_git_state,
        "git_head_changed_since_refresh_generation": git_head_changed,
        "git_state_policy": (
            "Git HEAD is provenance for the code/input state that generated the refresh. "
            "Freshness is decided by current file fingerprints and generated gate hashes, "
            "because committing generated artifacts necessarily changes HEAD."
        ),
        "checked_input_fingerprint_count": len(input_checks),
        "checked_step_file_count": len(step_checks),
        "input_fingerprint_mismatch_count": len(input_mismatches),
        "generated_step_file_mismatch_count": len(step_mismatches),
        "input_fingerprint_checks": input_checks,
        "generated_step_file_checks": step_checks,
        "failures": failures,
        "warnings": warnings,
        "remaining_blockers": remaining_blockers,
        "next_actions": [
            "If any fingerprint mismatch is present, rerun main_refresh_pre_submission_gates.",
            "Use final_submission_gate as the semantic readiness summary after freshness passes.",
            "Keep final_submission_ready=false until external proceedings metadata and private manual submission-system checks are closed.",
        ],
    }


def _write_md(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Pre-Submission Refresh Freshness Audit",
        "",
        f"Generated: {audit['created_at_utc']}",
        "",
        f"- OK: `{str(audit['ok']).lower()}`",
        f"- Refresh artifact fresh: `{str(audit['refresh_artifact_fresh']).lower()}`",
        f"- Final submission ready: `{str(audit['final_submission_ready']).lower()}`",
        f"- Refresh JSON: `{audit['refresh_json']}`",
        f"- Refresh verdict: `{audit.get('refresh_final_verdict', '')}`",
        f"- Input fingerprints checked: `{audit.get('checked_input_fingerprint_count', 0)}`",
        f"- Generated step files checked: `{audit.get('checked_step_file_count', 0)}`",
        f"- Input mismatches: `{audit.get('input_fingerprint_mismatch_count', 0)}`",
        f"- Generated output mismatches: `{audit.get('generated_step_file_mismatch_count', 0)}`",
        "",
        "## Git Provenance Policy",
        "",
        audit.get("git_state_policy", ""),
        "",
        f"- Refresh generation HEAD: `{(audit.get('refresh_git_state_before_refresh') or {}).get('head', '')}`",
        f"- Current HEAD: `{(audit.get('current_git_state') or {}).get('head', '')}`",
        f"- HEAD changed since refresh generation: `{str(audit.get('git_head_changed_since_refresh_generation')).lower()}`",
        "",
        "## Failures",
        "",
    ]
    failures = audit.get("failures") or []
    lines.extend(f"- `{item}`" for item in failures) if failures else lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = audit.get("warnings") or []
    lines.extend(f"- `{item}`" for item in warnings) if warnings else lines.append("- None")
    mismatches = [
        item
        for item in (audit.get("input_fingerprint_checks") or [])
        + (audit.get("generated_step_file_checks") or [])
        if not item.get("matches")
    ]
    lines.extend(["", "## Mismatched Files", ""])
    if mismatches:
        for item in mismatches:
            expected = item.get("expected") or {}
            actual = item.get("actual") or {}
            lines.extend(
                [
                    f"### `{item.get('path', '')}`",
                    "",
                    f"- Record type: `{item.get('record_type', '')}`",
                    f"- Owner: `{item.get('owner', '')}`",
                    f"- Mismatches: `{', '.join(item.get('mismatches') or [])}`",
                    f"- Expected exists/size/sha256: `{expected.get('exists')}` / `{expected.get('size_bytes')}` / `{expected.get('sha256')}`",
                    f"- Actual exists/size/sha256: `{actual.get('exists')}` / `{actual.get('size_bytes')}` / `{actual.get('sha256')}`",
                    "",
                ]
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Remaining Blockers", ""])
    blockers = audit.get("remaining_blockers") or []
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- None")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in audit.get("next_actions", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--refresh-json", default=str(DEFAULT_REFRESH_JSON))
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_pre_submission_refresh_freshness_audit(
        root=args.root,
        refresh_json=args.refresh_json,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        _write_md(Path(args.output_md), audit)
    if not args.output_json:
        print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
