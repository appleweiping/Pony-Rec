from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_FINAL_SUBMISSION_GATE = Path(
    "outputs/summary/paper_critical/final_submission_gate_20260612.json"
)
DEFAULT_REFRESH_FRESHNESS = Path(
    "outputs/summary/paper_critical/pre_submission_gate_refresh_freshness_20260612.json"
)
DEFAULT_SUBMISSION_SOURCE_PACKAGE = Path(
    "outputs/summary/paper_critical/submission_source_package_20260612.json"
)
DEFAULT_SUBMISSION_SOURCE_PACKAGE_REBUILD = Path(
    "outputs/summary/paper_critical/submission_source_package_rebuild_20260612.json"
)
DEFAULT_SUBMISSION_METADATA_PACKET = Path(
    "outputs/summary/paper_critical/submission_metadata_packet_20260612.json"
)
DEFAULT_MANUAL_CHECKLIST = Path(
    "outputs/summary/paper_critical/manual_submission_checklist_20260612.json"
)
DEFAULT_EXTERNAL_METADATA_AUDIT = Path(
    "outputs/summary/paper_critical/external_proceedings_metadata_recheck_20260612.json"
)


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


def _path_state(path: Path, root: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": _repo_relative(path, root),
            "exists": False,
            "is_file": False,
            "size_bytes": 0,
            "sha256": "",
        }
    if not path.is_file():
        return {
            "path": _repo_relative(path, root),
            "exists": True,
            "is_file": False,
            "size_bytes": 0,
            "sha256": "",
        }
    return {
        "path": _repo_relative(path, root),
        "exists": True,
        "is_file": True,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _read_json_object(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.exists():
        return {}, [f"missing_json:{path}"]
    if not path.is_file():
        return {}, [f"not_a_file:{path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, [f"invalid_json:{path}:{exc.msg}"]
    if not isinstance(payload, dict):
        return {}, [f"json_not_object:{path}"]
    return payload, []


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _stamp_from_path(path_text: str) -> str:
    matches = re.findall(r"20\d{6}", path_text)
    return matches[-1] if matches else ""


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


def _gate_record(
    *,
    gate_id: str,
    path: Path,
    root: Path,
    payload: dict[str, Any],
    load_failures: list[str],
    ready_field: str | None = None,
) -> dict[str, Any]:
    ready = payload.get(ready_field) is True if ready_field else None
    return {
        "gate_id": gate_id,
        "path": _path_state(path, root),
        "ok": payload.get("ok") is True,
        "ready_field": ready_field or "",
        "ready": ready,
        "final_submission_ready": payload.get("final_submission_ready") is True,
        "schema_version": payload.get("schema_version"),
        "created_at_utc": payload.get("created_at_utc"),
        "failures": list(payload.get("failures") or []) + load_failures,
        "warnings": list(payload.get("warnings") or []),
        "remaining_blockers": list(payload.get("remaining_blockers") or []),
    }


def _collect_manifest_sha_values(
    *,
    source_package: dict[str, Any],
    source_rebuild: dict[str, Any],
    metadata_packet: dict[str, Any],
) -> dict[str, str]:
    source_manifest = source_package.get("copied_manifest") or {}
    source_crosscheck = source_package.get("source_audit_crosscheck") or {}
    rebuild_crosscheck = source_rebuild.get("source_package_crosscheck") or {}
    metadata_crosscheck = metadata_packet.get("package_crosscheck") or {}
    return {
        "source_package_copied_manifest": str(source_manifest.get("manifest_sha256") or ""),
        "source_package_source_manifest": str(source_crosscheck.get("source_manifest_sha256") or ""),
        "source_rebuild_copied_manifest": str(rebuild_crosscheck.get("copied_manifest_sha256") or ""),
        "source_rebuild_source_manifest": str(rebuild_crosscheck.get("source_manifest_sha256") or ""),
        "metadata_packet_source_manifest": str(metadata_crosscheck.get("source_manifest_sha256") or ""),
    }


def _manifest_sha_check(values: dict[str, str]) -> dict[str, Any]:
    nonempty = {key: value for key, value in values.items() if value}
    missing = [key for key, value in values.items() if not value]
    distinct = sorted(set(nonempty.values()))
    valid_hex = all(len(value) == 64 and all(char in "0123456789abcdef" for char in value) for value in nonempty.values())
    return {
        "values": values,
        "missing_fields": missing,
        "distinct_sha256": distinct,
        "all_present": not missing,
        "all_64_hex": valid_hex and bool(nonempty),
        "all_match": len(distinct) == 1 and not missing and valid_hex,
        "canonical_manifest_sha256": distinct[0] if len(distinct) == 1 else "",
    }


def _command_returncode_check(source_rebuild: dict[str, Any]) -> dict[str, Any]:
    commands = source_rebuild.get("commands")
    if not isinstance(commands, list):
        return {
            "command_count": 0,
            "all_returncode_zero": False,
            "nonzero_commands": ["commands_not_list"],
        }
    nonzero: list[str] = []
    for index, command in enumerate(commands):
        if not isinstance(command, dict):
            nonzero.append(f"command_{index}:not_object")
            continue
        if command.get("returncode") != 0:
            nonzero.append(f"command_{index}:returncode={command.get('returncode')}")
    return {
        "command_count": len(commands),
        "all_returncode_zero": bool(commands) and not nonzero,
        "nonzero_commands": nonzero,
    }


def _step_summary(freshness: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in freshness.get("generated_step_file_checks") or []:
        if not isinstance(item, dict):
            continue
        records.append(
            {
                "owner": item.get("owner"),
                "path": item.get("path"),
                "matches": item.get("matches") is True,
                "record_type": item.get("record_type"),
                "mismatches": list(item.get("mismatches") or []),
            }
        )
    return records


def build_submission_release_candidate_packet(
    *,
    root: str | Path = ".",
    final_submission_gate_json: str | Path = DEFAULT_FINAL_SUBMISSION_GATE,
    refresh_freshness_json: str | Path = DEFAULT_REFRESH_FRESHNESS,
    submission_source_package_json: str | Path = DEFAULT_SUBMISSION_SOURCE_PACKAGE,
    submission_source_package_rebuild_json: str | Path = DEFAULT_SUBMISSION_SOURCE_PACKAGE_REBUILD,
    submission_metadata_packet_json: str | Path = DEFAULT_SUBMISSION_METADATA_PACKET,
    manual_checklist_json: str | Path = DEFAULT_MANUAL_CHECKLIST,
    external_metadata_audit_json: str | Path = DEFAULT_EXTERNAL_METADATA_AUDIT,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    paths = {
        "final_submission_gate": repo / final_submission_gate_json,
        "pre_submission_refresh_freshness": repo / refresh_freshness_json,
        "submission_source_package": repo / submission_source_package_json,
        "submission_source_package_rebuild": repo / submission_source_package_rebuild_json,
        "submission_metadata_packet": repo / submission_metadata_packet_json,
        "manual_submission_checklist": repo / manual_checklist_json,
        "external_proceedings_metadata": repo / external_metadata_audit_json,
    }

    payloads: dict[str, dict[str, Any]] = {}
    load_failures_by_gate: dict[str, list[str]] = {}
    for gate_id, path in paths.items():
        payload, load_failures = _read_json_object(path)
        payloads[gate_id] = payload
        load_failures_by_gate[gate_id] = load_failures

    gates = [
        _gate_record(
            gate_id="final_submission_gate",
            path=paths["final_submission_gate"],
            root=repo,
            payload=payloads["final_submission_gate"],
            load_failures=load_failures_by_gate["final_submission_gate"],
            ready_field="all_local_artifact_gates_ok",
        ),
        _gate_record(
            gate_id="pre_submission_refresh_freshness",
            path=paths["pre_submission_refresh_freshness"],
            root=repo,
            payload=payloads["pre_submission_refresh_freshness"],
            load_failures=load_failures_by_gate["pre_submission_refresh_freshness"],
            ready_field="refresh_artifact_fresh",
        ),
        _gate_record(
            gate_id="submission_source_package",
            path=paths["submission_source_package"],
            root=repo,
            payload=payloads["submission_source_package"],
            load_failures=load_failures_by_gate["submission_source_package"],
            ready_field="submission_source_package_ready",
        ),
        _gate_record(
            gate_id="submission_source_package_rebuild",
            path=paths["submission_source_package_rebuild"],
            root=repo,
            payload=payloads["submission_source_package_rebuild"],
            load_failures=load_failures_by_gate["submission_source_package_rebuild"],
            ready_field="submission_source_package_rebuild_ready",
        ),
        _gate_record(
            gate_id="submission_metadata_packet",
            path=paths["submission_metadata_packet"],
            root=repo,
            payload=payloads["submission_metadata_packet"],
            load_failures=load_failures_by_gate["submission_metadata_packet"],
            ready_field="submission_metadata_packet_ready",
        ),
        _gate_record(
            gate_id="manual_submission_checklist",
            path=paths["manual_submission_checklist"],
            root=repo,
            payload=payloads["manual_submission_checklist"],
            load_failures=load_failures_by_gate["manual_submission_checklist"],
            ready_field="manual_submission_checklist_ready",
        ),
        _gate_record(
            gate_id="external_proceedings_metadata",
            path=paths["external_proceedings_metadata"],
            root=repo,
            payload=payloads["external_proceedings_metadata"],
            load_failures=load_failures_by_gate["external_proceedings_metadata"],
            ready_field=None,
        ),
    ]

    final_gate = payloads["final_submission_gate"]
    freshness = payloads["pre_submission_refresh_freshness"]
    source_package = payloads["submission_source_package"]
    source_rebuild = payloads["submission_source_package_rebuild"]
    metadata_packet = payloads["submission_metadata_packet"]
    manual_checklist = payloads["manual_submission_checklist"]
    external_metadata = payloads["external_proceedings_metadata"]

    failures: list[str] = []
    warnings: list[str] = []
    remaining_blockers: list[str] = []
    for gate in gates:
        if not gate["path"]["exists"]:
            failures.append(f"{gate['gate_id']}:missing_json")
        if not gate["ok"]:
            failures.append(f"{gate['gate_id']}:not_ok")
        if gate["gate_id"] != "final_submission_gate" and gate["final_submission_ready"]:
            failures.append(f"{gate['gate_id']}:unexpected_local_final_submission_ready")
        failures.extend(f"{gate['gate_id']}:{item}" for item in gate["failures"])
        warnings.extend(f"{gate['gate_id']}:{item}" for item in gate["warnings"])
        remaining_blockers.extend(gate["remaining_blockers"])

    required_ready_fields = [
        ("final_submission_gate", "all_local_artifact_gates_ok"),
        ("pre_submission_refresh_freshness", "refresh_artifact_fresh"),
        ("submission_source_package", "submission_source_package_ready"),
        ("submission_source_package_rebuild", "submission_source_package_rebuild_ready"),
        ("submission_metadata_packet", "submission_metadata_packet_ready"),
        ("manual_submission_checklist", "manual_submission_checklist_ready"),
    ]
    for gate_id, ready_field in required_ready_fields:
        if payloads[gate_id].get(ready_field) is not True:
            failures.append(f"{gate_id}:{ready_field}_not_true")

    if final_gate.get("all_local_artifact_gates_ok") is not True:
        failures.append("final_submission_gate:all_local_artifact_gates_ok_not_true")
    if freshness.get("ok") is not True or freshness.get("refresh_artifact_fresh") is not True:
        failures.append("pre_submission_refresh_freshness:not_fresh")
    if int(freshness.get("input_fingerprint_mismatch_count") or 0) != 0:
        failures.append("pre_submission_refresh_freshness:input_fingerprint_mismatches_present")
    if int(freshness.get("generated_step_file_mismatch_count") or 0) != 0:
        failures.append("pre_submission_refresh_freshness:generated_step_file_mismatches_present")

    command_check = _command_returncode_check(source_rebuild)
    if not command_check["all_returncode_zero"]:
        failures.append("submission_source_package_rebuild:build_commands_not_all_zero")
        failures.extend(f"submission_source_package_rebuild:{item}" for item in command_check["nonzero_commands"])

    manifest_check = _manifest_sha_check(
        _collect_manifest_sha_values(
            source_package=source_package,
            source_rebuild=source_rebuild,
            metadata_packet=metadata_packet,
        )
    )
    if not manifest_check["all_match"]:
        failures.append("source_manifest_sha256_values_do_not_all_match")

    input_stamps = {
        key: _stamp_from_path(str(path.relative_to(repo)) if path.is_absolute() else str(path))
        for key, path in paths.items()
    }
    distinct_stamps = sorted({stamp for stamp in input_stamps.values() if stamp})
    stamp_check = {
        "input_stamps": input_stamps,
        "distinct_stamps": distinct_stamps,
        "all_dated_inputs_share_stamp": len(distinct_stamps) <= 1,
        "canonical_stamp": distinct_stamps[0] if len(distinct_stamps) == 1 else "",
    }
    if not stamp_check["all_dated_inputs_share_stamp"]:
        failures.append("input_gate_stamp_mismatch")

    final_submission_ready = final_gate.get("final_submission_ready") is True
    external_ready = external_metadata.get("external_proceedings_metadata_ready") is True
    manual_system_ready = manual_checklist.get("manual_submission_system_ready") is True
    manual_checklist_ready = manual_checklist.get("manual_submission_checklist_ready") is True
    local_release_candidate_ready = not _dedupe(failures)
    blocking_status = (
        "none"
        if final_submission_ready
        else "external_or_manual_blocked"
        if local_release_candidate_ready and (not external_ready or not manual_system_ready)
        else "local_artifact_repair_required"
    )

    if not external_ready:
        remaining_blockers.append("external_proceedings_metadata_not_ready")
    if not manual_system_ready:
        remaining_blockers.append("manual_submission_system_not_ready")

    verdict = (
        "FINAL_SUBMISSION_READY_FROM_FINAL_GATE"
        if local_release_candidate_ready and final_submission_ready
        else "LOCAL_RELEASE_CANDIDATE_READY_FINAL_BLOCKED"
        if local_release_candidate_ready
        else "LOCAL_RELEASE_CANDIDATE_NEEDS_REPAIR"
    )

    package_manifest = source_package.get("copied_manifest") or {}
    rebuild_build = source_rebuild.get("build") or {}
    metadata_fields = metadata_packet.get("submission_fields") or {}

    return {
        "schema_version": "2026-06-12.submission_release_candidate_packet.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_read_only_submission_release_candidate_packet",
        "aris_skill": "aris-paper-write",
        "readiness_scope": "local_artifacts_only",
        "local_only": True,
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "ok": local_release_candidate_ready,
        "local_release_candidate_ready": local_release_candidate_ready,
        "final_submission_ready": final_submission_ready,
        "external_proceedings_metadata_ready": external_ready,
        "manual_submission_checklist_ready": manual_checklist_ready,
        "manual_submission_system_ready": manual_system_ready,
        "blocking_status": blocking_status,
        "final_submission_ready_source": str(Path(final_submission_gate_json)),
        "final_submission_ready_policy": (
            "This field is copied exactly from final_submission_gate. "
            "It is not recomputed by the release-candidate packet."
        ),
        "verdict": verdict,
        "status_boundary": {
            "local_release_candidate_ready": (
                "Repo-side audited package, source rebuild, metadata packet, manual checklist, "
                "external audit health, and freshness index are internally consistent."
            ),
            "final_submission_ready": (
                "Submission may proceed only when the final submission gate also reports true; "
                "external proceedings metadata and private submission-system confirmation remain final blockers."
            ),
        },
        "git_state": _git_state(repo),
        "input_paths": {key: str(path.relative_to(repo)) if path.is_absolute() else str(path) for key, path in paths.items()},
        "input_file_states": {key: _path_state(path, repo) for key, path in paths.items()},
        "gates": gates,
        "freshness_summary": {
            "ok": freshness.get("ok") is True,
            "refresh_artifact_fresh": freshness.get("refresh_artifact_fresh") is True,
            "checked_input_fingerprint_count": int(freshness.get("checked_input_fingerprint_count") or 0),
            "checked_step_file_count": int(freshness.get("checked_step_file_count") or 0),
            "input_fingerprint_mismatch_count": int(freshness.get("input_fingerprint_mismatch_count") or 0),
            "generated_step_file_mismatch_count": int(freshness.get("generated_step_file_mismatch_count") or 0),
            "refresh_final_verdict": freshness.get("refresh_final_verdict") or "",
            "step_file_summary": _step_summary(freshness),
        },
        "source_package_summary": {
            "ready": source_package.get("submission_source_package_ready") is True,
            "file_count": int(package_manifest.get("file_count") or 0),
            "total_bytes": int(package_manifest.get("total_bytes") or 0),
            "manifest_sha256": package_manifest.get("manifest_sha256") or "",
            "files_dir": (source_package.get("output") or {}).get("files_dir", ""),
        },
        "source_rebuild_summary": {
            "ready": source_rebuild.get("submission_source_package_rebuild_ready") is True,
            "verified_file_count": int((source_rebuild.get("source_package_crosscheck") or {}).get("verified_file_count") or 0),
            "rebuilt_pdf_pages": int(rebuild_build.get("actual_pdf_page_count") or rebuild_build.get("page_count") or 0),
            "rebuilt_pdf_bytes": int((rebuild_build.get("pdf") or {}).get("size_bytes") or rebuild_build.get("logged_pdf_bytes") or 0),
            "bibtex_warning_count": int(rebuild_build.get("bibtex_warning_count") or 0),
            "overfull_hbox_count": int(rebuild_build.get("overfull_hbox_count") or 0),
            "command_check": command_check,
        },
        "metadata_summary": {
            "ready": metadata_packet.get("submission_metadata_packet_ready") is True,
            "title": metadata_fields.get("title") or "",
            "abstract_word_count": int(metadata_fields.get("abstract_word_count") or 0),
            "keyword_count": len(metadata_fields.get("keywords") or []),
            "topic_area_count": len(metadata_fields.get("topic_areas") or []),
            "pdf_pages": int((metadata_packet.get("package_crosscheck") or {}).get("pdf_pages") or 0),
        },
        "manual_summary": {
            "checklist_ready": manual_checklist.get("manual_submission_checklist_ready") is True,
            "submission_system_ready": manual_system_ready,
            "item_count": int(manual_checklist.get("item_count") or 0),
            "unconfirmed_required_item_ids": list(manual_checklist.get("unconfirmed_required_item_ids") or []),
            "private_confirmation_present": bool((manual_checklist.get("private_confirmation") or {}).get("exists")),
        },
        "external_metadata_summary": {
            "ok": external_metadata.get("ok") is True,
            "external_proceedings_metadata_ready": external_ready,
            "checked_entry_count": int(external_metadata.get("checked_entry_count") or 0),
            "network_mode": external_metadata.get("network_mode") or "",
        },
        "source_manifest_crosscheck": manifest_check,
        "input_stamp_check": stamp_check,
        "failures": _dedupe(failures),
        "warnings": _dedupe(warnings),
        "remaining_blockers": _dedupe(remaining_blockers),
        "next_actions": [
            "Use this packet as a local handoff index, not as final submission approval.",
            "Resolve ProMax final page range and DOI/Crossref visibility, then rerun the external metadata audit and refresh stack.",
            "Complete private submission-system fields with an untracked confirmation JSON, then rerun the manual checklist and refresh stack.",
            "Rerun the refresh, freshness audit, final gate, and release-candidate packet after any paper/source/BibTeX/package change.",
        ],
    }


def _write_md(path: Path, packet: dict[str, Any]) -> None:
    lines = [
        "# Submission Release-Candidate Packet",
        "",
        f"Generated: {packet['created_at_utc']}",
        "",
        f"- Verdict: `{packet['verdict']}`",
        f"- OK: `{str(packet['ok']).lower()}`",
        f"- Local release candidate ready: `{str(packet['local_release_candidate_ready']).lower()}`",
        f"- Final submission ready: `{str(packet['final_submission_ready']).lower()}`",
        f"- Final-ready source: `{packet['final_submission_ready_source']}`",
        "",
        "## Status Boundary",
        "",
        f"- Local RC: {packet['status_boundary']['local_release_candidate_ready']}",
        f"- Final submission: {packet['status_boundary']['final_submission_ready']}",
        "",
        "## Gate Summary",
        "",
    ]
    for gate in packet.get("gates", []):
        lines.append(
            f"- `{gate['gate_id']}`: ok=`{str(gate['ok']).lower()}`, "
            f"ready=`{str(gate['ready']).lower() if gate['ready'] is not None else 'n/a'}`, "
            f"final_ready=`{str(gate['final_submission_ready']).lower()}`, "
            f"path=`{gate['path']['path']}`"
        )
    lines.extend(
        [
            "",
            "## Package Summary",
            "",
            f"- Source files: `{packet['source_package_summary']['file_count']}`",
            f"- Source bytes: `{packet['source_package_summary']['total_bytes']}`",
            f"- Source manifest sha256: `{packet['source_package_summary']['manifest_sha256']}`",
            f"- Rebuild PDF pages: `{packet['source_rebuild_summary']['rebuilt_pdf_pages']}`",
            f"- Rebuild PDF bytes: `{packet['source_rebuild_summary']['rebuilt_pdf_bytes']}`",
            f"- BibTeX warnings: `{packet['source_rebuild_summary']['bibtex_warning_count']}`",
            f"- Overfull hbox warnings: `{packet['source_rebuild_summary']['overfull_hbox_count']}`",
            "",
            "## Remaining Blockers",
            "",
        ]
    )
    blockers = packet.get("remaining_blockers") or []
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- None")
    lines.extend(["", "## Failures", ""])
    failures = packet.get("failures") or []
    lines.extend(f"- `{item}`" for item in failures) if failures else lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = packet.get("warnings") or []
    lines.extend(f"- `{item}`" for item in warnings) if warnings else lines.append("- None")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in packet.get("next_actions", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--final-submission-gate-json", default=str(DEFAULT_FINAL_SUBMISSION_GATE))
    parser.add_argument("--refresh-freshness-json", default=str(DEFAULT_REFRESH_FRESHNESS))
    parser.add_argument("--submission-source-package-json", default=str(DEFAULT_SUBMISSION_SOURCE_PACKAGE))
    parser.add_argument(
        "--submission-source-package-rebuild-json",
        default=str(DEFAULT_SUBMISSION_SOURCE_PACKAGE_REBUILD),
    )
    parser.add_argument("--submission-metadata-packet-json", default=str(DEFAULT_SUBMISSION_METADATA_PACKET))
    parser.add_argument("--manual-checklist-json", default=str(DEFAULT_MANUAL_CHECKLIST))
    parser.add_argument("--external-metadata-audit-json", default=str(DEFAULT_EXTERNAL_METADATA_AUDIT))
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packet = build_submission_release_candidate_packet(
        root=args.root,
        final_submission_gate_json=args.final_submission_gate_json,
        refresh_freshness_json=args.refresh_freshness_json,
        submission_source_package_json=args.submission_source_package_json,
        submission_source_package_rebuild_json=args.submission_source_package_rebuild_json,
        submission_metadata_packet_json=args.submission_metadata_packet_json,
        manual_checklist_json=args.manual_checklist_json,
        external_metadata_audit_json=args.external_metadata_audit_json,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        _write_md(Path(args.output_md), packet)
    if not args.output_json:
        print(json.dumps(packet, indent=2, sort_keys=True))
    return 0 if packet["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
