from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.audit.main_build_manual_submission_private_confirmation_request_packet import (
    FORBIDDEN_PRIVATE_KEYS,
)


DEFAULT_OUTPUT_DIR = Path("outputs/summary/paper_critical")
DEFAULT_STAMP = "20260613"
DEFAULT_CONFIRMATION_JSON = Path("artifacts/private/manual_submission_private_confirmation_20260613.json")
DEFAULT_REQUEST_PACKET_JSON = Path(
    "outputs/summary/paper_critical/manual_submission_private_confirmation_request_packet_20260613.json"
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
        return "<outside_repo>"


def _path_state(path: Path, root: Path) -> dict[str, Any]:
    state = {
        "path": _repo_relative(path, root),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        "sha256": "",
    }
    if path.exists() and path.is_file():
        state["sha256"] = _sha256_file(path)
    return state


def _read_json(path: Path) -> tuple[dict[str, Any], list[str]]:
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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _is_under_artifacts_private(path: Path, root: Path) -> bool:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return True
    parts = [part.lower() for part in rel.parts]
    return len(parts) >= 2 and parts[0] == "artifacts" and parts[1] == "private"


def build_manual_private_confirmation_validation_packet(
    *,
    root: str | Path = ".",
    confirmation_json: str | Path = DEFAULT_CONFIRMATION_JSON,
    request_packet_json: str | Path = DEFAULT_REQUEST_PACKET_JSON,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    confirmation_path = (repo / confirmation_json).resolve()
    request_path = (repo / request_packet_json).resolve()

    request, request_errors = _read_json(request_path)
    confirmation, confirmation_errors = _read_json(confirmation_path)

    failures: list[str] = []
    failures.extend(f"request_packet:{error}" for error in request_errors)
    failures.extend(f"private_confirmation:{error}" for error in confirmation_errors)

    required = request.get("required_private_confirmation") or {}
    expected_schema = str(required.get("schema_version") or "")
    expected_target = str(request.get("target_profile_id") or "")
    expected_checklist = str(request.get("checklist_id") or "")
    expected_manifest = str(required.get("source_manifest_sha256") or "")
    full_item_ids = set(_string_list(required.get("completed_item_ids_for_full_manual_gate")))
    blocked_item_ids = set(_string_list(required.get("currently_blocked_item_ids")))
    current_required_item_ids = sorted(full_item_ids - blocked_item_ids)

    if request and request.get("request_packet_ready") is not True:
        failures.append("request_packet_not_ready")
    if request and request.get("final_submission_ready") is not False:
        failures.append("request_packet_unexpected_final_submission_ready")
    if not _is_under_artifacts_private(confirmation_path, repo):
        failures.append("private_confirmation_path_not_under_artifacts_private")

    present_forbidden = sorted(key for key in FORBIDDEN_PRIVATE_KEYS if key in confirmation)
    if present_forbidden:
        failures.append("private_confirmation_contains_forbidden_private_fields:" + ",".join(present_forbidden))

    if confirmation:
        if confirmation.get("schema_version") != expected_schema:
            failures.append(
                "private_confirmation_schema_mismatch:"
                f"{confirmation.get('schema_version')} != {expected_schema}"
            )
        if confirmation.get("target_profile_id") != expected_target:
            failures.append(
                "private_confirmation_target_profile_mismatch:"
                f"{confirmation.get('target_profile_id')} != {expected_target}"
            )
        if confirmation.get("checklist_id") != expected_checklist:
            failures.append(
                "private_confirmation_checklist_id_mismatch:"
                f"{confirmation.get('checklist_id')} != {expected_checklist}"
            )
        if confirmation.get("source_manifest_sha256") != expected_manifest:
            failures.append("private_confirmation_source_manifest_sha256_mismatch")
        for field in [
            "confirmed_in_submission_system",
            "private_fields_completed_in_submission_system",
            "no_private_fields_stored",
        ]:
            if confirmation.get(field) is not True:
                failures.append(f"private_confirmation_{field}_not_true")

    completed_raw = confirmation.get("completed_item_ids") if confirmation else None
    if completed_raw is None:
        completed_item_ids: list[str] = []
    elif not isinstance(completed_raw, list) or not all(isinstance(item, str) for item in completed_raw):
        failures.append("private_confirmation_completed_item_ids_invalid")
        completed_item_ids = []
    else:
        completed_item_ids = [item.strip() for item in completed_raw if item.strip()]

    duplicate_ids = sorted({item for item in completed_item_ids if completed_item_ids.count(item) > 1})
    if duplicate_ids:
        failures.append("private_confirmation_duplicate_item_ids:" + ",".join(duplicate_ids))
    completed_set = set(completed_item_ids)
    unknown_ids = sorted(completed_set - full_item_ids)
    missing_current_ids = sorted(set(current_required_item_ids) - completed_set)
    completed_blocked_ids = sorted(completed_set & blocked_item_ids)
    missing_full_ids = sorted(full_item_ids - completed_set)

    if unknown_ids:
        failures.append("private_confirmation_unknown_item_ids:" + ",".join(unknown_ids))
    if missing_current_ids:
        failures.append("private_confirmation_missing_current_required_item_ids:" + ",".join(missing_current_ids))
    if completed_blocked_ids:
        failures.append("private_confirmation_completed_currently_blocked_item_ids:" + ",".join(completed_blocked_ids))

    ready_to_use = not failures
    full_manual_gate_ready_shape = ready_to_use and not blocked_item_ids and not missing_full_ids

    warnings: list[str] = []
    if confirmation and confirmation.get("notes"):
        warnings.append("private_confirmation_notes_present_not_copied_to_public_status")

    return {
        "schema_version": "2026-06-13.manual_private_confirmation_validation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_read_only_manual_private_confirmation_validation",
        "project": "uncertainty",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "ok": ready_to_use,
        "ready_to_use_for_manual_checklist": ready_to_use,
        "full_manual_gate_ready_shape": full_manual_gate_ready_shape,
        "final_submission_ready": False,
        "input_paths": {
            "private_confirmation_json": _path_state(confirmation_path, repo),
            "manual_request_packet_json": _path_state(request_path, repo),
        },
        "expected": {
            "schema_version": expected_schema,
            "target_profile_id": expected_target,
            "checklist_id": expected_checklist,
            "source_manifest_sha256": expected_manifest,
            "current_required_item_ids": current_required_item_ids,
            "currently_blocked_item_ids": sorted(blocked_item_ids),
            "full_manual_gate_item_ids": sorted(full_item_ids),
        },
        "observed": {
            "schema_version": str(confirmation.get("schema_version") or ""),
            "target_profile_id_matches": confirmation.get("target_profile_id") == expected_target if confirmation else False,
            "checklist_id_matches": confirmation.get("checklist_id") == expected_checklist if confirmation else False,
            "source_manifest_sha256_matches": (
                confirmation.get("source_manifest_sha256") == expected_manifest if confirmation else False
            ),
            "completed_item_ids": sorted(completed_set),
            "unknown_item_ids": unknown_ids,
            "missing_current_required_item_ids": missing_current_ids,
            "completed_currently_blocked_item_ids": completed_blocked_ids,
            "missing_full_manual_gate_item_ids": missing_full_ids,
            "forbidden_private_keys_present": present_forbidden,
        },
        "privacy_policy": [
            "This validator reports only booleans, item IDs, key names, hashes, and path states.",
            "It must not print author names, conflicts, reviewer preferences, declarations, or account data.",
            "A confirmation under the repository must live under artifacts/private/.",
        ],
        "warnings": warnings,
        "failures": failures,
        "next_actions": [
            "If ok=true, rerun main_build_manual_submission_checklist with --private-confirmation-json.",
            "If completed_currently_blocked_item_ids is non-empty, refresh public gates first; do not confirm blocked items early.",
            "Keep final_submission_ready=false until manual, ProMax metadata, and explicit Claude Opus review gates all close.",
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: Path, packet: dict[str, Any]) -> None:
    lines = [
        "# Manual Private Confirmation Validation",
        "",
        f"- Created UTC: `{packet['created_at_utc']}`",
        f"- OK: `{str(packet['ok']).lower()}`",
        f"- Ready to use for manual checklist: `{str(packet['ready_to_use_for_manual_checklist']).lower()}`",
        f"- Full manual gate ready shape: `{str(packet['full_manual_gate_ready_shape']).lower()}`",
        f"- Final submission ready: `{str(packet['final_submission_ready']).lower()}`",
        "",
        "## Expected Item State",
        "",
    ]
    for key in ["current_required_item_ids", "currently_blocked_item_ids", "full_manual_gate_item_ids"]:
        values = packet["expected"].get(key) or []
        lines.append(f"- {key}: `{', '.join(values) or 'none'}`")
    lines.extend(["", "## Observed Item State", ""])
    observed = packet["observed"]
    for key in [
        "unknown_item_ids",
        "missing_current_required_item_ids",
        "completed_currently_blocked_item_ids",
        "missing_full_manual_gate_item_ids",
        "forbidden_private_keys_present",
    ]:
        values = observed.get(key) or []
        lines.append(f"- {key}: `{', '.join(values) or 'none'}`")
    lines.extend(["", "## Failures", ""])
    failures = packet.get("failures") or []
    lines.extend(f"- `{failure}`" for failure in failures) if failures else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    warnings = packet.get("warnings") or []
    lines.extend(f"- `{warning}`" for warning in warnings) if warnings else lines.append("- none")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in packet.get("next_actions") or [])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--stamp", default=DEFAULT_STAMP)
    parser.add_argument("--private-confirmation-json", default=str(DEFAULT_CONFIRMATION_JSON))
    parser.add_argument("--manual-request-packet-json", default=str(DEFAULT_REQUEST_PACKET_JSON))
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / DEFAULT_OUTPUT_DIR
    output_json = (
        Path(args.output_json)
        if args.output_json
        else out_dir / f"manual_private_confirmation_validation_{args.stamp}.json"
    )
    output_md = (
        Path(args.output_md)
        if args.output_md
        else out_dir / f"manual_private_confirmation_validation_{args.stamp}.md"
    )
    packet = build_manual_private_confirmation_validation_packet(
        root=root,
        confirmation_json=args.private_confirmation_json,
        request_packet_json=args.manual_request_packet_json,
    )
    _write_json(output_json, packet)
    _write_md(output_md, packet)
    print(json.dumps({"ok": packet["ok"], "output_json": str(output_json), "output_md": str(output_md)}, indent=2))
    return 0 if packet["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
