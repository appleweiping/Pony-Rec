from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/paper_manual_submission_checklist.json")
DEFAULT_METADATA_PACKET = Path("outputs/summary/paper_critical/submission_metadata_packet_20260612.json")
DEFAULT_SUBMISSION_PACKAGE_AUDIT = Path(
    "outputs/summary/paper_critical/submission_package_audit_20260612.json"
)
DEFAULT_EXTERNAL_METADATA_AUDIT = Path(
    "outputs/summary/paper_critical/external_proceedings_metadata_recheck_20260612.json"
)


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


def _get_nested(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _prefill_value(metadata_packet: dict[str, Any], source: str | None) -> dict[str, Any]:
    if not source:
        return {"available": False, "source": "", "value": None, "summary": ""}
    value = _get_nested(metadata_packet, source)
    if isinstance(value, list):
        summary = ", ".join(str(item) for item in value)
    elif value is None:
        summary = ""
    else:
        summary = str(value)
    return {
        "available": value is not None and summary != "",
        "source": source,
        "value": value,
        "summary": summary,
    }


def _item_status(
    item: dict[str, Any],
    *,
    metadata_packet: dict[str, Any],
    submission_package_audit: dict[str, Any],
    external_metadata_audit: dict[str, Any],
    confirmed_item_ids: set[str] | None = None,
) -> dict[str, Any]:
    prefill = _prefill_value(metadata_packet, item.get("prefill_source"))
    blockers: list[str] = []
    warnings: list[str] = []
    confirmed_item_ids = confirmed_item_ids or set()

    if item.get("prefill_source") and not prefill["available"]:
        blockers.append(f"{item['id']}:missing_prefill:{item['prefill_source']}")

    external_gate = item.get("external_gate")
    if external_gate:
        if external_metadata_audit.get(external_gate) is not True:
            blockers.append(f"{item['id']}:{external_gate}_not_closed")

    if item["id"] == "upload_pdf":
        crosscheck = metadata_packet.get("package_crosscheck") or {}
        if not crosscheck.get("pdf_path") or not crosscheck.get("pdf_size_bytes"):
            blockers.append("upload_pdf:missing_pdf_crosscheck")
    if item["id"] == "upload_source_if_required":
        crosscheck = metadata_packet.get("package_crosscheck") or {}
        if not crosscheck.get("source_manifest_sha256"):
            blockers.append("upload_source_if_required:missing_source_manifest_sha256")
    if item["id"] == "confirm_anonymous_shell":
        evidence_gates = submission_package_audit.get("evidence_gates") or {}
        target_profile = submission_package_audit.get("target_formatting_profile") or {}
        if evidence_gates.get("target_formatting_profile_ok") is not True:
            blockers.append("confirm_anonymous_shell:target_formatting_profile_not_ok")
        if target_profile.get("ok") is not True:
            blockers.append("confirm_anonymous_shell:target_profile_not_ok")

    if blockers:
        status = "blocked"
    elif item["id"] in confirmed_item_ids:
        status = "manual_private_confirmed_not_stored" if item.get("private") else "manual_confirmed"
    elif item.get("private"):
        status = "manual_private_not_stored"
    elif item.get("requires_submission_system"):
        status = "manual_pending"
    else:
        status = "evidence_pending_manual_confirmation"

    return {
        "id": item["id"],
        "category": item.get("category"),
        "label": item.get("label"),
        "storage_policy": item.get("storage_policy"),
        "private": bool(item.get("private")),
        "requires_submission_system": bool(item.get("requires_submission_system")),
        "status": status,
        "prefill": prefill,
        "blockers": blockers,
        "warnings": warnings,
    }


def _private_confirmation_state(
    *,
    repo: Path,
    raw_path: str | Path | None,
    config: dict[str, Any],
    metadata_packet: dict[str, Any],
    item_ids: set[str],
) -> tuple[dict[str, Any], set[str], list[str], list[str]]:
    state: dict[str, Any] = {
        "provided": False,
        "path": "",
        "exists": False,
        "sha256": "",
        "size_bytes": 0,
        "schema_version": "",
        "confirmed_in_submission_system": False,
        "no_private_fields_stored": False,
        "completed_item_ids": [],
    }
    failures: list[str] = []
    warnings: list[str] = []
    if raw_path is None or str(raw_path).strip() == "":
        return state, set(), failures, warnings

    path = Path(raw_path)
    if not path.is_absolute():
        path = repo / path
    state["provided"] = True
    state["path"] = str(Path(raw_path))
    state["exists"] = path.exists()
    if not path.exists():
        failures.append(f"private_confirmation_json_missing:{raw_path}")
        return state, set(), failures, warnings
    if not path.is_file():
        failures.append(f"private_confirmation_json_not_file:{raw_path}")
        return state, set(), failures, warnings

    state["sha256"] = _sha256_file(path)
    state["size_bytes"] = path.stat().st_size
    try:
        confirmation = _read_json(path)
    except (json.JSONDecodeError, ValueError) as exc:
        failures.append(f"private_confirmation_json_invalid:{exc}")
        return state, set(), failures, warnings

    forbidden_keys = {
        "authors",
        "author_names",
        "affiliations",
        "conflicts",
        "conflicts_of_interest",
        "reviewer_suggestions",
        "reviewer_exclusions",
        "submission_account",
        "account_metadata",
        "private_payload",
    }
    present_forbidden = sorted(key for key in forbidden_keys if key in confirmation)
    if present_forbidden:
        failures.append("private_confirmation_contains_forbidden_private_fields:" + ",".join(present_forbidden))

    state["schema_version"] = str(confirmation.get("schema_version") or "")
    if state["schema_version"] != "2026-06-12.manual_submission_private_confirmation.v1":
        failures.append(
            "private_confirmation_schema_mismatch:"
            f"{state['schema_version']} != 2026-06-12.manual_submission_private_confirmation.v1"
        )
    if confirmation.get("target_profile_id") != config.get("target_profile_id"):
        failures.append(
            "private_confirmation_target_profile_mismatch:"
            f"{confirmation.get('target_profile_id')} != {config.get('target_profile_id')}"
        )
    if confirmation.get("checklist_id") != config.get("checklist_id"):
        failures.append(
            "private_confirmation_checklist_id_mismatch:"
            f"{confirmation.get('checklist_id')} != {config.get('checklist_id')}"
        )
    expected_manifest = (metadata_packet.get("package_crosscheck") or {}).get("source_manifest_sha256")
    if confirmation.get("source_manifest_sha256") != expected_manifest:
        failures.append(
            "private_confirmation_source_manifest_sha256_mismatch:"
            f"{confirmation.get('source_manifest_sha256')} != {expected_manifest}"
        )

    state["confirmed_in_submission_system"] = confirmation.get("confirmed_in_submission_system") is True
    state["no_private_fields_stored"] = confirmation.get("no_private_fields_stored") is True
    if not state["confirmed_in_submission_system"]:
        failures.append("private_confirmation_not_confirmed_in_submission_system")
    if not state["no_private_fields_stored"]:
        failures.append("private_confirmation_does_not_declare_no_private_fields_stored")
    if confirmation.get("private_fields_completed_in_submission_system") is not True:
        failures.append("private_confirmation_private_fields_not_marked_completed_in_submission_system")

    completed = confirmation.get("completed_item_ids")
    if not isinstance(completed, list) or not all(isinstance(item, str) for item in completed):
        failures.append("private_confirmation_completed_item_ids_invalid")
        completed_ids: set[str] = set()
    else:
        completed_ids = {item.strip() for item in completed if item.strip()}
    unknown_ids = sorted(completed_ids - item_ids)
    if unknown_ids:
        failures.append("private_confirmation_unknown_item_ids:" + ",".join(unknown_ids))
    state["completed_item_ids"] = sorted(completed_ids)

    if confirmation.get("notes"):
        warnings.append("private_confirmation_notes_present_not_copied_to_public_status")
    return state, completed_ids, failures, warnings


def build_manual_submission_checklist(
    *,
    root: str | Path = ".",
    config_path: str | Path = DEFAULT_CONFIG,
    metadata_packet_json: str | Path = DEFAULT_METADATA_PACKET,
    submission_package_audit_json: str | Path = DEFAULT_SUBMISSION_PACKAGE_AUDIT,
    external_metadata_audit_json: str | Path = DEFAULT_EXTERNAL_METADATA_AUDIT,
    private_confirmation_json: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    config = _read_json(repo / config_path)
    metadata_packet = _read_json(repo / metadata_packet_json)
    submission_package_audit = _read_json(repo / submission_package_audit_json)
    external_metadata_audit = _read_json(repo / external_metadata_audit_json)

    failures: list[str] = []
    warnings: list[str] = []
    if metadata_packet.get("ok") is not True:
        failures.append("submission_metadata_packet_not_ok")
    if metadata_packet.get("submission_metadata_packet_ready") is not True:
        failures.append("submission_metadata_packet_not_ready")
    if submission_package_audit.get("ok") is not True:
        failures.append("submission_package_audit_not_ok")
    if submission_package_audit.get("submission_package_ready_for_target_formatting") is not True:
        failures.append("submission_package_not_ready_for_target_formatting")
    if external_metadata_audit.get("ok") is not True:
        failures.append("external_metadata_audit_not_ok")
    if metadata_packet.get("final_submission_ready") is True:
        failures.append("metadata_packet_unexpectedly_final_ready")
    if submission_package_audit.get("final_submission_ready") is True:
        failures.append("submission_package_unexpectedly_final_ready")
    if external_metadata_audit.get("final_submission_ready") is True:
        failures.append("external_metadata_audit_unexpectedly_final_ready")

    target_profile_id = config.get("target_profile_id")
    metadata_profile = (metadata_packet.get("submission_fields") or {}).get("target_profile_id")
    package_profile = (submission_package_audit.get("target_formatting_profile") or {}).get("profile_id")
    external_profile = external_metadata_audit.get("target_profile_id")
    if metadata_profile != target_profile_id:
        failures.append(f"metadata_packet_target_profile_mismatch:{metadata_profile} != {target_profile_id}")
    if package_profile != target_profile_id:
        failures.append(f"submission_package_target_profile_mismatch:{package_profile} != {target_profile_id}")
    if external_profile != target_profile_id:
        failures.append(f"external_metadata_target_profile_mismatch:{external_profile} != {target_profile_id}")

    config_items = list(config.get("items") or [])
    item_ids = {str(item.get("id")) for item in config_items if item.get("id")}
    private_confirmation, confirmed_item_ids, confirmation_failures, confirmation_warnings = (
        _private_confirmation_state(
            repo=repo,
            raw_path=private_confirmation_json,
            config=config,
            metadata_packet=metadata_packet,
            item_ids=item_ids,
        )
    )
    failures.extend(confirmation_failures)
    warnings.extend(confirmation_warnings)

    items = [
        _item_status(
            item,
            metadata_packet=metadata_packet,
            submission_package_audit=submission_package_audit,
            external_metadata_audit=external_metadata_audit,
            confirmed_item_ids=confirmed_item_ids,
        )
        for item in config_items
    ]
    item_blockers = [blocker for item in items for blocker in item["blockers"]]
    manual_private_items = [item["id"] for item in items if item["private"]]
    manual_pending_items = [
        item["id"]
        for item in items
        if item["status"]
        in {"manual_pending", "manual_private_not_stored", "evidence_pending_manual_confirmation"}
    ]
    unconfirmed_required_item_ids = [
        item["id"]
        for item in items
        if item["status"]
        in {"manual_pending", "manual_private_not_stored", "evidence_pending_manual_confirmation"}
        and not item["blockers"]
    ]

    ready = not failures
    manual_submission_system_ready = (
        ready
        and private_confirmation["provided"]
        and not unconfirmed_required_item_ids
        and not item_blockers
    )
    remaining_blockers = list(submission_package_audit.get("remaining_blockers") or [])
    for blocker in item_blockers:
        if blocker not in remaining_blockers:
            remaining_blockers.append(blocker)
    if (
        not manual_submission_system_ready
        and manual_pending_items
        and "manual_submission_system_items_not_confirmed" not in remaining_blockers
    ):
        remaining_blockers.append("manual_submission_system_items_not_confirmed")

    return {
        "schema_version": "2026-06-12.manual_submission_checklist.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_read_only_manual_submission_checklist",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "ok": ready,
        "manual_submission_checklist_ready": ready,
        "manual_submission_system_ready": manual_submission_system_ready,
        "final_submission_ready": False,
        "target_profile_id": target_profile_id,
        "checklist_id": config.get("checklist_id"),
        "input_paths": {
            "config": str(Path(config_path)),
            "metadata_packet": str(Path(metadata_packet_json)),
            "submission_package_audit": str(Path(submission_package_audit_json)),
            "external_metadata_audit": str(Path(external_metadata_audit_json)),
            "private_confirmation_json": str(Path(private_confirmation_json))
            if private_confirmation_json
            else "",
        },
        "crosscheck": {
            "metadata_packet_ok": metadata_packet.get("ok"),
            "submission_package_ok": submission_package_audit.get("ok"),
            "external_metadata_ok": external_metadata_audit.get("ok"),
            "submission_package_ready_for_target_formatting": submission_package_audit.get(
                "submission_package_ready_for_target_formatting"
            ),
            "external_proceedings_metadata_ready": external_metadata_audit.get(
                "external_proceedings_metadata_ready"
            ),
            "source_manifest_sha256": (
                metadata_packet.get("package_crosscheck") or {}
            ).get("source_manifest_sha256"),
        },
        "manual_private_fields_not_stored": config.get("manual_private_fields_not_stored") or [],
        "private_confirmation": private_confirmation,
        "manual_private_item_ids": manual_private_items,
        "manual_pending_item_ids": manual_pending_items,
        "unconfirmed_required_item_ids": unconfirmed_required_item_ids,
        "item_count": len(items),
        "items": items,
        "warnings": warnings,
        "failures": failures,
        "remaining_blockers": remaining_blockers,
        "next_actions": [
            "Use this checklist while filling the submission system; do not copy private author/COI/reviewer data into the repository.",
            "Optionally pass --private-confirmation-json to audit a local untracked confirmation file after the submission-system fields are completed.",
            "Rerun the external proceedings metadata audit immediately before final submission.",
            "Rerun submission package, metadata packet, and this checklist after any paper/source/BibTeX change.",
            "Keep final_submission_ready=false until manual submission-system items and external metadata blockers are all closed.",
        ],
    }


def _write_md(path: Path, checklist: dict[str, Any]) -> None:
    lines = [
        "# Manual Submission Checklist",
        "",
        f"Generated: {checklist['created_at_utc']}",
        "",
        f"- OK: `{str(checklist['ok']).lower()}`",
        "- Manual submission checklist ready: "
        f"`{str(checklist['manual_submission_checklist_ready']).lower()}`",
        f"- Manual submission system ready: `{str(checklist['manual_submission_system_ready']).lower()}`",
        f"- Final submission ready: `{str(checklist['final_submission_ready']).lower()}`",
        f"- Target profile: `{checklist['target_profile_id']}`",
        f"- Checklist ID: `{checklist['checklist_id']}`",
        f"- Items: `{checklist['item_count']}`",
        f"- Source manifest sha256: `{checklist['crosscheck']['source_manifest_sha256']}`",
        "",
        "## Private Confirmation",
        "",
        f"- Provided: `{str(checklist['private_confirmation']['provided']).lower()}`",
        f"- Exists: `{str(checklist['private_confirmation']['exists']).lower()}`",
        f"- SHA256: `{checklist['private_confirmation']['sha256']}`",
        f"- Completed item count: `{len(checklist['private_confirmation']['completed_item_ids'])}`",
        "",
        "## Items",
        "",
    ]
    for item in checklist["items"]:
        prefill = item["prefill"]
        lines.extend(
            [
                f"### `{item['id']}`",
                "",
                f"- Category: `{item['category']}`",
                f"- Label: {item['label']}",
                f"- Status: `{item['status']}`",
                f"- Private: `{str(item['private']).lower()}`",
                f"- Storage policy: `{item['storage_policy']}`",
            ]
        )
        if prefill["available"]:
            summary = str(prefill["summary"])
            if len(summary) > 300:
                summary = summary[:297] + "..."
            lines.append(f"- Prefill source: `{prefill['source']}`")
            lines.append(f"- Prefill summary: {summary}")
        blockers = item.get("blockers") or []
        lines.append("- Blockers: " + (", ".join(f"`{blocker}`" for blocker in blockers) if blockers else "None"))
        lines.append("")

    lines.extend(["## Manual Private Fields Not Stored", ""])
    lines.extend(f"- {item}" for item in checklist.get("manual_private_fields_not_stored", []))
    lines.extend(["", "## Remaining Blockers", ""])
    blockers = checklist.get("remaining_blockers") or []
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- None")
    lines.extend(["", "## Failures", ""])
    failures = checklist.get("failures") or []
    lines.extend(f"- `{item}`" for item in failures) if failures else lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = checklist.get("warnings") or []
    lines.extend(f"- `{item}`" for item in warnings) if warnings else lines.append("- None")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in checklist.get("next_actions", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG))
    parser.add_argument("--metadata-packet-json", default=str(DEFAULT_METADATA_PACKET))
    parser.add_argument("--submission-package-audit-json", default=str(DEFAULT_SUBMISSION_PACKAGE_AUDIT))
    parser.add_argument("--external-metadata-audit-json", default=str(DEFAULT_EXTERNAL_METADATA_AUDIT))
    parser.add_argument(
        "--private-confirmation-json",
        help="Optional untracked/private JSON confirming submission-system fields were completed without storing private values.",
    )
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checklist = build_manual_submission_checklist(
        root=args.root,
        config_path=args.config_path,
        metadata_packet_json=args.metadata_packet_json,
        submission_package_audit_json=args.submission_package_audit_json,
        external_metadata_audit_json=args.external_metadata_audit_json,
        private_confirmation_json=args.private_confirmation_json,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(checklist, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        _write_md(Path(args.output_md), checklist)
    if not args.output_json:
        print(json.dumps(checklist, indent=2, sort_keys=True))
    return 0 if checklist["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
