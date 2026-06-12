from __future__ import annotations

import argparse
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
) -> dict[str, Any]:
    prefill = _prefill_value(metadata_packet, item.get("prefill_source"))
    blockers: list[str] = []
    warnings: list[str] = []

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

    if item.get("private"):
        status = "manual_private_not_stored"
    elif blockers:
        status = "blocked"
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


def build_manual_submission_checklist(
    *,
    root: str | Path = ".",
    config_path: str | Path = DEFAULT_CONFIG,
    metadata_packet_json: str | Path = DEFAULT_METADATA_PACKET,
    submission_package_audit_json: str | Path = DEFAULT_SUBMISSION_PACKAGE_AUDIT,
    external_metadata_audit_json: str | Path = DEFAULT_EXTERNAL_METADATA_AUDIT,
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

    items = [
        _item_status(
            item,
            metadata_packet=metadata_packet,
            submission_package_audit=submission_package_audit,
            external_metadata_audit=external_metadata_audit,
        )
        for item in config.get("items") or []
    ]
    item_blockers = [blocker for item in items for blocker in item["blockers"]]
    manual_private_items = [item["id"] for item in items if item["private"]]
    manual_pending_items = [
        item["id"]
        for item in items
        if item["status"] in {"manual_pending", "manual_private_not_stored"}
    ]

    ready = not failures
    remaining_blockers = list(submission_package_audit.get("remaining_blockers") or [])
    for blocker in item_blockers:
        if blocker not in remaining_blockers:
            remaining_blockers.append(blocker)
    if manual_pending_items and "manual_submission_system_items_not_confirmed" not in remaining_blockers:
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
        "manual_submission_system_ready": False,
        "final_submission_ready": False,
        "target_profile_id": target_profile_id,
        "checklist_id": config.get("checklist_id"),
        "input_paths": {
            "config": str(Path(config_path)),
            "metadata_packet": str(Path(metadata_packet_json)),
            "submission_package_audit": str(Path(submission_package_audit_json)),
            "external_metadata_audit": str(Path(external_metadata_audit_json)),
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
        "manual_private_item_ids": manual_private_items,
        "manual_pending_item_ids": manual_pending_items,
        "item_count": len(items),
        "items": items,
        "warnings": warnings,
        "failures": failures,
        "remaining_blockers": remaining_blockers,
        "next_actions": [
            "Use this checklist while filling the submission system; do not copy private author/COI/reviewer data into the repository.",
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
