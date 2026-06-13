from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs/summary/paper_critical")
DEFAULT_STAMP = "20260613"
DEFAULT_MANUAL_CHECKLIST_JSON = Path(
    "outputs/summary/paper_critical/manual_submission_checklist_20260613.json"
)
DEFAULT_TEMPLATE_JSON = Path("configs/paper_manual_submission_private_confirmation.template.json")


FORBIDDEN_PRIVATE_KEYS = [
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
]


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


def _as_list(values: Any) -> list[Any]:
    return values if isinstance(values, list) else []


def _string_list(values: Any) -> list[str]:
    return [str(item) for item in _as_list(values) if str(item).strip()]


def _checklist_item_ids(checklist: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for item in _as_list(checklist.get("items")):
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
    return ids


def _public_prefill_index(checklist: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _as_list(checklist.get("items")):
        if not isinstance(item, dict):
            continue
        prefill = item.get("prefill") or {}
        if item.get("private") or not prefill.get("available"):
            continue
        summary = str(prefill.get("summary") or "")
        rows.append(
            {
                "id": str(item.get("id") or ""),
                "label": str(item.get("label") or ""),
                "prefill_source": str(prefill.get("source") or ""),
                "prefill_summary": summary,
            }
        )
    return rows


def _safe_confirmation_skeleton(
    *,
    template: dict[str, Any],
    checklist: dict[str, Any],
    source_manifest_sha256: str,
    completed_item_ids: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": template.get("schema_version", ""),
        "target_profile_id": checklist.get("target_profile_id", ""),
        "checklist_id": checklist.get("checklist_id", ""),
        "confirmed_in_submission_system": True,
        "private_fields_completed_in_submission_system": True,
        "no_private_fields_stored": True,
        "source_manifest_sha256": source_manifest_sha256,
        "completed_item_ids": completed_item_ids,
    }


def build_manual_submission_private_confirmation_request_packet(
    *,
    root: str | Path = ".",
    manual_checklist_json: str | Path = DEFAULT_MANUAL_CHECKLIST_JSON,
    template_json: str | Path = DEFAULT_TEMPLATE_JSON,
    stamp: str = DEFAULT_STAMP,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    checklist_path = (repo / manual_checklist_json).resolve()
    template_path = (repo / template_json).resolve()

    checklist, checklist_errors = _read_json(checklist_path)
    template, template_errors = _read_json(template_path)
    failures = [f"manual_checklist:{error}" for error in checklist_errors]
    failures.extend(f"template:{error}" for error in template_errors)

    source_manifest_sha256 = str(((checklist.get("crosscheck") or {}).get("source_manifest_sha256") or ""))
    all_item_ids = _checklist_item_ids(checklist)
    template_item_ids = _string_list(template.get("completed_item_ids"))
    unconfirmed_required_item_ids = _string_list(checklist.get("unconfirmed_required_item_ids"))
    blocked_item_ids = [
        str(item.get("id"))
        for item in _as_list(checklist.get("items"))
        if isinstance(item, dict) and item.get("id") and item.get("blockers")
    ]
    manual_private_item_ids = _string_list(checklist.get("manual_private_item_ids"))

    if checklist and checklist.get("ok") is not True:
        failures.append("manual_submission_checklist_not_ok")
    if checklist and checklist.get("manual_submission_checklist_ready") is not True:
        failures.append("manual_submission_checklist_not_ready")
    if checklist and checklist.get("final_submission_ready") is not False:
        failures.append("manual_checklist_unexpected_final_submission_ready_state")
    if checklist and not source_manifest_sha256:
        failures.append("source_manifest_sha256_missing_from_manual_checklist")
    if checklist and template:
        if template.get("target_profile_id") != checklist.get("target_profile_id"):
            failures.append(
                "template_target_profile_mismatch:"
                f"{template.get('target_profile_id')} != {checklist.get('target_profile_id')}"
            )
        if template.get("checklist_id") != checklist.get("checklist_id"):
            failures.append(
                "template_checklist_id_mismatch:"
                f"{template.get('checklist_id')} != {checklist.get('checklist_id')}"
            )
        if sorted(template_item_ids) != sorted(all_item_ids):
            failures.append("template_completed_item_ids_do_not_match_manual_checklist_items")
        if template.get("no_private_fields_stored") is not True:
            failures.append("template_must_keep_no_private_fields_stored_true")
        for key in FORBIDDEN_PRIVATE_KEYS:
            if key in template:
                failures.append(f"template_contains_forbidden_private_field:{key}")

    manual_system_ready = checklist.get("manual_submission_system_ready") is True
    confirmation_needed = not manual_system_ready
    safe_skeleton = _safe_confirmation_skeleton(
        template=template,
        checklist=checklist,
        source_manifest_sha256=source_manifest_sha256,
        completed_item_ids=all_item_ids,
    )

    recommended_path = f"artifacts/private/manual_submission_private_confirmation_{stamp}.json"
    request_packet_path = (
        f"outputs/summary/paper_critical/manual_submission_private_confirmation_request_packet_{stamp}.json"
    )
    follow_up_validation_command = (
        "python -m scripts.audit.main_validate_manual_submission_private_confirmation_json "
        f"--private-confirmation-json {recommended_path} "
        f"--manual-request-packet-json {request_packet_path} "
        f"--output-json outputs/summary/paper_critical/manual_private_confirmation_validation_{stamp}.json "
        f"--output-md outputs/summary/paper_critical/manual_private_confirmation_validation_{stamp}.md"
    )
    follow_up_checklist_command = (
        "python -m scripts.audit.main_build_manual_submission_checklist "
        f"--private-confirmation-json {recommended_path} "
        f"--output-json outputs/summary/paper_critical/manual_submission_checklist_{stamp}.json "
        f"--output-md outputs/summary/paper_critical/manual_submission_checklist_{stamp}.md"
    )
    follow_up_stack_command = (
        "python -m scripts.audit.main_refresh_submission_release_candidate_stack "
        f"--stamp {stamp} "
        f"--manual-private-confirmation-json {recommended_path} "
        "--external-timeout-seconds 45 "
        f"--output-json outputs/summary/paper_critical/submission_release_candidate_stack_refresh_{stamp}.json "
        f"--output-md outputs/summary/paper_critical/submission_release_candidate_stack_refresh_{stamp}.md"
    )

    return {
        "schema_version": "2026-06-13.manual_submission_private_confirmation_request_packet.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_read_only_manual_submission_private_confirmation_request_packet",
        "project": "uncertainty",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "ok": not failures,
        "request_packet_ready": not failures,
        "manual_confirmation_needed": confirmation_needed,
        "manual_submission_system_ready": manual_system_ready,
        "final_submission_ready": False,
        "target_profile_id": checklist.get("target_profile_id", ""),
        "checklist_id": checklist.get("checklist_id", ""),
        "input_paths": {
            "manual_checklist_json": _path_state(checklist_path, repo),
            "private_confirmation_template_json": _path_state(template_path, repo),
        },
        "gate_state": {
            "manual_submission_checklist_ready": checklist.get("manual_submission_checklist_ready") is True,
            "manual_submission_system_ready": manual_system_ready,
            "external_proceedings_metadata_ready": (
                (checklist.get("crosscheck") or {}).get("external_proceedings_metadata_ready") is True
            ),
            "private_confirmation_provided": ((checklist.get("private_confirmation") or {}).get("provided") is True),
            "remaining_blockers": _string_list(checklist.get("remaining_blockers")),
        },
        "required_private_confirmation": {
            "recommended_untracked_path": recommended_path,
            "template_path": _repo_relative(template_path, repo),
            "schema_version": template.get("schema_version", ""),
            "source_manifest_sha256": source_manifest_sha256,
            "completed_item_ids_for_full_manual_gate": all_item_ids,
            "currently_unconfirmed_required_item_ids": unconfirmed_required_item_ids,
            "currently_blocked_item_ids": blocked_item_ids,
            "manual_private_item_ids": manual_private_item_ids,
            "boolean_fields_that_must_be_true_after_human_completion": [
                "confirmed_in_submission_system",
                "private_fields_completed_in_submission_system",
                "no_private_fields_stored",
            ],
            "safe_confirmation_skeleton": safe_skeleton,
        },
        "public_prefill_index": _public_prefill_index(checklist),
        "forbidden_private_fields": {
            "human_readable": _string_list(checklist.get("manual_private_fields_not_stored")),
            "json_keys_rejected_by_checker": FORBIDDEN_PRIVATE_KEYS,
        },
        "privacy_rules": [
            "Keep the filled confirmation JSON under an ignored path such as artifacts/private/.",
            "Do not commit the filled confirmation JSON.",
            "Do not store author names, affiliations, COI details, reviewer preferences, declarations, account metadata, or submission-account data.",
            "Only record booleans, source_manifest_sha256, completed_item_ids, and non-sensitive notes if absolutely needed.",
            "Do not set completed_item_ids until the corresponding action is genuinely complete in the submission system.",
            "Run the private confirmation validator before consuming the JSON in the public manual checklist.",
            "This request packet does not close ProMax public metadata or Claude Opus review blockers.",
        ],
        "private_confirmation_validation_command": follow_up_validation_command,
        "follow_up_commands_after_human_completion": [
            follow_up_validation_command,
            follow_up_checklist_command,
            follow_up_stack_command,
            (
                "python -m scripts.audit.main_build_final_submission_gate "
                f"--manual-checklist-json outputs/summary/paper_critical/manual_submission_checklist_{stamp}.json "
                "--output-json outputs/summary/paper_critical/final_submission_gate_YYYYMMDD.json "
                "--output-md outputs/summary/paper_critical/final_submission_gate_YYYYMMDD.md"
            ),
        ],
        "notes": [
            "This packet is a public-safe handoff request only; it is not a private confirmation.",
            "The manual gate remains open until the checklist consumes a validated untracked private confirmation JSON.",
            "Final submission readiness must stay false while external metadata or review coverage blockers remain.",
        ],
        "failures": failures,
    }


def render_markdown(packet: dict[str, Any]) -> str:
    required = packet["required_private_confirmation"]
    lines = [
        "# Manual Submission Private Confirmation Request Packet",
        "",
        f"- Created UTC: `{packet['created_at_utc']}`",
        f"- OK: `{str(packet['ok']).lower()}`",
        f"- Request packet ready: `{str(packet['request_packet_ready']).lower()}`",
        f"- Manual confirmation needed: `{str(packet['manual_confirmation_needed']).lower()}`",
        f"- Manual submission system ready: `{str(packet['manual_submission_system_ready']).lower()}`",
        f"- Final submission ready: `{str(packet['final_submission_ready']).lower()}`",
        f"- Target profile: `{packet['target_profile_id']}`",
        f"- Checklist ID: `{packet['checklist_id']}`",
        f"- Recommended untracked path: `{required['recommended_untracked_path']}`",
        f"- Source manifest sha256: `{required['source_manifest_sha256']}`",
        "",
        "## Safe Confirmation Skeleton",
        "",
        "```json",
        json.dumps(required["safe_confirmation_skeleton"], indent=2, sort_keys=True),
        "```",
        "",
        "## Item IDs",
        "",
        "Full manual gate item IDs:",
    ]
    lines.extend(f"- `{item}`" for item in required["completed_item_ids_for_full_manual_gate"])
    lines.extend(["", "Currently unconfirmed required item IDs:"])
    unconfirmed = required.get("currently_unconfirmed_required_item_ids") or []
    lines.extend(f"- `{item}`" for item in unconfirmed) if unconfirmed else lines.append("- None")
    lines.extend(["", "Currently blocked item IDs:"])
    blocked = required.get("currently_blocked_item_ids") or []
    lines.extend(f"- `{item}`" for item in blocked) if blocked else lines.append("- None")
    lines.extend(["", "## Forbidden Private Fields", ""])
    human_fields = packet["forbidden_private_fields"].get("human_readable") or []
    lines.extend(f"- {item}" for item in human_fields)
    lines.extend(["", "Rejected JSON keys:"])
    lines.extend(f"- `{item}`" for item in packet["forbidden_private_fields"]["json_keys_rejected_by_checker"])
    lines.extend(["", "## Privacy Rules", ""])
    lines.extend(f"- {item}" for item in packet.get("privacy_rules", []))
    lines.extend(["", "## Follow-Up Commands", ""])
    for command in packet.get("follow_up_commands_after_human_completion", []):
        lines.extend(["```bash", command, "```", ""])
    lines.extend(["## Remaining Blockers", ""])
    blockers = (packet.get("gate_state") or {}).get("remaining_blockers") or []
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- None")
    lines.extend(["", "## Failures", ""])
    failures = packet.get("failures") or []
    lines.extend(f"- `{item}`" for item in failures) if failures else lines.append("- None")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--stamp", default=DEFAULT_STAMP)
    parser.add_argument("--manual-checklist-json", default=str(DEFAULT_MANUAL_CHECKLIST_JSON))
    parser.add_argument("--template-json", default=str(DEFAULT_TEMPLATE_JSON))
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args()

    out_dir = Path(args.root) / DEFAULT_OUTPUT_DIR
    output_json = (
        Path(args.output_json)
        if args.output_json
        else out_dir / f"manual_submission_private_confirmation_request_packet_{args.stamp}.json"
    )
    output_md = (
        Path(args.output_md)
        if args.output_md
        else out_dir / f"manual_submission_private_confirmation_request_packet_{args.stamp}.md"
    )

    packet = build_manual_submission_private_confirmation_request_packet(
        root=args.root,
        manual_checklist_json=args.manual_checklist_json,
        template_json=args.template_json,
        stamp=args.stamp,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(packet), encoding="utf-8")
    print(json.dumps({"ok": packet["ok"], "output_json": str(output_json), "output_md": str(output_md)}, indent=2))
    return 0 if packet["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
