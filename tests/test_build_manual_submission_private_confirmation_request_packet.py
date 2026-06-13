import json
from pathlib import Path

from scripts.audit.main_build_manual_submission_private_confirmation_request_packet import (
    build_manual_submission_private_confirmation_request_packet,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _seed_inputs(tmp_path: Path, *, checklist_overrides: dict | None = None, template_overrides: dict | None = None):
    items = [
        {
            "id": "paste_title",
            "label": "Paste title",
            "private": False,
            "status": "manual_pending",
            "blockers": [],
            "prefill": {"available": True, "source": "submission_fields.title", "summary": "Paper Title"},
        },
        {
            "id": "enter_authors",
            "label": "Enter authors",
            "private": True,
            "status": "manual_private_not_stored",
            "blockers": [],
            "prefill": {"available": False, "source": "", "summary": ""},
        },
        {
            "id": "confirm_external_proceedings_metadata",
            "label": "Confirm external metadata",
            "private": False,
            "status": "blocked",
            "blockers": ["confirm_external_proceedings_metadata:external_proceedings_metadata_ready_not_closed"],
            "prefill": {"available": False, "source": "", "summary": ""},
        },
    ]
    checklist = {
        "ok": True,
        "manual_submission_checklist_ready": True,
        "manual_submission_system_ready": False,
        "final_submission_ready": False,
        "target_profile_id": "sigir2026_full_paper_acm_anonymous",
        "checklist_id": "sigir2026_anonymous_full_paper_manual_submission",
        "crosscheck": {
            "source_manifest_sha256": "abc123",
            "external_proceedings_metadata_ready": False,
        },
        "manual_private_fields_not_stored": [
            "author names and affiliations",
            "conflicts of interest",
        ],
        "manual_private_item_ids": ["enter_authors"],
        "unconfirmed_required_item_ids": ["paste_title", "enter_authors"],
        "items": items,
        "private_confirmation": {"provided": False},
        "remaining_blockers": [
            "manual_submission_system_items_not_confirmed",
            "promax:doi_resolver_not_visible:status=404",
        ],
    }
    if checklist_overrides:
        checklist.update(checklist_overrides)

    template = {
        "schema_version": "2026-06-12.manual_submission_private_confirmation.v1",
        "target_profile_id": "sigir2026_full_paper_acm_anonymous",
        "checklist_id": "sigir2026_anonymous_full_paper_manual_submission",
        "confirmed_in_submission_system": False,
        "private_fields_completed_in_submission_system": False,
        "no_private_fields_stored": True,
        "source_manifest_sha256": "REPLACE_WITH_CURRENT_SOURCE_MANIFEST_SHA256",
        "completed_item_ids": [item["id"] for item in items],
    }
    if template_overrides:
        template.update(template_overrides)

    return {
        "checklist": _write_json(tmp_path / "manual_checklist.json", checklist),
        "template": _write_json(tmp_path / "private_template.json", template),
    }


def test_private_confirmation_request_packet_builds_public_safe_skeleton(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)

    packet = build_manual_submission_private_confirmation_request_packet(
        root=tmp_path,
        manual_checklist_json=paths["checklist"].relative_to(tmp_path),
        template_json=paths["template"].relative_to(tmp_path),
        stamp="20260613",
    )

    assert packet["ok"] is True
    assert packet["manual_confirmation_needed"] is True
    assert packet["will_start_experiment"] is False
    required = packet["required_private_confirmation"]
    assert required["recommended_untracked_path"] == "artifacts/private/manual_submission_private_confirmation_20260613.json"
    assert required["source_manifest_sha256"] == "abc123"
    assert required["currently_blocked_item_ids"] == ["confirm_external_proceedings_metadata"]
    skeleton = required["safe_confirmation_skeleton"]
    assert skeleton["confirmed_in_submission_system"] is True
    assert skeleton["private_fields_completed_in_submission_system"] is True
    assert skeleton["no_private_fields_stored"] is True
    assert skeleton["source_manifest_sha256"] == "abc123"
    assert skeleton["completed_item_ids"] == [
        "paste_title",
        "enter_authors",
        "confirm_external_proceedings_metadata",
    ]
    assert "authors" not in skeleton
    assert "conflicts" not in skeleton
    assert "Paper Title" in packet["public_prefill_index"][0]["prefill_summary"]
    commands = packet["follow_up_commands_after_human_completion"]
    assert "main_validate_manual_submission_private_confirmation_json" in commands[0]
    assert "--manual-request-packet-json outputs/summary/paper_critical/manual_submission_private_confirmation_request_packet_20260613.json" in commands[0]
    assert "--private-confirmation-json artifacts/private/manual_submission_private_confirmation_20260613.json" in commands[0]
    assert packet["private_confirmation_validation_command"] == commands[0]
    assert "main_build_manual_submission_checklist" in commands[1]


def test_private_confirmation_request_packet_fails_closed_when_checklist_not_ready(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path, checklist_overrides={"manual_submission_checklist_ready": False})

    packet = build_manual_submission_private_confirmation_request_packet(
        root=tmp_path,
        manual_checklist_json=paths["checklist"].relative_to(tmp_path),
        template_json=paths["template"].relative_to(tmp_path),
    )

    assert packet["ok"] is False
    assert "manual_submission_checklist_not_ready" in packet["failures"]


def test_private_confirmation_request_packet_rejects_template_item_mismatch(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path, template_overrides={"completed_item_ids": ["paste_title"]})

    packet = build_manual_submission_private_confirmation_request_packet(
        root=tmp_path,
        manual_checklist_json=paths["checklist"].relative_to(tmp_path),
        template_json=paths["template"].relative_to(tmp_path),
    )

    assert packet["ok"] is False
    assert "template_completed_item_ids_do_not_match_manual_checklist_items" in packet["failures"]
