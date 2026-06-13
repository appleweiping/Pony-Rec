import json
from pathlib import Path

from scripts.audit.main_validate_manual_submission_private_confirmation_json import (
    build_manual_private_confirmation_validation_packet,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _seed_request(tmp_path: Path, *, blocked: list[str] | None = None) -> Path:
    blocked = ["confirm_external_proceedings_metadata"] if blocked is None else blocked
    request = {
        "ok": True,
        "request_packet_ready": True,
        "final_submission_ready": False,
        "target_profile_id": "test_profile",
        "checklist_id": "test_checklist",
        "required_private_confirmation": {
            "schema_version": "2026-06-12.manual_submission_private_confirmation.v1",
            "source_manifest_sha256": "a" * 64,
            "completed_item_ids_for_full_manual_gate": [
                "paste_title",
                "enter_authors",
                "confirm_external_proceedings_metadata",
            ],
            "currently_blocked_item_ids": blocked,
        },
    }
    return _write_json(tmp_path / "request.json", request)


def _confirmation(**overrides: object) -> dict:
    payload = {
        "schema_version": "2026-06-12.manual_submission_private_confirmation.v1",
        "target_profile_id": "test_profile",
        "checklist_id": "test_checklist",
        "confirmed_in_submission_system": True,
        "private_fields_completed_in_submission_system": True,
        "no_private_fields_stored": True,
        "source_manifest_sha256": "a" * 64,
        "completed_item_ids": ["paste_title", "enter_authors"],
    }
    payload.update(overrides)
    return payload


def test_manual_private_confirmation_validation_accepts_current_unblocked_items(tmp_path: Path) -> None:
    request = _seed_request(tmp_path)
    confirmation = _write_json(
        tmp_path / "artifacts" / "private" / "manual_submission_private_confirmation_20260613.json",
        _confirmation(),
    )

    packet = build_manual_private_confirmation_validation_packet(
        root=tmp_path,
        confirmation_json=confirmation.relative_to(tmp_path),
        request_packet_json=request.relative_to(tmp_path),
    )

    assert packet["ok"] is True
    assert packet["ready_to_use_for_manual_checklist"] is True
    assert packet["full_manual_gate_ready_shape"] is False
    assert packet["expected"]["currently_blocked_item_ids"] == ["confirm_external_proceedings_metadata"]
    assert packet["observed"]["missing_current_required_item_ids"] == []
    assert packet["observed"]["completed_currently_blocked_item_ids"] == []


def test_manual_private_confirmation_validation_rejects_blocked_item_completion(tmp_path: Path) -> None:
    request = _seed_request(tmp_path)
    confirmation = _write_json(
        tmp_path / "artifacts" / "private" / "manual_submission_private_confirmation_20260613.json",
        _confirmation(
            completed_item_ids=[
                "paste_title",
                "enter_authors",
                "confirm_external_proceedings_metadata",
            ]
        ),
    )

    packet = build_manual_private_confirmation_validation_packet(
        root=tmp_path,
        confirmation_json=confirmation.relative_to(tmp_path),
        request_packet_json=request.relative_to(tmp_path),
    )

    assert packet["ok"] is False
    assert (
        "private_confirmation_completed_currently_blocked_item_ids:confirm_external_proceedings_metadata"
        in packet["failures"]
    )


def test_manual_private_confirmation_validation_rejects_forbidden_private_keys(tmp_path: Path) -> None:
    request = _seed_request(tmp_path)
    confirmation = _write_json(
        tmp_path / "artifacts" / "private" / "manual_submission_private_confirmation_20260613.json",
        _confirmation(authors=["Private Author"]),
    )

    packet = build_manual_private_confirmation_validation_packet(
        root=tmp_path,
        confirmation_json=confirmation.relative_to(tmp_path),
        request_packet_json=request.relative_to(tmp_path),
    )

    assert packet["ok"] is False
    assert "private_confirmation_contains_forbidden_private_fields:authors" in packet["failures"]
    assert packet["observed"]["forbidden_private_keys_present"] == ["authors"]


def test_manual_private_confirmation_validation_rejects_repo_path_outside_private_artifacts(
    tmp_path: Path,
) -> None:
    request = _seed_request(tmp_path)
    confirmation = _write_json(tmp_path / "manual_confirmation.json", _confirmation())

    packet = build_manual_private_confirmation_validation_packet(
        root=tmp_path,
        confirmation_json=confirmation.relative_to(tmp_path),
        request_packet_json=request.relative_to(tmp_path),
    )

    assert packet["ok"] is False
    assert "private_confirmation_path_not_under_artifacts_private" in packet["failures"]


def test_manual_private_confirmation_validation_full_gate_shape_when_no_items_blocked(
    tmp_path: Path,
) -> None:
    request = _seed_request(tmp_path, blocked=[])
    confirmation = _write_json(
        tmp_path / "artifacts" / "private" / "manual_submission_private_confirmation_20260613.json",
        _confirmation(
            completed_item_ids=[
                "paste_title",
                "enter_authors",
                "confirm_external_proceedings_metadata",
            ]
        ),
    )

    packet = build_manual_private_confirmation_validation_packet(
        root=tmp_path,
        confirmation_json=confirmation.relative_to(tmp_path),
        request_packet_json=request.relative_to(tmp_path),
    )

    assert packet["ok"] is True
    assert packet["ready_to_use_for_manual_checklist"] is True
    assert packet["full_manual_gate_ready_shape"] is True
