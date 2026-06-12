import json
from pathlib import Path

from scripts.audit.main_build_manual_submission_checklist import build_manual_submission_checklist


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _seed_inputs(tmp_path: Path) -> dict[str, Path]:
    config = _write_json(
        tmp_path / "manual_config.json",
        {
            "target_profile_id": "test_profile",
            "checklist_id": "test_manual_checklist",
            "manual_private_fields_not_stored": ["authors", "conflicts"],
            "items": [
                {
                    "id": "paste_title",
                    "category": "metadata",
                    "label": "Paste title",
                    "storage_policy": "repo_prefill_public_value",
                    "private": False,
                    "requires_submission_system": True,
                    "prefill_source": "submission_fields.title",
                },
                {
                    "id": "enter_authors",
                    "category": "private_metadata",
                    "label": "Enter authors",
                    "storage_policy": "not_stored_in_repo",
                    "private": True,
                    "requires_submission_system": True,
                },
                {
                    "id": "confirm_external_proceedings_metadata",
                    "category": "external_metadata",
                    "label": "Confirm external proceedings metadata",
                    "storage_policy": "repo_external_audit_plus_manual_confirmation",
                    "private": False,
                    "requires_submission_system": False,
                    "external_gate": "external_proceedings_metadata_ready",
                },
            ],
        },
    )
    metadata = _write_json(
        tmp_path / "metadata_packet.json",
        {
            "ok": True,
            "submission_metadata_packet_ready": True,
            "final_submission_ready": False,
            "submission_fields": {"target_profile_id": "test_profile", "title": "Paper Title"},
            "package_crosscheck": {
                "pdf_path": "Paper/main.pdf",
                "pdf_size_bytes": 123456,
                "source_manifest_sha256": "abc123",
            },
        },
    )
    package = _write_json(
        tmp_path / "submission_package.json",
        {
            "ok": True,
            "submission_package_ready_for_target_formatting": True,
            "final_submission_ready": False,
            "target_formatting_profile": {"profile_id": "test_profile", "ok": True},
            "evidence_gates": {"target_formatting_profile_ok": True},
            "remaining_blockers": ["promax:final_page_range_missing_in_bib"],
        },
    )
    external = _write_json(
        tmp_path / "external_metadata.json",
        {
            "ok": True,
            "target_profile_id": "test_profile",
            "external_proceedings_metadata_ready": False,
            "final_submission_ready": False,
        },
    )
    return {"config": config, "metadata": metadata, "package": package, "external": external}


def test_manual_submission_checklist_preserves_manual_and_external_blockers(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)

    checklist = build_manual_submission_checklist(
        root=tmp_path,
        config_path=paths["config"].relative_to(tmp_path),
        metadata_packet_json=paths["metadata"].relative_to(tmp_path),
        submission_package_audit_json=paths["package"].relative_to(tmp_path),
        external_metadata_audit_json=paths["external"].relative_to(tmp_path),
    )

    assert checklist["ok"] is True
    assert checklist["manual_submission_checklist_ready"] is True
    assert checklist["manual_submission_system_ready"] is False
    assert checklist["final_submission_ready"] is False
    assert "enter_authors" in checklist["manual_private_item_ids"]
    assert "manual_submission_system_items_not_confirmed" in checklist["remaining_blockers"]
    assert "confirm_external_proceedings_metadata:external_proceedings_metadata_ready_not_closed" in checklist[
        "remaining_blockers"
    ]
    items = {item["id"]: item for item in checklist["items"]}
    assert items["paste_title"]["prefill"]["summary"] == "Paper Title"
    assert items["enter_authors"]["status"] == "manual_private_not_stored"


def test_manual_submission_checklist_fails_on_profile_mismatch(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)
    external = json.loads(paths["external"].read_text(encoding="utf-8"))
    external["target_profile_id"] = "other_profile"
    paths["external"].write_text(json.dumps(external), encoding="utf-8")

    checklist = build_manual_submission_checklist(
        root=tmp_path,
        config_path=paths["config"].relative_to(tmp_path),
        metadata_packet_json=paths["metadata"].relative_to(tmp_path),
        submission_package_audit_json=paths["package"].relative_to(tmp_path),
        external_metadata_audit_json=paths["external"].relative_to(tmp_path),
    )

    assert checklist["ok"] is False
    assert "external_metadata_target_profile_mismatch:other_profile != test_profile" in checklist["failures"]
