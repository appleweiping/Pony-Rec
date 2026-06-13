import json
from pathlib import Path

from scripts.audit.main_build_final_submission_gate import build_final_submission_gate


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _seed_inputs(tmp_path: Path) -> dict[str, Path]:
    package = _write_json(
        tmp_path / "submission_package.json",
        {
            "schema_version": "package.v1",
            "created_at_utc": "2026-06-12T00:00:00Z",
            "ok": True,
            "submission_package_ready_for_target_formatting": True,
            "final_submission_ready": False,
            "warnings": [],
            "failures": [],
            "remaining_blockers": ["promax:final_page_range_missing_in_bib"],
        },
    )
    metadata = _write_json(
        tmp_path / "metadata_packet.json",
        {
            "schema_version": "metadata.v1",
            "created_at_utc": "2026-06-12T00:00:00Z",
            "ok": True,
            "submission_metadata_packet_ready": True,
            "final_submission_ready": False,
            "warnings": [],
            "failures": [],
            "remaining_blockers": ["promax:final_page_range_missing_in_bib"],
        },
    )
    source_rebuild = _write_json(
        tmp_path / "source_rebuild.json",
        {
            "schema_version": "source_rebuild.v1",
            "created_at_utc": "2026-06-12T00:00:00Z",
            "ok": True,
            "submission_source_package_rebuild_ready": True,
            "final_submission_ready": False,
            "warnings": [],
            "failures": [],
            "remaining_blockers": ["promax:final_page_range_missing_in_bib"],
        },
    )
    external = _write_json(
        tmp_path / "external_metadata.json",
        {
            "schema_version": "external.v1",
            "created_at_utc": "2026-06-12T00:00:00Z",
            "ok": True,
            "external_proceedings_metadata_ready": False,
            "final_submission_ready": False,
            "warnings": ["proex:crossref_not_visible:status=404"],
            "failures": [],
            "remaining_blockers": ["promax:crossref_registry_not_visible:status=404"],
        },
    )
    manual = _write_json(
        tmp_path / "manual_checklist.json",
        {
            "schema_version": "manual.v1",
            "created_at_utc": "2026-06-12T00:00:00Z",
            "ok": True,
            "manual_submission_checklist_ready": True,
            "manual_submission_system_ready": False,
            "final_submission_ready": False,
            "warnings": [],
            "failures": [],
            "remaining_blockers": ["manual_submission_system_items_not_confirmed"],
        },
    )
    review = _write_json(
        tmp_path / "review_continuation.json",
        {
            "schema_version": "review.v1",
            "created_at_utc": "2026-06-13T00:00:00Z",
            "ok": True,
            "review_continuation_ready": True,
            "final_panel_coverage_complete": False,
            "final_submission_ready": False,
            "warnings": [],
            "failures": [],
            "remaining_blockers": ["explicit_claude_opus_review"],
        },
    )
    return {
        "package": package,
        "metadata": metadata,
        "source_rebuild": source_rebuild,
        "external": external,
        "manual": manual,
        "review": review,
    }


def test_final_submission_gate_aggregates_external_and_manual_blockers(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)

    gate = build_final_submission_gate(
        root=tmp_path,
        submission_package_audit_json=paths["package"].relative_to(tmp_path),
        submission_metadata_packet_json=paths["metadata"].relative_to(tmp_path),
        submission_source_package_rebuild_json=paths["source_rebuild"].relative_to(tmp_path),
        external_metadata_audit_json=paths["external"].relative_to(tmp_path),
        manual_checklist_json=paths["manual"].relative_to(tmp_path),
        review_continuation_packet_json=paths["review"].relative_to(tmp_path),
    )

    assert gate["ok"] is True
    assert gate["all_local_artifact_gates_ok"] is True
    assert gate["external_proceedings_metadata_ready"] is False
    assert gate["manual_submission_system_ready"] is False
    assert gate["review_continuation_ready"] is True
    assert gate["review_panel_coverage_complete"] is False
    assert gate["final_submission_ready"] is False
    assert gate["verdict"] == "LOCAL_PACKAGE_READY_BUT_EXTERNAL_MANUAL_OR_REVIEW_BLOCKED"
    assert "external_proceedings_metadata_not_ready" in gate["remaining_blockers"]
    assert "manual_submission_system_not_ready" in gate["remaining_blockers"]
    assert "review_panel_coverage_not_complete" in gate["remaining_blockers"]
    assert "explicit_claude_opus_review" in gate["remaining_blockers"]
    assert "promax:crossref_registry_not_visible:status=404" in gate["remaining_blockers"]
    assert "manual_submission_system_items_not_confirmed" in gate["remaining_blockers"]
    assert gate["gates"][2]["gate_id"] == "submission_source_package_rebuild"
    assert gate["warnings"] == ["external_proceedings_metadata:proex:crossref_not_visible:status=404"]


def test_final_submission_gate_fails_closed_on_unexpected_subgate_final_ready(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)
    package = json.loads(paths["package"].read_text(encoding="utf-8"))
    package["final_submission_ready"] = True
    paths["package"].write_text(json.dumps(package), encoding="utf-8")

    gate = build_final_submission_gate(
        root=tmp_path,
        submission_package_audit_json=paths["package"].relative_to(tmp_path),
        submission_metadata_packet_json=paths["metadata"].relative_to(tmp_path),
        submission_source_package_rebuild_json=paths["source_rebuild"].relative_to(tmp_path),
        external_metadata_audit_json=paths["external"].relative_to(tmp_path),
        manual_checklist_json=paths["manual"].relative_to(tmp_path),
        review_continuation_packet_json=paths["review"].relative_to(tmp_path),
    )

    assert gate["ok"] is False
    assert gate["final_submission_ready"] is False
    assert "submission_package:unexpected_local_final_submission_ready" in gate["failures"]
    assert gate["verdict"] == "FINAL_SUBMISSION_GATE_NEEDS_REPAIR"


def test_final_submission_gate_fails_closed_when_source_rebuild_not_ready(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)
    source_rebuild = json.loads(paths["source_rebuild"].read_text(encoding="utf-8"))
    source_rebuild["submission_source_package_rebuild_ready"] = False
    source_rebuild["failures"] = ["rebuilt_pdf_missing_or_too_small"]
    paths["source_rebuild"].write_text(json.dumps(source_rebuild), encoding="utf-8")

    gate = build_final_submission_gate(
        root=tmp_path,
        submission_package_audit_json=paths["package"].relative_to(tmp_path),
        submission_metadata_packet_json=paths["metadata"].relative_to(tmp_path),
        submission_source_package_rebuild_json=paths["source_rebuild"].relative_to(tmp_path),
        external_metadata_audit_json=paths["external"].relative_to(tmp_path),
        manual_checklist_json=paths["manual"].relative_to(tmp_path),
        review_continuation_packet_json=paths["review"].relative_to(tmp_path),
    )

    assert gate["ok"] is False
    assert gate["final_submission_ready"] is False
    assert "submission_source_package_rebuild:not_ready" in gate["failures"]
    assert "submission_source_package_rebuild:rebuilt_pdf_missing_or_too_small" in gate["failures"]


def test_final_submission_gate_can_mark_ready_when_all_gates_ready(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)
    for key, ready_field in [
        ("external", "external_proceedings_metadata_ready"),
        ("manual", "manual_submission_system_ready"),
    ]:
        payload = json.loads(paths[key].read_text(encoding="utf-8"))
        payload[ready_field] = True
        payload["remaining_blockers"] = []
        paths[key].write_text(json.dumps(payload), encoding="utf-8")
    for key in ["package", "metadata", "source_rebuild"]:
        payload = json.loads(paths[key].read_text(encoding="utf-8"))
        payload["remaining_blockers"] = []
        paths[key].write_text(json.dumps(payload), encoding="utf-8")
    review = json.loads(paths["review"].read_text(encoding="utf-8"))
    review["final_panel_coverage_complete"] = True
    review["remaining_blockers"] = []
    paths["review"].write_text(json.dumps(review), encoding="utf-8")

    gate = build_final_submission_gate(
        root=tmp_path,
        submission_package_audit_json=paths["package"].relative_to(tmp_path),
        submission_metadata_packet_json=paths["metadata"].relative_to(tmp_path),
        submission_source_package_rebuild_json=paths["source_rebuild"].relative_to(tmp_path),
        external_metadata_audit_json=paths["external"].relative_to(tmp_path),
        manual_checklist_json=paths["manual"].relative_to(tmp_path),
        review_continuation_packet_json=paths["review"].relative_to(tmp_path),
    )

    assert gate["ok"] is True
    assert gate["all_local_artifact_gates_ok"] is True
    assert gate["external_proceedings_metadata_ready"] is True
    assert gate["manual_submission_system_ready"] is True
    assert gate["review_panel_coverage_complete"] is True
    assert gate["final_submission_ready"] is True
    assert gate["verdict"] == "FINAL_SUBMISSION_READY"
    assert gate["remaining_blockers"] == []
