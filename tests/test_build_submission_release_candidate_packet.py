import json
from pathlib import Path

from scripts.audit.main_build_submission_release_candidate_packet import (
    build_submission_release_candidate_packet,
)


MANIFEST_SHA = "4f2a9856f722c98ffaf6b7073af27f6890c3086fffe23fa596ebe9fc62aa3cfa"


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _seed_inputs(tmp_path: Path) -> dict[str, Path]:
    final_gate = _write_json(
        tmp_path / "final_gate.json",
        {
            "schema_version": "final_gate.v1",
            "created_at_utc": "2026-06-12T00:00:00Z",
            "ok": True,
            "all_local_artifact_gates_ok": True,
            "external_proceedings_metadata_ready": False,
            "manual_submission_system_ready": False,
            "review_continuation_ready": True,
            "review_panel_coverage_complete": False,
            "final_submission_ready": False,
            "warnings": [],
            "failures": [],
            "remaining_blockers": [
                "external_proceedings_metadata_not_ready",
                "manual_submission_system_not_ready",
                "review_panel_coverage_not_complete",
                "explicit_claude_opus_review",
            ],
        },
    )
    freshness = _write_json(
        tmp_path / "freshness.json",
        {
            "schema_version": "freshness.v1",
            "created_at_utc": "2026-06-12T00:00:00Z",
            "ok": True,
            "refresh_artifact_fresh": True,
            "checked_input_fingerprint_count": 21,
            "checked_step_file_count": 14,
            "input_fingerprint_mismatch_count": 0,
            "generated_step_file_mismatch_count": 0,
            "refresh_final_verdict": "LOCAL_PACKAGE_READY_BUT_EXTERNAL_OR_MANUAL_BLOCKED",
            "final_submission_ready": False,
            "generated_step_file_checks": [
                {
                    "owner": "final_submission_gate",
                    "path": "final_gate.json",
                    "matches": True,
                    "record_type": "step_json",
                    "mismatches": [],
                }
            ],
            "warnings": [],
            "failures": [],
            "remaining_blockers": ["manual_submission_system_not_ready"],
        },
    )
    source_package = _write_json(
        tmp_path / "source_package.json",
        {
            "schema_version": "source_package.v1",
            "created_at_utc": "2026-06-12T00:00:00Z",
            "ok": True,
            "submission_source_package_ready": True,
            "final_submission_ready": False,
            "copied_manifest": {
                "file_count": 21,
                "total_bytes": 652691,
                "manifest_sha256": MANIFEST_SHA,
            },
            "source_audit_crosscheck": {
                "source_manifest_sha256": MANIFEST_SHA,
            },
            "output": {
                "files_dir": "artifacts/submission_source_package_20260612/files",
            },
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
            "commands": [{"returncode": 0}, {"returncode": 0}, {"returncode": 0}, {"returncode": 0}],
            "source_package_crosscheck": {
                "copied_manifest_sha256": MANIFEST_SHA,
                "source_manifest_sha256": MANIFEST_SHA,
                "verified_file_count": 21,
            },
            "build": {
                "actual_pdf_page_count": 9,
                "page_count": 9,
                "pdf": {"size_bytes": 546669},
                "bibtex_warning_count": 0,
                "overfull_hbox_count": 0,
            },
            "warnings": ["rebuilt_underfull_layout_warnings:hbox=6,vbox=8"],
            "failures": [],
            "remaining_blockers": ["promax:crossref_registry_not_visible:status=404"],
        },
    )
    metadata = _write_json(
        tmp_path / "metadata.json",
        {
            "schema_version": "metadata.v1",
            "created_at_utc": "2026-06-12T00:00:00Z",
            "ok": True,
            "submission_metadata_packet_ready": True,
            "final_submission_ready": False,
            "package_crosscheck": {
                "source_manifest_sha256": MANIFEST_SHA,
                "pdf_pages": 9,
            },
            "submission_fields": {
                "title": "Actionable Uncertainty for LLM-Based Recommendation",
                "abstract_word_count": 212,
                "keywords": ["LLM-based recommendation", "uncertainty estimation"],
                "topic_areas": ["Recommender systems", "Evaluation and reproducibility"],
            },
            "warnings": [],
            "failures": [],
            "remaining_blockers": [],
        },
    )
    manual = _write_json(
        tmp_path / "manual.json",
        {
            "schema_version": "manual.v1",
            "created_at_utc": "2026-06-12T00:00:00Z",
            "ok": True,
            "manual_submission_checklist_ready": True,
            "manual_submission_system_ready": False,
            "final_submission_ready": False,
            "item_count": 14,
            "unconfirmed_required_item_ids": ["select_track_and_paper_type"],
            "private_confirmation": {"exists": False},
            "warnings": [],
            "failures": [],
            "remaining_blockers": ["manual_submission_system_items_not_confirmed"],
        },
    )
    external = _write_json(
        tmp_path / "external.json",
        {
            "schema_version": "external.v1",
            "created_at_utc": "2026-06-12T00:00:00Z",
            "ok": True,
            "external_proceedings_metadata_ready": False,
            "final_submission_ready": False,
            "checked_entry_count": 2,
            "network_mode": "live_public_metadata",
            "warnings": ["proex:crossref_not_visible:status=404"],
            "failures": [],
            "remaining_blockers": [
                "promax:final_page_range_missing_in_bib",
                "promax:doi_resolver_not_visible:status=404",
            ],
        },
    )
    return {
        "final_gate": final_gate,
        "freshness": freshness,
        "source_package": source_package,
        "source_rebuild": source_rebuild,
        "metadata": metadata,
        "manual": manual,
        "external": external,
    }


def _build(tmp_path: Path, paths: dict[str, Path]) -> dict:
    return build_submission_release_candidate_packet(
        root=tmp_path,
        final_submission_gate_json=paths["final_gate"].relative_to(tmp_path),
        refresh_freshness_json=paths["freshness"].relative_to(tmp_path),
        submission_source_package_json=paths["source_package"].relative_to(tmp_path),
        submission_source_package_rebuild_json=paths["source_rebuild"].relative_to(tmp_path),
        submission_metadata_packet_json=paths["metadata"].relative_to(tmp_path),
        manual_checklist_json=paths["manual"].relative_to(tmp_path),
        external_metadata_audit_json=paths["external"].relative_to(tmp_path),
    )


def test_release_candidate_ready_while_final_submission_blocked(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)

    packet = _build(tmp_path, paths)

    assert packet["ok"] is True
    assert packet["local_release_candidate_ready"] is True
    assert packet["final_submission_ready"] is False
    assert packet["verdict"] == "LOCAL_RELEASE_CANDIDATE_READY_FINAL_BLOCKED"
    assert packet["readiness_scope"] == "local_artifacts_only"
    assert packet["blocking_status"] == "external_manual_or_review_blocked"
    assert packet["external_proceedings_metadata_ready"] is False
    assert packet["manual_submission_checklist_ready"] is True
    assert packet["manual_submission_system_ready"] is False
    assert packet["review_panel_coverage_complete"] is False
    assert packet["source_manifest_crosscheck"]["all_match"] is True
    assert packet["input_stamp_check"]["all_dated_inputs_share_stamp"] is True
    assert packet["source_rebuild_summary"]["command_check"]["all_returncode_zero"] is True
    assert "manual_submission_system_not_ready" in packet["remaining_blockers"]
    assert "external_proceedings_metadata_not_ready" in packet["remaining_blockers"]
    assert "review_panel_coverage_not_complete" in packet["remaining_blockers"]
    assert "explicit_claude_opus_review" in packet["remaining_blockers"]
    assert "copied exactly from final_submission_gate" in packet["final_submission_ready_policy"]
    assert not packet["failures"]


def test_release_candidate_fails_on_freshness_mismatch(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)
    payload = json.loads(paths["freshness"].read_text(encoding="utf-8"))
    payload["input_fingerprint_mismatch_count"] = 1
    payload["failures"] = ["input_fingerprint_mismatch:Paper/main.tex"]
    paths["freshness"].write_text(json.dumps(payload), encoding="utf-8")

    packet = _build(tmp_path, paths)

    assert packet["ok"] is False
    assert packet["local_release_candidate_ready"] is False
    assert "pre_submission_refresh_freshness:input_fingerprint_mismatches_present" in packet["failures"]
    assert "pre_submission_refresh_freshness:input_fingerprint_mismatch:Paper/main.tex" in packet["failures"]


def test_release_candidate_fails_on_nonzero_rebuild_command(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)
    payload = json.loads(paths["source_rebuild"].read_text(encoding="utf-8"))
    payload["commands"][2]["returncode"] = 1
    paths["source_rebuild"].write_text(json.dumps(payload), encoding="utf-8")

    packet = _build(tmp_path, paths)

    assert packet["ok"] is False
    assert packet["local_release_candidate_ready"] is False
    assert "submission_source_package_rebuild:build_commands_not_all_zero" in packet["failures"]
    assert "submission_source_package_rebuild:command_2:returncode=1" in packet["failures"]


def test_release_candidate_fails_on_manifest_sha_mismatch(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)
    payload = json.loads(paths["source_rebuild"].read_text(encoding="utf-8"))
    payload["source_package_crosscheck"]["copied_manifest_sha256"] = (
        "0" * 64
    )
    paths["source_rebuild"].write_text(json.dumps(payload), encoding="utf-8")

    packet = _build(tmp_path, paths)

    assert packet["ok"] is False
    assert packet["source_manifest_crosscheck"]["all_match"] is False
    assert "source_manifest_sha256_values_do_not_all_match" in packet["failures"]


def test_release_candidate_mirrors_final_ready_from_final_gate(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)
    final_gate = json.loads(paths["final_gate"].read_text(encoding="utf-8"))
    final_gate["external_proceedings_metadata_ready"] = True
    final_gate["manual_submission_system_ready"] = True
    final_gate["review_panel_coverage_complete"] = True
    final_gate["final_submission_ready"] = True
    final_gate["remaining_blockers"] = []
    paths["final_gate"].write_text(json.dumps(final_gate), encoding="utf-8")

    packet = _build(tmp_path, paths)

    assert packet["ok"] is True
    assert packet["local_release_candidate_ready"] is True
    assert packet["final_submission_ready"] is True
    assert packet["verdict"] == "FINAL_SUBMISSION_READY_FROM_FINAL_GATE"
    assert packet["blocking_status"] == "none"


def test_release_candidate_fails_when_final_gate_local_artifacts_not_ok(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)
    final_gate = json.loads(paths["final_gate"].read_text(encoding="utf-8"))
    final_gate["all_local_artifact_gates_ok"] = False
    paths["final_gate"].write_text(json.dumps(final_gate), encoding="utf-8")

    packet = _build(tmp_path, paths)

    assert packet["ok"] is False
    assert packet["local_release_candidate_ready"] is False
    assert "final_submission_gate:all_local_artifact_gates_ok_not_true" in packet["failures"]


def test_release_candidate_fails_on_non_final_subgate_final_ready(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)
    payload = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    payload["final_submission_ready"] = True
    paths["metadata"].write_text(json.dumps(payload), encoding="utf-8")

    packet = _build(tmp_path, paths)

    assert packet["ok"] is False
    assert packet["local_release_candidate_ready"] is False
    assert "submission_metadata_packet:unexpected_local_final_submission_ready" in packet["failures"]


def test_release_candidate_fails_on_input_stamp_mismatch(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)
    final_renamed = tmp_path / "final_gate_20260612.json"
    paths["final_gate"].rename(final_renamed)
    paths["final_gate"] = final_renamed
    renamed = tmp_path / "freshness_20260613.json"
    paths["freshness"].rename(renamed)
    paths["freshness"] = renamed

    packet = _build(tmp_path, paths)

    assert packet["ok"] is False
    assert packet["local_release_candidate_ready"] is False
    assert "input_gate_stamp_mismatch" in packet["failures"]
