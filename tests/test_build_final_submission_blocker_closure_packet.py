import json
from pathlib import Path

from scripts.audit.main_build_final_submission_blocker_closure_packet import (
    _infer_stamp_from_output_path,
    _write_md,
    build_final_submission_blocker_closure_packet,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _seed_inputs(tmp_path: Path) -> dict[str, Path]:
    out = tmp_path / "out"
    final_gate = _write_json(
        out / "final_submission_gate_test.json",
        {
            "schema_version": "final.v1",
            "created_at_utc": "2026-06-12T00:00:00+00:00",
            "ok": True,
            "final_submission_ready": False,
            "external_proceedings_metadata_ready": False,
            "manual_submission_system_ready": False,
            "review_continuation_ready": True,
            "review_panel_coverage_complete": False,
            "remaining_blockers": [
                "promax:final_page_range_missing_in_bib",
                "external_proceedings_metadata_not_ready",
                "manual_submission_system_items_not_confirmed",
                "review_panel_coverage_not_complete",
                "explicit_claude_opus_review",
            ],
            "warnings": [],
        },
    )
    external = _write_json(
        out / "external_proceedings_metadata_recheck_test.json",
        {
            "schema_version": "external.v1",
            "created_at_utc": "2026-06-12T00:00:00+00:00",
            "ok": True,
            "external_proceedings_metadata_ready": False,
            "final_submission_ready": False,
            "checked_entries": {
                "promax": {
                    "external_metadata_ready": False,
                    "bibtex": {
                        "doi": "10.1145/3805712.3809600",
                        "pages": "",
                        "numpages": "11",
                        "isbn": "979-8-4007-2599-9",
                        "location": "Melbourne, VIC, Australia",
                        "eprint": "2604.26231",
                        "booktitle": "Proceedings of SIGIR",
                    },
                    "network": {
                        "crossref": {"status_code": 404},
                        "doi_resolver": {"status_code": 404},
                    },
                    "source_checks": [
                        {
                            "name": "arxiv_html_promax_acm_metadata",
                            "ok": True,
                            "status_code": 200,
                            "missing_patterns": [],
                        }
                    ],
                    "blockers": [
                        "promax:final_page_range_missing_in_bib",
                        "promax:crossref_registry_not_visible:status=404",
                        "promax:doi_resolver_not_visible:status=404",
                    ],
                    "warnings": [],
                }
            },
            "remaining_blockers": [
                "promax:final_page_range_missing_in_bib",
                "promax:crossref_registry_not_visible:status=404",
                "promax:doi_resolver_not_visible:status=404",
            ],
        },
    )
    manual = _write_json(
        out / "manual_submission_checklist_test.json",
        {
            "schema_version": "manual.v1",
            "created_at_utc": "2026-06-12T00:00:00+00:00",
            "ok": True,
            "manual_submission_checklist_ready": True,
            "manual_submission_system_ready": False,
            "final_submission_ready": False,
            "crosscheck": {"source_manifest_sha256": "a" * 64},
            "unconfirmed_required_item_ids": ["paste_title", "enter_authors"],
            "manual_private_item_ids": ["enter_authors"],
            "private_confirmation": {"provided": False},
            "remaining_blockers": ["manual_submission_system_items_not_confirmed"],
        },
    )
    stack = _write_json(
        out / "submission_release_candidate_stack_refresh_test.json",
        {
            "schema_version": "stack.v1",
            "created_at_utc": "2026-06-12T00:00:00+00:00",
            "ok": True,
            "local_release_candidate_ready": True,
            "readiness_scope": "local_artifacts_only",
            "blocking_status": "external_or_manual_blocked",
            "final_submission_ready": False,
            "failures": [],
            "warnings": [],
            "remaining_blockers": [
                "external_proceedings_metadata_not_ready",
                "manual_submission_system_not_ready",
            ],
        },
    )
    probe = _write_json(
        out / "promax_public_metadata_probe_test.json",
        {
            "schema_version": "probe.v1",
            "created_at_utc": "2026-06-13T00:00:00+00:00",
            "ok": True,
            "promax_public_metadata_ready": False,
            "final_submission_ready": False,
            "direct_checks": {
                "crossref": {"status_code": 404},
                "doi_resolver": {"status_code": 404},
                "acm_dl": {"status_code": 403},
            },
            "source_probes": [
                {
                    "name": "arxiv_html_promax_acm_metadata",
                    "ok": True,
                    "status_code": 200,
                    "missing_patterns": [],
                }
            ],
            "remaining_blockers": ["promax:crossref_registry_not_visible"],
            "warnings": ["acm_dl_not_accessible:status=403"],
        },
    )
    return {
        "final": final_gate,
        "external": external,
        "manual": manual,
        "stack": stack,
        "probe": probe,
    }


def test_final_submission_blocker_closure_packet_groups_external_and_manual(
    tmp_path: Path,
) -> None:
    paths = _seed_inputs(tmp_path)

    packet = build_final_submission_blocker_closure_packet(
        root=tmp_path,
        final_gate_json=paths["final"].relative_to(tmp_path),
        external_metadata_json=paths["external"].relative_to(tmp_path),
        manual_checklist_json=paths["manual"].relative_to(tmp_path),
        release_candidate_stack_json=paths["stack"].relative_to(tmp_path),
        promax_probe_json=paths["probe"].relative_to(tmp_path),
    )

    assert packet["ok"] is True
    assert packet["closure_packet_ready"] is True
    assert packet["local_release_candidate_ready"] is True
    assert packet["final_submission_ready"] is False
    assert packet["review_panel_coverage_complete"] is False
    assert packet["ready_for_human_handoff"] is True
    assert "promax:final_page_range_missing_in_bib" in packet["classified_remaining_blockers"][
        "external_proceedings_metadata"
    ]
    assert "manual_submission_system_items_not_confirmed" in packet["classified_remaining_blockers"][
        "manual_submission_system"
    ]
    assert "explicit_claude_opus_review" in packet["classified_remaining_blockers"][
        "review_panel_coverage"
    ]
    groups = {item["group_id"]: item for item in packet["closure_groups"]}
    assert groups["review_panel_coverage"]["status"] == "blocked"
    assert groups["external_proceedings_metadata"]["status"] == "blocked"
    assert groups["manual_submission_system"]["status"] == "manual_private_pending"
    assert groups["manual_submission_system"]["public_safe"] is False
    assert groups["manual_submission_system"]["current_evidence"]["source_manifest_sha256"] == "a" * 64
    assert (
        groups["external_proceedings_metadata"]["current_evidence"]["bibtex"]["isbn"]
        == "979-8-4007-2599-9"
    )
    probe = groups["external_proceedings_metadata"]["latest_public_probe"]
    assert probe["provided"] is True
    assert probe["created_at_utc"] == "2026-06-13T00:00:00+00:00"
    assert probe["crossref_status_code"] == 404


def test_final_submission_blocker_closure_packet_markdown_has_commands(
    tmp_path: Path,
) -> None:
    paths = _seed_inputs(tmp_path)
    packet = build_final_submission_blocker_closure_packet(
        root=tmp_path,
        final_gate_json=paths["final"].relative_to(tmp_path),
        external_metadata_json=paths["external"].relative_to(tmp_path),
        manual_checklist_json=paths["manual"].relative_to(tmp_path),
        release_candidate_stack_json=paths["stack"].relative_to(tmp_path),
        promax_probe_json=paths["probe"].relative_to(tmp_path),
    )
    output = tmp_path / "packet.md"

    _write_md(output, packet)

    text = output.read_text(encoding="utf-8")
    assert "Final Submission Blocker Closure Packet" in text
    assert "external_proceedings_metadata" in text
    assert "manual_submission_system" in text
    assert "review_panel_coverage" in text
    assert "main_build_review_continuation_packet" in text
    assert "main_build_manual_submission_private_confirmation_request_packet" in text
    assert "Source manifest sha256: `" + "a" * 64 + "`" in text
    assert "main_probe_promax_public_metadata" in text
    assert "main_refresh_submission_release_candidate_stack" in text
    assert "Latest public probe" in text
    assert "Latest public source probes" in text
    assert "arxiv_html_promax_acm_metadata" in text
    assert "ACM DL status: `403`" in text


def test_infer_stamp_from_output_path_uses_dated_artifact_name() -> None:
    assert (
        _infer_stamp_from_output_path(
            "outputs/summary/paper_critical/final_submission_blocker_closure_packet_20260613.json"
        )
        == "20260613"
    )
    assert _infer_stamp_from_output_path("outputs/summary/paper_critical/closure_packet_latest.json") is None


def test_final_submission_blocker_closure_packet_reads_default_promax_probe_by_stamp(
    tmp_path: Path,
) -> None:
    paths = _seed_inputs(tmp_path)
    default_probe = (
        tmp_path
        / "outputs"
        / "summary"
        / "paper_critical"
        / "promax_public_metadata_probe_20260613.json"
    )
    default_probe.parent.mkdir(parents=True, exist_ok=True)
    default_probe.write_text(paths["probe"].read_text(encoding="utf-8"), encoding="utf-8")

    packet = build_final_submission_blocker_closure_packet(
        root=tmp_path,
        stamp="20260613",
        final_gate_json=paths["final"].relative_to(tmp_path),
        external_metadata_json=paths["external"].relative_to(tmp_path),
        manual_checklist_json=paths["manual"].relative_to(tmp_path),
        release_candidate_stack_json=paths["stack"].relative_to(tmp_path),
    )

    groups = {item["group_id"]: item for item in packet["closure_groups"]}
    probe = groups["external_proceedings_metadata"]["latest_public_probe"]
    assert packet["input_paths"]["promax_probe_json"]["exists"] is True
    assert packet["input_paths"]["promax_probe_json"]["path"].endswith(
        "outputs\\summary\\paper_critical\\promax_public_metadata_probe_20260613.json"
    ) or packet["input_paths"]["promax_probe_json"]["path"].endswith(
        "outputs/summary/paper_critical/promax_public_metadata_probe_20260613.json"
    )
    assert probe["provided"] is True
    assert probe["acm_dl_status_code"] == 403
