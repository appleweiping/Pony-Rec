import json
from pathlib import Path

from scripts.audit.main_build_final_submission_blocker_closure_packet import (
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
            "remaining_blockers": [
                "promax:final_page_range_missing_in_bib",
                "external_proceedings_metadata_not_ready",
                "manual_submission_system_items_not_confirmed",
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
    return {
        "final": final_gate,
        "external": external,
        "manual": manual,
        "stack": stack,
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
    )

    assert packet["ok"] is True
    assert packet["closure_packet_ready"] is True
    assert packet["local_release_candidate_ready"] is True
    assert packet["final_submission_ready"] is False
    assert packet["ready_for_human_handoff"] is True
    assert "promax:final_page_range_missing_in_bib" in packet["classified_remaining_blockers"][
        "external_proceedings_metadata"
    ]
    assert "manual_submission_system_items_not_confirmed" in packet["classified_remaining_blockers"][
        "manual_submission_system"
    ]
    groups = {item["group_id"]: item for item in packet["closure_groups"]}
    assert groups["external_proceedings_metadata"]["status"] == "blocked"
    assert groups["manual_submission_system"]["status"] == "manual_private_pending"
    assert groups["manual_submission_system"]["public_safe"] is False
    assert groups["manual_submission_system"]["current_evidence"]["source_manifest_sha256"] == "a" * 64
    assert (
        groups["external_proceedings_metadata"]["current_evidence"]["bibtex"]["isbn"]
        == "979-8-4007-2599-9"
    )


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
    )
    output = tmp_path / "packet.md"

    _write_md(output, packet)

    text = output.read_text(encoding="utf-8")
    assert "Final Submission Blocker Closure Packet" in text
    assert "external_proceedings_metadata" in text
    assert "manual_submission_system" in text
    assert "Source manifest sha256: `" + "a" * 64 + "`" in text
    assert "main_probe_promax_public_metadata" in text
    assert "main_refresh_submission_release_candidate_stack" in text
