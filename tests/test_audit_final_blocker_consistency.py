import json
from pathlib import Path

from scripts.audit.main_audit_final_blocker_consistency import (
    audit_final_blocker_consistency,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _seed_inputs(tmp_path: Path, *, overrides: dict[str, dict] | None = None) -> dict[str, Path]:
    overrides = overrides or {}
    out = tmp_path / "out"
    final_gate = {
        "ok": True,
        "final_submission_ready": False,
        "review_panel_coverage_complete": False,
        "remaining_blockers": [
            "external_proceedings_metadata_not_ready",
            "manual_submission_system_not_ready",
            "review_panel_coverage_not_complete",
            "explicit_claude_opus_review",
        ],
        "warnings": ["submission_package:underfull_layout_warnings:hbox=6,vbox=8"],
    }
    stack = {
        "ok": True,
        "final_submission_ready": False,
        "local_release_candidate_ready": True,
        "blocking_status": "external_manual_or_review_blocked",
        "remaining_blockers": ["manual_submission_system_not_ready"],
        "warnings": ["pre_submission_gate_refresh:underfull_layout_warnings:hbox=6,vbox=8"],
    }
    closure = {
        "ok": True,
        "final_submission_ready": False,
        "ready_for_human_handoff": True,
        "remaining_blockers": ["explicit_claude_opus_review"],
        "input_paths": {
            "promax_probe_json": {
                "exists": True,
                "path": "out/promax.json",
                "size_bytes": 1,
            }
        },
        "closure_groups": [
            {
                "group_id": "review_panel_coverage",
                "remaining_blockers": [
                    "review_panel_coverage_not_complete",
                    "explicit_claude_opus_review",
                ],
            },
            {
                "group_id": "external_proceedings_metadata",
                "latest_public_probe": {
                    "provided": True,
                    "crossref_status_code": 404,
                    "doi_resolver_status_code": 404,
                    "acm_dl_status_code": 403,
                },
                "remaining_blockers": [
                    "promax:final_page_range_missing_in_bib",
                    "promax:crossref_registry_not_visible",
                    "promax:doi_resolver_not_visible",
                ],
            },
            {
                "group_id": "manual_submission_system",
                "next_commands": [
                    "python -m scripts.audit.main_build_manual_submission_private_confirmation_request_packet --output-json outputs/summary/paper_critical/manual_submission_private_confirmation_request_packet_20260613.json --output-md outputs/summary/paper_critical/manual_submission_private_confirmation_request_packet_20260613.md",
                    "python -m scripts.audit.main_validate_manual_submission_private_confirmation_json --private-confirmation-json path/to/untracked_private_confirmation.json --manual-request-packet-json outputs/summary/paper_critical/manual_submission_private_confirmation_request_packet_20260613.json --output-json outputs/summary/paper_critical/manual_private_confirmation_validation_20260613.json --output-md outputs/summary/paper_critical/manual_private_confirmation_validation_20260613.md",
                    "python -m scripts.audit.main_build_manual_submission_checklist --private-confirmation-json path/to/untracked_private_confirmation.json --output-json outputs/summary/paper_critical/manual_submission_checklist_20260613.json --output-md outputs/summary/paper_critical/manual_submission_checklist_20260613.md",
                ],
            },
        ],
        "warnings": ["underfull_layout_warnings:hbox=6,vbox=8"],
    }
    review = {
        "ok": True,
        "final_submission_ready": False,
        "required_claude_blocker_ack_groups": [
            "manual_submission_system",
            "promax_public_metadata",
        ],
        "reviewer_coverage": {
            "explicit_claude_opus_present": False,
            "final_panel_coverage_complete": False,
            "missing_perspectives": ["explicit_claude_opus_review"],
        },
        "failed_review_attempts": [
            {
                "reviewer": "claude-opus",
                "status": "failed",
                "valid_review_evidence": False,
                "error": "Claude CLI did not return JSON output",
            },
            {
                "reviewer": "claude-opus",
                "status": "failed",
                "valid_review_evidence": False,
                "error": "Claude CLI did not return JSON output",
            },
        ],
        "remaining_blockers": ["explicit_claude_opus_review"],
        "warnings": [],
    }
    claude_request = {
        "ok": True,
        "claude_review_needed": True,
        "failed_claude_attempt_summary": {"count": 2},
        "expected_additional_review_json": {
            "response_template": {
                "reviewer": "claude-opus",
                "valid_review_evidence": False,
                "remaining_blockers_acknowledged": [
                    "promax_public_metadata: final page range, Crossref, and DOI resolver visibility remain open",
                    "manual_submission_system: private submission-system confirmation remains open",
                ],
            },
            "response_template_sha256": "b" * 64,
            "must_count_as_coverage": [
                "remaining_blockers_acknowledged names the ProMax public metadata blocker",
                "remaining_blockers_acknowledged names the private manual submission-system blocker",
            ],
        },
        "warnings": [],
    }
    promax = {
        "ok": True,
        "final_submission_ready": False,
        "promax_public_metadata_ready": False,
        "remaining_blockers": [
            "promax:final_page_range_missing_in_bib",
            "promax:crossref_registry_not_visible",
            "promax:doi_resolver_not_visible",
        ],
        "direct_checks": {
            "crossref": {"status_code": 404},
            "doi_resolver": {"status_code": 404},
            "acm_dl": {"status_code": 403},
        },
        "warnings": ["acm_dl_not_accessible:status=403"],
    }
    manual_request = {
        "ok": True,
        "final_submission_ready": False,
        "request_packet_ready": True,
        "manual_confirmation_needed": True,
        "manual_submission_system_ready": False,
        "required_private_confirmation": {
            "source_manifest_sha256": "a" * 64,
            "completed_item_ids_for_full_manual_gate": ["paste_title", "enter_authors"],
        },
        "private_confirmation_validation_command": (
            "python -m scripts.audit.main_validate_manual_submission_private_confirmation_json "
            "--private-confirmation-json artifacts/private/manual_submission_private_confirmation_20260613.json "
            "--manual-request-packet-json outputs/summary/paper_critical/manual_submission_private_confirmation_request_packet_20260613.json "
            "--output-json outputs/summary/paper_critical/manual_private_confirmation_validation_20260613.json "
            "--output-md outputs/summary/paper_critical/manual_private_confirmation_validation_20260613.md"
        ),
        "follow_up_commands_after_human_completion": [
            "python -m scripts.audit.main_validate_manual_submission_private_confirmation_json --private-confirmation-json artifacts/private/manual_submission_private_confirmation_20260613.json --manual-request-packet-json outputs/summary/paper_critical/manual_submission_private_confirmation_request_packet_20260613.json --output-json outputs/summary/paper_critical/manual_private_confirmation_validation_20260613.json --output-md outputs/summary/paper_critical/manual_private_confirmation_validation_20260613.md",
            "python -m scripts.audit.main_build_manual_submission_checklist --private-confirmation-json artifacts/private/manual_submission_private_confirmation_20260613.json --output-json outputs/summary/paper_critical/manual_submission_checklist_20260613.json --output-md outputs/summary/paper_critical/manual_submission_checklist_20260613.md",
        ],
        "warnings": [],
    }
    payloads = {
        "final_gate": final_gate,
        "stack": stack,
        "closure": closure,
        "review": review,
        "claude_request": claude_request,
        "promax": promax,
        "manual_request": manual_request,
    }
    for key, update in overrides.items():
        payloads[key].update(update)
    return {
        "final": _write_json(out / "final.json", payloads["final_gate"]),
        "stack": _write_json(out / "stack.json", payloads["stack"]),
        "closure": _write_json(out / "closure.json", payloads["closure"]),
        "review": _write_json(out / "review.json", payloads["review"]),
        "claude": _write_json(out / "claude.json", payloads["claude_request"]),
        "promax": _write_json(out / "promax.json", payloads["promax"]),
        "manual": _write_json(out / "manual.json", payloads["manual_request"]),
    }


def _run(paths: dict[str, Path], tmp_path: Path) -> dict:
    return audit_final_blocker_consistency(
        root=tmp_path,
        final_gate_json=paths["final"].relative_to(tmp_path),
        stack_json=paths["stack"].relative_to(tmp_path),
        closure_json=paths["closure"].relative_to(tmp_path),
        review_json=paths["review"].relative_to(tmp_path),
        claude_request_json=paths["claude"].relative_to(tmp_path),
        promax_probe_json=paths["promax"].relative_to(tmp_path),
        manual_request_json=paths["manual"].relative_to(tmp_path),
    )


def test_final_blocker_consistency_audit_passes_on_expected_blocked_state(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)

    audit = _run(paths, tmp_path)

    assert audit["ok"] is True
    assert audit["final_submission_ready"] is False
    assert audit["summary"]["review_failed_claude_attempt_count"] == 2
    assert audit["summary"]["claude_request_failed_attempt_count"] == 2
    assert audit["summary"]["claude_request_has_response_template"] is True
    assert audit["summary"]["claude_request_template_valid_review_evidence"] is False
    assert audit["summary"]["review_required_claude_ack_groups"] == [
        "manual_submission_system",
        "promax_public_metadata",
    ]
    assert audit["summary"]["promax_direct_status"] == {
        "crossref": 404,
        "doi_resolver": 404,
        "acm_dl": 403,
    }
    assert audit["summary"]["manual_request_has_private_confirmation_validator"] is True
    assert audit["summary"]["closure_manual_group_has_private_confirmation_validator"] is True
    assert audit["summary"]["recursive_warning_regression_count"] == 0
    assert "explicit_claude_opus_review" in audit["required_open_blockers"]["review_missing_perspectives"]


def test_final_blocker_consistency_audit_defaults_inputs_to_stamp(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)
    out = tmp_path / "outputs" / "summary" / "paper_critical"
    out.mkdir(parents=True, exist_ok=True)
    stamped_inputs = {
        "final": "final_submission_gate_20260615.json",
        "stack": "submission_release_candidate_stack_refresh_20260615.json",
        "closure": "final_submission_blocker_closure_packet_20260615.json",
        "review": "review_continuation_packet_20260615.json",
        "claude": "claude_opus_review_request_packet_20260615.json",
        "promax": "promax_public_metadata_probe_20260615.json",
        "manual": "manual_submission_private_confirmation_request_packet_20260615.json",
    }
    for key, filename in stamped_inputs.items():
        (out / filename).write_text(paths[key].read_text(encoding="utf-8"), encoding="utf-8")

    audit = audit_final_blocker_consistency(root=tmp_path, stamp="20260615")

    assert audit["ok"] is True
    assert audit["input_paths"]["final_gate"]["path"].replace("\\", "/").endswith(
        "final_submission_gate_20260615.json"
    )
    assert audit["input_paths"]["manual_request"]["path"].replace("\\", "/").endswith(
        "manual_submission_private_confirmation_request_packet_20260615.json"
    )


def test_final_blocker_consistency_audit_accepts_expected_local_repair_state(tmp_path: Path) -> None:
    paths = _seed_inputs(
        tmp_path,
        overrides={
            "final_gate": {"ok": False},
            "stack": {
                "ok": False,
                "local_release_candidate_ready": False,
                "blocking_status": "local_artifact_repair_required",
                "remaining_blockers": [
                    "confirm_anonymous_shell:target_formatting_profile_not_ok",
                    "external_proceedings_metadata_not_ready",
                    "manual_submission_system_not_ready",
                    "explicit_claude_opus_review",
                ],
            },
            "closure": {
                "ok": False,
                "ready_for_human_handoff": False,
                "remaining_blockers": [
                    "confirm_anonymous_shell:target_formatting_profile_not_ok",
                    "external_proceedings_metadata_not_ready",
                    "manual_submission_system_not_ready",
                    "explicit_claude_opus_review",
                ],
            },
            "review": {"ok": False},
        },
    )

    audit = _run(paths, tmp_path)

    assert audit["ok"] is True
    assert audit["final_submission_ready"] is False
    assert audit["summary"]["blocking_status"] == "local_artifact_repair_required"
    assert "confirm_anonymous_shell:target_formatting_profile_not_ok" in audit["required_open_blockers"]["release_stack"]


def test_final_blocker_consistency_audit_rejects_failed_claude_count_mismatch(
    tmp_path: Path,
) -> None:
    paths = _seed_inputs(tmp_path, overrides={"claude_request": {"failed_claude_attempt_summary": {"count": 1}}})

    audit = _run(paths, tmp_path)

    assert audit["ok"] is False
    assert "claude_failed_attempt_count_mismatch:1 != 2" in audit["failures"]


def test_final_blocker_consistency_audit_rejects_recursive_warning_regression(
    tmp_path: Path,
) -> None:
    paths = _seed_inputs(
        tmp_path,
        overrides={
            "stack": {
                "warnings": [
                    "pre_submission_gate_refresh:final_submission_gate:review_continuation:underfull_layout_warnings:hbox=6,vbox=8"
                ]
            }
        },
    )

    audit = _run(paths, tmp_path)

    assert audit["ok"] is False
    assert "recursive_warning_prefix_regressions:1" in audit["failures"]
    assert audit["warning_regressions"][0]["artifact"] == "release_stack"


def test_final_blocker_consistency_audit_rejects_closure_missing_promax_probe(
    tmp_path: Path,
) -> None:
    paths = _seed_inputs(
        tmp_path,
        overrides={
            "closure": {
                "input_paths": {"promax_probe_json": {"exists": False, "path": "", "size_bytes": 0}},
                "closure_groups": [
                    {
                        "group_id": "review_panel_coverage",
                        "remaining_blockers": [
                            "review_panel_coverage_not_complete",
                            "explicit_claude_opus_review",
                        ],
                    },
                    {
                        "group_id": "external_proceedings_metadata",
                        "latest_public_probe": {"provided": False},
                    },
                ],
            }
        },
    )

    audit = _run(paths, tmp_path)

    assert audit["ok"] is False
    assert "closure_missing_promax_probe_input" in audit["failures"]
    assert "closure_missing_latest_public_promax_probe" in audit["failures"]


def test_final_blocker_consistency_audit_rejects_missing_claude_response_template(
    tmp_path: Path,
) -> None:
    paths = _seed_inputs(
        tmp_path,
        overrides={"claude_request": {"expected_additional_review_json": {}}},
    )

    audit = _run(paths, tmp_path)

    assert audit["ok"] is False
    assert "claude_request_missing_response_template" in audit["failures"]
    assert "claude_request_missing_response_template_sha256" in audit["failures"]


def test_final_blocker_consistency_audit_rejects_missing_current_claude_ack_group(
    tmp_path: Path,
) -> None:
    paths = _seed_inputs(
        tmp_path,
        overrides={
            "review": {
                "required_claude_blocker_ack_groups": ["promax_public_metadata"],
            }
        },
    )

    audit = _run(paths, tmp_path)

    assert audit["ok"] is False
    assert "review_missing_required_claude_ack_group:manual_submission_system" in audit["failures"]


def test_final_blocker_consistency_audit_rejects_missing_manual_private_validator(
    tmp_path: Path,
) -> None:
    paths = _seed_inputs(
        tmp_path,
        overrides={
            "manual_request": {
                "private_confirmation_validation_command": "",
                "follow_up_commands_after_human_completion": [
                    "python -m scripts.audit.main_build_manual_submission_checklist --private-confirmation-json artifacts/private/manual_submission_private_confirmation_20260613.json"
                ],
            }
        },
    )

    audit = _run(paths, tmp_path)

    assert audit["ok"] is False
    assert "manual_request_missing_private_confirmation_validator_command" in audit["failures"]
    assert "manual_request_followups_missing_private_confirmation_validator" in audit["failures"]
