import json
from pathlib import Path

from scripts.audit.main_build_review_continuation_packet import build_review_continuation_packet


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _seed_inputs(tmp_path: Path) -> dict[str, Path]:
    panel = _write_json(
        tmp_path / "panel.json",
        {
            "reviewer_consensus": {
                "score_floor": "8.0/10",
                "claim_boundary_ok": True,
                "final_submission_ready": False,
            },
            "panel_reviews": [
                {"reviewer": "Faraday existing subagent", "score_10": "8.0/10"},
                {"reviewer": "Meitner existing subagent", "score_10": "8.0/10"},
            ],
        },
    )
    claim = _write_json(
        tmp_path / "claim.json",
        {
            "ok": True,
            "paper_evidence_ready_for_drafting": True,
            "final_submission_ready": False,
            "remaining_blockers": [],
        },
    )
    package = _write_json(
        tmp_path / "package.json",
        {
            "ok": True,
            "final_submission_ready": False,
            "evidence_gates": {"claim_audit_ok": True, "panel_review_ok": True},
            "remaining_blockers": ["promax:final_page_range_missing_in_bib"],
        },
    )
    stack = _write_json(
        tmp_path / "stack.json",
        {
            "ok": True,
            "local_release_candidate_ready": True,
            "final_submission_ready": False,
            "remaining_blockers": ["manual_submission_system_not_ready"],
        },
    )
    closure = _write_json(
        tmp_path / "closure.json",
        {
            "ok": True,
            "closure_packet_ready": True,
            "ready_for_human_handoff": True,
            "final_submission_ready": False,
            "external_proceedings_metadata_ready": False,
            "manual_submission_system_ready": False,
            "classified_remaining_blockers": {
                "external_proceedings_metadata": ["promax:crossref_registry_not_visible:status=404"],
                "manual_submission_system": ["manual_submission_system_items_not_confirmed"],
                "other": [],
            },
            "remaining_blockers": ["external_proceedings_metadata_not_ready"],
        },
    )
    promax = _write_json(
        tmp_path / "promax.json",
        {
            "ok": True,
            "promax_public_metadata_ready": False,
            "final_submission_ready": False,
            "remaining_blockers": ["promax:doi_resolver_not_visible"],
        },
    )
    return {
        "panel": panel,
        "claim": claim,
        "package": package,
        "stack": stack,
        "closure": closure,
        "promax": promax,
    }


def test_review_continuation_ready_but_missing_claude_is_blocker(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)

    packet = build_review_continuation_packet(
        root=tmp_path,
        panel_review_json=paths["panel"].relative_to(tmp_path),
        claim_audit_json=paths["claim"].relative_to(tmp_path),
        submission_package_audit_json=paths["package"].relative_to(tmp_path),
        release_candidate_stack_json=paths["stack"].relative_to(tmp_path),
        closure_packet_json=paths["closure"].relative_to(tmp_path),
        promax_probe_json=paths["promax"].relative_to(tmp_path),
    )

    assert packet["ok"] is True
    assert packet["review_continuation_ready"] is True
    assert packet["final_submission_ready"] is False
    assert packet["final_panel_coverage_complete"] is False
    assert "explicit_claude_opus_review" in packet["remaining_blockers"]
    assert packet["gate_summary"]["release_candidate_stack_ok"] is True


def test_review_continuation_accepts_additional_claude_review(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)
    claude = _write_json(
        tmp_path / "claude.json",
        {
            "reviewer": "claude-opus",
            "score_0_to_10": 8.1,
            "verdict": "CONDITIONAL_PASS",
            "valid_review_evidence": True,
            "claim_boundary_ok": True,
            "final_submission_ready_claim_allowed": False,
            "kill_argument": "Final blockers remain external/manual, not method evidence.",
            "major_concerns": ["ProMax direct metadata remains unresolved."],
            "required_changes": ["Keep final readiness false until final gates close."],
            "remaining_blockers_acknowledged": ["promax:doi_resolver_not_visible"],
        },
    )

    packet = build_review_continuation_packet(
        root=tmp_path,
        panel_review_json=paths["panel"].relative_to(tmp_path),
        claim_audit_json=paths["claim"].relative_to(tmp_path),
        submission_package_audit_json=paths["package"].relative_to(tmp_path),
        release_candidate_stack_json=paths["stack"].relative_to(tmp_path),
        closure_packet_json=paths["closure"].relative_to(tmp_path),
        promax_probe_json=paths["promax"].relative_to(tmp_path),
        additional_review_jsons=[claude.relative_to(tmp_path)],
    )

    assert packet["ok"] is True
    assert packet["final_panel_coverage_complete"] is True
    assert packet["reviewer_coverage"]["explicit_claude_opus_present"] is True
    assert packet["reviewer_coverage"]["score_floor_0_to_10"] == 8.0
    assert packet["additional_review_validation"] == [
        {
            "index": 0,
            "reviewer": "claude-opus",
            "counted_for_coverage": True,
            "validation_failures": [],
        }
    ]


def test_minimal_claude_additional_review_is_not_coverage(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)
    claude = _write_json(
        tmp_path / "claude_minimal.json",
        {"reviewer": "claude-opus", "score_0_to_10": 8.1, "verdict": "CONDITIONAL_PASS"},
    )

    packet = build_review_continuation_packet(
        root=tmp_path,
        panel_review_json=paths["panel"].relative_to(tmp_path),
        claim_audit_json=paths["claim"].relative_to(tmp_path),
        submission_package_audit_json=paths["package"].relative_to(tmp_path),
        release_candidate_stack_json=paths["stack"].relative_to(tmp_path),
        closure_packet_json=paths["closure"].relative_to(tmp_path),
        promax_probe_json=paths["promax"].relative_to(tmp_path),
        additional_review_jsons=[claude.relative_to(tmp_path)],
    )

    assert packet["ok"] is True
    assert packet["final_panel_coverage_complete"] is False
    assert packet["reviewer_coverage"]["explicit_claude_opus_present"] is False
    assert packet["reviewer_coverage"]["additional_review_count"] == 0
    validation = packet["additional_review_validation"][0]
    assert validation["counted_for_coverage"] is False
    assert "claude_valid_review_evidence_not_true" in validation["validation_failures"]
    assert "claude_claim_boundary_ok_not_true" in validation["validation_failures"]
    assert "explicit_claude_opus_review" in packet["remaining_blockers"]


def test_failed_claude_attempt_is_recorded_but_not_coverage(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)
    failed = _write_json(
        tmp_path / "failed_claude.json",
        {
            "reviewer": "claude-opus",
            "status": "failed",
            "valid_review_evidence": False,
            "error": "Claude CLI did not return JSON output",
            "source": "claude_review.review_start job test",
        },
    )

    packet = build_review_continuation_packet(
        root=tmp_path,
        panel_review_json=paths["panel"].relative_to(tmp_path),
        claim_audit_json=paths["claim"].relative_to(tmp_path),
        submission_package_audit_json=paths["package"].relative_to(tmp_path),
        release_candidate_stack_json=paths["stack"].relative_to(tmp_path),
        closure_packet_json=paths["closure"].relative_to(tmp_path),
        promax_probe_json=paths["promax"].relative_to(tmp_path),
        failed_review_attempt_jsons=[failed.relative_to(tmp_path)],
    )

    assert packet["ok"] is True
    assert packet["final_panel_coverage_complete"] is False
    assert packet["reviewer_coverage"]["explicit_claude_opus_present"] is False
    assert packet["failed_review_attempts"] == [
        {
            "reviewer": "claude-opus",
            "status": "failed",
            "valid_review_evidence": False,
            "error": "Claude CLI did not return JSON output",
            "source": "claude_review.review_start job test",
        }
    ]
