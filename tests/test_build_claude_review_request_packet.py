import json
from pathlib import Path

from scripts.audit.main_build_claude_review_request_packet import (
    build_claude_review_request_packet,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_claude_review_request_packet_preserves_missing_claude_gap(tmp_path: Path) -> None:
    review_packet = _write_json(
        tmp_path / "review_packet.json",
        {
            "review_continuation_ready": True,
            "final_submission_ready": False,
            "local_release_candidate_ready": True,
            "reviewer_coverage": {
                "score_floor_0_to_10": 8.0,
                "explicit_claude_opus_present": False,
                "missing_perspectives": ["explicit_claude_opus_review"],
                "reviewer_names": ["Faraday existing subagent", "gpt-5.5-xhigh"],
            },
            "gate_summary": {
                "promax_public_metadata_ready": False,
                "external_proceedings_metadata_ready": False,
                "manual_submission_system_ready": False,
            },
            "classified_remaining_blockers": {
                "external_proceedings_metadata": ["promax:doi_resolver_not_visible:status=404"],
                "manual_submission_system": ["manual_submission_system_items_not_confirmed"],
                "other": [],
            },
            "failed_review_attempts": [
                {
                    "reviewer": "claude-opus",
                    "status": "failed",
                    "valid_review_evidence": False,
                    "error": "Claude CLI did not return JSON output",
                    "source": "test",
                }
            ],
            "failed_review_attempt_paths": [
                {"path": "outputs\\summary\\paper_critical\\claude_opus_review_attempt_20260613.json"},
                {"path": "outputs\\summary\\paper_critical\\claude_opus_review_attempt_retry_20260613.json"},
            ],
        },
    )
    panel = _write_json(
        tmp_path / "panel.json",
        {
            "reviewer_consensus": {
                "score_floor": "8.0/10",
                "claim_boundary_ok": True,
                "final_submission_ready": False,
            }
        },
    )
    claim = _write_json(
        tmp_path / "claim.json",
        {
            "ok": True,
            "paper_evidence_ready_for_drafting": True,
            "final_submission_ready": False,
            "claims": [
                {
                    "id": "C5",
                    "status": "CONTRADICTED",
                    "allowed_wording": "Do not make this claim.",
                    "forbidden_wording": "every component is necessary",
                }
            ],
        },
    )
    gpt55 = _write_json(
        tmp_path / "gpt55.json",
        {
            "reviewer": "gpt-5.5-xhigh",
            "score_0_to_10": 8.0,
            "verdict": "CONDITIONAL_PASS",
            "critical_blockers": ["Final submission gate is explicitly false."],
        },
    )

    packet = build_claude_review_request_packet(
        root=tmp_path,
        review_continuation_packet_json=review_packet.relative_to(tmp_path),
        panel_review_json=panel.relative_to(tmp_path),
        claim_audit_json=claim.relative_to(tmp_path),
        gpt55_review_json=gpt55.relative_to(tmp_path),
    )

    assert packet["ok"] is True
    assert packet["claude_review_needed"] is True
    assert packet["failed_claude_attempt_summary"]["count"] == 1
    assert packet["expected_additional_review_json"]["recommended_path"].endswith("claude_opus_review_20260613.json")
    assert "valid_review_evidence" in packet["expected_additional_review_json"]["schema"]
    assert "main_audit_claude_review_connector_health" in packet["connector_health_command_before_retry"]
    assert "main_validate_claude_opus_review_json" in packet["validation_command_before_attach"]
    assert "promax:doi_resolver_not_visible:status=404" in packet["claude_review_prompt"]
    assert "final_submission_ready_claim_allowed" in packet["claude_review_prompt"]
    assert "claude_opus_review_attempt_20260613.json" in packet["review_continuation_command_after_valid_review"]
    assert "claude_opus_review_attempt_retry_20260613.json" in packet["review_continuation_command_after_valid_review"]
    assert packet["will_start_experiment"] is False


def test_claude_review_request_packet_fails_closed_on_not_ready_review_packet(tmp_path: Path) -> None:
    review_packet = _write_json(
        tmp_path / "review_packet.json",
        {
            "review_continuation_ready": False,
            "final_submission_ready": False,
            "reviewer_coverage": {
                "explicit_claude_opus_present": False,
                "missing_perspectives": ["explicit_claude_opus_review"],
            },
        },
    )
    panel = _write_json(tmp_path / "panel.json", {})
    claim = _write_json(tmp_path / "claim.json", {})
    gpt55 = _write_json(tmp_path / "gpt55.json", {})

    packet = build_claude_review_request_packet(
        root=tmp_path,
        review_continuation_packet_json=review_packet.relative_to(tmp_path),
        panel_review_json=panel.relative_to(tmp_path),
        claim_audit_json=claim.relative_to(tmp_path),
        gpt55_review_json=gpt55.relative_to(tmp_path),
    )

    assert packet["ok"] is False
    assert "review_continuation_packet_not_ready" in packet["failures"]
