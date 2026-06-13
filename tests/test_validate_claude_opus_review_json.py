import json
from pathlib import Path

from scripts.audit.main_validate_claude_opus_review_json import (
    build_claude_opus_review_validation_packet,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _seed_request_inputs(tmp_path: Path) -> dict[str, Path]:
    request = _write_json(
        tmp_path / "request.json",
        {
            "claude_review_needed": True,
            "review_continuation_command_after_valid_review": (
                "python -m scripts.audit.main_build_review_continuation_packet "
                "--additional-review-json outputs/summary/paper_critical/claude_opus_review_20260613.json"
            ),
        },
    )
    continuation = _write_json(
        tmp_path / "continuation.json",
        {"review_continuation_ready": True, "final_submission_ready": False},
    )
    return {"request": request, "continuation": continuation}


def _valid_claude_review(**overrides: object) -> dict:
    review = {
        "reviewer": "claude-opus",
        "created_at_utc": "2026-06-13T05:00:00+00:00",
        "source": "external claude review",
        "score_0_to_10": 8.2,
        "verdict": "CONDITIONAL_PASS",
        "claim_boundary_ok": True,
        "final_submission_ready_claim_allowed": False,
        "kill_argument": "Final blockers remain external/manual rather than evidence gaps.",
        "major_concerns": ["ProMax direct metadata remains unresolved."],
        "required_changes": ["Keep final readiness false until final gates close."],
        "remaining_blockers_acknowledged": [
            "promax:doi_resolver_not_visible",
            "manual_submission_system:not_ready_private_confirmation_pending",
        ],
        "valid_review_evidence": True,
    }
    review.update(overrides)
    return review


def test_claude_opus_review_validation_accepts_gate_ready_json(tmp_path: Path) -> None:
    paths = _seed_request_inputs(tmp_path)
    review = _write_json(tmp_path / "claude.json", _valid_claude_review())

    packet = build_claude_opus_review_validation_packet(
        root=tmp_path,
        review_json=review.relative_to(tmp_path),
        review_request_packet_json=paths["request"].relative_to(tmp_path),
        review_continuation_packet_json=paths["continuation"].relative_to(tmp_path),
    )

    assert packet["ok"] is True
    assert packet["ready_to_attach_for_review_gate"] is True
    assert packet["explicit_claude_opus_reviewer"] is True
    assert packet["score_floor_meets_8"] is True
    assert packet["schema_key_state"]["missing_keys"] == []


def test_claude_sonnet_review_does_not_close_explicit_opus_gate(tmp_path: Path) -> None:
    paths = _seed_request_inputs(tmp_path)
    review = _write_json(tmp_path / "claude_sonnet.json", _valid_claude_review(reviewer="claude-sonnet"))

    packet = build_claude_opus_review_validation_packet(
        root=tmp_path,
        review_json=review.relative_to(tmp_path),
        review_request_packet_json=paths["request"].relative_to(tmp_path),
        review_continuation_packet_json=paths["continuation"].relative_to(tmp_path),
    )

    assert packet["ok"] is False
    assert packet["explicit_claude_opus_reviewer"] is False
    assert "claude_reviewer_not_explicit_opus" in packet["failures"]
    assert "reviewer_not_explicit_claude_opus" in packet["failures"]


def test_claude_opus_review_below_8_is_valid_but_not_gate_ready(tmp_path: Path) -> None:
    paths = _seed_request_inputs(tmp_path)
    review = _write_json(tmp_path / "claude_low_score.json", _valid_claude_review(score_0_to_10=7.9))

    packet = build_claude_opus_review_validation_packet(
        root=tmp_path,
        review_json=review.relative_to(tmp_path),
        review_request_packet_json=paths["request"].relative_to(tmp_path),
        review_continuation_packet_json=paths["continuation"].relative_to(tmp_path),
    )

    assert packet["ok"] is False
    assert packet["valid_review_evidence"] is True
    assert packet["score_floor_meets_8"] is False
    assert "score_floor_below_8" in packet["failures"]


def test_claude_opus_review_must_acknowledge_manual_submission_blocker(tmp_path: Path) -> None:
    paths = _seed_request_inputs(tmp_path)
    review = _write_json(
        tmp_path / "claude_missing_manual.json",
        _valid_claude_review(remaining_blockers_acknowledged=["promax:crossref_registry_not_visible"]),
    )

    packet = build_claude_opus_review_validation_packet(
        root=tmp_path,
        review_json=review.relative_to(tmp_path),
        review_request_packet_json=paths["request"].relative_to(tmp_path),
        review_continuation_packet_json=paths["continuation"].relative_to(tmp_path),
    )

    assert packet["ok"] is False
    assert packet["valid_review_evidence"] is False
    assert "claude_remaining_blockers_missing_manual_submission_system" in packet["failures"]
