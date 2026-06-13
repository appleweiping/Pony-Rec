import json
from pathlib import Path

from scripts.audit.main_audit_claude_review_connector_health import (
    build_claude_review_connector_health_packet,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_connector_health_recommends_external_route_after_repeated_error(tmp_path: Path) -> None:
    request = _write_json(tmp_path / "request.json", {"ok": True, "claude_review_needed": True})
    attempts = []
    for idx in range(3):
        attempts.append(
            _write_json(
                tmp_path / f"attempt_{idx}.json",
                {
                    "created_at_utc": f"2026-06-13T05:0{idx}:00Z",
                    "completed_at_utc": f"2026-06-13T05:0{idx}:01Z",
                    "reviewer": "claude-opus",
                    "status": "failed",
                    "valid_review_evidence": False,
                    "error": "Claude CLI did not return JSON output",
                    "source": f"job {idx}",
                },
            )
        )

    packet = build_claude_review_connector_health_packet(
        root=tmp_path,
        failed_attempt_jsons=[path.relative_to(tmp_path) for path in attempts],
        review_request_packet_json=request.relative_to(tmp_path),
    )

    assert packet["ok"] is True
    assert packet["failed_attempt_count"] == 3
    assert packet["same_error_tail_streak"] == 3
    assert packet["connector_unhealthy"] is True
    assert packet["same_route_retry_recommended"] is False
    assert packet["recommended_next_route"] == "external_claude_opus_json_via_request_packet_and_validator"
    assert packet["final_submission_ready"] is False


def test_connector_health_allows_retry_below_threshold(tmp_path: Path) -> None:
    request = _write_json(tmp_path / "request.json", {"ok": True, "claude_review_needed": True})
    attempt = _write_json(
        tmp_path / "attempt.json",
        {
            "created_at_utc": "2026-06-13T05:00:00Z",
            "completed_at_utc": "2026-06-13T05:00:01Z",
            "reviewer": "claude-opus",
            "status": "failed",
            "valid_review_evidence": False,
            "error": "temporary",
            "source": "job",
        },
    )

    packet = build_claude_review_connector_health_packet(
        root=tmp_path,
        failed_attempt_jsons=[attempt.relative_to(tmp_path)],
        review_request_packet_json=request.relative_to(tmp_path),
    )

    assert packet["connector_unhealthy"] is False
    assert packet["same_route_retry_recommended"] is True
