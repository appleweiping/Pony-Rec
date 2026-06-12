from pathlib import Path

from scripts.audit.main_refresh_submission_release_candidate_stack import (
    refresh_submission_release_candidate_stack,
)


def _refresh_payload() -> dict:
    return {
        "schema_version": "refresh.v1",
        "created_at_utc": "2026-06-12T00:00:00+00:00",
        "ok": True,
        "final_submission_ready": False,
        "final_verdict": "LOCAL_PACKAGE_READY_BUT_EXTERNAL_OR_MANUAL_BLOCKED",
        "external_network_mode": "fixture",
        "stamp": "test",
        "git_state_before_refresh": {
            "head": "abc",
            "tracked_dirty": False,
        },
        "input_fingerprints": [],
        "steps": [
            {
                "step_id": "final_submission_gate",
                "json": {"path": "out/final_submission_gate_test.json"},
                "ok": True,
                "ready": False,
            }
        ],
        "failures": [],
        "warnings": [],
        "remaining_blockers": ["manual_submission_system_not_ready"],
        "next_actions": ["Refresh again after changes."],
    }


def _freshness_payload(*, ok: bool = True) -> dict:
    return {
        "schema_version": "freshness.v1",
        "created_at_utc": "2026-06-12T00:00:00+00:00",
        "ok": ok,
        "refresh_artifact_fresh": ok,
        "final_submission_ready": False,
        "refresh_json": "out/pre_submission_gate_refresh_test.json",
        "refresh_final_verdict": "LOCAL_PACKAGE_READY_BUT_EXTERNAL_OR_MANUAL_BLOCKED",
        "checked_input_fingerprint_count": 0,
        "checked_step_file_count": 0,
        "input_fingerprint_mismatch_count": 0 if ok else 1,
        "generated_step_file_mismatch_count": 0,
        "git_state_policy": "Git HEAD is provenance.",
        "refresh_git_state_before_refresh": {"head": "abc"},
        "current_git_state": {"head": "abc"},
        "git_head_changed_since_refresh_generation": False,
        "input_fingerprint_checks": [],
        "generated_step_file_checks": [],
        "failures": [] if ok else ["input_fingerprint_mismatch:Paper/main.tex"],
        "warnings": [],
        "remaining_blockers": ["manual_submission_system_not_ready"],
        "next_actions": ["Rerun refresh."],
    }


def _candidate_payload() -> dict:
    return {
        "schema_version": "candidate.v1",
        "created_at_utc": "2026-06-12T00:00:00+00:00",
        "ok": True,
        "local_release_candidate_ready": True,
        "readiness_scope": "local_artifacts_only",
        "blocking_status": "external_or_manual_blocked",
        "final_submission_ready": False,
        "final_submission_ready_source": "out/final_submission_gate_test.json",
        "verdict": "LOCAL_RELEASE_CANDIDATE_READY_FINAL_BLOCKED",
        "status_boundary": {
            "local_release_candidate_ready": "Local artifacts are consistent.",
            "final_submission_ready": "Final submission still uses final gate.",
        },
        "gates": [],
        "source_package_summary": {
            "file_count": 21,
            "total_bytes": 652691,
            "manifest_sha256": "a" * 64,
        },
        "source_rebuild_summary": {
            "rebuilt_pdf_pages": 9,
            "rebuilt_pdf_bytes": 546669,
            "bibtex_warning_count": 0,
            "overfull_hbox_count": 0,
        },
        "failures": [],
        "warnings": [],
        "remaining_blockers": ["manual_submission_system_not_ready"],
        "next_actions": ["Resolve blockers."],
    }


def test_refresh_submission_release_candidate_stack_runs_in_order(tmp_path: Path) -> None:
    calls: list[str] = []

    def refresh_runner(**kwargs):
        calls.append("refresh")
        assert kwargs["stamp"] == "test"
        return _refresh_payload()

    def freshness_builder(**kwargs):
        calls.append("freshness")
        assert str(kwargs["refresh_json"]).replace("\\", "/").endswith(
            "out/pre_submission_gate_refresh_test.json"
        )
        return _freshness_payload()

    def candidate_builder(**kwargs):
        calls.append("candidate")
        assert str(kwargs["refresh_freshness_json"]).replace("\\", "/").endswith(
            "out/pre_submission_gate_refresh_freshness_test.json"
        )
        return _candidate_payload()

    stack = refresh_submission_release_candidate_stack(
        root=tmp_path,
        output_dir="out",
        stamp="test",
        refresh_runner=refresh_runner,
        freshness_builder=freshness_builder,
        candidate_builder=candidate_builder,
    )

    assert calls == ["refresh", "freshness", "candidate"]
    assert stack["ok"] is True
    assert stack["local_release_candidate_ready"] is True
    assert stack["final_submission_ready"] is False
    assert stack["blocking_status"] == "external_or_manual_blocked"
    assert [step["step_id"] for step in stack["steps"]] == [
        "pre_submission_gate_refresh",
        "pre_submission_refresh_freshness",
        "submission_release_candidate",
    ]
    for name in [
        "pre_submission_gate_refresh_test.json",
        "pre_submission_gate_refresh_test.md",
        "pre_submission_gate_refresh_freshness_test.json",
        "pre_submission_gate_refresh_freshness_test.md",
        "submission_release_candidate_test.json",
        "submission_release_candidate_test.md",
    ]:
        assert (tmp_path / "out" / name).exists()


def test_refresh_submission_release_candidate_stack_fails_when_freshness_fails(
    tmp_path: Path,
) -> None:
    stack = refresh_submission_release_candidate_stack(
        root=tmp_path,
        output_dir="out",
        stamp="test",
        refresh_runner=lambda **kwargs: _refresh_payload(),
        freshness_builder=lambda **kwargs: _freshness_payload(ok=False),
        candidate_builder=lambda **kwargs: _candidate_payload(),
    )

    assert stack["ok"] is False
    assert stack["refresh_artifact_fresh"] is False
    assert "pre_submission_refresh_freshness:input_fingerprint_mismatch:Paper/main.tex" in stack[
        "failures"
    ]
