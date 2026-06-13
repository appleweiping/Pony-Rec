import json
from pathlib import Path

from scripts.audit.main_audit_final_blocker_doc_status import (
    audit_final_blocker_doc_status,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _consistency(tmp_path: Path) -> Path:
    return _consistency_with_count(tmp_path, 9)


def _consistency_with_count(tmp_path: Path, count: int) -> Path:
    return _write_json(
        tmp_path / "out" / "final_blocker_consistency_audit_20260613.json",
        {
            "ok": True,
            "final_blocker_consistency_ok": True,
            "final_submission_ready": False,
            "summary": {
                "review_failed_claude_attempt_count": count,
                "explicit_claude_opus_present": False,
                "final_panel_coverage_complete": False,
                "promax_public_metadata_ready": False,
                "manual_confirmation_needed": True,
                "manual_submission_system_ready": False,
                "recursive_warning_regression_count": 0,
            },
        },
    )


def _doc(tmp_path: Path, rel: str, body: str) -> Path:
    return _write(tmp_path / rel, body)


def _current_body(*, stale: bool = False, missing_manual: bool = False) -> str:
    count_line = (
        "The refreshed packet still reports failed Claude attempts `8`."
        if stale
        else "The refreshed packet now records failed Claude attempts `9`."
    )
    manual_line = (
        "Manual details are omitted."
        if missing_manual
        else "`manual_confirmation_needed=true` and `manual_submission_system_ready=false`."
    )
    return f"""
## Current Checkpoint (2026-06-13)

`final_submission_ready=false`; {count_line}
`explicit_claude_opus_present=false` and `final_panel_coverage_complete=false`.
`promax_public_metadata_ready=false`; Crossref `404` / DOI resolver `404` / ACM DL `403`.
{manual_line}
recursive warning regressions `0`.

## Hard Invariants
Historical line outside current section: failed Claude attempts `8`.
"""


def _current_body_with_count(count: int) -> str:
    return f"""
## Current Checkpoint (2026-06-13)

`final_submission_ready=false`. The refreshed packet now records failed Claude attempts `{count}`.
`explicit_claude_opus_present=false` and `final_panel_coverage_complete=false`.
`promax_public_metadata_ready=false`; Crossref `404` / DOI resolver `404` / ACM DL `403`.
`manual_confirmation_needed=true` and `manual_submission_system_ready=false`.
recursive warning regressions `0`.

## Hard Invariants
"""


def test_doc_status_audit_passes_when_current_sections_match_latest_truth(tmp_path: Path) -> None:
    consistency = _consistency(tmp_path)
    docs = [
        _doc(tmp_path, "docs/active_todo_pony_uncertainty.md", _current_body()),
        _doc(
            tmp_path,
            "docs/paper_claims_and_status.md",
            _current_body().replace("## Current Checkpoint (2026-06-13)", "## Paper-critical readiness modules").replace(
                "## Hard Invariants", "## Not primary claims"
            ),
        ),
        _doc(
            tmp_path,
            "docs/milestones/README.md",
            _current_body()
            .replace("## Current Checkpoint (2026-06-13)", "## Current Evidence Integrity (updated 2026-06-13)")
            .replace("## Hard Invariants", "## Current Evidence Integrity (updated 2026-06-06)"),
        ),
        _doc(
            tmp_path,
            "docs/server_runbook.md",
            _current_body().replace("## Current Checkpoint (2026-06-13)", "## Current Priority Order (2026-06-13)").replace(
                "## Hard Invariants", "## Key Scripts"
            ),
        ),
    ]

    audit = audit_final_blocker_doc_status(
        root=tmp_path,
        consistency_audit_json=consistency.relative_to(tmp_path),
        docs=[path.relative_to(tmp_path) for path in docs],
    )

    assert audit["ok"] is True
    assert audit["final_submission_ready"] is False
    assert audit["expected_current_truth"]["final_submission_ready"] is False
    assert audit["expected_current_truth"]["failed_claude_attempts"] == 9


def test_doc_status_audit_rejects_stale_current_failed_claude_count(tmp_path: Path) -> None:
    consistency = _consistency(tmp_path)
    doc = _doc(tmp_path, "docs/active_todo_pony_uncertainty.md", _current_body(stale=True))

    audit = audit_final_blocker_doc_status(
        root=tmp_path,
        consistency_audit_json=consistency.relative_to(tmp_path),
        docs=[doc.relative_to(tmp_path)],
    )

    assert audit["ok"] is False
    assert audit["doc_results"]["docs/active_todo_pony_uncertainty.md"]["stale_count_hits"]
    assert any("stale_current_failed_claude_count" in failure for failure in audit["failures"])


def test_doc_status_audit_rejects_missing_manual_blocker_in_current_section(tmp_path: Path) -> None:
    consistency = _consistency(tmp_path)
    doc = _doc(tmp_path, "docs/active_todo_pony_uncertainty.md", _current_body(missing_manual=True))

    audit = audit_final_blocker_doc_status(
        root=tmp_path,
        consistency_audit_json=consistency.relative_to(tmp_path),
        docs=[doc.relative_to(tmp_path)],
    )

    assert audit["ok"] is False
    assert (
        "missing_current_observation:manual_submission_blocked"
        in audit["doc_results"]["docs/active_todo_pony_uncertainty.md"]["failures"]
    )


def test_doc_status_audit_rejects_any_stale_currentish_failed_count(tmp_path: Path) -> None:
    consistency = _consistency(tmp_path)
    doc = _doc(
        tmp_path,
        "docs/milestones/README.md",
        """
## Current Evidence Integrity (updated 2026-06-13)

`final_submission_ready=false`. The review-continuation packet now records three failed Claude attempts.
`explicit_claude_opus_present=false`; `final_panel_coverage_complete=false`.
`promax_public_metadata_ready=false`; Crossref `404` / DOI resolver `404` / ACM DL `403`.
`manual_confirmation_needed=true`; `manual_submission_system_ready=false`.
recursive warning regressions `0`.

## Current Evidence Integrity (updated 2026-06-06)
""",
    )

    audit = audit_final_blocker_doc_status(
        root=tmp_path,
        consistency_audit_json=consistency.relative_to(tmp_path),
        docs=[doc.relative_to(tmp_path)],
    )

    assert audit["ok"] is False
    assert any("stale_current_failed_claude_count" in failure for failure in audit["failures"])


def test_doc_status_audit_rejects_stale_two_blocker_taxonomy(tmp_path: Path) -> None:
    consistency = _consistency(tmp_path)
    doc = _doc(
        tmp_path,
        "docs/milestones/README.md",
        """
## Current Evidence Integrity (updated 2026-06-13)

`final_submission_ready=false`; failed Claude attempts `9`.
`explicit_claude_opus_present=false`; `final_panel_coverage_complete=false`.
`promax_public_metadata_ready=false`; Crossref `404` / DOI resolver `404` / ACM DL `403`.
`manual_confirmation_needed=true`; `manual_submission_system_ready=false`.
recursive warning regressions `0`.
This is now the compact first-read artifact for the two remaining blocker classes.

## Current Evidence Integrity (updated 2026-06-06)
""",
    )

    audit = audit_final_blocker_doc_status(
        root=tmp_path,
        consistency_audit_json=consistency.relative_to(tmp_path),
        docs=[doc.relative_to(tmp_path)],
    )

    assert audit["ok"] is False
    assert any("stale_blocker_taxonomy" in failure for failure in audit["failures"])


def test_doc_status_audit_treats_ten_as_complete_count_not_one(tmp_path: Path) -> None:
    consistency = _consistency_with_count(tmp_path, 10)
    doc = _doc(tmp_path, "docs/active_todo_pony_uncertainty.md", _current_body_with_count(10))

    audit = audit_final_blocker_doc_status(
        root=tmp_path,
        consistency_audit_json=consistency.relative_to(tmp_path),
        docs=[doc.relative_to(tmp_path)],
    )

    assert audit["ok"] is True
    assert audit["doc_results"]["docs/active_todo_pony_uncertainty.md"]["stale_count_hits"] == []
