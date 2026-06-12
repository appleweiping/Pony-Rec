import hashlib
import json
from pathlib import Path

import scripts.audit.main_audit_pre_submission_refresh_freshness as freshness
from scripts.audit.main_audit_pre_submission_refresh_freshness import (
    build_pre_submission_refresh_freshness_audit,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(root: Path, path: Path) -> dict:
    return {
        "exists": True,
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _seed_refresh(root: Path) -> Path:
    paper = _write(root / "Paper" / "main.tex", "\\documentclass{article}\n")
    bib = _write(root / "Paper" / "references.bib", "@inproceedings{x, title={X}}\n")
    external_json = _write_json(
        root / "out" / "external_proceedings_metadata_recheck_test.json",
        {
            "ok": True,
            "external_proceedings_metadata_ready": False,
            "final_submission_ready": False,
            "remaining_blockers": ["promax:final_page_range_missing_in_bib"],
        },
    )
    external_md = _write(root / "out" / "external_proceedings_metadata_recheck_test.md", "# External\n")
    final_json = _write_json(
        root / "out" / "final_submission_gate_test.json",
        {
            "ok": True,
            "final_submission_ready": False,
            "remaining_blockers": ["manual_submission_system_not_ready"],
        },
    )
    final_md = _write(root / "out" / "final_submission_gate_test.md", "# Final\n")
    refresh = {
        "schema_version": "2026-06-12.pre_submission_gate_refresh.v1",
        "created_at_utc": "2026-06-12T00:00:00+00:00",
        "ok": True,
        "final_submission_ready": False,
        "final_verdict": "LOCAL_PACKAGE_READY_BUT_EXTERNAL_OR_MANUAL_BLOCKED",
        "git_state_before_refresh": {
            "available": True,
            "head": "abc123",
            "tracked_dirty": False,
            "tracked_dirty_paths": [],
            "error": "",
        },
        "input_fingerprints": [_record(root, paper), _record(root, bib)],
        "remaining_blockers": [
            "promax:final_page_range_missing_in_bib",
            "manual_submission_system_not_ready",
        ],
        "steps": [
            {
                "step_id": "external_proceedings_metadata",
                "json": _record(root, external_json),
                "md": _record(root, external_md),
                "ok": True,
                "ready": False,
                "ready_field": "external_proceedings_metadata_ready",
                "final_submission_ready": False,
                "remaining_blockers": ["promax:final_page_range_missing_in_bib"],
            },
            {
                "step_id": "final_submission_gate",
                "json": _record(root, final_json),
                "md": _record(root, final_md),
                "ok": True,
                "ready": False,
                "ready_field": "final_submission_ready",
                "final_submission_ready": False,
                "remaining_blockers": ["manual_submission_system_not_ready"],
            },
        ],
    }
    return _write_json(root / "out" / "pre_submission_gate_refresh_test.json", refresh)


def test_pre_submission_refresh_freshness_passes_when_hashes_match(tmp_path: Path) -> None:
    refresh = _seed_refresh(tmp_path)

    audit = build_pre_submission_refresh_freshness_audit(
        root=tmp_path,
        refresh_json=refresh.relative_to(tmp_path),
    )

    assert audit["ok"] is True
    assert audit["refresh_artifact_fresh"] is True
    assert audit["final_submission_ready"] is False
    assert audit["checked_input_fingerprint_count"] == 2
    assert audit["checked_step_file_count"] == 4
    assert audit["input_fingerprint_mismatch_count"] == 0
    assert audit["generated_step_file_mismatch_count"] == 0
    assert "promax:final_page_range_missing_in_bib" in audit["remaining_blockers"]
    assert "Git HEAD is provenance" in audit["git_state_policy"]
    assert audit["refresh_final_submission_ready"] is False


def test_pre_submission_refresh_freshness_fails_on_input_mismatch(tmp_path: Path) -> None:
    refresh = _seed_refresh(tmp_path)
    _write(tmp_path / "Paper" / "main.tex", "\\documentclass{article}\nchanged\n")

    audit = build_pre_submission_refresh_freshness_audit(
        root=tmp_path,
        refresh_json=refresh.relative_to(tmp_path),
    )

    assert audit["ok"] is False
    assert audit["refresh_artifact_fresh"] is False
    assert audit["input_fingerprint_mismatch_count"] == 1
    assert any(item.startswith("input_fingerprint_mismatch:") for item in audit["failures"])


def test_pre_submission_refresh_freshness_reports_missing_refresh_json(tmp_path: Path) -> None:
    audit = build_pre_submission_refresh_freshness_audit(
        root=tmp_path,
        refresh_json="out/missing_refresh.json",
    )

    assert audit["ok"] is False
    assert audit["refresh_artifact_fresh"] is False
    assert audit["final_submission_ready"] is False
    assert audit["failures"] == ["refresh_json_missing:out/missing_refresh.json"]


def test_pre_submission_refresh_freshness_reports_missing_files(tmp_path: Path) -> None:
    refresh = _seed_refresh(tmp_path)
    (tmp_path / "Paper" / "references.bib").unlink()
    (tmp_path / "out" / "final_submission_gate_test.md").unlink()

    audit = build_pre_submission_refresh_freshness_audit(
        root=tmp_path,
        refresh_json=refresh.relative_to(tmp_path),
    )

    assert audit["ok"] is False
    assert audit["input_fingerprint_mismatch_count"] == 1
    assert audit["generated_step_file_mismatch_count"] == 1
    assert "input_fingerprint_mismatch:Paper/references.bib" in {
        item.replace("\\", "/") for item in audit["failures"]
    }
    assert any(
        item.replace("\\", "/").startswith(
            "generated_step_file_mismatch:final_submission_gate:out/final_submission_gate_test.md"
        )
        for item in audit["failures"]
    )


def test_pre_submission_refresh_freshness_fails_on_generated_gate_mismatch(tmp_path: Path) -> None:
    refresh = _seed_refresh(tmp_path)
    _write_json(
        tmp_path / "out" / "final_submission_gate_test.json",
        {
            "ok": True,
            "final_submission_ready": True,
            "remaining_blockers": [],
        },
    )

    audit = build_pre_submission_refresh_freshness_audit(
        root=tmp_path,
        refresh_json=refresh.relative_to(tmp_path),
    )

    assert audit["ok"] is False
    assert audit["refresh_artifact_fresh"] is False
    assert audit["generated_step_file_mismatch_count"] == 1
    assert any("generated_step_file_mismatch:final_submission_gate" in item for item in audit["failures"])


def test_pre_submission_refresh_freshness_does_not_fail_on_head_change(
    tmp_path: Path, monkeypatch
) -> None:
    refresh = _seed_refresh(tmp_path)
    monkeypatch.setattr(
        freshness,
        "_git_state",
        lambda root: {
            "available": True,
            "head": "def456",
            "tracked_dirty": False,
            "tracked_dirty_paths": [],
            "error": "",
        },
    )

    audit = build_pre_submission_refresh_freshness_audit(
        root=tmp_path,
        refresh_json=refresh.relative_to(tmp_path),
    )

    assert audit["ok"] is True
    assert audit["refresh_artifact_fresh"] is True
    assert audit["git_head_changed_since_refresh_generation"] is True
    assert not audit["failures"]


def test_pre_submission_refresh_freshness_handles_directory_record(tmp_path: Path) -> None:
    refresh = _seed_refresh(tmp_path)
    payload = json.loads(refresh.read_text(encoding="utf-8"))
    payload["input_fingerprints"].append(
        {
            "exists": True,
            "path": "Paper",
            "sha256": "not-a-real-file-sha",
            "size_bytes": 123,
        }
    )
    _write_json(refresh, payload)

    audit = build_pre_submission_refresh_freshness_audit(
        root=tmp_path,
        refresh_json=refresh.relative_to(tmp_path),
    )

    assert audit["ok"] is False
    directory_check = next(
        item for item in audit["input_fingerprint_checks"] if item["path"].replace("\\", "/") == "Paper"
    )
    assert "not_a_regular_file" in directory_check["mismatches"]
