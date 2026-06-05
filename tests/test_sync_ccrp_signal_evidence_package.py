import json
from pathlib import Path

from scripts.audit.main_sync_ccrp_signal_evidence_package import (
    SIGNAL_SYNC_MANIFEST,
    build_package_audit,
    sha256_file,
    signal_role,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _json(path: Path, payload: dict) -> Path:
    return _write(path, json.dumps(payload, indent=2) + "\n")


def _complete_signal_package(root: Path) -> None:
    signal_path = _write(
        root / "valid_ccrp_signal_rows.csv",
        "source_event_id,user_id,candidate_item_id,item_id,candidate_idx,"
        "relevance_probability,calibrated_relevance_probability,evidence_support,"
        "counterevidence_strength,reason,parse_success,signal_schema_version\n"
        "e1,u1,i1,i1,0,0.8,0.75,0.7,0.1,match,true,ccrp_v3_signal_rows.2026-06-06\n"
        "e1,u1,i2,i2,1,0.2,0.25,0.2,0.5,weak,true,ccrp_v3_signal_rows.2026-06-06\n"
        "e2,u2,i3,i3,0,0.6,0.55,0.5,0.2,match,true,ccrp_v3_signal_rows.2026-06-06\n"
        "e2,u2,i4,i4,1,0.1,0.15,0.1,0.8,weak,true,ccrp_v3_signal_rows.2026-06-06\n",
    )
    provenance_path = _json(
        root / "valid_ccrp_signal_rows_provenance.json",
        {
            "status_label": "ccrp_v3_recomputable_signal_rows_generated",
            "artifact_class": "paper_critical_signal_rows",
            "domain": "sports",
            "split": "valid",
            "git_commit": "abc123",
            "data_sha256": "d" * 64,
            "expected_candidates_per_event": 2,
            "expected_signal_rows": 4,
            "prompt_count": 4,
            "meta_row_count": 4,
            "raw_result_count": 4,
            "n_events": 2,
            "n_signal_rows": 4,
            "signal_rows_sha256": sha256_file(signal_path),
            "parse_failure_count": 0,
            "parse_failure_rate": 0.0,
            "max_parse_failure_rate": 0.005,
        },
    )
    _json(
        root / "valid_ccrp_signal_source_audit.json",
        {
            "sources": [
                {
                    "status": "recomputable_signal_rows",
                    "candidate_key_coverage_rate": 1.0,
                    "matched_candidate_keys": 4,
                    "extra_source_keys": 0,
                    "missing_candidate_keys": 0,
                }
            ]
        },
    )
    _json(
        root / SIGNAL_SYNC_MANIFEST,
        {
            "ok": True,
            "checked_files": [
                {
                    "rel_path": "valid_ccrp_signal_rows.csv",
                    "size_ok": True,
                    "sha256_ok": True,
                    "sha256": sha256_file(signal_path),
                },
                {
                    "rel_path": "valid_ccrp_signal_rows_provenance.json",
                    "size_ok": True,
                    "sha256_ok": True,
                    "sha256": sha256_file(provenance_path),
                },
                {
                    "rel_path": "valid_ccrp_signal_source_audit.json",
                    "size_ok": True,
                    "sha256_ok": True,
                    "sha256": sha256_file(root / "valid_ccrp_signal_source_audit.json"),
                },
            ],
        },
    )


def test_signal_package_audit_accepts_complete_package(tmp_path):
    _complete_signal_package(tmp_path)

    audit = build_package_audit(
        package_dir=tmp_path,
        expected_events=2,
        expected_candidates_per_event=2,
    )

    assert audit["ok"] is True
    assert audit["failures"] == []
    assert audit["expected_signal_rows"] == 4


def test_signal_package_audit_requires_sync_hash_evidence(tmp_path):
    _complete_signal_package(tmp_path)
    manifest = json.loads((tmp_path / SIGNAL_SYNC_MANIFEST).read_text(encoding="utf-8"))
    manifest["checked_files"] = [
        row for row in manifest["checked_files"] if row["rel_path"] != "valid_ccrp_signal_rows.csv"
    ]
    _json(tmp_path / SIGNAL_SYNC_MANIFEST, manifest)

    audit = build_package_audit(
        package_dir=tmp_path,
        expected_events=2,
        expected_candidates_per_event=2,
    )

    assert audit["ok"] is False
    assert "sync_manifest_missing_hash_evidence:valid_ccrp_signal_rows.csv" in audit["failures"]


def test_signal_package_audit_rejects_source_audit_coverage_gap(tmp_path):
    _complete_signal_package(tmp_path)
    _json(
        tmp_path / "valid_ccrp_signal_source_audit.json",
        {
            "sources": [
                {
                    "status": "recomputable_signal_rows",
                    "candidate_key_coverage_rate": 0.75,
                    "matched_candidate_keys": 3,
                    "extra_source_keys": 0,
                    "missing_candidate_keys": 1,
                }
            ]
        },
    )

    audit = build_package_audit(
        package_dir=tmp_path,
        expected_events=2,
        expected_candidates_per_event=2,
    )

    assert audit["ok"] is False
    assert "source_audit_candidate_key_coverage_not_one:0.75" in audit["failures"]
    assert "source_audit_matched_candidate_keys_mismatch" in audit["failures"]
    assert "source_audit_missing_candidate_keys" in audit["failures"]


def test_signal_package_audit_requires_source_audit_or_candidate_items(tmp_path):
    _complete_signal_package(tmp_path)
    (tmp_path / "valid_ccrp_signal_source_audit.json").unlink()

    audit = build_package_audit(
        package_dir=tmp_path,
        expected_events=2,
        expected_candidates_per_event=2,
    )

    assert audit["ok"] is False
    assert "missing_source_audit_json_or_candidate_items_path" in audit["failures"]


def test_signal_role_can_exclude_bulk_signal_rows():
    assert signal_role("valid_ccrp_signal_rows.csv", include_signal_rows=True) == "signal_rows"
    assert signal_role("valid_ccrp_signal_rows.csv", include_signal_rows=False) == ""
    assert signal_role("valid_ccrp_signal_rows_provenance.json") == "signal_provenance"
