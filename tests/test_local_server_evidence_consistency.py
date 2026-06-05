import json

from scripts.audit.main_audit_local_server_evidence_consistency import audit_package, build_audit


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _sha(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_valid_package(base):
    files = {}
    files["fairness_provenance.json"] = _write(
        base / "fairness_provenance.json",
        json.dumps(
            {
                "implementation_status": "official_completed",
                "blockers": [],
                "score_coverage_rate": 1.0,
                "comparison_variant": "official_code_qwen3base_default_hparams_declared_adaptation",
                "score_schema": ["source_event_id", "user_id", "item_id", "score"],
            }
        ).encode(),
    )
    files["server_final_evidence_audit.json"] = _write(base / "server_final_evidence_audit.json", b'{"ok": true}\n')
    files["method_score_audit.json"] = _write(base / "method_score_audit.json", b'{"audit_ok": true}\n')
    files["method_same_candidate_score_audit.txt"] = _write(
        base / "method_same_candidate_score_audit.txt",
        b"audit_ok=True\n",
    )
    files["method_run_summary.json"] = _write(base / "method_run_summary.json", b'{"ok": true}\n')
    files["tables/ranking_metrics.csv"] = _write(
        base / "tables" / "ranking_metrics.csv",
        (
            "sample_count,avg_candidates,HR@5,HR@10,HR@20,NDCG@5,NDCG@10,NDCG@20,MRR\n"
            "2,3,0.1,0.2,0.3,0.01,0.02,0.03,0.04\n"
        ).encode(),
    )
    files["tables/same_candidate_external_baseline_summary.csv"] = _write(
        base / "tables" / "same_candidate_external_baseline_summary.csv",
        b"method,MRR\nx,0.1\n",
    )
    files["tables/external_score_coverage.csv"] = _write(
        base / "tables" / "external_score_coverage.csv",
        b"score_coverage_rate\n1.0\n",
    )
    files["tables/ranking_exposure_distribution.csv"] = _write(
        base / "tables" / "ranking_exposure_distribution.csv",
        b"bucket,count\n1,2\n",
    )
    files["tables/ranking_eval_records.csv"] = _write(
        base / "tables" / "ranking_eval_records.csv",
        b"user_id,rank\nu1,1\nu2,2\n",
    )
    server_manifest = {
        "ok": True,
        "file_count": 3,
        "model_artifact_count": 1,
        "failures": [],
        "require_model_artifact": True,
        "files": [
            {"rel_path": "predictions/rank_predictions.jsonl", "size": 100, "sha256": "a"},
            {"rel_path": "scores.csv", "size": 200, "sha256": "b"},
            {"rel_path": "method_official_model.pt", "size": 300, "sha256": "c"},
        ],
    }
    files["server_large_artifact_manifest.json"] = _write(
        base / "server_large_artifact_manifest.json",
        json.dumps(server_manifest).encode(),
    )
    files["server_large_artifact_manifest.sha256"] = _write(
        base / "server_large_artifact_manifest.sha256",
        b"a  predictions/rank_predictions.jsonl\nb  scores.csv\nc  method_official_model.pt\n",
    )
    checked_files = []
    for rel_path, path in files.items():
        checked_files.append(
            {
                "rel_path": rel_path,
                "local_path": str(path),
                "size": path.stat().st_size,
                "sha256": _sha(path),
                "size_ok": True,
                "sha256_ok": True,
            }
        )
    sync_manifest = {
        "ok": True,
        "allowed_file_count": len(checked_files),
        "excluded_file_count": 3,
        "remote_evidence_dir": "/remote/evidence",
        "checked_files": checked_files,
        "failures": [],
    }
    _write(base / "light_evidence_sync_manifest.json", json.dumps(sync_manifest).encode())


def test_audit_package_passes_for_consistent_local_light_and_server_manifests(tmp_path):
    package = tmp_path / "pkg"
    _write_valid_package(package)

    result = audit_package(local_dir=package, expected_users=2, expected_candidates_per_user=3)

    assert result["ok"] is True
    assert result["failures"] == []
    assert result["light_sync_manifest"]["checked_file_count"] > 0
    assert result["server_large_artifact_manifest"]["file_count"] == 3


def test_audit_package_fails_when_manifest_checked_file_is_missing(tmp_path):
    package = tmp_path / "pkg"
    _write_valid_package(package)
    (package / "method_run_summary.json").unlink()

    result = audit_package(local_dir=package, expected_users=2, expected_candidates_per_user=3)

    assert result["ok"] is False
    assert "missing_checked_local_file:method_run_summary.json" in result["failures"]


def test_audit_package_fails_when_server_only_large_artifact_is_local(tmp_path):
    package = tmp_path / "pkg"
    _write_valid_package(package)
    _write(package / "scores.csv", b"large score csv should stay server side\n")

    result = audit_package(local_dir=package, expected_users=2, expected_candidates_per_user=3)

    assert result["ok"] is False
    assert "server_only_large_artifact_present_locally:scores.csv" in result["failures"]


def test_audit_package_accepts_certified_missing_prediction_in_server_manifest(tmp_path):
    package = tmp_path / "pkg"
    _write_valid_package(package)
    manifest_path = package / "server_large_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        row for row in manifest["files"] if row["rel_path"] != "predictions/rank_predictions.jsonl"
    ]
    manifest["file_count"] = len(manifest["files"])
    manifest["certified_missing_artifacts"] = [
        {
            "rel_path": "predictions/rank_predictions.jsonl",
            "status": "certified_missing_after_post_gate_cleanup",
            "certified_by": "server_final_evidence_audit.json",
            "certified_size": 100,
            "certified_lines": 10000,
        }
    ]
    manifest["certified_missing_artifact_count"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sync_manifest_path = package / "light_evidence_sync_manifest.json"
    sync_manifest = json.loads(sync_manifest_path.read_text(encoding="utf-8"))
    for row in sync_manifest["checked_files"]:
        if row["rel_path"] == "server_large_artifact_manifest.json":
            row["size"] = manifest_path.stat().st_size
            row["sha256"] = _sha(manifest_path)
    sync_manifest_path.write_text(json.dumps(sync_manifest), encoding="utf-8")

    result = audit_package(local_dir=package, expected_users=2, expected_candidates_per_user=3)

    assert result["ok"] is True
    assert result["server_large_artifact_manifest"]["certified_missing_rel_paths"] == [
        "predictions/rank_predictions.jsonl"
    ]


def test_build_audit_checks_expected_domain_method_rows(tmp_path):
    root = tmp_path
    local_root = root / "outputs" / "baselines" / "official_adapters"
    _write_valid_package(
        local_root / "tools_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate"
    )

    result = build_audit(
        root=root,
        local_root="outputs/baselines/official_adapters",
        domains=["tools"],
        methods=["llm2rec_sasrec", "llmesr_sasrec"],
        expected_users=2,
        expected_candidates_per_user=3,
    )

    assert result["ok"] is False
    assert result["row_count"] == 2
    assert result["ok_count"] == 1
    assert "tools/llmesr_sasrec:missing_local_evidence_dir" in result["failures"]
