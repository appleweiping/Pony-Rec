import json

from scripts.audit.main_build_server_large_artifact_manifest import build_manifest, write_manifest


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_build_server_large_artifact_manifest_records_required_and_model_files(tmp_path):
    evidence = tmp_path / "evidence"
    _write(evidence / "scores.csv", b"source_event_id,user_id,item_id,score\n")
    _write(evidence / "predictions" / "rank_predictions.jsonl", b'{"ok":true}\n')
    _write(evidence / "rlmrec_official_model.pt", b"model")
    _write(evidence / "tables" / "ranking_metrics.csv", b"sample_count\n1\n")

    result = build_manifest(evidence_dir=evidence)

    assert result["ok"] is True
    assert result["failures"] == []
    assert [row["rel_path"] for row in result["files"]] == [
        "predictions/rank_predictions.jsonl",
        "rlmrec_official_model.pt",
        "scores.csv",
    ]
    assert result["model_artifact_count"] == 1


def test_write_server_large_artifact_manifest_outputs_sha256_and_json(tmp_path):
    evidence = tmp_path / "evidence"
    _write(evidence / "scores.csv", b"score\n")
    _write(evidence / "predictions" / "rank_predictions.jsonl", b"{}\n")
    _write(evidence / "proex_official_model.pt", b"weights")
    result = build_manifest(evidence_dir=evidence)

    sha_path = evidence / "server_large_artifact_manifest.sha256"
    json_path = evidence / "server_large_artifact_manifest.json"
    write_manifest(result, output_sha256=sha_path, output_json=json_path)

    sha_text = sha_path.read_text(encoding="utf-8")
    assert "scores.csv" in sha_text
    assert "predictions/rank_predictions.jsonl" in sha_text
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["ok"] is True
    assert saved["file_count"] == 3


def test_server_large_artifact_manifest_fails_when_model_is_required(tmp_path):
    evidence = tmp_path / "evidence"
    _write(evidence / "scores.csv", b"score\n")
    _write(evidence / "predictions" / "rank_predictions.jsonl", b"{}\n")

    result = build_manifest(evidence_dir=evidence)

    assert result["ok"] is False
    assert "missing_model_artifact" in result["failures"]


def test_server_large_artifact_manifest_can_include_extra_suffix(tmp_path):
    evidence = tmp_path / "evidence"
    _write(evidence / "scores.csv", b"score\n")
    _write(evidence / "predictions" / "rank_predictions.jsonl", b"{}\n")
    _write(evidence / "adapter.faiss", b"index")

    result = build_manifest(
        evidence_dir=evidence,
        include_suffixes=["faiss"],
        require_model_artifact=False,
    )

    assert result["ok"] is True
    assert "adapter.faiss" in {row["rel_path"] for row in result["files"]}


def test_server_large_artifact_manifest_certifies_missing_prediction_from_server_final_audit(tmp_path):
    evidence = tmp_path / "evidence"
    _write(evidence / "scores.csv", b"score\n")
    _write(evidence / "proex_official_model.pt", b"weights")
    _write(
        evidence / "server_final_evidence_audit.json",
        json.dumps(
            {
                "ok": True,
                "files": {
                    "predictions/rank_predictions.jsonl": {
                        "present": True,
                        "size": 1234,
                        "lines": 10000,
                    }
                },
            }
        ).encode(),
    )

    result = build_manifest(
        evidence_dir=evidence,
        allow_certified_missing_prediction_jsonl=True,
    )

    assert result["ok"] is True
    assert "missing_required_large_artifact:predictions/rank_predictions.jsonl" not in result["failures"]
    assert result["certified_missing_artifact_count"] == 1
    assert result["certified_missing_artifacts"][0]["rel_path"] == "predictions/rank_predictions.jsonl"
    assert result["certified_missing_artifacts"][0]["certified_lines"] == 10000
