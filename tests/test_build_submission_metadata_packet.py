import json
from pathlib import Path

from scripts.audit.main_build_submission_metadata_packet import build_submission_metadata_packet


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _seed_inputs(tmp_path: Path) -> dict[str, Path]:
    paper = tmp_path / "Paper"
    _write(
        paper / "main.tex",
        "\\title{Actionable Uncertainty for LLM-Based Recommendation}\n",
    )
    _write(
        paper / "sections" / "abstract.tex",
        "\\begin{abstract}\n"
        + " ".join(f"word{i}" for i in range(130))
        + "\n\\end{abstract}\n",
    )
    config = _write_json(
        tmp_path / "metadata.json",
        {
            "target_profile_id": "test_profile",
            "paper_type": "full research paper",
            "submission_title": "Actionable Uncertainty for LLM-Based Recommendation",
            "anonymous_submission": True,
            "suggested_keywords": ["recommendation", "uncertainty", "calibration"],
            "suggested_topic_areas": ["Recommender systems", "Evaluation"],
            "manual_fields_not_stored": ["conflicts"],
        },
    )
    audit = _write_json(
        tmp_path / "audit.json",
        {
            "ok": True,
            "submission_package_ready_for_target_formatting": True,
            "final_submission_ready": False,
            "target_formatting_profile": {"profile_id": "test_profile", "ok": True},
            "build": {"pdf": {"path": "Paper/main.pdf", "size_bytes": 123}, "page_count": 9},
            "source_package_manifest": {"file_count": 5, "manifest_sha256": "abc"},
            "remaining_blockers": ["external metadata"],
        },
    )
    return {"paper": paper, "config": config, "audit": audit}


def test_submission_metadata_packet_extracts_submission_fields(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)

    packet = build_submission_metadata_packet(
        root=tmp_path,
        paper_dir=paths["paper"].relative_to(tmp_path),
        metadata_config=paths["config"].relative_to(tmp_path),
        submission_audit_json=paths["audit"].relative_to(tmp_path),
    )

    assert packet["ok"] is True
    assert packet["submission_metadata_packet_ready"] is True
    assert packet["final_submission_ready"] is False
    assert packet["submission_fields"]["title"] == "Actionable Uncertainty for LLM-Based Recommendation"
    assert packet["submission_fields"]["abstract_word_count"] == 130
    assert packet["submission_fields"]["keywords"] == ["recommendation", "uncertainty", "calibration"]
    assert packet["package_crosscheck"]["source_manifest_sha256"] == "abc"
    assert packet["remaining_blockers"] == ["external metadata"]


def test_submission_metadata_packet_fails_on_title_mismatch(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    config["submission_title"] = "Wrong Title"
    paths["config"].write_text(json.dumps(config), encoding="utf-8")

    packet = build_submission_metadata_packet(
        root=tmp_path,
        paper_dir=paths["paper"].relative_to(tmp_path),
        metadata_config=paths["config"].relative_to(tmp_path),
        submission_audit_json=paths["audit"].relative_to(tmp_path),
    )

    assert packet["ok"] is False
    assert any(failure.startswith("title_mismatch") for failure in packet["failures"])


def test_submission_metadata_packet_fails_on_target_profile_mismatch(tmp_path: Path) -> None:
    paths = _seed_inputs(tmp_path)
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    audit["target_formatting_profile"]["profile_id"] = "other_profile"
    paths["audit"].write_text(json.dumps(audit), encoding="utf-8")

    packet = build_submission_metadata_packet(
        root=tmp_path,
        paper_dir=paths["paper"].relative_to(tmp_path),
        metadata_config=paths["config"].relative_to(tmp_path),
        submission_audit_json=paths["audit"].relative_to(tmp_path),
    )

    assert packet["ok"] is False
    assert "target_profile_mismatch" in packet["failures"]
