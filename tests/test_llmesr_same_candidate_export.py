from __future__ import annotations

import csv
import json
import pickle

import numpy as np
import pytest

import main_export_llmesr_same_candidate_task as exporter
import main_generate_llmesr_sentence_embeddings as sentence_embeddings
from main_run_llmesr_scaffold_four_domain import _raw_metadata_args
from main_audit_llmesr_adapter_package import audit
from main_enrich_llmesr_item_text_seed import enrich_item_text_seed
from main_generate_llmesr_text_embeddings import generate_embeddings
from main_score_llmesr_same_candidate_adapter import score_adapter


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_llmesr_export_writes_mapped_adapter_package(tmp_path, monkeypatch):
    task_dir = tmp_path / "beauty_week8_same_candidate_external"
    _write_csv(
        task_dir / "train_interactions.csv",
        [
            {"user_id": "u1", "item_id": "i1", "timestamp": 1, "sequence_index": 0},
            {"user_id": "u1", "item_id": "i2", "timestamp": 2, "sequence_index": 1},
            {"user_id": "u2", "item_id": "i2", "timestamp": 1, "sequence_index": 0},
            {"user_id": "u2", "item_id": "i3", "timestamp": 2, "sequence_index": 1},
        ],
        ["user_id", "item_id", "timestamp", "sequence_index"],
    )
    _write_csv(
        task_dir / "candidate_items.csv",
        [
            {
                "source_event_id": "u1::3",
                "user_id": "u1",
                "timestamp": 3,
                "split_name": "test",
                "candidate_index": 0,
                "item_id": "i3",
                "label": 1,
                "is_positive": 1,
                "popularity_group": "longtail",
                "candidate_title": "Target Item",
            },
            {
                "source_event_id": "u1::3",
                "user_id": "u1",
                "timestamp": 3,
                "split_name": "test",
                "candidate_index": 1,
                "item_id": "i4",
                "label": 0,
                "is_positive": 0,
                "popularity_group": "head",
                "candidate_title": "Negative Item",
            },
        ],
        [
            "source_event_id",
            "user_id",
            "timestamp",
            "split_name",
            "candidate_index",
            "item_id",
            "label",
            "is_positive",
            "popularity_group",
            "candidate_title",
        ],
    )
    (task_dir / "metadata.json").write_text(json.dumps({"exp_name": "source"}), encoding="utf-8")

    output_root = tmp_path / "outputs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "main_export_llmesr_same_candidate_task.py",
            "--task_dir",
            str(task_dir),
            "--exp_name",
            "beauty_llmesr_adapter",
            "--output_root",
            str(output_root),
            "--top_sim_users",
            "2",
        ],
    )

    exporter.main()

    package_dir = output_root / "baselines" / "paper_adapters" / "beauty_llmesr_adapter"
    metadata = json.loads((package_dir / "adapter_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "adapter_package_only"
    assert metadata["users"] == 2
    assert metadata["items"] == 4
    assert metadata["candidate_rows"] == 2

    inter_lines = (package_dir / "llm_esr" / "handled" / "inter.txt").read_text(encoding="utf-8").splitlines()
    assert inter_lines == ["1 1", "1 2", "2 2", "2 3"]

    mapped_candidates = list(csv.DictReader((package_dir / "candidate_items_mapped.csv").open(encoding="utf-8")))
    assert mapped_candidates[0]["llmesr_user_idx"] == "1"
    assert mapped_candidates[0]["llmesr_item_idx"] == "3"
    assert mapped_candidates[1]["llmesr_item_idx"] == "4"

    item_text_rows = list(csv.DictReader((package_dir / "item_text_seed.csv").open(encoding="utf-8")))
    assert item_text_rows[2]["embedding_text"] == "Target Item"
    assert item_text_rows[3]["embedding_text"] == "Negative Item"

    with (package_dir / "llm_esr" / "handled" / "sim_user_100.pkl").open("rb") as fh:
        sim_users = pickle.load(fh)
    assert len(sim_users) == 2
    assert all(len(row) == 2 for row in sim_users)
    assert set(sim_users[0]) <= {0, 1}
    assert set(sim_users[1]) <= {0, 1}

    audit_row = audit(package_dir)
    assert audit_row["diagnosis"] == "adapter_core_ready_embeddings_missing_or_invalid"
    assert audit_row["ready_for_embedding_generation"] is True
    assert audit_row["ready_for_scoring"] is False
    assert audit_row["itm_emb_status"] == "missing"

    processed_dir = tmp_path / "processed"
    _write_csv(
        processed_dir / "items.csv",
        [
            {
                "item_id": "i1",
                "title": "History One",
                "categories": "Catalog",
                "description": "Seen in train history",
                "candidate_text": "Title: History One Categories: Catalog Description: Seen in train history",
            },
            {
                "item_id": "i3",
                "title": "Target Catalog Title",
                "categories": "Catalog",
                "description": "Better target text",
                "candidate_text": "Title: Target Catalog Title Categories: Catalog Description: Better target text",
            },
        ],
        ["item_id", "title", "categories", "description", "candidate_text"],
    )
    enrich_summary = enrich_item_text_seed(adapter_dir=package_dir, processed_dir=processed_dir, raw_metadata_paths=[])
    assert enrich_summary["embedding_text_coverage_after"] == 1.0
    item_text_rows = list(csv.DictReader((package_dir / "item_text_seed.csv").open(encoding="utf-8")))
    assert item_text_rows[2]["candidate_title"] == "Target Item"
    assert "Better target text" in item_text_rows[2]["embedding_text"]

    embedding_summary = generate_embeddings(package_dir, embedding_dim=16, pca_dim=64, backend="deterministic_text_hash")
    assert embedding_summary["artifact_class"] == "adapter_scaffold_embedding"
    audit_row = audit(package_dir)
    assert audit_row["diagnosis"] == "ready_for_llmesr_scorer_wrapper"
    assert audit_row["ready_for_scoring"] is True
    assert audit_row["itm_emb_dim"] == 16
    assert audit_row["pca64_emb_dim"] == 64

    score_summary = score_adapter(package_dir)
    assert score_summary["artifact_class"] == "adapter_scaffold_score"
    assert score_summary["score_coverage_rate"] == 1.0
    score_rows = list(csv.DictReader((package_dir / "llmesr_scaffold_scores.csv").open(encoding="utf-8")))
    assert len(score_rows) == 2
    assert set(score_rows[0]) == {"source_event_id", "user_id", "item_id", "score"}
    assert all(float(row["score"]) == float(row["score"]) for row in score_rows)

    monkeypatch.setattr(
        sentence_embeddings,
        "_encode_hf_mean_pool",
        lambda texts, **kwargs: np.ones((len(texts), 8), dtype=np.float32),
    )
    hf_summary = sentence_embeddings.generate_sentence_embeddings(
        package_dir,
        backend="hf_mean_pool",
        model_name="local-qwen3-8b",
        batch_size=2,
        pca_dim=64,
    )
    assert hf_summary["backend"] == "hf_mean_pool"
    assert hf_summary["model_name"] == "local-qwen3-8b"
    assert hf_summary["artifact_class"] == "adapter_text_embedding"
    assert hf_summary["embedding_dim"] == 8


def test_raw_metadata_args_fail_fast_when_root_or_domain_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _raw_metadata_args(tmp_path / "missing", "movies", allow_missing=False)

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    with pytest.raises(FileNotFoundError, match="no raw metadata file"):
        _raw_metadata_args(raw_root, "movies", allow_missing=False)

    assert _raw_metadata_args(raw_root, "movies", allow_missing=True) == []


def test_raw_metadata_args_emit_movie_metadata_path(tmp_path):
    raw_path = tmp_path / "amazon_movies" / "meta_Movies_and_TV.jsonl.gz"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("{}", encoding="utf-8")

    assert _raw_metadata_args(tmp_path, "movies", allow_missing=False) == ["--raw_metadata_path", str(raw_path)]
