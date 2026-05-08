from __future__ import annotations

import csv
import json

import numpy as np

from main_audit_llm2rec_adapter_package import audit
from main_export_llm2rec_same_candidate_task import export_llm2rec_package
from main_generate_llm2rec_sentence_embeddings import generate_embeddings


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_llm2rec_export_writes_adapter_package(tmp_path):
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
                "candidate_index": 0,
                "item_id": "i3",
                "label": 1,
                "is_positive": 1,
                "candidate_title": "Target Item",
            },
            {
                "source_event_id": "u1::3",
                "user_id": "u1",
                "timestamp": 3,
                "candidate_index": 1,
                "item_id": "i4",
                "label": 0,
                "is_positive": 0,
                "candidate_title": "Negative Item",
            },
        ],
        [
            "source_event_id",
            "user_id",
            "timestamp",
            "candidate_index",
            "item_id",
            "label",
            "is_positive",
            "candidate_title",
        ],
    )
    (task_dir / "metadata.json").write_text(json.dumps({"exp_name": "source"}), encoding="utf-8")

    metadata = export_llm2rec_package(
        task_dir,
        exp_name="beauty_llm2rec_adapter",
        output_root=tmp_path / "outputs",
        dataset_alias="BeautySameCandidate",
    )

    package_dir = tmp_path / "outputs" / "baselines" / "paper_adapters" / "beauty_llm2rec_adapter"
    assert metadata["status"] == "adapter_package_only"
    assert metadata["users"] == 2
    assert metadata["items"] == 4
    assert metadata["candidate_events"] == 1
    assert metadata["candidate_rows"] == 2

    item_titles = json.loads((package_dir / "llm2rec" / "data" / "BeautySameCandidate" / "downstream" / "item_titles.json").read_text())
    assert item_titles["3"] == "Target Item"
    assert item_titles["4"] == "Negative Item"

    train_lines = (package_dir / "llm2rec" / "data" / "BeautySameCandidate" / "downstream" / "train_data.txt").read_text().splitlines()
    assert train_lines == ["1 2", "2 3"]
    test_lines = (package_dir / "llm2rec" / "data" / "BeautySameCandidate" / "downstream" / "test_data.txt").read_text().splitlines()
    assert test_lines == ["1 2 3"]

    mapped_candidates = list(csv.DictReader((package_dir / "candidate_items_mapped.csv").open(encoding="utf-8")))
    assert mapped_candidates[0]["llm2rec_user_idx"] == "1"
    assert mapped_candidates[0]["llm2rec_item_idx"] == "3"
    assert mapped_candidates[0]["llm2rec_test_row_idx"] == "0"
    assert mapped_candidates[1]["llm2rec_item_idx"] == "4"

    audit_row = audit(package_dir)
    assert audit_row["diagnosis"] == "ready_for_llm2rec_upstream_wrapper"
    assert audit_row["ready_for_embedding_generation"] is True
    assert audit_row["ready_for_upstream_wrapper"] is True
    assert audit_row["item_titles_valid"] is True
    assert audit_row["sequence_files_valid"] is True
    assert audit_row["same_candidate_events_valid"] is True


def test_llm2rec_sentence_embedding_generator_writes_padded_upstream_npy(tmp_path, monkeypatch):
    task_dir = tmp_path / "beauty_week8_same_candidate_external"
    _write_csv(
        task_dir / "train_interactions.csv",
        [
            {"user_id": "u1", "item_id": "i1", "timestamp": 1, "sequence_index": 0},
            {"user_id": "u1", "item_id": "i2", "timestamp": 2, "sequence_index": 1},
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
                "candidate_index": 0,
                "item_id": "i3",
                "label": 1,
                "is_positive": 1,
                "candidate_title": "Target Item",
            },
            {
                "source_event_id": "u1::3",
                "user_id": "u1",
                "timestamp": 3,
                "candidate_index": 1,
                "item_id": "i4",
                "label": 0,
                "is_positive": 0,
                "candidate_title": "Negative Item",
            },
        ],
        [
            "source_event_id",
            "user_id",
            "timestamp",
            "candidate_index",
            "item_id",
            "label",
            "is_positive",
            "candidate_title",
        ],
    )

    export_llm2rec_package(
        task_dir,
        exp_name="beauty_llm2rec_adapter",
        output_root=tmp_path / "outputs",
        dataset_alias="beauty_same_candidate",
    )
    adapter_dir = tmp_path / "outputs" / "baselines" / "paper_adapters" / "beauty_llm2rec_adapter"
    repo_dir = tmp_path / "LLM2Rec"

    monkeypatch.setattr(
        "main_generate_llm2rec_sentence_embeddings._encode_hf_mean_pool",
        lambda texts, **kwargs: np.ones((len(texts), 8), dtype=np.float32),
    )
    summary = generate_embeddings(
        adapter_dir,
        backend="hf_mean_pool",
        model_name="local-qwen3-8b",
        batch_size=2,
        llm2rec_repo_dir=repo_dir,
        save_info="pony_qwen3_8b",
    )

    matrix = np.load(adapter_dir / "llm2rec_item_embeddings.npy")
    assert summary["artifact_class"] == "llm2rec_style_text_embedding"
    assert summary["embedding_rows"] == 5
    assert matrix.shape == (5, 8)
    assert np.allclose(matrix[0], 0.0)
    assert np.allclose(matrix[1:], 1.0)
    upstream_path = repo_dir / "item_info" / "beauty_same_candidate" / "pony_qwen3_8b_title_item_embs.npy"
    assert upstream_path.exists()
    assert np.load(upstream_path).shape == (5, 8)


def test_llm2rec_export_uses_separate_validation_task(tmp_path):
    task_dir = tmp_path / "books_test_same_candidate"
    valid_task_dir = tmp_path / "books_valid_same_candidate"
    _write_csv(
        task_dir / "train_interactions.csv",
        [
            {"user_id": "u1", "item_id": "i1", "timestamp": 1, "sequence_index": 0},
            {"user_id": "u1", "item_id": "i2", "timestamp": 2, "sequence_index": 1},
        ],
        ["user_id", "item_id", "timestamp", "sequence_index"],
    )
    rows = [
        {
            "source_event_id": "u1::4",
            "user_id": "u1",
            "timestamp": 4,
            "candidate_index": 0,
            "item_id": "i3",
            "label": 1,
            "is_positive": 1,
            "candidate_title": "Test Positive",
        },
        {
            "source_event_id": "u1::4",
            "user_id": "u1",
            "timestamp": 4,
            "candidate_index": 1,
            "item_id": "i4",
            "label": 0,
            "is_positive": 0,
            "candidate_title": "Test Negative",
        },
    ]
    valid_rows = [
        {
            "source_event_id": "u1::3",
            "user_id": "u1",
            "timestamp": 3,
            "candidate_index": 0,
            "item_id": "i5",
            "label": 1,
            "is_positive": 1,
            "candidate_title": "Valid Positive",
        },
        {
            "source_event_id": "u1::3",
            "user_id": "u1",
            "timestamp": 3,
            "candidate_index": 1,
            "item_id": "i6",
            "label": 0,
            "is_positive": 0,
            "candidate_title": "Valid Negative",
        },
    ]
    fieldnames = [
        "source_event_id",
        "user_id",
        "timestamp",
        "candidate_index",
        "item_id",
        "label",
        "is_positive",
        "candidate_title",
    ]
    _write_csv(task_dir / "candidate_items.csv", rows, fieldnames)
    _write_csv(valid_task_dir / "candidate_items.csv", valid_rows, fieldnames)

    metadata = export_llm2rec_package(
        task_dir,
        exp_name="books_llm2rec_adapter",
        output_root=tmp_path / "outputs",
        dataset_alias="BooksSameCandidate",
        valid_task_dir=valid_task_dir,
    )

    data_dir = tmp_path / "outputs" / "baselines" / "paper_adapters" / "books_llm2rec_adapter" / "llm2rec" / "data" / "BooksSameCandidate" / "downstream"
    assert metadata["valid_candidate_rows"] == 2
    assert metadata["candidate_rows"] == 2
    assert (data_dir / "val_data.txt").read_text(encoding="utf-8").strip().endswith("5")
    assert (data_dir / "test_data.txt").read_text(encoding="utf-8").strip().endswith("3")
