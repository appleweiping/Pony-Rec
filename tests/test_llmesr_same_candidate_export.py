from __future__ import annotations

import csv
import json
import pickle

import main_export_llmesr_same_candidate_task as exporter
from main_audit_llmesr_adapter_package import audit
from main_generate_llmesr_text_embeddings import generate_embeddings


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

    embedding_summary = generate_embeddings(package_dir, embedding_dim=16, pca_dim=64, backend="deterministic_text_hash")
    assert embedding_summary["artifact_class"] == "adapter_scaffold_embedding"
    audit_row = audit(package_dir)
    assert audit_row["diagnosis"] == "ready_for_llmesr_scorer_wrapper"
    assert audit_row["ready_for_scoring"] is True
    assert audit_row["itm_emb_dim"] == 16
    assert audit_row["pca64_emb_dim"] == 64
