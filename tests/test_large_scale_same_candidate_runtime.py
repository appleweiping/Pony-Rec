from __future__ import annotations

import csv
import json

import pandas as pd

from main_build_large_scale_same_candidate_runtime import (
    LargeScaleRuntimeConfig,
    build_large_scale_runtime,
)
from main_export_llmesr_same_candidate_task import main as export_llmesr_main


def _read_jsonl(path):
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_large_scale_runtime_builds_valid_and_test_task_packages(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed" / "toy"
    processed_dir.mkdir(parents=True)

    interactions = []
    for user_idx in range(1, 5):
        for seq_idx in range(1, 5):
            interactions.append(
                {
                    "user_id": f"u{user_idx}",
                    "item_id": f"u{user_idx}_i{seq_idx}",
                    "timestamp": user_idx * 100 + seq_idx,
                }
            )
    pd.DataFrame(interactions).to_csv(processed_dir / "interactions.csv", index=False)

    item_ids = [f"u{user_idx}_i{seq_idx}" for user_idx in range(1, 5) for seq_idx in range(1, 5)]
    item_ids.extend(f"neg{i}" for i in range(1, 15))
    pd.DataFrame(
        [
            {
                "item_id": item_id,
                "title": f"Title {item_id}",
                "candidate_text": f"Full text for {item_id}",
            }
            for item_id in item_ids
        ]
    ).to_csv(processed_dir / "items.csv", index=False)
    pd.DataFrame(
        [
            {
                "item_id": item_id,
                "popularity_group": "head" if idx % 3 == 0 else "tail",
            }
            for idx, item_id in enumerate(item_ids)
        ]
    ).to_csv(processed_dir / "popularity_stats.csv", index=False)

    summary = build_large_scale_runtime(
        LargeScaleRuntimeConfig(
            processed_dir=processed_dir,
            domain="toy",
            dataset_name="toy_dataset",
            output_root=tmp_path / "outputs",
            exp_prefix="toy_large3_2neg",
            user_limit=3,
            num_negatives=2,
            max_history_len=3,
            seed=7,
            splits=("valid", "test"),
            negative_sampling="popularity",
        )
    )

    assert summary["selected_users"] == 3
    valid_task = tmp_path / "outputs" / "baselines" / "external_tasks" / "toy_large3_2neg_valid_same_candidate"
    test_task = tmp_path / "outputs" / "baselines" / "external_tasks" / "toy_large3_2neg_test_same_candidate"

    valid_rows = _read_jsonl(valid_task / "ranking_valid.jsonl")
    test_rows = _read_jsonl(test_task / "ranking_test.jsonl")
    assert len(valid_rows) == 3
    assert len(test_rows) == 3
    assert all(row["num_candidates"] == 3 for row in test_rows)
    assert all(sum(row["candidate_labels"]) == 1 for row in test_rows)
    assert all("history_item_ids" in row for row in test_rows)

    valid_train = _read_csv(valid_task / "train_interactions.csv")
    test_train = _read_csv(test_task / "train_interactions.csv")
    assert len(valid_train) == 6
    assert len(test_train) == 9

    item_metadata = _read_csv(test_task / "item_metadata.csv")
    assert any(row["candidate_text"] == "Full text for u1_i1" for row in item_metadata)

    monkeypatch.setattr(
        "sys.argv",
        [
            "main_export_llmesr_same_candidate_task.py",
            "--task_dir",
            str(test_task),
            "--exp_name",
            "toy_llmesr_adapter",
            "--output_root",
            str(tmp_path / "outputs"),
        ],
    )
    export_llmesr_main()
    item_text_seed = _read_csv(
        tmp_path
        / "outputs"
        / "baselines"
        / "paper_adapters"
        / "toy_llmesr_adapter"
        / "item_text_seed.csv"
    )
    train_only = next(row for row in item_text_seed if row["item_id"] == "u1_i1")
    assert train_only["embedding_text"] == "Full text for u1_i1"
