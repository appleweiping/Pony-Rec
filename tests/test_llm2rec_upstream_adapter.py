from __future__ import annotations

import json

import numpy as np

from main_prepare_llm2rec_upstream_adapter import prepare_upstream_adapter
from main_score_llm2rec_same_candidate_adapter import (
    _candidate_groups,
    _load_item_embeddings,
    _load_upstream_config,
)


def test_prepare_llm2rec_upstream_adapter_installs_data_and_patches_maps(tmp_path):
    adapter_dir = tmp_path / "adapter"
    data_dir = adapter_dir / "llm2rec" / "data" / "beauty_same_candidate" / "downstream"
    data_dir.mkdir(parents=True)
    (data_dir / "data.txt").write_text("1 2 3\n", encoding="utf-8")
    (adapter_dir / "adapter_metadata.json").write_text(
        json.dumps(
            {
                "adapter_name": "llm2rec_same_candidate",
                "dataset_alias": "beauty_same_candidate",
            }
        ),
        encoding="utf-8",
    )

    repo_dir = tmp_path / "LLM2Rec"
    (repo_dir / "seqrec").mkdir(parents=True)
    (repo_dir / "seqrec" / "recdata.py").write_text(
        'def load_data():\n'
        '    source_dict = {\n'
        '        "Games_5core": "Video_Games/5-core/downstream",\n'
        '    }\n',
        encoding="utf-8",
    )
    (repo_dir / "extract_llm_embedding.py").write_text(
        'dataset_name_mappings = {\n'
        '    "Games_5core": "Video_Games/5-core/downstream",\n'
        '}\n',
        encoding="utf-8",
    )

    summary = prepare_upstream_adapter(adapter_dir, repo_dir)
    prepare_upstream_adapter(adapter_dir, repo_dir)

    assert summary["status"] == "llm2rec_upstream_adapter_prepared"
    assert (repo_dir / "data" / "beauty_same_candidate" / "downstream" / "data.txt").read_text(
        encoding="utf-8"
    ) == "1 2 3\n"
    recdata_text = (repo_dir / "seqrec" / "recdata.py").read_text(encoding="utf-8")
    extract_text = (repo_dir / "extract_llm_embedding.py").read_text(encoding="utf-8")
    assert recdata_text.count("# PONY_SAME_CANDIDATE_DATASETS: beauty_same_candidate") == 1
    assert extract_text.count("# PONY_SAME_CANDIDATE_DATASETS: beauty_same_candidate") == 1
    assert '    source_dict["beauty_same_candidate"] = "beauty_same_candidate/downstream"' in recdata_text
    assert 'dataset_name_mappings["beauty_same_candidate"] = "beauty_same_candidate/downstream"' in extract_text


def test_llm2rec_item_embedding_padding_auto_prepends_null_row(tmp_path):
    path = tmp_path / "emb.npy"
    np.save(path, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    matrix = _load_item_embeddings(path, expected_items=2, padding_mode="auto")

    assert matrix.shape == (3, 2)
    assert np.allclose(matrix[0], [0.0, 0.0])
    assert np.allclose(matrix[1], [1.0, 2.0])


def test_llm2rec_candidate_groups_sort_by_event_and_candidate_index():
    rows = [
        {"llm2rec_test_row_idx": "1", "candidate_index": "1", "item_id": "i4"},
        {"llm2rec_test_row_idx": "0", "candidate_index": "1", "item_id": "i2"},
        {"llm2rec_test_row_idx": "0", "candidate_index": "0", "item_id": "i1"},
    ]

    groups = _candidate_groups(rows)

    assert [row["item_id"] for row in groups[0]] == ["i1", "i2"]
    assert [row["item_id"] for row in groups[1]] == ["i4"]


def test_llm2rec_upstream_config_uses_adapter_item_pool(tmp_path):
    repo_dir = tmp_path / "LLM2Rec"
    (repo_dir / "seqrec" / "models" / "SASRec").mkdir(parents=True)
    (repo_dir / "seqrec" / "default.yaml").write_text("max_seq_length: 50\ndropout: 0.1\n", encoding="utf-8")
    (repo_dir / "seqrec" / "models" / "SASRec" / "config.yaml").write_text(
        "hidden_size: 128\nadapter_dims: [-1]\n",
        encoding="utf-8",
    )

    config = _load_upstream_config(
        repo_dir,
        model="SASRec",
        item_count=1183,
        max_seq_length=10,
        hidden_size=64,
        dropout=None,
    )

    assert config["item_num"] == 1183
    assert config["select_pool"] == [1, 1184]
    assert config["eos_token"] == 1184
    assert config["max_seq_length"] == 10
    assert config["hidden_size"] == 64
