from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.backends.openai_compatible_backend import build_openai_chat_payload, stable_cache_key
from src.baselines.recbole_adapter import ensure_recbole_installed, export_recbole_atomic
from src.cli.export_paper_tables import main as export_tables_main
from src.cli.infer import _run
from src.cli.train_lora import _format_sft_rows
from src.data.protocol import write_jsonl
from src.utils.manifest import backend_type_from_name, build_manifest, is_paper_result


def test_backend_type_rerank_maps() -> None:
    assert backend_type_from_name("rerank") == "rerank"


def test_deepseek_request_payload_shape_no_key() -> None:
    payload = build_openai_chat_payload(prompt="hello", model="deepseek-v4-flash", temperature=0.0, max_tokens=8)
    assert payload["model"] == "deepseek-v4-flash"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["temperature"] == 0.0


def test_deepseek_cache_key_stability() -> None:
    payload = build_openai_chat_payload(prompt="hello", model="deepseek-v4-pro")
    assert stable_cache_key(payload) == stable_cache_key(dict(reversed(list(payload.items()))))


def test_deepseek_payload_avoids_deprecated_defaults() -> None:
    payload = build_openai_chat_payload(prompt="x", model="deepseek-v4-flash")
    assert payload["model"] not in {"deepseek-chat", "deepseek-reasoner"}


def test_infer_resume_from_partial_output(tmp_path: Path) -> None:
    import asyncio

    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame(
        [
            {"item_id": "i1", "candidate_text": "Item 1"},
            {"item_id": "i2", "candidate_text": "Item 2"},
        ]
    ).to_csv(processed / "items.csv", index=False)
    samples = [
        {"user_id": "u1", "history_item_ids": [], "candidate_item_ids": ["i1", "i2"], "target_item_id": "i1"},
        {"user_id": "u2", "history_item_ids": [], "candidate_item_ids": ["i1", "i2"], "target_item_id": "i2"},
    ]
    input_path = tmp_path / "test_candidates.jsonl"
    output_path = tmp_path / "out" / "predictions" / "test_raw.jsonl"
    write_jsonl(samples, input_path)
    write_jsonl([{"already": True}], output_path)
    cfg = {
        "run_type": "smoke",
        "seed": 1,
        "method": "mock",
        "output_dir": str(tmp_path / "out"),
        "dataset": {"dataset": "d", "domain": "D", "processed_dir": str(processed)},
        "backend": {"backend": "mock"},
        "inference": {"prompt_id": "listwise_ranking_v1", "topk": 2},
    }
    args = argparse.Namespace(config="", split="test", input_path=str(input_path), output_path=str(output_path), max_samples=None, resume=True)
    asyncio.run(_run(cfg, args))
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["already"] is True


def test_manifest_paper_result_flags() -> None:
    manifest = build_manifest(config={"run_type": "full"}, dataset="d", domain="D", method="m", backend="deepseek", model="deepseek-v4-flash", prompt_template="p")
    assert manifest["backend_type"] == "api"
    assert manifest["is_paper_result"] is True
    assert is_paper_result("smoke", "api") is False
    assert backend_type_from_name("mock") == "mock"
    assert backend_type_from_name("lora") == "lora"


def test_recbole_export_atomic_format(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rating": 5, "timestamp": 1}]).to_csv(processed / "interactions.csv", index=False)
    result = export_recbole_atomic(processed_dir=processed, output_dir=tmp_path / "recbole", dataset_name="tiny")
    text = Path(result["inter_file"]).read_text(encoding="utf-8")
    assert "user_id:token" in text
    assert "item_id:token" in text


def test_recbole_missing_install_fails_clearly() -> None:
    try:
        ensure_recbole_installed()
    except ImportError as exc:
        assert "pip install recbole" in str(exc)
    else:
        assert True


def test_lora_standard_sft_data_format() -> None:
    rows = [{"history_item_ids": ["i1"], "candidate_item_ids": ["i2"], "target_item_id": "i2"}]
    out = _format_sft_rows(rows)
    assert out[0]["data_mode"] == "standard_sft"
    assert out[0]["weight"] == 1.0


def test_lora_uncertainty_pruned_data_format() -> None:
    rows = [
        {"history_item_ids": [], "candidate_item_ids": ["i1"], "target_item_id": "i1", "uncertainty_score": 0.2},
        {"history_item_ids": [], "candidate_item_ids": ["i2"], "target_item_id": "i2", "uncertainty_score": 0.9},
    ]
    out = _format_sft_rows(rows, data_mode="uncertainty_pruned", uncertainty_threshold=0.5)
    assert len(out) == 1
    assert "i1" in out[0]["response"]


def test_lora_uncertainty_weighted_data_format() -> None:
    rows = [{"history_item_ids": [], "candidate_item_ids": ["i1"], "target_item_id": "i1", "uncertainty_score": 0.25}]
    out = _format_sft_rows(rows, data_mode="uncertainty_weighted")
    assert out[0]["weight"] == pytest.approx(0.75)


def test_lora_curriculum_orders_by_uncertainty() -> None:
    rows = [
        {"history_item_ids": [], "candidate_item_ids": ["high"], "target_item_id": "high", "uncertainty_score": 0.9},
        {"history_item_ids": [], "candidate_item_ids": ["low"], "target_item_id": "low", "uncertainty_score": 0.1},
    ]
    out = _format_sft_rows(rows, data_mode="curriculum_uncertainty")
    assert "low" in out[0]["response"]


def test_export_paper_tables_refuses_smoke_mock_outputs(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate.csv"
    with aggregate.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metadata.dataset", "metadata.domain", "metadata.seed", "metadata.method", "metadata.config_hash", "metadata.backend_type", "metadata.run_type", "metadata.is_paper_result"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "metadata.dataset": "d",
                "metadata.domain": "D",
                "metadata.seed": "1",
                "metadata.method": "m",
                "metadata.config_hash": "abc",
                "metadata.backend_type": "mock",
                "metadata.run_type": "smoke",
                "metadata.is_paper_result": "False",
            }
        )
    old_argv = sys.argv
    sys.argv = ["export", "--aggregate_csv", str(aggregate), "--output_dir", str(tmp_path / "tables")]
    try:
        with pytest.raises(ValueError):
            export_tables_main()
    finally:
        sys.argv = old_argv


def test_export_paper_tables_allow_smoke_writes_tables(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate.csv"
    with aggregate.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "ranking.HR@1"])
        writer.writeheader()
        writer.writerow({"path": "smoke", "ranking.HR@1": "0.0"})
    old_argv = sys.argv
    sys.argv = ["export", "--aggregate_csv", str(aggregate), "--output_dir", str(tmp_path / "tables"), "--allow_smoke"]
    try:
        export_tables_main()
    finally:
        sys.argv = old_argv
    assert (tmp_path / "tables" / "main_results.md").exists()
