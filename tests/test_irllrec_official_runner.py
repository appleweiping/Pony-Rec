from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from src.baselines.official_runner.contract import resolve_method_config
from src.baselines.official_runner.irllrec import run_irllrec_official


PINNED_IRLLREC_COMMIT = "ee8330b456e568f87869ec6c0c553c55d43fce6e"


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_task(tmp_path: Path) -> Path:
    task_dir = tmp_path / "task"
    _write_csv(
        task_dir / "train_interactions.csv",
        [
            {"user_id": "u1", "item_id": "i1", "timestamp": 1, "sequence_index": 1},
            {"user_id": "u1", "item_id": "i2", "timestamp": 2, "sequence_index": 2},
            {"user_id": "u2", "item_id": "i2", "timestamp": 1, "sequence_index": 1},
            {"user_id": "u2", "item_id": "i3", "timestamp": 2, "sequence_index": 2},
        ],
        ["user_id", "item_id", "timestamp", "sequence_index"],
    )
    _write_csv(
        task_dir / "candidate_items.csv",
        [
            {"source_event_id": "e1", "user_id": "u1", "item_id": "i2", "candidate_index": 0, "candidate_title": "Two"},
            {"source_event_id": "e1", "user_id": "u1", "item_id": "i3", "candidate_index": 1, "candidate_title": "Three"},
        ],
        ["source_event_id", "user_id", "item_id", "candidate_index", "candidate_title"],
    )
    (task_dir / "ranking_test.jsonl").write_text(json.dumps({"source_event_id": "e1"}) + "\n", encoding="utf-8")
    return task_dir


def _make_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "IRLLRec"
    for rel_path in [
        "encoder/train_encoder.py",
        "encoder/config/modelconf/lightgcn_int.yml",
        "encoder/models/general_cf/lightgcn_int.py",
        "encoder/models/general_cf/lightgcn.py",
        "encoder/models/loss_utils.py",
        "encoder/models/base_model.py",
    ]:
        path = repo_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# stub\n", encoding="utf-8")
    return repo_dir


def _args(tmp_path: Path, *, task_dir: Path, repo_dir: Path, dry_run: bool = False) -> argparse.Namespace:
    output_dir = tmp_path / "official"
    return argparse.Namespace(
        method="irllrec",
        stage="run",
        domain="beauty",
        task_dir=str(task_dir),
        valid_task_dir="",
        output_scores_path=str(output_dir / "irllrec_official_scores.csv"),
        provenance_output_path=str(output_dir / "fairness_provenance.json"),
        fairness_policy_id="official_code_qwen3base_default_hparams_declared_adaptation_v1",
        comparison_variant="official_code_qwen3base_default_hparams_declared_adaptation",
        backbone_model_family="Qwen3-8B",
        backbone_path=str(tmp_path / "Qwen3-8B"),
        llm_adaptation_mode="representation_learner",
        implementation_status="",
        repo_dir=str(repo_dir),
        work_dir="",
        adapter_or_checkpoint_path="",
        hparam_policy="official_default_or_recommended",
        baseline_hyperparameter_overrides_json="{}",
        validation_selection_metric="none",
        run_id="",
        seed=2023,
        device="cpu",
        output_root=str(tmp_path / "outputs"),
        dry_run=dry_run,
        force_embeddings=False,
        embedding_backend="hf_mean_pool",
        embedding_batch_size=2,
        embedding_max_text_chars=1200,
        embedding_max_length=16,
        torch_dtype="auto",
        hf_device_map="",
        trust_remote_code=False,
        deterministic_dim=384,
        score_batch_size=128,
        irllrec_adapter_exp_name="",
        irllrec_adapter_dir="",
        irllrec_epochs=3000,
        irllrec_train_batch_size=4096,
        irllrec_lr=0.001,
        irllrec_weight_decay=0.0,
        irllrec_embedding_size=32,
        irllrec_intent_num=128,
        irllrec_layer_num=3,
        irllrec_reg_weight=1.0e-5,
        irllrec_kd_weight=1.0e-2,
        irllrec_kd_temperature=0.2,
        irllrec_kd_int_weight=2.0e-2,
        irllrec_kd_int_temperature=0.2,
        irllrec_kd_int_weight_2=1.0e-7,
        irllrec_kd_int_weight_3=1.0e-7,
        irllrec_keep_rate=1.0,
        irllrec_log_every=10,
        irllrec_reuse_existing_scores=False,
    )


def _config() -> tuple[dict, dict, dict]:
    return resolve_method_config("configs/official_external_baselines.yaml", "irllrec")


def test_irllrec_official_runner_marks_completed_only_after_exact_scores(tmp_path, monkeypatch):
    task_dir = _make_task(tmp_path)
    repo_dir = _make_repo(tmp_path)
    (tmp_path / "Qwen3-8B").mkdir()
    args = _args(tmp_path, task_dir=task_dir, repo_dir=repo_dir)
    cfg, method_cfg, contract = _config()

    monkeypatch.setattr("src.baselines.official_runner.irllrec.git_commit", lambda _repo: PINNED_IRLLREC_COMMIT)

    def fake_embeddings(adapter_dir: Path, **_: object) -> dict[str, str]:
        handled = adapter_dir / "llm_esr" / "handled"
        with (handled / "itm_emb_np.pkl").open("wb") as fh:
            pickle.dump(np.ones((3, 8), dtype=np.float32), fh)
        with (handled / "pca64_itm_emb_np.pkl").open("wb") as fh:
            pickle.dump(np.ones((3, 4), dtype=np.float32), fh)
        (adapter_dir / "llmesr_embedding_metadata.json").write_text(
            json.dumps({"backend": "hf_mean_pool", "artifact_class": "adapter_text_embedding"}) + "\n",
            encoding="utf-8",
        )
        return {"itm_emb_path": str(handled / "itm_emb_np.pkl")}

    def fake_train(train_args: argparse.Namespace) -> dict[str, object]:
        candidate_rows = list(csv.DictReader((task_dir / "candidate_items.csv").open(newline="", encoding="utf-8")))
        _write_csv(
            Path(train_args.output_scores_path),
            [
                {"source_event_id": row["source_event_id"], "user_id": row["user_id"], "item_id": row["item_id"], "score": 1.0 - idx}
                for idx, row in enumerate(candidate_rows)
            ],
            ["source_event_id", "user_id", "item_id", "score"],
        )
        Path(train_args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        Path(train_args.checkpoint_path).write_bytes(b"checkpoint")
        return {
            "status": "completed",
            "checkpoint_path": train_args.checkpoint_path,
            "official_data_dir": str(tmp_path / "irllrec_data"),
            "user_embedding_source": "train_history_mean_of_qwen_item_embeddings",
            "user_intent_embedding_source": "train_history_mean_of_centered_qwen_pca64_item_embeddings",
            "trainable_params": 12,
        }

    monkeypatch.setattr("src.baselines.official_runner.irllrec.generate_sentence_embeddings", fake_embeddings)
    monkeypatch.setattr("src.baselines.official_runner.irllrec.train_and_score", fake_train)

    provenance = run_irllrec_official(args=args, cfg=cfg, method_cfg=method_cfg, contract=contract)

    assert provenance["implementation_status"] == "official_completed"
    assert provenance["blockers"] == []
    assert provenance["score_coverage_rate"] == 1.0
    assert provenance["runner_support_level"] == "official_irllrec_qwen3base_lightgcn_int"
    assert provenance["official_training_config"]["model_name"] == "lightgcn_int"
    assert provenance["same_candidate_score_audit"]["audit_ok"] is True


def test_irllrec_official_runner_blocks_hash_embeddings(tmp_path, monkeypatch):
    task_dir = _make_task(tmp_path)
    repo_dir = _make_repo(tmp_path)
    (tmp_path / "Qwen3-8B").mkdir()
    args = _args(tmp_path, task_dir=task_dir, repo_dir=repo_dir, dry_run=True)
    args.embedding_backend = "deterministic_text_hash"
    cfg, method_cfg, contract = _config()

    monkeypatch.setattr("src.baselines.official_runner.irllrec.git_commit", lambda _repo: PINNED_IRLLREC_COMMIT)

    provenance = run_irllrec_official(args=args, cfg=cfg, method_cfg=method_cfg, contract=contract)

    assert provenance["implementation_status"] == "official_blocked"
    assert "irllrec_official_requires_real_text_embeddings_not_deterministic_hash" in provenance["blockers"]
    assert provenance["score_coverage_rate"] is None
