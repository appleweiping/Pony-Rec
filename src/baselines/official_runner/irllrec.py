from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from main_audit_llmesr_adapter_package import audit as audit_llmesr_adapter
from main_export_llmesr_same_candidate_task import export_llmesr_package
from main_generate_llmesr_sentence_embeddings import generate_sentence_embeddings
from main_train_score_irllrec_upstream_adapter import train_and_score
from src.baselines.official_runner.contract import (
    build_base_provenance,
    candidate_key_count,
    git_commit,
    resolve_repo_dir,
    sha256_file,
    text,
)


IRLLREC_OFFICIAL_ENTRYPOINT = (
    "encoder/train_encoder.py audited; runner imports encoder.models.general_cf."
    "lightgcn_int.LightGCN_int from the pinned repo and preserves the official "
    "BPR, semantic alignment, and intent representation losses while exporting "
    "exact same-candidate scores."
)
IRLLREC_HPARAM_SOURCE = (
    "pinned IRLLRec encoder/config/modelconf/lightgcn_int.yml defaults "
    "(lightgcn_int, embedding_size=32, intent_num=128, layer_num=3, "
    "kd_weight=0.01, kd_int_weight=0.02, lr=1e-3, batch_size=4096, epoch=3000)"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _domain_exp_prefix(domain: str) -> str:
    if domain == "beauty":
        return "beauty_supplementary_smallerN_100neg"
    return f"{domain}_large10000_100neg"


def _default_adapter_exp_name(args: argparse.Namespace) -> str:
    explicit = text(getattr(args, "irllrec_adapter_exp_name", ""))
    if explicit:
        return explicit
    return f"{_domain_exp_prefix(args.domain)}_irllrec_official_adapter"


def _adapter_exp_name(args: argparse.Namespace) -> str:
    explicit_dir = text(getattr(args, "irllrec_adapter_dir", ""))
    if explicit_dir:
        return Path(explicit_dir).expanduser().name
    return _default_adapter_exp_name(args)


def _adapter_output_root(args: argparse.Namespace) -> Path:
    explicit_dir = text(getattr(args, "irllrec_adapter_dir", ""))
    if explicit_dir:
        adapter_dir = Path(explicit_dir).expanduser()
        parts = adapter_dir.parts
        if len(parts) >= 4 and parts[-3:-1] == ("baselines", "paper_adapters"):
            return Path(*parts[:-3])
        raise ValueError(
            "--irllrec_adapter_dir must end with outputs/baselines/paper_adapters/<adapter_name> "
            "so export metadata and provenance stay aligned."
        )
    return Path(getattr(args, "output_root", "outputs")).expanduser()


def _adapter_dir(args: argparse.Namespace) -> Path:
    return _adapter_output_root(args) / "baselines" / "paper_adapters" / _adapter_exp_name(args)


def _official_adapter_dir(args: argparse.Namespace) -> Path:
    return Path(args.provenance_output_path).expanduser().parent


def _repo_pin_blockers(repo_dir: Path, method_cfg: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not repo_dir.exists():
        return [f"missing_official_repo:{repo_dir}"]
    commit = git_commit(repo_dir)
    pinned = text(method_cfg.get("pinned_commit"))
    if not commit:
        blockers.append(f"cannot_read_official_repo_commit:{repo_dir}")
    elif pinned and commit != pinned:
        blockers.append(f"official_repo_commit_mismatch:{commit}!={pinned}")
    for rel_path in (
        "encoder/train_encoder.py",
        "encoder/config/modelconf/lightgcn_int.yml",
        "encoder/models/general_cf/lightgcn_int.py",
        "encoder/models/general_cf/lightgcn.py",
        "encoder/models/loss_utils.py",
        "encoder/models/base_model.py",
    ):
        if not (repo_dir / rel_path).exists():
            blockers.append(f"missing_irllrec_official_file:{rel_path}")
    return blockers


def _score_audit(candidate_items_path: Path, scores_path: Path) -> dict[str, Any]:
    candidate_rows = _read_csv(candidate_items_path)
    score_rows = _read_csv(scores_path) if scores_path.exists() else []
    candidate_keys = {
        (text(row.get("source_event_id")), text(row.get("user_id")), text(row.get("item_id")))
        for row in candidate_rows
    }
    score_keys = {
        (text(row.get("source_event_id")), text(row.get("user_id")), text(row.get("item_id")))
        for row in score_rows
    }
    finite_scores = 0
    for row in score_rows:
        try:
            value = float(row.get("score"))
        except Exception:
            continue
        if math.isfinite(value):
            finite_scores += 1
    matched = candidate_keys & score_keys
    missing = candidate_keys - score_keys
    extra = score_keys - candidate_keys
    duplicate_score_keys = len(score_rows) - len(score_keys)
    coverage = float(len(matched) / len(candidate_keys)) if candidate_keys else 0.0
    return {
        "candidate_rows": len(candidate_rows),
        "score_rows": len(score_rows),
        "candidate_keys": len(candidate_keys),
        "score_keys": len(score_keys),
        "matched_keys": len(matched),
        "missing_keys": len(missing),
        "extra_keys": len(extra),
        "duplicate_score_keys": duplicate_score_keys,
        "finite_scores": finite_scores,
        "score_coverage_rate": coverage,
        "audit_ok": (
            len(candidate_rows) == len(score_rows)
            and coverage == 1.0
            and not missing
            and not extra
            and duplicate_score_keys == 0
            and finite_scores == len(score_rows)
        ),
    }


def _official_hparams(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_name": "lightgcn_int",
        "epochs": int(getattr(args, "irllrec_epochs", 3000)),
        "batch_size": int(getattr(args, "irllrec_train_batch_size", 4096)),
        "seed": int(getattr(args, "seed", 2023)),
        "lr": float(getattr(args, "irllrec_lr", 0.001)),
        "weight_decay": float(getattr(args, "irllrec_weight_decay", 0.0)),
        "embedding_size": int(getattr(args, "irllrec_embedding_size", 32)),
        "intent_num": int(getattr(args, "irllrec_intent_num", 128)),
        "layer_num": int(getattr(args, "irllrec_layer_num", 3)),
        "reg_weight": float(getattr(args, "irllrec_reg_weight", 1.0e-5)),
        "kd_weight": float(getattr(args, "irllrec_kd_weight", 1.0e-2)),
        "kd_temperature": float(getattr(args, "irllrec_kd_temperature", 0.2)),
        "kd_int_weight": float(getattr(args, "irllrec_kd_int_weight", 2.0e-2)),
        "kd_int_temperature": float(getattr(args, "irllrec_kd_int_temperature", 0.2)),
        "kd_int_weight_2": float(getattr(args, "irllrec_kd_int_weight_2", 1.0e-7)),
        "kd_int_weight_3": float(getattr(args, "irllrec_kd_int_weight_3", 1.0e-7)),
        "ssl_con_max_nodes": int(getattr(args, "irllrec_ssl_con_max_nodes", 4096)),
        "keep_rate": float(getattr(args, "irllrec_keep_rate", 1.0)),
    }


def _train_args(args: argparse.Namespace, *, adapter_dir: Path, repo_dir: Path, official_dir: Path) -> argparse.Namespace:
    hparams = _official_hparams(args)
    checkpoint_arg = text(getattr(args, "adapter_or_checkpoint_path", ""))
    checkpoint_path = Path(checkpoint_arg).expanduser() if checkpoint_arg else official_dir / "irllrec_official_model.pt"
    return argparse.Namespace(
        adapter_dir=str(adapter_dir),
        irllrec_repo_dir=str(repo_dir),
        output_scores_path=str(Path(args.output_scores_path).expanduser()),
        checkpoint_path=str(checkpoint_path),
        model_name=hparams["model_name"],
        epochs=hparams["epochs"],
        batch_size=hparams["batch_size"],
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
        embedding_size=hparams["embedding_size"],
        intent_num=hparams["intent_num"],
        layer_num=hparams["layer_num"],
        reg_weight=hparams["reg_weight"],
        kd_weight=hparams["kd_weight"],
        kd_temperature=hparams["kd_temperature"],
        kd_int_weight=hparams["kd_int_weight"],
        kd_int_temperature=hparams["kd_int_temperature"],
        kd_int_weight_2=hparams["kd_int_weight_2"],
        kd_int_weight_3=hparams["kd_int_weight_3"],
        ssl_con_max_nodes=hparams["ssl_con_max_nodes"],
        keep_rate=hparams["keep_rate"],
        device=getattr(args, "device", "auto"),
        seed=hparams["seed"],
        log_every=int(getattr(args, "irllrec_log_every", 10)),
    )


def _existing_score_training_summary(
    *,
    args: argparse.Namespace,
    adapter_dir: Path,
    official_dir: Path,
    score_audit: dict[str, Any],
) -> dict[str, Any]:
    metadata_path = adapter_dir / "adapter_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    hparams = _official_hparams(args)
    checkpoint_arg = text(getattr(args, "adapter_or_checkpoint_path", ""))
    checkpoint_path = Path(checkpoint_arg).expanduser() if checkpoint_arg else official_dir / "irllrec_official_model.pt"
    return {
        "status": "reused_existing_scores_after_post_scoring_failure",
        "recovery_mode": "irllrec_reuse_existing_scores",
        "recovery_reason": (
            "Existing exact same-candidate scores were audited after an earlier run "
            "completed training/scoring but failed while serializing the large checkpoint."
        ),
        "adapter_dir": str(adapter_dir),
        "baseline_name": "irllrec_official_qwen3base_intent",
        "artifact_class": "official_irllrec_same_candidate_score",
        "official_result_gate": "provenance_coverage_and_import_required",
        "upstream_repo": "https://github.com/wangyu0627/IRLLRec",
        "official_model_class": "encoder.models.general_cf.lightgcn_int.LightGCN_int",
        "users": int(metadata.get("users", 0) or 0),
        "items": int(metadata.get("items", 0) or 0),
        "epochs": hparams["epochs"],
        "batch_size": hparams["batch_size"],
        "lr": hparams["lr"],
        "embedding_size": hparams["embedding_size"],
        "intent_num": hparams["intent_num"],
        "layer_num": hparams["layer_num"],
        "kd_weight": hparams["kd_weight"],
        "kd_int_weight": hparams["kd_int_weight"],
        "ssl_con_max_nodes": hparams["ssl_con_max_nodes"],
        "seed": hparams["seed"],
        "trainable_params": None,
        "final_train_loss": None,
        "official_data_dir": str(adapter_dir / "irllrec" / "handled"),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else "",
        "checkpoint_storage_decision": "not_required_for_same_candidate_evidence_recovery",
        "score_coverage_rate": score_audit.get("score_coverage_rate"),
        "scored_rows": score_audit.get("score_rows"),
        "candidate_rows": score_audit.get("candidate_rows"),
        "output_scores_path": str(Path(args.output_scores_path).expanduser()),
        "note": (
            "Recovered from an existing score CSV after post-scoring checkpoint serialization failed. "
            "The score file still passes the exact same-candidate audit before provenance is marked complete."
        ),
    }


def run_irllrec_official(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    method_cfg: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    repo_dir = resolve_repo_dir(method_cfg, getattr(args, "repo_dir", "")).resolve()
    adapter_dir = _adapter_dir(args)
    official_dir = _official_adapter_dir(args)
    official_dir.mkdir(parents=True, exist_ok=True)
    score_audit_path = official_dir / "irllrec_official_score_audit.json"
    run_summary_path = official_dir / "irllrec_official_run_summary.json"
    blockers = _repo_pin_blockers(repo_dir, method_cfg)

    if not text(getattr(args, "backbone_path", "")) or not Path(args.backbone_path).expanduser().exists():
        blockers.append(f"missing_backbone_path:{getattr(args, 'backbone_path', '')}")
    if getattr(args, "embedding_backend", "") == "deterministic_text_hash":
        blockers.append("irllrec_official_requires_real_text_embeddings_not_deterministic_hash")

    adapter_metadata: dict[str, Any] = {}
    adapter_audit: dict[str, Any] = {}
    embedding_summary: dict[str, Any] = {}
    training_summary: dict[str, Any] = {}
    score_audit: dict[str, Any] = {}
    reuse_existing_scores = bool(getattr(args, "irllrec_reuse_existing_scores", False))

    if reuse_existing_scores:
        output_scores_path = Path(args.output_scores_path).expanduser()
        embedding_summary = {"status": "skipped_embedding_generation_reused_existing_scores"}
        if not output_scores_path.exists():
            blockers.append(f"irllrec_reuse_existing_scores_missing_score_file:{output_scores_path}")
        elif not blockers:
            score_audit = _score_audit(Path(args.task_dir).expanduser() / "candidate_items.csv", output_scores_path)
            _write_json(score_audit, score_audit_path)
            if not score_audit.get("audit_ok"):
                blockers.append("irllrec_existing_same_candidate_score_audit_failed")
            else:
                training_summary = _existing_score_training_summary(
                    args=args,
                    adapter_dir=adapter_dir,
                    official_dir=official_dir,
                    score_audit=score_audit,
                )

    if not reuse_existing_scores and not blockers:
        adapter_metadata = export_llmesr_package(
            Path(args.task_dir).expanduser(),
            exp_name=_adapter_exp_name(args),
            output_root=_adapter_output_root(args),
            top_sim_users=100,
        )
        adapter_audit = audit_llmesr_adapter(adapter_dir)
        if not adapter_audit.get("ready_for_embedding_generation"):
            blockers.append("irllrec_adapter_audit_not_ready_for_embedding_generation")

    if not reuse_existing_scores and not blockers:
        if adapter_audit.get("ready_for_scoring") and not bool(getattr(args, "force_embeddings", False)):
            embedding_summary = {
                "status": "reused_existing_IRLLRec_item_embeddings",
                "itm_emb_path": str(adapter_dir / "llm_esr" / "handled" / "itm_emb_np.pkl"),
            }
        else:
            embedding_summary = generate_sentence_embeddings(
                adapter_dir,
                backend=getattr(args, "embedding_backend", "hf_mean_pool"),
                model_name=getattr(args, "backbone_path", ""),
                batch_size=int(getattr(args, "embedding_batch_size", 8)),
                device=getattr(args, "device", "auto"),
                pca_dim=64,
                max_text_chars=int(getattr(args, "embedding_max_text_chars", 1200)),
                max_length=int(getattr(args, "embedding_max_length", 128)),
                trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
                torch_dtype=getattr(args, "torch_dtype", "auto"),
                hf_device_map=getattr(args, "hf_device_map", ""),
            )
        adapter_audit = audit_llmesr_adapter(adapter_dir)
        if not (adapter_dir / "llm_esr" / "handled" / "itm_emb_np.pkl").exists():
            blockers.append("irllrec_missing_qwen_item_embeddings")

    if not reuse_existing_scores and not blockers:
        if bool(getattr(args, "dry_run", False)):
            training_summary = {"status": "dry_run_planned"}
            blockers.append("dry_run_no_irllrec_training_or_scores")
        else:
            training_summary = train_and_score(
                _train_args(args, adapter_dir=adapter_dir, repo_dir=repo_dir, official_dir=official_dir)
            )
            score_audit = _score_audit(Path(args.task_dir).expanduser() / "candidate_items.csv", Path(args.output_scores_path).expanduser())
            _write_json(score_audit, score_audit_path)
            if not score_audit.get("audit_ok"):
                blockers.append("irllrec_same_candidate_score_audit_failed")

    run_summary = {
        "method": "irllrec",
        "domain": args.domain,
        "adapter_dir": str(adapter_dir),
        "official_adapter_dir": str(official_dir),
        "official_repo_dir": str(repo_dir),
        "official_entrypoint": IRLLREC_OFFICIAL_ENTRYPOINT,
        "adapter_metadata": adapter_metadata,
        "adapter_audit": adapter_audit,
        "embedding_summary": embedding_summary,
        "training_summary": training_summary,
        "score_audit": score_audit,
        "blockers": blockers,
    }
    _write_json(run_summary, run_summary_path)

    score_coverage_rate = score_audit.get("score_coverage_rate") if score_audit else None
    status = "official_completed" if not blockers and score_coverage_rate == 1.0 else "official_blocked"
    hparams = _official_hparams(args)
    provenance = build_base_provenance(
        args=args,
        cfg=cfg,
        method_cfg=method_cfg,
        contract=contract,
        implementation_status=status,
        stage="run",
        blockers=blockers,
        official_entrypoint=IRLLREC_OFFICIAL_ENTRYPOINT,
        score_coverage_rate=score_coverage_rate,
        extra={
            "runner_support_level": "official_irllrec_qwen3base_lightgcn_int",
            "adapter_dir": str(adapter_dir),
            "adapter_audit": adapter_audit,
            "official_training_or_adaptation_entrypoint": IRLLREC_OFFICIAL_ENTRYPOINT,
            "default_hparam_source_file_or_url": IRLLREC_HPARAM_SOURCE,
            "official_training_config": hparams,
            "baseline_hyperparameter_overrides": {},
            "scalability_bridge": training_summary.get("scalability_bridge", {}),
            "qwen3_item_embedding_path": text(embedding_summary.get("itm_emb_path")),
            "qwen3_item_embedding_sha256": sha256_file(embedding_summary.get("itm_emb_path") or ""),
            "official_irllrec_data_dir": text(training_summary.get("official_data_dir")),
            "official_irllrec_storage_policy": "semantic_embeddings_loaded_via_config_not_duplicated",
            "adapter_or_checkpoint_path": text(training_summary.get("checkpoint_path")),
            "adapter_or_checkpoint_sha256": sha256_file(training_summary.get("checkpoint_path") or ""),
            "adapter_or_checkpoint_kind": "official_irllrec_intent_checkpoint",
            "user_embedding_source": text(training_summary.get("user_embedding_source")),
            "same_candidate_score_audit": score_audit,
            "same_candidate_score_audit_path": str(score_audit_path),
            "run_summary_path": str(run_summary_path),
            "candidate_key_count": candidate_key_count(args.task_dir),
            "wrapper_scope_note": (
                "The pinned IRLLRec LightGCN_int model and official BPR, semantic "
                "alignment, and intent representation losses are authoritative. "
                "Local code only adapts same-candidate graph data, Qwen3 item "
                "embeddings, Qwen3-PCA64 intent artifacts, train-history user "
                "semantic/intent embeddings, and exact score export; native "
                "full-catalog metrics are not imported."
            ),
        },
    )
    return provenance
