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
from main_train_score_proex_upstream_adapter import train_and_score
from src.baselines.official_runner.contract import (
    build_base_provenance,
    candidate_key_count,
    git_commit,
    resolve_repo_dir,
    sha256_file,
    text,
)


PROEX_OFFICIAL_ENTRYPOINT = (
    "encoder/train_encoder.py audited; runner imports encoder.models.general_cf."
    "lightgcn_proex.LightGCN_proex from the pinned ProRec repo and preserves "
    "the official BPR, invariant, and profile extrapolation losses while "
    "exporting exact same-candidate scores."
)
PROEX_HPARAM_SOURCE = (
    "pinned ProRec encoder/config/modelconf/lightgcn_proex.yml defaults "
    "(amazon discriminative LightGCN-ProEx: layer_num=3, reg_weight=1e-6, "
    "num_envs=4, inv_weight=10, ex_weight=0.4, alpha=0.1, "
    "embedding_size=32, lr=1e-3, batch_size=4096, epoch=3000)."
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
    explicit = text(getattr(args, "proex_adapter_exp_name", ""))
    if explicit:
        return explicit
    return f"{_domain_exp_prefix(args.domain)}_proex_official_adapter"


def _adapter_exp_name(args: argparse.Namespace) -> str:
    explicit_dir = text(getattr(args, "proex_adapter_dir", ""))
    if explicit_dir:
        return Path(explicit_dir).expanduser().name
    return _default_adapter_exp_name(args)


def _adapter_output_root(args: argparse.Namespace) -> Path:
    explicit_dir = text(getattr(args, "proex_adapter_dir", ""))
    if explicit_dir:
        adapter_dir = Path(explicit_dir).expanduser()
        parts = adapter_dir.parts
        if len(parts) >= 4 and parts[-3:-1] == ("baselines", "paper_adapters"):
            return Path(*parts[:-3])
        raise ValueError(
            "--proex_adapter_dir must end with outputs/baselines/paper_adapters/<adapter_name> "
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
        "README.md",
        "encoder/train_encoder.py",
        "encoder/config/modelconf/lightgcn_proex.yml",
        "encoder/models/general_cf/lightgcn_proex.py",
        "encoder/models/loss_utils.py",
        "encoder/models/base_model.py",
        "encoder/data_utils/data_handler_general_cf.py",
    ):
        if not (repo_dir / rel_path).exists():
            blockers.append(f"missing_proex_official_file:{rel_path}")
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
        "model_name": "lightgcn_proex",
        "epochs": int(getattr(args, "proex_epochs", 3000)),
        "batch_size": int(getattr(args, "proex_train_batch_size", 4096)),
        "seed": int(getattr(args, "seed", 2025)),
        "lr": float(getattr(args, "proex_lr", 0.001)),
        "weight_decay": float(getattr(args, "proex_weight_decay", 0.0)),
        "embedding_size": int(getattr(args, "proex_embedding_size", 32)),
        "layer_num": int(getattr(args, "proex_layer_num", 3)),
        "reg_weight": float(getattr(args, "proex_reg_weight", 1.0e-6)),
        "num_envs": int(getattr(args, "proex_num_envs", 4)),
        "inv_weight": float(getattr(args, "proex_inv_weight", 10.0)),
        "ex_weight": float(getattr(args, "proex_ex_weight", 0.4)),
        "alpha": float(getattr(args, "proex_alpha", 0.1)),
        "keep_rate": float(getattr(args, "proex_keep_rate", 1.0)),
    }


def _train_args(args: argparse.Namespace, *, adapter_dir: Path, repo_dir: Path, official_dir: Path) -> argparse.Namespace:
    hparams = _official_hparams(args)
    checkpoint_arg = text(getattr(args, "adapter_or_checkpoint_path", ""))
    checkpoint_path = Path(checkpoint_arg).expanduser() if checkpoint_arg else official_dir / "proex_official_model.pt"
    return argparse.Namespace(
        adapter_dir=str(adapter_dir),
        prorec_repo_dir=str(repo_dir),
        output_scores_path=str(Path(args.output_scores_path).expanduser()),
        checkpoint_path=str(checkpoint_path),
        dataset_name=f"pony_proex_{_adapter_exp_name(args)}",
        epochs=hparams["epochs"],
        batch_size=hparams["batch_size"],
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
        embedding_size=hparams["embedding_size"],
        layer_num=hparams["layer_num"],
        reg_weight=hparams["reg_weight"],
        num_envs=hparams["num_envs"],
        inv_weight=hparams["inv_weight"],
        ex_weight=hparams["ex_weight"],
        alpha=hparams["alpha"],
        keep_rate=hparams["keep_rate"],
        device=getattr(args, "device", "auto"),
        seed=hparams["seed"],
        log_every=int(getattr(args, "proex_log_every", 10)),
    )


def run_proex_official(
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
    score_audit_path = official_dir / "proex_official_score_audit.json"
    run_summary_path = official_dir / "proex_official_run_summary.json"
    blockers = _repo_pin_blockers(repo_dir, method_cfg)

    if not text(getattr(args, "backbone_path", "")) or not Path(args.backbone_path).expanduser().exists():
        blockers.append(f"missing_backbone_path:{getattr(args, 'backbone_path', '')}")
    if getattr(args, "embedding_backend", "") == "deterministic_text_hash":
        blockers.append("proex_official_requires_real_text_embeddings_not_deterministic_hash")

    adapter_metadata: dict[str, Any] = {}
    adapter_audit: dict[str, Any] = {}
    embedding_summary: dict[str, Any] = {}
    training_summary: dict[str, Any] = {}
    score_audit: dict[str, Any] = {}

    if not blockers:
        adapter_metadata = export_llmesr_package(
            Path(args.task_dir).expanduser(),
            exp_name=_adapter_exp_name(args),
            output_root=_adapter_output_root(args),
            top_sim_users=100,
        )
        adapter_audit = audit_llmesr_adapter(adapter_dir)
        if not adapter_audit.get("ready_for_embedding_generation"):
            blockers.append("proex_adapter_audit_not_ready_for_embedding_generation")

    if not blockers:
        if adapter_audit.get("ready_for_scoring") and not bool(getattr(args, "force_embeddings", False)):
            embedding_summary = {
                "status": "reused_existing_proex_item_embeddings",
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
            blockers.append("proex_missing_qwen_item_embeddings")

    if not blockers:
        if bool(getattr(args, "dry_run", False)):
            training_summary = {"status": "dry_run_planned"}
            blockers.append("dry_run_no_proex_training_or_scores")
        else:
            training_summary = train_and_score(
                _train_args(args, adapter_dir=adapter_dir, repo_dir=repo_dir, official_dir=official_dir)
            )
            score_audit = _score_audit(Path(args.task_dir).expanduser() / "candidate_items.csv", Path(args.output_scores_path).expanduser())
            _write_json(score_audit, score_audit_path)
            if not score_audit.get("audit_ok"):
                blockers.append("proex_same_candidate_score_audit_failed")

    run_summary = {
        "method": "proex",
        "domain": args.domain,
        "adapter_dir": str(adapter_dir),
        "official_adapter_dir": str(official_dir),
        "official_repo_dir": str(repo_dir),
        "official_entrypoint": PROEX_OFFICIAL_ENTRYPOINT,
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
        official_entrypoint=PROEX_OFFICIAL_ENTRYPOINT,
        score_coverage_rate=score_coverage_rate,
        extra={
            "runner_support_level": "official_proex_qwen3base_profile",
            "adapter_dir": str(adapter_dir),
            "adapter_audit": adapter_audit,
            "official_training_or_adaptation_entrypoint": PROEX_OFFICIAL_ENTRYPOINT,
            "default_hparam_source_file_or_url": PROEX_HPARAM_SOURCE,
            "official_training_config": hparams,
            "baseline_hyperparameter_overrides": {
                "profile_source": {
                    "official_default": "downloaded LLM original and CoT-derived profile arrays",
                    "runner_value": "Qwen3-8B item representation plus train-graph profile views",
                    "reason": (
                        "experiment-wide unified Qwen3-8B backbone policy; preserves ProEx "
                        "LightGCN_proex architecture and losses while replacing profile-source "
                        "files with same-candidate-domain Qwen3-derived profile arrays"
                    ),
                }
            },
            "qwen3_item_embedding_path": text(embedding_summary.get("itm_emb_path")),
            "qwen3_item_embedding_sha256": sha256_file(embedding_summary.get("itm_emb_path") or ""),
            "official_proex_data_dir": text(training_summary.get("official_data_dir")),
            "proex_profile_bridge_manifest_path": text(training_summary.get("profile_bridge_manifest_path")),
            "official_proex_storage_policy": "profile_arrays_externalized_and_manifest_recorded",
            "adapter_or_checkpoint_path": text(training_summary.get("checkpoint_path")),
            "adapter_or_checkpoint_sha256": sha256_file(training_summary.get("checkpoint_path") or ""),
            "adapter_or_checkpoint_kind": "official_proex_qwen3_profile_checkpoint",
            "same_candidate_score_audit": score_audit,
            "same_candidate_score_audit_path": str(score_audit_path),
            "run_summary_path": str(run_summary_path),
            "candidate_key_count": candidate_key_count(args.task_dir),
            "wrapper_scope_note": (
                "The pinned ProRec LightGCN_proex model, BPR/invariant/profile "
                "extrapolation losses, and official amazon hyperparameters are "
                "authoritative. Local code only adapts same-candidate graph data, "
                "Qwen3-derived profile arrays, and exact score export; native "
                "full-catalog metrics are not imported."
            ),
        },
    )
    return provenance
