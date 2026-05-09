from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from main_audit_llm2rec_adapter_package import audit as audit_llm2rec_adapter
from main_export_llm2rec_same_candidate_task import export_llm2rec_package
from main_generate_llm2rec_sentence_embeddings import generate_embeddings
from main_prepare_llm2rec_upstream_adapter import prepare_upstream_adapter
from main_score_llm2rec_same_candidate_adapter import score_adapter
from src.baselines.official_runner.contract import (
    build_base_provenance,
    candidate_key_count,
    git_commit,
    resolve_repo_dir,
    sha256_file,
    text,
)


LLM2REC_OFFICIAL_ENTRYPOINT = "evaluate_with_seqrec.py -> seqrec.runner.Runner(model_name=SASRec).run()"
DEFAULT_DATASET_ALIASES = {
    "beauty": "BeautySupplementary100Neg",
    "books": "BooksLarge10000_100Neg",
    "electronics": "ElectronicsLarge10000_100Neg",
    "movies": "MoviesLarge10000_100Neg",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _json_arg(value: str) -> dict[str, Any]:
    if not text(value):
        return {}
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    return payload


def _domain_exp_prefix(domain: str) -> str:
    if domain == "beauty":
        return "beauty_supplementary_smallerN_100neg"
    return f"{domain}_large10000_100neg"


def _default_adapter_exp_name(args: argparse.Namespace) -> str:
    explicit = text(getattr(args, "llm2rec_adapter_exp_name", ""))
    if explicit:
        return explicit
    return f"{_domain_exp_prefix(args.domain)}_llm2rec_official_adapter"


def _adapter_exp_name(args: argparse.Namespace) -> str:
    explicit_dir = text(getattr(args, "llm2rec_adapter_dir", ""))
    if explicit_dir:
        return Path(explicit_dir).expanduser().name
    return _default_adapter_exp_name(args)


def _adapter_output_root(args: argparse.Namespace) -> Path:
    explicit_dir = text(getattr(args, "llm2rec_adapter_dir", ""))
    if explicit_dir:
        adapter_dir = Path(explicit_dir).expanduser()
        parts = adapter_dir.parts
        if len(parts) >= 4 and parts[-3:-1] == ("baselines", "paper_adapters"):
            return Path(*parts[:-3])
        raise ValueError(
            "--llm2rec_adapter_dir must end with outputs/baselines/paper_adapters/<adapter_name> "
            "so export metadata and provenance stay aligned."
        )
    return Path(getattr(args, "output_root", "outputs")).expanduser()


def _dataset_alias(args: argparse.Namespace) -> str:
    explicit = text(getattr(args, "llm2rec_dataset_alias", ""))
    if explicit:
        return explicit
    return DEFAULT_DATASET_ALIASES.get(args.domain, f"{args.domain.title()}SameCandidate100Neg")


def _adapter_dir(args: argparse.Namespace) -> Path:
    return _adapter_output_root(args) / "baselines" / "paper_adapters" / _adapter_exp_name(args)


def _official_adapter_dir(args: argparse.Namespace) -> Path:
    provenance_path = Path(args.provenance_output_path).expanduser()
    return provenance_path.parent


def _path_exists(path_text: str) -> bool:
    return bool(text(path_text)) and Path(path_text).expanduser().exists()


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
    for rel_path in ("evaluate_with_seqrec.py", "seqrec/runner.py", "seqrec/trainer.py", "seqrec/models/SASRec/config.yaml"):
        if not (repo_dir / rel_path).exists():
            blockers.append(f"missing_llm2rec_official_file:{rel_path}")
    return blockers


def _score_audit(candidate_items_path: Path, scores_path: Path) -> dict[str, Any]:
    candidate_rows = _read_csv(candidate_items_path)
    score_rows = _read_csv(scores_path)
    candidate_keys = {
        (text(row.get("source_event_id")), text(row.get("user_id")), text(row.get("item_id")))
        for row in candidate_rows
    }
    score_keys = {
        (text(row.get("source_event_id")), text(row.get("user_id")), text(row.get("item_id")))
        for row in score_rows
    }
    duplicate_score_keys = len(score_rows) - len(score_keys)
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
    coverage = float(len(matched) / len(candidate_keys)) if candidate_keys else 0.0
    audit = {
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
    return audit


def _training_config(args: argparse.Namespace, *, dataset_alias: str, item_embedding_path: Path, repo_dir: Path) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model": "SASRec",
        "dataset": dataset_alias,
        "exp_type": "srec",
        "embedding": str(item_embedding_path),
        "seq_embedding": "",
        "run_id": text(getattr(args, "run_id", "")) or f"LLM2Rec_official_qwen3base_{args.domain}_100neg",
        "save": True,
    }
    if getattr(args, "llm2rec_ckpt_dir", ""):
        config["ckpt_dir"] = str(Path(args.llm2rec_ckpt_dir).expanduser())
    else:
        config["ckpt_dir"] = str((_official_adapter_dir(args) / "seqrec_ckpt").resolve())
    if getattr(args, "llm2rec_log_dir", ""):
        config["log_dir"] = str(Path(args.llm2rec_log_dir).expanduser())
    if getattr(args, "llm2rec_tensorboard_log_dir", ""):
        config["tensorboard_log_dir"] = str(Path(args.llm2rec_tensorboard_log_dir).expanduser())
    overrides = _json_arg(getattr(args, "baseline_hyperparameter_overrides_json", "{}"))
    config.update(overrides)
    return config


def _train_with_official_entrypoint(
    *,
    repo_dir: Path,
    config: dict[str, Any],
    log_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    argv = [
        "python",
        "evaluate_with_seqrec.py",
        "--model=SASRec",
        f"--dataset={config['dataset']}",
        "--exp_type=srec",
        f"--embedding={config['embedding']}",
    ]
    for key, value in sorted(config.items()):
        if key in {"model", "dataset", "exp_type", "embedding", "seq_embedding"}:
            continue
        argv.append(f"--{key}={repr(value)}")
    if text(config.get("seq_embedding")):
        argv.append(f"--seq_embedding={config['seq_embedding']}")

    summary = {
        "official_training_entrypoint": str(repo_dir / "evaluate_with_seqrec.py"),
        "official_training_command": argv,
        "official_training_log_path": str(log_path),
        "official_training_config": config,
    }
    if dry_run:
        summary["status"] = "dry_run_planned"
        return summary

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")
    with log_path.open("w", encoding="utf-8") as log_fh:
        result = subprocess.run(
            argv,
            cwd=repo_dir,
            env=env,
            text=True,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
    summary["returncode"] = result.returncode
    if result.returncode != 0:
        summary["status"] = "failed"
        summary["blocker"] = f"llm2rec_official_training_failed:returncode={result.returncode}"
        return summary

    ckpt_dir = Path(config["ckpt_dir"]).expanduser()
    candidates = sorted(ckpt_dir.glob("*.pth"), key=lambda item: item.stat().st_mtime, reverse=True)
    checkpoint_path = candidates[0] if candidates else None
    summary["status"] = "completed" if checkpoint_path else "failed"
    summary["checkpoint_path"] = str(checkpoint_path) if checkpoint_path else ""
    if checkpoint_path is None:
        summary["blocker"] = f"llm2rec_official_training_missing_checkpoint:{ckpt_dir}"
    return summary


def _compact_llm2rec_checkpoint(
    checkpoint_path: Path,
    *,
    keep_full_checkpoint: bool,
) -> dict[str, Any]:
    if keep_full_checkpoint or not checkpoint_path.exists():
        return {
            "status": "kept_full_checkpoint",
            "full_checkpoint_path": str(checkpoint_path),
            "compact_checkpoint_path": "",
            "removed_keys": [],
            "full_checkpoint_removed": False,
        }
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("state_dict") if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict) else payload
    if not isinstance(state_dict, dict):
        return {
            "status": "unsupported_checkpoint_payload",
            "full_checkpoint_path": str(checkpoint_path),
            "compact_checkpoint_path": "",
            "removed_keys": [],
            "full_checkpoint_removed": False,
        }
    remove_keys = [
        key
        for key, value in list(state_dict.items())
        if key.endswith("item_embedding.weight") and hasattr(value, "ndim") and getattr(value, "ndim", 0) == 2
    ]
    if not remove_keys:
        return {
            "status": "no_large_item_embedding_key_found",
            "full_checkpoint_path": str(checkpoint_path),
            "compact_checkpoint_path": str(checkpoint_path),
            "removed_keys": [],
            "full_checkpoint_removed": False,
        }
    compact_path = checkpoint_path.with_name(f"{checkpoint_path.stem}.compact_no_item_embedding{checkpoint_path.suffix}")
    for key in remove_keys:
        state_dict.pop(key, None)
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload["state_dict"] = state_dict
        torch.save(payload, compact_path)
    else:
        torch.save(state_dict, compact_path)
    full_size = checkpoint_path.stat().st_size
    compact_size = compact_path.stat().st_size
    checkpoint_path.unlink()
    return {
        "status": "compacted_externalized_item_embedding",
        "full_checkpoint_path": str(checkpoint_path),
        "compact_checkpoint_path": str(compact_path),
        "removed_keys": remove_keys,
        "full_checkpoint_size_bytes": full_size,
        "compact_checkpoint_size_bytes": compact_size,
        "full_checkpoint_removed": True,
        "note": (
            "The official LLM2Rec training path saved the frozen/precomputed item embedding table inside the "
            "SASRec checkpoint. The compact checkpoint removes that duplicated table and the scorer injects "
            "the same external Qwen3 item embedding .npy before loading the model."
        ),
    }


def run_llm2rec_official(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    method_cfg: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    repo_dir = resolve_repo_dir(method_cfg, getattr(args, "repo_dir", "")).resolve()
    dataset_alias = _dataset_alias(args)
    adapter_dir = _adapter_dir(args)
    official_dir = _official_adapter_dir(args)
    official_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = official_dir / "llm2rec_official_training.log"
    score_audit_path = official_dir / "llm2rec_official_score_audit.json"
    run_summary_path = official_dir / "llm2rec_official_run_summary.json"
    dry_run = bool(getattr(args, "dry_run", False))
    blockers = _repo_pin_blockers(repo_dir, method_cfg)

    if not _path_exists(getattr(args, "backbone_path", "")):
        blockers.append(f"missing_backbone_path:{getattr(args, 'backbone_path', '')}")

    adapter_metadata: dict[str, Any] = {}
    adapter_audit: dict[str, Any] = {}
    prepare_summary: dict[str, Any] = {}
    embedding_summary: dict[str, Any] = {}
    training_summary: dict[str, Any] = {}
    checkpoint_compaction_summary: dict[str, Any] = {}
    scoring_summary: dict[str, Any] = {}
    score_audit: dict[str, Any] = {}

    if not blockers:
        adapter_metadata = export_llm2rec_package(
            Path(args.task_dir).expanduser(),
            exp_name=_adapter_exp_name(args),
            output_root=_adapter_output_root(args),
            dataset_alias=dataset_alias,
            valid_task_dir=Path(args.valid_task_dir).expanduser() if text(getattr(args, "valid_task_dir", "")) else None,
        )
        adapter_audit = audit_llm2rec_adapter(adapter_dir)
        if not adapter_audit.get("ready_for_upstream_wrapper"):
            blockers.append("llm2rec_adapter_audit_not_ready")

    if not blockers:
        prepare_summary = prepare_upstream_adapter(
            adapter_dir,
            repo_dir,
            dataset_alias=dataset_alias,
            link_mode=getattr(args, "llm2rec_link_mode", "copy"),
            skip_patch=bool(getattr(args, "llm2rec_skip_patch", False)),
        )

    if not blockers:
        existing_embedding_arg = text(getattr(args, "llm2rec_item_embedding_path", ""))
        existing_embedding = Path(existing_embedding_arg).expanduser() if existing_embedding_arg else Path()
        if existing_embedding_arg and existing_embedding.exists() and not bool(getattr(args, "force_embeddings", False)):
            embedding_summary = {
                "status": "reused_existing_embedding",
                "output_path": str(existing_embedding),
                "upstream_item_embedding_path": str(existing_embedding),
                "backend": "external_precomputed",
            }
        else:
            embedding_summary = generate_embeddings(
                adapter_dir,
                backend=getattr(args, "embedding_backend", "hf_mean_pool"),
                model_name=getattr(args, "backbone_path", ""),
                batch_size=int(getattr(args, "embedding_batch_size", 8)),
                device=getattr(args, "device", "auto"),
                max_text_chars=int(getattr(args, "embedding_max_text_chars", 1200)),
                max_length=int(getattr(args, "embedding_max_length", 128)),
                trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
                torch_dtype=getattr(args, "torch_dtype", "auto"),
                hf_device_map=getattr(args, "hf_device_map", ""),
                deterministic_dim=int(getattr(args, "deterministic_dim", 384)),
                llm2rec_repo_dir=repo_dir,
                save_info=getattr(args, "llm2rec_save_info", "pony_qwen3_8b"),
            )
        item_embedding_path = Path(
            embedding_summary.get("upstream_item_embedding_path")
            or embedding_summary.get("output_path")
            or ""
        ).expanduser()
        if not item_embedding_path.exists():
            blockers.append(f"missing_llm2rec_item_embedding_path:{item_embedding_path}")

    if not blockers:
        training_config = _training_config(args, dataset_alias=dataset_alias, item_embedding_path=item_embedding_path, repo_dir=repo_dir)
        existing_checkpoint_arg = text(getattr(args, "adapter_or_checkpoint_path", ""))
        existing_checkpoint = Path(existing_checkpoint_arg).expanduser() if existing_checkpoint_arg else Path()
        if existing_checkpoint_arg:
            if existing_checkpoint.exists():
                training_summary = {
                    "status": "completed",
                    "checkpoint_path": str(existing_checkpoint),
                    "official_training_config": training_config,
                    "official_training_log_path": str(training_log_path),
                    "reused_existing_checkpoint": True,
                    "note": "Reused --adapter_or_checkpoint_path; official training was not rerun in this recovery pass.",
                }
            else:
                blockers.append(f"missing_adapter_or_checkpoint_path:{existing_checkpoint}")
        else:
            training_summary = _train_with_official_entrypoint(
                repo_dir=repo_dir,
                config=training_config,
                log_path=training_log_path,
                dry_run=dry_run,
            )
            if training_summary.get("status") != "completed":
                blockers.append(text(training_summary.get("blocker")) or text(training_summary.get("status")) or "llm2rec_official_training_incomplete")

    checkpoint_path = Path(text(training_summary.get("checkpoint_path"))).expanduser() if training_summary.get("checkpoint_path") else Path()
    if not blockers and checkpoint_path.exists():
        checkpoint_compaction_summary = _compact_llm2rec_checkpoint(
            checkpoint_path,
            keep_full_checkpoint=bool(getattr(args, "llm2rec_keep_full_checkpoint", False)),
        )
        compact_path = text(checkpoint_compaction_summary.get("compact_checkpoint_path"))
        if compact_path:
            checkpoint_path = Path(compact_path).expanduser()
            training_summary["checkpoint_path"] = str(checkpoint_path)
        if not checkpoint_path.exists():
            blockers.append(f"llm2rec_compact_checkpoint_missing:{checkpoint_path}")

    if not blockers and checkpoint_path.exists():
        scoring_summary = score_adapter(
            adapter_dir,
            llm2rec_repo_dir=repo_dir,
            checkpoint_path=checkpoint_path,
            item_embedding_path=item_embedding_path,
            model_name="SASRec",
            output_scores_path=Path(args.output_scores_path).expanduser(),
            device=getattr(args, "device", "auto"),
            batch_size=int(getattr(args, "score_batch_size", 128)),
            max_seq_length=int(training_summary.get("official_training_config", {}).get("max_seq_length", training_config.get("max_seq_length", 10))),
            embedding_padding="auto",
        )
        score_audit = _score_audit(Path(args.task_dir).expanduser() / "candidate_items.csv", Path(args.output_scores_path).expanduser())
        _write_json(score_audit, score_audit_path)
        if not score_audit.get("audit_ok"):
            blockers.append("llm2rec_same_candidate_score_audit_failed")

    run_summary = {
        "method": "llm2rec",
        "domain": args.domain,
        "dataset_alias": dataset_alias,
        "adapter_dir": str(adapter_dir),
        "official_adapter_dir": str(official_dir),
        "official_repo_dir": str(repo_dir),
        "adapter_metadata": adapter_metadata,
        "adapter_audit": adapter_audit,
        "prepare_summary": prepare_summary,
        "embedding_summary": embedding_summary,
        "training_summary": training_summary,
        "checkpoint_compaction_summary": checkpoint_compaction_summary,
        "scoring_summary": scoring_summary,
        "score_audit": score_audit,
        "blockers": blockers,
    }
    _write_json(run_summary, run_summary_path)

    score_coverage_rate = score_audit.get("score_coverage_rate") if score_audit else None
    status = "official_completed" if not blockers and score_coverage_rate == 1.0 else "official_blocked"
    provenance = build_base_provenance(
        args=args,
        cfg=cfg,
        method_cfg=method_cfg,
        contract=contract,
        implementation_status=status,
        stage="run",
        blockers=blockers,
        official_entrypoint=LLM2REC_OFFICIAL_ENTRYPOINT,
        score_coverage_rate=score_coverage_rate,
        extra={
            "runner_support_level": "official_llm2rec_qwen3base_sasrec",
            "dataset_alias": dataset_alias,
            "adapter_dir": str(adapter_dir),
            "adapter_audit": adapter_audit,
            "llm2rec_prepare_summary_path": str(adapter_dir / "llm2rec_upstream_prepare_summary.json"),
            "qwen3_item_embedding_path": text(
                embedding_summary.get("upstream_item_embedding_path") or embedding_summary.get("output_path")
            ),
            "qwen3_item_embedding_sha256": sha256_file(
                embedding_summary.get("upstream_item_embedding_path") or embedding_summary.get("output_path") or ""
            ),
            "official_training_or_adaptation_entrypoint": LLM2REC_OFFICIAL_ENTRYPOINT,
            "official_training_log_path": str(training_log_path),
            "official_training_config": training_summary.get("official_training_config", {}),
            "official_training_command": training_summary.get("official_training_command", []),
            "adapter_or_checkpoint_path": text(training_summary.get("checkpoint_path")),
            "adapter_or_checkpoint_sha256": sha256_file(training_summary.get("checkpoint_path") or ""),
            "adapter_or_checkpoint_kind": "official_llm2rec_sasrec_checkpoint",
            "checkpoint_compaction_summary": checkpoint_compaction_summary,
            "same_candidate_score_audit": score_audit,
            "same_candidate_score_audit_path": str(score_audit_path),
            "run_summary_path": str(run_summary_path),
            "baseline_hyperparameter_overrides": _json_arg(getattr(args, "baseline_hyperparameter_overrides_json", "{}")),
            "candidate_key_count": candidate_key_count(args.task_dir),
        },
    )
    return provenance
