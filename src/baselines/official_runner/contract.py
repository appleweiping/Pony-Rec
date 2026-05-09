from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.exp_io import load_yaml


REQUIRED_SCORE_SCHEMA = ["source_event_id", "user_id", "item_id", "score"]
PRIMARY_VARIANT = "official_code_qwen3base_default_hparams_declared_adaptation"


def text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    if not text(path):
        return ""
    file_path = Path(path).expanduser()
    if not file_path.exists() or not file_path.is_file():
        return ""
    digest = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_dir: str | Path) -> str:
    repo_path = Path(repo_dir).expanduser()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def resolve_method_config(config_path: str | Path, method: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cfg = load_yaml(config_path)
    method_cfg = (cfg.get("official_baselines") or {}).get(method)
    if not isinstance(method_cfg, dict):
        raise ValueError(f"Unknown official baseline method {method!r} in {config_path}")
    return cfg, method_cfg, method_cfg.get("fairness_contract", {}) or {}


def resolve_repo_dir(method_cfg: dict[str, Any], repo_dir_arg: str = "") -> Path:
    if repo_dir_arg:
        return Path(repo_dir_arg).expanduser()
    env_name = text(method_cfg.get("local_repo_env"))
    default = text(method_cfg.get("local_repo_default"))
    return Path(os.environ.get(env_name, default)).expanduser()


def resolve_repo_dir_text(method_cfg: dict[str, Any], repo_dir_arg: str = "") -> str:
    if repo_dir_arg:
        return repo_dir_arg
    env_name = text(method_cfg.get("local_repo_env"))
    default = text(method_cfg.get("local_repo_default"))
    return os.environ.get(env_name, default)


def task_sources(task_dir: str | Path, valid_task_dir: str = "") -> dict[str, str]:
    task_path = Path(task_dir).expanduser()
    valid_path = Path(valid_task_dir).expanduser() if valid_task_dir else task_path
    return {
        "train_source": str(task_path / "train_interactions.csv"),
        "valid_source": str(valid_path / "ranking_valid.jsonl"),
        "test_source": str(task_path / "ranking_test.jsonl"),
        "candidate_source": str(task_path / "candidate_items.csv"),
    }


def candidate_key_count(task_dir: str | Path) -> int:
    candidate_path = Path(task_dir).expanduser() / "candidate_items.csv"
    if not candidate_path.exists():
        return 0
    return len(read_csv(candidate_path))


def inspect_task_package(task_dir: str | Path, valid_task_dir: str = "") -> list[str]:
    blockers: list[str] = []
    sources = task_sources(task_dir, valid_task_dir)
    required = ["train_source", "test_source", "candidate_source"]
    if valid_task_dir:
        required.append("valid_source")
    for key in required:
        if not Path(sources[key]).expanduser().exists():
            blockers.append(f"missing_{key}:{sources[key]}")
    return blockers


def build_base_provenance(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    method_cfg: dict[str, Any],
    contract: dict[str, Any],
    implementation_status: str,
    stage: str,
    blockers: list[str],
    official_entrypoint: str,
    score_coverage_rate: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_dir = resolve_repo_dir(method_cfg, getattr(args, "repo_dir", ""))
    repo_dir_text = resolve_repo_dir_text(method_cfg, getattr(args, "repo_dir", ""))
    policy = cfg.get("fairness_policy", {}) or {}
    baseline_name = text(method_cfg.get("target_baseline_name"))
    sources = task_sources(args.task_dir, getattr(args, "valid_task_dir", ""))
    score_path = text(getattr(args, "output_scores_path", ""))
    provenance = {
        "schema_version": "official_runner_provenance_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": text(getattr(args, "run_id", "")),
        "stage": stage,
        "method": args.method,
        "domain": text(getattr(args, "domain", "")),
        "baseline_name": baseline_name,
        "implementation_status": implementation_status,
        "blockers": blockers,
        "fairness_policy_id": text(getattr(args, "fairness_policy_id", "")) or text(policy.get("policy_id")),
        "comparison_variant": text(getattr(args, "comparison_variant", "")) or text(policy.get("primary_table_variant")),
        "comparison_tier": text(contract.get("comparison_tier")),
        "official_code_required": bool(contract.get("official_code_required", True)),
        "official_repo": text(method_cfg.get("official_repo")),
        "official_repo_commit": git_commit(repo_dir),
        "pinned_commit": text(method_cfg.get("pinned_commit")),
        "official_entrypoint": official_entrypoint,
        "local_repo_env": text(method_cfg.get("local_repo_env")),
        "resolved_local_repo_path": repo_dir_text,
        "baseline_hyperparameter_source": text(getattr(args, "hparam_policy", ""))
        or text(contract.get("hparam_policy"))
        or text(policy.get("baseline_hyperparameter_policy")),
        "baseline_hyperparameter_overrides": {},
        "baseline_extra_tuning_allowed": bool(contract.get("extra_baseline_tuning_allowed", False)),
        "backbone_model_family": text(getattr(args, "backbone_model_family", "")) or text(policy.get("unified_backbone_family")),
        "backbone_model_path": text(getattr(args, "backbone_path", "")) or text(policy.get("unified_backbone_path")),
        "llm_adaptation_mode": text(getattr(args, "llm_adaptation_mode", "")),
        "adapter_or_checkpoint_path": text(getattr(args, "adapter_or_checkpoint_path", "")),
        "trainable_parameter_count": None,
        "total_parameter_count": None,
        "validation_selection_metric": text(getattr(args, "validation_selection_metric", "")) or "none",
        "test_set_model_selection_allowed": False,
        "same_candidate_task_sources": sources,
        "candidate_key_count": candidate_key_count(args.task_dir),
        "score_output_path": score_path,
        "score_output_sha256": sha256_file(score_path),
        "score_schema": REQUIRED_SCORE_SCHEMA,
        "score_coverage_rate": score_coverage_rate,
        "command_argv": sys.argv,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if extra:
        provenance.update(extra)
    return provenance
