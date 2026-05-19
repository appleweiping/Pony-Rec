from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.utils.exp_io import load_yaml


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    except ModuleNotFoundError:
        text = json.dumps(data, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")


def _domain_list(cfg: dict[str, Any], selected: str) -> list[str]:
    domains = cfg.get("domains") or {}
    if selected == "all":
        return list(domains)
    return [item.strip() for item in selected.split(",") if item.strip()]


def _base_lora_training() -> dict[str, Any]:
    return {
        "dry_run": False,
        "max_train_samples": None,
        "max_valid_samples": None,
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 0.0002,
        "warmup_ratio": 0.03,
        "max_seq_length": 1024,
        "logging_steps": 10,
        "save_strategy": "no",
        "eval_strategy": "epoch",
        "gradient_checkpointing": True,
        "bf16": True,
        "fp16": False,
        "include_reason": False,
        "bias": "none",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }


def _model_cfg() -> dict[str, Any]:
    return {
        "model_name_or_path": "/home/ajifang/models/Qwen/Qwen3-8B",
        "tokenizer_name_or_path": "/home/ajifang/models/Qwen/Qwen3-8B",
        "device": "cuda",
        "device_map": "cuda:0",
        "dtype": "bfloat16",
        "local_files_only": True,
        "trust_remote_code": True,
    }


def _signal_lora_config(cfg: dict[str, Any], domain: str) -> dict[str, Any]:
    shadow_cfg = cfg["shadow"]
    run_name = f"{domain}_large10000_100neg_signal_lora_shadow_v1"
    exp_prefix = cfg["domains"][domain]["exp_prefix"]
    return {
        "run_name": run_name,
        "domain": domain,
        "task": "shadow_signal_pointwise",
        "method_family": "shadow_signal_lora",
        "method_variant": "shadow_v1_signal_teacher_sft_large10000_100neg",
        "model_name": "qwen3_8b_local",
        "base_model_config": cfg["model_config"],
        "prompt_path": shadow_cfg["signal_prompt_path"],
        "train_input_path": f"data/processed/{exp_prefix}/shadow_signal_lora_train.jsonl",
        "valid_input_path": f"data/processed/{exp_prefix}/shadow_signal_lora_valid.jsonl",
        "eval_input_path": f"outputs/summary/week8_large10000_100neg_shadow_inputs/{domain}_shadow_test_pointwise.jsonl",
        "adapter_output_dir": f"artifacts/adapters/{run_name}",
        "framework_output_dir": f"outputs/{run_name}",
        "logs_dir": f"artifacts/logs/{run_name}",
        "output_root": cfg.get("output_root", "outputs"),
        "topk": cfg.get("topk", 10),
        "seed": cfg.get("seed", 20260506),
        "model": _model_cfg(),
        "training": _base_lora_training(),
        "evaluation": {
            "framework_exp_name": run_name,
            "run_inference_after_train": False,
            "metrics_output_path": f"outputs/{run_name}/tables/shadow_signal_metrics.csv",
        },
        "support_signals": {
            "teacher_signal_path": f"outputs/{domain}_large10000_100neg_qwen3_shadow_v1/calibrated/test_calibrated.jsonl",
            "teacher_variant": shadow_cfg["winner_signal_variant"],
            "protocol": cfg["protocol_name"],
        },
        "summary": {
            "startup_check_path": f"outputs/summary/{run_name}_startup_check.json",
            "dataset_preview_path": f"outputs/summary/{run_name}_dataset_preview.csv",
            "training_summary_path": f"artifacts/logs/{run_name}/training_summary.csv",
            "framework_manifest_path": f"outputs/{run_name}/framework_run_manifest.json",
        },
        "status_note": "Scaffold only. Build train/valid teacher data before running LoRA.",
    }


def _decision_lora_config(cfg: dict[str, Any], domain: str) -> dict[str, Any]:
    run_name = f"{domain}_large10000_100neg_decision_lora_shadow_v6"
    exp_prefix = cfg["domains"][domain]["exp_prefix"]
    return {
        "run_name": run_name,
        "domain": domain,
        "task": "shadow_v6_accept_revise_fallback",
        "method_family": "shadow_decision_lora",
        "method_variant": "validation_selected_shadow_v6_gate_sft_large10000_100neg",
        "model_name": "qwen3_8b_local",
        "base_model_config": cfg["model_config"],
        "prompt_path": "prompts/shadow_v6_signal_to_decision.txt",
        "train_input_path": f"data/processed/{exp_prefix}/shadow_decision_lora_train.jsonl",
        "valid_input_path": f"data/processed/{exp_prefix}/shadow_decision_lora_valid.jsonl",
        "eval_input_path": f"outputs/summary/week8_large10000_100neg_shadow_v6_gate_sweep/{domain}/test_selected_bridge_rows.jsonl",
        "adapter_output_dir": f"artifacts/adapters/{run_name}",
        "framework_output_dir": f"outputs/{run_name}",
        "logs_dir": f"artifacts/logs/{run_name}",
        "output_root": cfg.get("output_root", "outputs"),
        "topk": cfg.get("topk", 10),
        "seed": cfg.get("seed", 20260506),
        "model": _model_cfg(),
        "training": _base_lora_training(),
        "evaluation": {
            "framework_exp_name": run_name,
            "run_inference_after_train": False,
            "metrics_output_path": f"outputs/{run_name}/tables/decision_metrics.csv",
        },
        "support_signals": {
            "selected_gate_metrics_path": f"outputs/summary/week8_large10000_100neg_shadow_v6_gate_sweep/{domain}/selected_gate_test_metrics.csv",
            "bridge_rows_path": f"outputs/summary/week8_large10000_100neg_shadow_v6_gate_sweep/{domain}/test_selected_bridge_rows.jsonl",
            "protocol": cfg["protocol_name"],
        },
        "summary": {
            "startup_check_path": f"outputs/summary/{run_name}_startup_check.json",
            "dataset_preview_path": f"outputs/summary/{run_name}_dataset_preview.csv",
            "training_summary_path": f"artifacts/logs/{run_name}/training_summary.csv",
            "framework_manifest_path": f"outputs/{run_name}/framework_run_manifest.json",
        },
        "status_note": "Scaffold only. Requires validation-selected v6 bridge rows before training.",
    }


def _generative_lora_config(cfg: dict[str, Any], domain: str) -> dict[str, Any]:
    run_name = f"{domain}_large10000_100neg_generative_lora_title"
    exp_prefix = cfg["domains"][domain]["exp_prefix"]
    return {
        "run_name": run_name,
        "domain": domain,
        "task": "generated_title_recommendation_with_verification",
        "method_family": "shadow_verified_generative_lora",
        "method_variant": "catalog_grounded_generated_title_sft_large10000_100neg",
        "model_name": "qwen3_8b_local",
        "base_model_config": cfg["model_config"],
        "prompt_path": "prompts/candidate_ranking.txt",
        "train_input_path": f"data/processed/{exp_prefix}/generative_title_lora_train.jsonl",
        "valid_input_path": f"data/processed/{exp_prefix}/generative_title_lora_valid.jsonl",
        "eval_input_path": f"outputs/summary/week8_large10000_100neg_generated_title_verification/{domain}/test_verification_scaffold.jsonl",
        "adapter_output_dir": f"artifacts/adapters/{run_name}",
        "framework_output_dir": f"outputs/{run_name}",
        "logs_dir": f"artifacts/logs/{run_name}",
        "output_root": cfg.get("output_root", "outputs"),
        "topk": cfg.get("topk", 10),
        "seed": cfg.get("seed", 20260506),
        "model": _model_cfg(),
        "training": _base_lora_training(),
        "evaluation": {
            "framework_exp_name": run_name,
            "run_inference_after_train": False,
            "metrics_output_path": f"outputs/{run_name}/tables/generative_title_verification_metrics.csv",
        },
        "support_signals": {
            "verification_scaffold_path": f"outputs/summary/week8_large10000_100neg_generated_title_verification/{domain}/test_verification_scaffold.jsonl",
            "shadow_verifier_path": f"outputs/summary/week8_large10000_100neg_shadow_v6_gate_sweep/{domain}/selected_gate_test_metrics.csv",
            "protocol": cfg["protocol_name"],
        },
        "summary": {
            "startup_check_path": f"outputs/summary/{run_name}_startup_check.json",
            "dataset_preview_path": f"outputs/summary/{run_name}_dataset_preview.csv",
            "training_summary_path": f"artifacts/logs/{run_name}/training_summary.csv",
            "framework_manifest_path": f"outputs/{run_name}/framework_run_manifest.json",
        },
        "status_note": "Scaffold only. Replace catalog-title proxy with generated titles before final claims.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Week8 future LoRA config scaffolds.")
    parser.add_argument("--config", default="configs/week8_large_scale_future_framework.yaml")
    parser.add_argument("--domains", default="all")
    parser.add_argument("--output_dir", default="configs/week8_future_lora")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    domains = _domain_list(cfg, args.domains)
    output_dir = Path(args.output_dir)
    written: list[str] = []
    for domain in domains:
        configs = {
            f"{domain}_signal_lora_shadow_v1.yaml": _signal_lora_config(cfg, domain),
            f"{domain}_decision_lora_shadow_v6.yaml": _decision_lora_config(cfg, domain),
            f"{domain}_generative_lora_title.yaml": _generative_lora_config(cfg, domain),
        }
        for name, data in configs.items():
            path = output_dir / name
            _write_yaml(path, data)
            written.append(str(path))

    manifest = {
        "config_source": args.config,
        "domains": domains,
        "written_configs": written,
        "status": "scaffold_only",
        "note": "Run startup checks only after the referenced train/valid data builders exist.",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved LoRA scaffold manifest: {manifest_path}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
