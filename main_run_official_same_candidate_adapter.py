from __future__ import annotations

import argparse
from pathlib import Path

from src.baselines.official_runner.adapters import inspect_official_adapter, run_official_adapter
from src.baselines.official_runner.contract import resolve_method_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified official external-baseline same-candidate runner.")
    parser.add_argument("--config", default="configs/official_external_baselines.yaml")
    parser.add_argument("--method", required=True)
    parser.add_argument("--stage", choices=["inspect", "run"], default="inspect")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--task_dir", required=True)
    parser.add_argument("--valid_task_dir", default="")
    parser.add_argument("--output_scores_path", required=True)
    parser.add_argument("--provenance_output_path", required=True)
    parser.add_argument("--fairness_policy_id", required=True)
    parser.add_argument("--comparison_variant", required=True)
    parser.add_argument("--backbone_model_family", default="Qwen3-8B")
    parser.add_argument("--backbone_path", required=True)
    parser.add_argument("--llm_adaptation_mode", default="frozen_base_embedding")
    parser.add_argument("--implementation_status", default="")
    parser.add_argument("--repo_dir", default="")
    parser.add_argument("--work_dir", default="")
    parser.add_argument("--adapter_or_checkpoint_path", default="")
    parser.add_argument("--hparam_policy", default="official_default_or_recommended")
    parser.add_argument("--baseline_hyperparameter_overrides_json", default="{}")
    parser.add_argument("--validation_selection_metric", default="none")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force_embeddings", action="store_true")
    parser.add_argument("--embedding_backend", default="hf_mean_pool", choices=["hf_mean_pool", "sentence_transformers", "deterministic_text_hash"])
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--embedding_max_text_chars", type=int, default=1200)
    parser.add_argument("--embedding_max_length", type=int, default=128)
    parser.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--hf_device_map", default="")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--deterministic_dim", type=int, default=384)
    parser.add_argument("--score_batch_size", type=int, default=128)
    parser.add_argument("--llm2rec_dataset_alias", default="")
    parser.add_argument("--llm2rec_adapter_exp_name", default="")
    parser.add_argument("--llm2rec_adapter_dir", default="")
    parser.add_argument("--llm2rec_save_info", default="pony_qwen3_8b")
    parser.add_argument("--llm2rec_item_embedding_path", default="")
    parser.add_argument("--llm2rec_link_mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--llm2rec_skip_patch", action="store_true")
    parser.add_argument("--llm2rec_keep_full_checkpoint", action="store_true")
    parser.add_argument("--llm2rec_ckpt_dir", default="")
    parser.add_argument("--llm2rec_log_dir", default="")
    parser.add_argument("--llm2rec_tensorboard_log_dir", default="")
    parser.add_argument(
        "--allow_blocked_exit_zero",
        action="store_true",
        help="Return success for inspect/blocker reports. Run stage still returns non-zero when blocked.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg, method_cfg, contract = resolve_method_config(args.config, args.method)
    if args.stage == "inspect":
        provenance = inspect_official_adapter(args=args, cfg=cfg, method_cfg=method_cfg, contract=contract)
    else:
        provenance = run_official_adapter(args=args, cfg=cfg, method_cfg=method_cfg, contract=contract)

    output_path = Path(args.provenance_output_path).expanduser()
    write_json(provenance, output_path)
    status = provenance.get("implementation_status", "")
    blockers = provenance.get("blockers", [])
    print(f"Saved provenance: {output_path}")
    print(f"method={args.method} stage={args.stage} implementation_status={status}")
    print(f"blockers={len(blockers)}")
    for blocker in blockers:
        print(f"  - {blocker}")

    if args.stage == "inspect" and args.allow_blocked_exit_zero:
        return 0
    return 0 if status == "official_completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
