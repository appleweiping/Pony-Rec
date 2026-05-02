from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.cli.infer import _run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a local model or LoRA adapter through the common inference path.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    import asyncio

    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        lora_cfg = yaml.safe_load(f) or {}
    cfg = lora_cfg.get("evaluation_experiment")
    if cfg is None:
        adapter_path = str(lora_cfg.get("adapter_output_dir", ""))
        processed_dir = str(lora_cfg.get("processed_dir", "data/processed/amazon_beauty"))
        cfg = {
            "run_type": str(lora_cfg.get("run_type", "pilot")),
            "seed": int(lora_cfg.get("seed", 42)),
            "method": "local_lora_listwise",
            "output_dir": str(lora_cfg.get("eval_output_dir", "outputs/lora/eval")),
            "dataset": {
                "dataset": str(lora_cfg.get("dataset", "amazon_beauty")),
                "domain": str(lora_cfg.get("domain", "Beauty")),
                "processed_dir": processed_dir,
            },
            "backend": {
                "backend": "hf_local",
                "model": str(lora_cfg.get("base_model", "local")),
                "model_name_or_path": str(lora_cfg.get("base_model", "")),
                "adapter_path": adapter_path,
                "runtime": {
                    "device_map": lora_cfg.get("device_map", "auto"),
                    "load_in_4bit": bool(lora_cfg.get("qlora", False)),
                    "trust_remote_code": bool(lora_cfg.get("trust_remote_code", False)),
                    "batch_size": int(lora_cfg.get("eval_batch_size", 1)),
                },
                "generation": {"max_new_tokens": int(lora_cfg.get("max_new_tokens", 512)), "temperature": 0.0},
            },
            "inference": {"prompt_id": "listwise_ranking_v1", "topk": int(lora_cfg.get("topk", 10))},
        }
    infer_args = argparse.Namespace(
        config=args.config,
        split=args.split,
        input_path=None,
        output_path=None,
        max_samples=args.max_samples,
        resume=False,
    )
    asyncio.run(_run(cfg, infer_args))


if __name__ == "__main__":
    main()
