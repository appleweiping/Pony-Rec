"""CARE-LoRA debug: build data (if needed), train two tiny adapters, infer beauty valid/test."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.cli.run_lora_debug_reprocessed import _gpu_mem_snapshot, _infer_split, validate_reprocess_candidate_rows
from src.data.protocol import read_jsonl, write_jsonl
from src.parsing import write_failure_taxonomy_csv
from src.utils.exp_io import load_yaml
from src.utils.manifest import backend_type_from_name, build_manifest, write_manifest
from src.utils.research_artifacts import config_hash, git_commit_or_unknown, utc_timestamp


def _apply_yaml_to_args(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    if cfg.get("domain"):
        args.domain = str(cfg["domain"])
    paths = cfg.get("paths") or {}
    if paths.get("output_root"):
        args.output_root = str(paths["output_root"])
    if paths.get("reprocess_dir"):
        args.reprocess_dir = str(paths["reprocess_dir"])
    if paths.get("processed_dir"):
        args.processed_dir = str(paths["processed_dir"])
    if paths.get("base_model"):
        args.base_model = str(paths["base_model"])
    tr = cfg.get("training") or {}
    if tr.get("max_train_steps") is not None:
        args.max_train_steps = int(tr["max_train_steps"])
    if tr.get("per_device_train_batch_size") is not None:
        args.per_device_train_batch_size = int(tr["per_device_train_batch_size"])
    if tr.get("gradient_accumulation_steps") is not None:
        args.gradient_accumulation_steps = int(tr["gradient_accumulation_steps"])
    if tr.get("learning_rate") is not None:
        args.learning_rate = float(tr["learning_rate"])
    if tr.get("lora_r") is not None:
        args.lora_r = int(tr["lora_r"])
    if tr.get("lora_alpha") is not None:
        args.lora_alpha = int(tr["lora_alpha"])
    if tr.get("max_seq_length") is not None:
        args.max_seq_length = int(tr["max_seq_length"])
    if tr.get("topk") is not None:
        args.topk = int(tr["topk"])
    if tr.get("prompt_id"):
        args.prompt_id = str(tr["prompt_id"])
    inf = cfg.get("inference") or {}
    if inf.get("max_new_tokens") is not None:
        args.max_new_tokens = int(inf["max_new_tokens"])
    if cfg.get("seed") is not None:
        args.seed = int(cfg["seed"])
    # stash extra paths for subprocess (not on argparse object by default)
    setattr(args, "_deepseek_root", str(paths.get("deepseek_pilot_root", "outputs/pilots/deepseek_v4_flash_processed_20u_c19_seed42")))
    setattr(args, "_care_rerank_config", str(paths.get("care_rerank_config", "configs/methods/care_rerank_pilot.yaml")))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=None, help="YAML config (e.g. configs/lora/care_qwen3_8b_debug.yaml). CLI flags override YAML.")
    p.add_argument("--domain", default="amazon_beauty")
    p.add_argument("--reprocess_dir", default="outputs/reprocessed_processed_source")
    p.add_argument("--processed_dir", default="data/processed/amazon_beauty")
    p.add_argument("--base_model", default="/home/ajifang/models/Qwen/Qwen3-8B")
    p.add_argument("--output_root", default="outputs/pilots/care_lora_qwen3_8b_beauty_20u_c19_seed42_debug")
    p.add_argument("--data_split", default="valid", choices=("valid", "test"), help="Split used to align CARE training rows (20 pilot users).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_steps", type=int, default=12)
    p.add_argument("--prompt_id", default="listwise_ranking_v1")
    p.add_argument("--topk", type=int, default=19)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--skip_build_data", action="store_true")
    p.add_argument("--skip_train", action="store_true", help="Only run inference using existing adapters.")
    ns = p.parse_args(argv)
    if ns.config:
        cfg = load_yaml(ns.config)
        if not isinstance(cfg, dict):
            raise ValueError(f"Config must be a mapping: {ns.config}")
        _apply_yaml_to_args(ns, cfg)
    if not getattr(ns, "_deepseek_root", None):
        setattr(ns, "_deepseek_root", "outputs/pilots/deepseek_v4_flash_processed_20u_c19_seed42")
    if not getattr(ns, "_care_rerank_config", None):
        setattr(ns, "_care_rerank_config", "configs/methods/care_rerank_pilot.yaml")
    return ns


class LossLogCallback(TrainerCallback):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not logs or "loss" not in logs:
            return
        rec = {"step": int(getattr(state, "global_step", 0)), "loss": float(logs["loss"])}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class WeightedTrainer(Trainer):
    """Scale LM loss by per-sequence sample_weight (batch size 1 debug)."""

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch: int | None = None):
        w = inputs.pop("sample_weight", None)
        outputs = model(**inputs)
        loss = outputs.loss
        if w is not None:
            wt = w.to(loss.device).to(loss.dtype)
            loss = loss * wt.view(-1)[0]
        return (loss, outputs) if return_outputs else loss


def _tokenize_sft(batch: dict[str, list], tokenizer: Any, max_len: int) -> dict[str, Any]:
    eos = tokenizer.eos_token or ""
    texts = [f"{p}\n{r}{eos}" for p, r in zip(batch["prompt"], batch["response"])]
    out = tokenizer(texts, truncation=True, max_length=max_len)
    if "sample_weight" in batch:
        out["sample_weight"] = list(batch["sample_weight"])
    return out


class WeightedLMDataCollator:
    """Wraps MLM collator and forwards per-row sample_weight for WeightedTrainer."""

    def __init__(self, tokenizer: Any) -> None:
        self.inner = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        feats = []
        weights: list[float] = []
        for f in features:
            f = dict(f)
            weights.append(float(f.pop("sample_weight", 1.0)))
            feats.append(f)
        batch = self.inner(feats)
        batch["sample_weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch


def _train_adapter_from_jsonl(
    *,
    jsonl_path: Path,
    args: argparse.Namespace,
    adapter_dir: Path,
    loss_path: Path,
    use_sample_weights: bool,
) -> dict[str, Any]:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(jsonl_path)
    if not rows:
        raise ValueError(f"No rows in {jsonl_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    records = []
    for r in rows:
        rec: dict[str, Any] = {"prompt": r["prompt"], "response": r["response"]}
        if use_sample_weights:
            rec["sample_weight"] = float(r.get("sample_weight", 1.0))
        records.append(rec)

    snap_before = _gpu_mem_snapshot()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
    )
    lora_cfg = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    snap_loaded = _gpu_mem_snapshot()

    ds = Dataset.from_list(records)
    drop_cols = [c for c in ("prompt", "response") if c in ds.column_names]
    ds = ds.map(
        lambda b: _tokenize_sft(b, tokenizer, int(args.max_seq_length)),
        batched=True,
        remove_columns=drop_cols,
    )

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    tout = TrainingArguments(
        output_dir=str(adapter_dir / "trainer_state"),
        max_steps=int(args.max_train_steps),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.learning_rate),
        logging_steps=1,
        save_strategy="no",
        prediction_loss_only=True,
        report_to=[],
        bf16=use_bf16,
        fp16=False,
        gradient_checkpointing=True,
        seed=int(args.seed),
    )
    if use_sample_weights:
        collator: Any = WeightedLMDataCollator(tokenizer)
        trainer_cls: type = WeightedTrainer
    else:
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        trainer_cls = Trainer
    trainer = trainer_cls(
        model=model,
        args=tout,
        train_dataset=ds,
        data_collator=collator,
        callbacks=[LossLogCallback(loss_path)],
    )
    train_out = trainer.train()
    snap_after = _gpu_mem_snapshot()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    loss_history: list[dict[str, Any]] = []
    if loss_path.is_file():
        for line in loss_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    loss_history.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    def _max_gpu_bytes(snaps: list[dict[str, Any]]) -> int | None:
        best = 0
        found = False
        for s in snaps:
            if not isinstance(s, dict):
                continue
            for k in ("reserved_bytes", "allocated_bytes"):
                v = s.get(k)
                if isinstance(v, int) and v > best:
                    best = v
                    found = True
        return best if found else None

    gpu_snaps = [snap_before, snap_loaded, snap_after]
    return {
        "trainable_parameters": trainable,
        "train_rows": len(records),
        "max_train_steps": int(args.max_train_steps),
        "train_global_steps": int(getattr(train_out, "global_step", 0) or 0),
        "lora_config": {
            "r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "adapter_save_path": str(adapter_dir.resolve()),
        "gpu_mem_before_load": snap_before,
        "gpu_mem_after_peft": snap_loaded,
        "gpu_mem_after_train": snap_after,
        "gpu_mem_reserved_max_bytes": _max_gpu_bytes(gpu_snaps),
        "loss_log_path": str(loss_path),
        "loss_history": loss_history,
        "weighted_loss": bool(use_sample_weights),
    }


def _run_infer_for_adapter(
    *,
    run_name: str,
    adapter_path: Path,
    args: argparse.Namespace,
    domain: str,
    rep: Path,
    out_root: Path,
    pilot_cfg: dict[str, Any],
) -> None:
    backend_cfg = {
        "backend": "hf_local",
        "model": Path(args.base_model).name,
        "model_name_or_path": str(args.base_model),
        "adapter_path": str(adapter_path),
        "runtime": {
            "device_map": "auto",
            "load_in_4bit": False,
            "trust_remote_code": True,
            "batch_size": 1,
            "dtype": "bfloat16",
            "use_chat_template": True,
            "enable_thinking": False,
        },
        "generation": {"max_new_tokens": int(args.max_new_tokens), "temperature": 0.0},
    }
    from src.cli.run_pilot_reprocessed_deepseek import _load_item_lookup

    item_lookup = _load_item_lookup(Path(args.processed_dir))

    for split in ("valid", "test"):
        in_path = rep / f"{split}_candidates.jsonl"
        samples = read_jsonl(in_path)
        # Flat layout: eval_runs/<adapter>/<split>/*.jsonl (beauty-only debug)
        run_dir = out_root / "eval_runs" / run_name / split
        run_dir.mkdir(parents=True, exist_ok=True)
        pred_dir = run_dir
        meta = {
            "dataset": domain,
            "domain": domain,
            "split": split,
            "seed": int(args.seed),
            "method": f"care_lora_{run_name}",
            "prompt_template_id": args.prompt_id,
            "config_hash": config_hash(pilot_cfg),
            "git_commit": git_commit_or_unknown("."),
            "run_type": "pilot",
            "backend_type": backend_type_from_name("lora"),
            "is_paper_result": False,
        }
        raw_rows, parsed_rows, pred_rows, taxonomy_rows, repair_summary = asyncio.run(
            _infer_split(
                samples=samples,
                item_lookup=item_lookup,
                backend_cfg=backend_cfg,
                prompt_id=args.prompt_id,
                topk=int(args.topk),
                meta=meta,
            )
        )
        for pr in pred_rows:
            pr["run_type"] = "pilot"
            pr["is_paper_result"] = False
            pr["confidence_available"] = bool(
                not pr.get("missing_confidence") and pr.get("raw_confidence") is not None
            )
        write_jsonl(raw_rows, pred_dir / "raw_responses.jsonl")
        write_jsonl(parsed_rows, pred_dir / "parsed_responses.jsonl")
        pred_path = pred_dir / "rank_predictions.jsonl"
        write_jsonl(pred_rows, pred_path)
        write_failure_taxonomy_csv(taxonomy_rows, pred_dir / "format_failure_taxonomy.csv")
        (pred_dir / "repair_summary.json").write_text(json.dumps(repair_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_manifest(
            run_dir / "manifest.json",
            build_manifest(
                config=pilot_cfg,
                dataset=domain,
                domain=domain,
                raw_data_paths=[],
                processed_data_paths=[str(Path(args.processed_dir).resolve()), str(in_path.resolve())],
                method=f"care_lora_{run_name}",
                backend="lora",
                model=Path(args.base_model).name,
                prompt_template=args.prompt_id,
                seed=int(args.seed),
                candidate_size=len(samples[0]["candidate_item_ids"]) if samples else 19,
                calibration_source=None,
                command=sys.argv,
                mock_data_used=False,
            ),
        )
        eval_dir = run_dir / "eval"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli.evaluate",
                "--predictions_path",
                str(pred_path),
                "--output_dir",
                str(eval_dir),
                "--candidates_source_path",
                str(in_path),
            ],
            check=True,
            cwd=str(Path.cwd()),
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    domain = str(args.domain)
    rep = Path(args.reprocess_dir) / domain
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    adapters_root = out_root / "adapters"
    train_logs = out_root / "train_logs"
    data_dir = out_root / "data"

    for split in ("train", "valid", "test"):
        validate_reprocess_candidate_rows(read_jsonl(rep / f"{split}_candidates.jsonl"), split=split)

    if not args.skip_build_data:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli.build_care_lora_data",
                "--domain",
                domain,
                "--split",
                str(args.data_split),
                "--reprocess_dir",
                str(args.reprocess_dir),
                "--output_root",
                str(out_root),
                "--processed_dir",
                str(args.processed_dir),
                "--topk",
                str(args.topk),
                "--seed",
                str(args.seed),
                "--prompt_id",
                str(args.prompt_id),
                "--deepseek_root",
                str(getattr(args, "_deepseek_root", "")),
                "--care_rerank_config",
                str(getattr(args, "_care_rerank_config", "")),
            ],
            check=True,
            cwd=str(Path.cwd()),
        )

    pilot_cfg: dict[str, Any] = {
        "run_type": "pilot",
        "seed": int(args.seed),
        "method": "care_lora_debug",
        "base_model": str(args.base_model),
        "domain": domain,
        "output_root": str(out_root.resolve()),
        "max_train_steps": int(args.max_train_steps),
    }

    train_results: dict[str, Any] = {}
    if not args.skip_train:
        vanilla_jsonl = data_dir / "vanilla_lora_baseline_train.jsonl"
        care_jsonl = data_dir / "care_full_train.jsonl"
        train_results["vanilla_lora_baseline"] = _train_adapter_from_jsonl(
            jsonl_path=vanilla_jsonl,
            args=args,
            adapter_dir=adapters_root / "vanilla_lora_baseline",
            loss_path=train_logs / "vanilla_lora_baseline_loss.jsonl",
            use_sample_weights=False,
        )
        train_results["care_full_training"] = _train_adapter_from_jsonl(
            jsonl_path=care_jsonl,
            args=args,
            adapter_dir=adapters_root / "care_full_training",
            loss_path=train_logs / "care_full_training_loss.jsonl",
            use_sample_weights=True,
        )

    for name in ("vanilla_lora_baseline", "care_full_training"):
        adapter_path = adapters_root / name
        if not (adapter_path / "adapter_config.json").is_file():
            print(f"[care_lora_debug] skip infer, missing adapter {adapter_path}", file=sys.stderr)
            continue
        _run_infer_for_adapter(
            run_name=name,
            adapter_path=adapter_path,
            args=args,
            domain=domain,
            rep=rep,
            out_root=out_root,
            pilot_cfg=pilot_cfg,
        )

    manifest = {
        "created_at": utc_timestamp(),
        "git_commit": git_commit_or_unknown("."),
        "run_type": "pilot",
        "backend_type": "lora",
        "is_paper_result": False,
        "domain": domain,
        "output_root": str(out_root.resolve()),
        "base_model": str(Path(args.base_model).resolve()),
        "max_train_steps": int(args.max_train_steps),
        "lora_hyperparameters": {
            "lora_r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "per_device_train_batch_size": int(args.per_device_train_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "learning_rate": float(args.learning_rate),
        },
        "train": train_results,
        "adapter_paths": {
            "vanilla_lora_baseline": str((adapters_root / "vanilla_lora_baseline").resolve()),
            "care_full_training": str((adapters_root / "care_full_training").resolve()),
        },
        "eval_runs_root": str((out_root / "eval_runs").resolve()),
        "note": "CARE-LoRA debug: tiny steps, beauty only; eval_runs/<adapter>/<split>/ holds predictions + eval.",
    }
    (out_root / "train_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[care_lora_debug] done output_root={out_root}")


if __name__ == "__main__":
    main(None)
