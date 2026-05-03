"""Debug pilot: PEFT LoRA on reprocessed amazon_beauty candidates (tiny steps, Qwen3-8B local)."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.backends import GenerationRequest, build_backend
from src.cli.run_pilot_reprocessed_deepseek import _build_ranking_prompt, _load_item_lookup, _merge_item_texts
from src.data.protocol import read_jsonl, write_jsonl
from src.parsing import (
    build_repair_summary,
    failure_taxonomy_row,
    safe_repair_ranking,
    write_failure_taxonomy_csv,
)
from src.prompts import parse_ranking_output, ranking_parse_strict_for_prompt
from src.uncertainty.interface import VerbalizedConfidenceEstimator
from src.utils.manifest import backend_type_from_name, build_manifest, write_manifest
from src.utils.research_artifacts import config_hash, git_commit_or_unknown, utc_timestamp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA debug train+infer on reprocessed candidate JSONL (single domain).")
    p.add_argument("--domain", default="amazon_beauty")
    p.add_argument("--reprocess_dir", default="outputs/reprocessed_processed_source")
    p.add_argument("--processed_dir", default="data/processed/amazon_beauty")
    p.add_argument("--base_model", default="/home/ajifang/models/Qwen/Qwen3-8B")
    p.add_argument("--output_root", default="outputs/pilots/lora_qwen3_8b_processed_20u_c19_seed42_debug")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_steps", type=int, default=12, help="Optimizer steps (5–20 typical).")
    p.add_argument("--prompt_id", default="listwise_ranking_v1")
    p.add_argument("--topk", type=int, default=19)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_new_tokens", type=int, default=384)
    return p.parse_args()


def validate_reprocess_candidate_rows(rows: list[dict[str, Any]], *, split: str) -> None:
    errs: list[str] = []
    for idx, row in enumerate(rows):
        uid = row.get("user_id")
        tid = str(row.get("target_item_id", ""))
        cids = [str(x) for x in row.get("candidate_item_ids", [])]
        hist = [str(x) for x in row.get("history_item_ids", [])]
        if not tid:
            errs.append(f"{split}[{idx}] user={uid} missing target_item_id")
            continue
        if tid not in cids:
            errs.append(f"{split}[{idx}] user={uid} target not in candidate_item_ids")
        if tid in hist:
            errs.append(f"{split}[{idx}] user={uid} target in history")
        negs = set(cids) - {tid}
        leaked = sorted(negs & set(hist))
        if leaked:
            errs.append(f"{split}[{idx}] user={uid} history in negatives: {leaked[:3]}")
    if errs:
        raise ValueError("Candidate validation failed:\n" + "\n".join(errs[:40]))


def _teacher_response_json(row: dict[str, Any]) -> str:
    tid = str(row["target_item_id"])
    order = [tid] + [str(x) for x in row.get("candidate_item_ids", []) if str(x) != tid]
    return json.dumps(
        {"ranked_item_ids": order, "confidence": 0.88, "reason": "teacher_rank_target_first"},
        ensure_ascii=False,
    )


def _sft_rows_from_candidates(rows: list[dict[str, Any]], *, item_lookup: dict[str, str], prompt_id: str, topk: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        prompt = _build_ranking_prompt(row, prompt_id, _merge_item_texts(row, item_lookup), topk)
        out.append({"prompt": prompt, "response": _teacher_response_json(row)})
    return out


def _gpu_mem_snapshot() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False}
        free, total = torch.cuda.mem_get_info()
        return {
            "cuda": True,
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "mem_free_bytes": int(free),
            "mem_total_bytes": int(total),
        }
    except Exception as exc:
        return {"cuda": None, "error": repr(exc)}


def _train_lora(
    *,
    train_rows: list[dict[str, Any]],
    item_lookup: dict[str, str],
    args: argparse.Namespace,
    adapter_dir: Path,
    meta_dir: Path,
) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, TrainerCallback

    class LossCb(TrainerCallback):
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

    adapter_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    sft = _sft_rows_from_candidates(train_rows, item_lookup=item_lookup, prompt_id=args.prompt_id, topk=args.topk)
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

    def tokenize(batch: dict[str, list]) -> dict[str, Any]:
        eos = tokenizer.eos_token or ""
        texts = [f"{p}\n{r}{eos}" for p, r in zip(batch["prompt"], batch["response"])]
        return tokenizer(texts, truncation=True, max_length=int(args.max_seq_length))

    ds = Dataset.from_list(sft).map(tokenize, batched=True, remove_columns=["prompt", "response"])
    loss_path = meta_dir / "train_loss.jsonl"
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
    trainer = Trainer(
        model=model,
        args=tout,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[LossCb(loss_path)],
    )
    train_out = trainer.train()
    snap_after_train = _gpu_mem_snapshot()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "trainable_parameters": trainable,
        "lora_config": {
            "r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "train_rows": len(sft),
        "max_seq_length": int(args.max_seq_length),
        "max_train_steps": int(args.max_train_steps),
        "train_global_steps": int(getattr(train_out, "global_step", 0) or 0),
        "gpu_mem_before_load": snap_before,
        "gpu_mem_after_peft": snap_loaded,
        "gpu_mem_after_train": snap_after_train,
        "loss_log_path": str(loss_path),
    }


async def _infer_split(
    *,
    samples: list[dict[str, Any]],
    item_lookup: dict[str, str],
    backend_cfg: dict[str, Any],
    prompt_id: str,
    topk: int,
    meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    backend = build_backend(backend_cfg)
    estimator = VerbalizedConfidenceEstimator()
    raw_rows, parsed_rows, pred_rows = [], [], []
    taxonomy_rows: list[dict[str, Any]] = []
    for sample in samples:
        prompt = _build_ranking_prompt(sample, prompt_id, _merge_item_texts(sample, item_lookup), topk)
        resp = (await backend.abatch_generate([GenerationRequest(prompt=prompt, request_id=f"{meta['split']}:{sample.get('user_id')}")]))[0]
        raw_text = str(resp.raw_text or "")
        allowed = [str(x) for x in sample.get("candidate_item_ids", [])]
        parsed = parse_ranking_output(
            raw_text,
            allowed_item_ids=allowed,
            topk=topk,
            strict_json_only=ranking_parse_strict_for_prompt(prompt_id),
        )
        strict_ranking = list(parsed.ranked_item_ids or [])
        strict_conf_avail = bool(parsed.is_valid and parsed.confidence is not None and not parsed.missing_confidence)
        repaired = safe_repair_ranking(
            raw_text=raw_text,
            allowed_item_ids=allowed,
            strict_json_valid=bool(parsed.is_valid),
            strict_ranking=strict_ranking,
            strict_confidence_available=strict_conf_avail,
        )
        ranking = list(repaired.ranking)
        predicted = ranking[0] if ranking else ""
        raw_conf = float(parsed.confidence) if strict_conf_avail and parsed.confidence is not None else 0.0
        est = estimator.estimate({"raw_confidence": raw_conf})
        target = str(sample.get("target_item_id", ""))
        request_id = f"{meta['split']}:{sample.get('user_id')}"
        raw_rows.append(
            {
                "timestamp": utc_timestamp(),
                "request_id": request_id,
                "raw_response": raw_text,
                "api_response": None,
                "token_usage": resp.usage or {},
                "latency_seconds": resp.latency_seconds,
                "error": resp.error,
                "model_name": str(backend_cfg.get("model_name_or_path", "")),
            }
        )
        parsed_rows.append(
            {
                "timestamp": utc_timestamp(),
                "user_id": sample.get("user_id"),
                "is_valid": repaired.usable_ranking,
                "strict_json_valid": parsed.is_valid,
                "invalid_output": not repaired.usable_ranking,
                "invalid_output_strict": parsed.invalid_output,
                "hallucinated_item": parsed.hallucinated_item,
                "duplicate_item": parsed.duplicate_item,
                "missing_confidence": parsed.missing_confidence,
                "output_not_in_candidate_set": parsed.output_not_in_candidate_set,
                "malformed_json": parsed.malformed_json,
                "repaired_json": parsed.repaired_json,
                "strict_ranked_item_ids": strict_ranking,
                "ranked_item_ids": ranking,
                "confidence": parsed.confidence,
                "confidence_available_strict": strict_conf_avail,
                "confidence_available": repaired.confidence_available_after_repair,
                "usable_ranking": repaired.usable_ranking,
                "repaired_by": repaired.repaired_by,
                "repair_reason": repaired.repair_reason,
                "error": resp.error,
            }
        )
        taxonomy_rows.append(
            failure_taxonomy_row(
                user_id=str(sample.get("user_id", "")),
                request_id=request_id,
                raw_text=raw_text,
                strict_json_valid=bool(parsed.is_valid),
                strict_ranking=strict_ranking,
                allowed_item_ids=allowed,
                missing_confidence=bool(parsed.missing_confidence),
                malformed_json=bool(parsed.malformed_json),
                duplicate_item=bool(parsed.duplicate_item),
                output_not_in_candidate_set=bool(parsed.output_not_in_candidate_set),
                max_new_tokens=int(backend_cfg.get("generation", {}).get("max_new_tokens", 0) or 0),
            )
        )
        pred_rows.append(
            {
                **meta,
                "timestamp": utc_timestamp(),
                "user_id": sample.get("user_id"),
                "history_length": len(sample.get("history_item_ids", [])),
                "candidate_item_ids": sample.get("candidate_item_ids", []),
                "candidate_popularity_counts": sample.get("candidate_popularity_counts", []),
                "candidate_popularity_buckets": sample.get("candidate_popularity_buckets", []),
                "predicted_ranking": ranking,
                "predicted_item_id": predicted,
                "target_item_id": target,
                "correctness": bool(predicted == target),
                "raw_confidence": raw_conf,
                "calibrated_confidence": None,
                "uncertainty_score": est.uncertainty_score,
                "uncertainty_estimator_name": est.estimator_name,
                "item_popularity_count": sample.get("target_popularity_count"),
                "item_popularity_bucket": sample.get("target_popularity_bucket"),
                "is_valid": repaired.usable_ranking,
                "strict_json_valid": parsed.is_valid,
                "usable_ranking": repaired.usable_ranking,
                "hallucinated_item": parsed.hallucinated_item,
                "duplicate_item": parsed.duplicate_item,
                "missing_confidence": parsed.missing_confidence,
                "output_not_in_candidate_set": parsed.output_not_in_candidate_set,
                "strict_predicted_ranking": strict_ranking,
                "repaired_by": repaired.repaired_by,
                "repair_reason": repaired.repair_reason,
                "confidence_available_strict": strict_conf_avail,
                "confidence_available": repaired.confidence_available_after_repair,
                "backend": "lora",
                "model": Path(str(backend_cfg.get("model_name_or_path", ""))).name,
                "raw_response": raw_text,
                "usage": resp.usage or {},
                "latency_seconds": resp.latency_seconds,
                "retry_count": resp.retry_count,
                "cache_hit": resp.cache_hit,
                "error": resp.error,
            }
        )
    return raw_rows, parsed_rows, pred_rows, taxonomy_rows, build_repair_summary(pred_rows, taxonomy_rows)


def main() -> None:
    args = parse_args()
    domain = str(args.domain)
    rep = Path(args.reprocess_dir) / domain
    proc = Path(args.processed_dir)
    out_root = Path(args.output_root)
    adapter_dir = out_root / "adapter"
    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    train_path = rep / "train_candidates.jsonl"
    for split in ("train", "valid", "test"):
        p = rep / f"{split}_candidates.jsonl"
        rows = read_jsonl(p)
        validate_reprocess_candidate_rows(rows, split=split)
    train_rows = read_jsonl(train_path)
    item_lookup = _load_item_lookup(proc)

    pilot_cfg: dict[str, Any] = {
        "run_type": "pilot",
        "seed": int(args.seed),
        "method": "local_lora_listwise",
        "base_model": str(args.base_model),
        "domain": domain,
        "reprocess_dir": str(args.reprocess_dir),
        "output_root": str(out_root),
        "prompt_id": args.prompt_id,
        "max_train_steps": int(args.max_train_steps),
        "lora_r": int(args.lora_r),
    }

    train_meta = _train_lora(train_rows=train_rows, item_lookup=item_lookup, args=args, adapter_dir=adapter_dir, meta_dir=meta_dir)
    train_meta["adapter_dir"] = str(adapter_dir)
    (meta_dir / "train_debug.json").write_text(json.dumps(train_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    backend_cfg = {
        "backend": "hf_local",
        "model": Path(args.base_model).name,
        "model_name_or_path": str(args.base_model),
        "adapter_path": str(adapter_dir),
        "runtime": {
            "device_map": "auto",
            "load_in_4bit": False,
            "trust_remote_code": True,
            "batch_size": 1,
            "dtype": "bfloat16",
        },
        "generation": {"max_new_tokens": int(args.max_new_tokens), "temperature": 0.0},
    }

    snap_before_infer = _gpu_mem_snapshot()
    for split in ("valid", "test"):
        in_path = rep / f"{split}_candidates.jsonl"
        samples = read_jsonl(in_path)
        out_dir = out_root / domain / split
        pred_dir = out_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "dataset": domain,
            "domain": domain,
            "split": split,
            "seed": int(args.seed),
            "method": "local_lora_listwise",
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
        write_jsonl(raw_rows, pred_dir / "raw_responses.jsonl")
        write_jsonl(parsed_rows, pred_dir / "parsed_responses.jsonl")
        pred_path = pred_dir / "rank_predictions.jsonl"
        write_jsonl(pred_rows, pred_path)
        write_failure_taxonomy_csv(taxonomy_rows, pred_dir / "format_failure_taxonomy.csv")
        (pred_dir / "repair_summary.json").write_text(json.dumps(repair_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_manifest(
            out_dir / "manifest.json",
            build_manifest(
                config=pilot_cfg,
                dataset=domain,
                domain=domain,
                raw_data_paths=[],
                processed_data_paths=[str(proc), str(in_path)],
                method="local_lora_listwise",
                backend="lora",
                model=Path(args.base_model).name,
                prompt_template=args.prompt_id,
                seed=int(args.seed),
                candidate_size=len(samples[0]["candidate_item_ids"]) if samples else None,
                calibration_source=None,
                command=sys.argv,
                api_key_env=None,
                mock_data_used=False,
            ),
        )
        eval_dir = out_dir / "eval"
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

    snap_after_infer = _gpu_mem_snapshot()
    summary = {
        "created_at": utc_timestamp(),
        "git_commit": git_commit_or_unknown("."),
        "run_type": "pilot",
        "backend_type": "lora",
        "is_paper_result": False,
        "domain": domain,
        "adapter_dir": str(adapter_dir),
        "train_debug": train_meta,
        "gpu_mem_before_infer_reload": snap_before_infer,
        "gpu_mem_after_infer": snap_after_infer,
        "note": "LoRA inference uses HFLocalBackend with adapter_path; prediction files mirror DeepSeek pilot layout.",
    }
    (out_root / "debug_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[lora_debug] done output_root={out_root}")


if __name__ == "__main__":
    main()
