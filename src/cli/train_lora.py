from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.data.protocol import read_jsonl, write_json
from src.utils.research_artifacts import utc_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA/QLoRA ranking adapter.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke_check", action="store_true", help="Validate data/config without loading a model.")
    return parser.parse_args()


def _prepare_lora_rows(rows: list[dict[str, Any]], *, data_mode: str, uncertainty_threshold: float | None = None) -> list[dict[str, Any]]:
    prepared = list(rows)
    if data_mode == "uncertainty_pruned":
        threshold = 0.7 if uncertainty_threshold is None else float(uncertainty_threshold)
        prepared = [row for row in prepared if float(row.get("uncertainty_score", row.get("uncertainty", 0.0)) or 0.0) <= threshold]
    elif data_mode == "curriculum_uncertainty":
        prepared = sorted(prepared, key=lambda row: float(row.get("uncertainty_score", row.get("uncertainty", 0.0)) or 0.0))
    return prepared


def _format_sft_rows(
    rows: list[dict[str, Any]],
    *,
    data_mode: str = "standard_sft",
    uncertainty_threshold: float | None = None,
) -> list[dict[str, Any]]:
    out = []
    for row in _prepare_lora_rows(rows, data_mode=data_mode, uncertainty_threshold=uncertainty_threshold):
        uncertainty = float(row.get("uncertainty_score", row.get("uncertainty", 0.0)) or 0.0)
        weight = max(0.05, 1.0 - uncertainty) if data_mode == "uncertainty_weighted" else 1.0
        prompt = (
            "Rank candidate items for the user.\n"
            f"History item IDs: {row.get('history_item_ids', [])}\n"
            f"Candidate item IDs: {row.get('candidate_item_ids', [])}\n"
            "Return JSON ranked_item_ids."
        )
        target = json.dumps({"ranked_item_ids": [row["target_item_id"]]}, ensure_ascii=False)
        out.append(
            {
                "prompt": prompt,
                "response": target,
                "weight": weight,
                "data_mode": data_mode,
                "uncertainty_score": uncertainty,
            }
        )
    return out


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    train_path = Path(str(cfg["train_path"]))
    valid_path = Path(str(cfg["valid_path"]))
    output_dir = Path(str(cfg.get("adapter_output_dir", "outputs/lora/adapter")))
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(train_path)
    valid_rows = read_jsonl(valid_path)
    data_mode = str(cfg.get("data_mode", "standard_sft"))
    uncertainty_threshold = cfg.get("uncertainty_threshold")
    train_sft = _format_sft_rows(train_rows, data_mode=data_mode, uncertainty_threshold=uncertainty_threshold)
    valid_sft = _format_sft_rows(valid_rows, data_mode="standard_sft")
    if not train_sft:
        raise ValueError(f"LoRA data mode {data_mode} produced zero training examples.")
    if not valid_sft:
        raise ValueError("LoRA validation data produced zero examples.")
    (output_dir / "train_sft_preview.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in train_sft[:100]) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "created_at": utc_timestamp(),
        "base_model": cfg.get("base_model"),
        "qlora": bool(cfg.get("qlora", False)),
        "train_examples": len(train_sft),
        "valid_examples": len(valid_sft),
        "adapter_output_dir": str(output_dir),
        "status": "smoke_check" if args.smoke_check else "training_requested",
        "data_mode": data_mode,
        "uncertainty_threshold": uncertainty_threshold,
    }
    if args.smoke_check:
        write_json(manifest, output_dir / "adapter_manifest.json")
        print(f"[train_lora] smoke_check passed train={len(train_sft)} valid={len(valid_sft)}")
        return
    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
    except ImportError as exc:
        raise ImportError("Real LoRA training requires transformers, peft, datasets, accelerate, and torch.") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(cfg["base_model"]), trust_remote_code=bool(cfg.get("trust_remote_code", False)))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(cfg["base_model"]),
        device_map=cfg.get("device_map", "auto"),
        load_in_4bit=bool(cfg.get("qlora", False)),
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
    )
    if cfg.get("qlora", False):
        model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        task_type="CAUSAL_LM",
        target_modules=cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    model = get_peft_model(model, lora_cfg)

    def tokenize(batch):
        texts = [f"{p}\n{r}{tokenizer.eos_token or ''}" for p, r in zip(batch["prompt"], batch["response"])]
        return tokenizer(texts, truncation=True, max_length=int(cfg.get("max_seq_length", 2048)))

    train_ds = Dataset.from_list(train_sft).map(tokenize, batched=True, remove_columns=list(train_sft[0].keys()))
    valid_ds = Dataset.from_list(valid_sft).map(tokenize, batched=True, remove_columns=list(valid_sft[0].keys()))
    args_out = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_strategy=str(cfg.get("save_strategy", "epoch")),
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args_out,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    manifest["status"] = "trained"
    write_json(manifest, output_dir / "adapter_manifest.json")
    print(f"[train_lora] saved={output_dir}")


if __name__ == "__main__":
    main()
