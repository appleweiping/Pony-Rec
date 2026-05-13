from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import shutil
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train pinned upstream SETRec Qwen4Rec on a same-candidate adapter "
            "package and emit exact candidate scores."
        )
    )
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--setrec_repo_dir", required=True)
    parser.add_argument("--backbone_path", required=True)
    parser.add_argument("--output_scores_path", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--model_class", default="Qwen4Rec")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--micro_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--val_set_size", type=int, default=2000)
    parser.add_argument("--n_sem", type=int, default=4)
    parser.add_argument("--n_cf", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--ae_layers", default="512,256,128")
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--bn", action="store_true")
    parser.add_argument("--loss_type", default="mse")
    parser.add_argument("--sem_encoder", default="qwen")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.02)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj,o_proj")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr_scheduler", default="cosine")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score_batch_size", type=int, default=32)
    return parser.parse_args()


def _import_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import numpy as np
        import torch
        import transformers
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError("The SETRec upstream wrapper requires numpy, PyTorch, and transformers.") from exc
    return np, torch, transformers, DataLoader, torch.optim.AdamW, torch.nn


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_int(value: Any) -> int | None:
    try:
        value_text = str(value).strip()
        if not value_text:
            return None
        return int(float(value_text))
    except Exception:
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


@contextmanager
def _pushd(path: Path) -> Iterator[None]:
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _set_seed(seed: int, torch: Any, np: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_arg: str, torch: Any) -> Any:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _load_interactions(path: Path) -> dict[int, list[int]]:
    sequences: dict[int, list[int]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            user_idx = _as_int(parts[0])
            item_idx = _as_int(parts[1])
            if user_idx is None or item_idx is None:
                continue
            sequences.setdefault(user_idx - 1, []).append(item_idx - 1)
    return sequences


def _candidate_groups(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        source_event_id = _text(row.get("source_event_id"))
        user_id = _text(row.get("user_id"))
        item_id = _text(row.get("item_id"))
        user_idx = _as_int(row.get("llmesr_user_idx"))
        item_idx = _as_int(row.get("llmesr_item_idx"))
        if not source_event_id or not user_id or not item_id or user_idx is None or item_idx is None:
            raise ValueError("candidate_items_mapped.csv contains an incomplete mapped row")
        key = (source_event_id, user_id)
        if key not in groups:
            groups[key] = {
                "source_event_id": source_event_id,
                "user_id": user_id,
                "setrec_user_idx": user_idx - 1,
                "candidate_rows": [],
            }
        try:
            candidate_index = int(float(row.get("candidate_index", 0)))
        except Exception:
            candidate_index = 0
        groups[key]["candidate_rows"].append(
            {"candidate_index": candidate_index, "item_id": item_id, "setrec_item_idx": item_idx - 1}
        )
    events = list(groups.values())
    for event in events:
        event["candidate_rows"].sort(key=lambda item: item["candidate_index"])
    events.sort(key=lambda item: (item["user_id"], item["source_event_id"]))
    return events


def _load_pickle_matrix(path: Path, np: Any) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required SETRec matrix not found: {path}")
    with path.open("rb") as fh:
        matrix = pickle.load(fh)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix at {path}, got shape={matrix.shape}")
    return matrix


def _load_feature_matrix(path: Path, np: Any) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required SETRec feature matrix not found: {path}")
    if path.suffix.lower() == ".npy":
        matrix = np.load(path, allow_pickle=True)
    else:
        with path.open("rb") as fh:
            matrix = pickle.load(fh)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix at {path}, got shape={matrix.shape}")
    return matrix


def _write_official_dataset(
    *,
    adapter_dir: Path,
    setrec_repo_dir: Path,
    dataset_alias: str,
    args: argparse.Namespace,
    np: Any,
) -> tuple[Path, dict[str, Any]]:
    metadata = json.loads((adapter_dir / "adapter_metadata.json").read_text(encoding="utf-8"))
    user_count = int(metadata.get("users", 0))
    item_count = int(metadata.get("items", 0))
    sequences = _load_interactions(adapter_dir / "llm_esr" / "handled" / "inter.txt")
    item_embed_path = adapter_dir / "llm_esr" / "handled" / "itm_emb_np.pkl"
    sem_embed_path = adapter_dir / "llm_esr" / "handled" / "pca64_itm_emb_np.pkl"
    item_embed = _load_pickle_matrix(item_embed_path, np)
    sem_embed = _load_feature_matrix(sem_embed_path, np)
    if item_embed.shape[0] != item_count:
        raise ValueError(f"SASRec/CF embedding rows={item_embed.shape[0]} but adapter metadata items={item_count}")
    if sem_embed.shape[0] != item_count:
        raise ValueError(f"Semantic embedding rows={sem_embed.shape[0]} but adapter metadata items={item_count}")

    validation: dict[int, list[int]] = {}
    testing: dict[int, list[int]] = {}
    for event in _candidate_groups(adapter_dir / "candidate_items_mapped.csv"):
        user_idx = int(event["setrec_user_idx"])
        positives = [
            int(row["setrec_item_idx"])
            for row in event["candidate_rows"]
            if int(row.get("candidate_index", 0)) == 0
        ]
        testing[user_idx] = positives[:1] if positives else []
        validation.setdefault(user_idx, [])

    training = {user_idx: list(items) for user_idx, items in sequences.items()}
    for user_idx in range(user_count):
        training.setdefault(user_idx, [])
        validation.setdefault(user_idx, [])
        testing.setdefault(user_idx, [])

    out_dir = setrec_repo_dir / "data" / dataset_alias
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "training_dict.npy", training, allow_pickle=True)
    np.save(out_dir / "validation_dict.npy", validation, allow_pickle=True)
    np.save(out_dir / "testing_dict.npy", testing, allow_pickle=True)
    np.save(out_dir / "warm_item.npy", np.arange(item_count, dtype=np.int64), allow_pickle=True)
    np.save(out_dir / "cold_item.npy", np.array([], dtype=np.int64), allow_pickle=True)
    np.save(out_dir / "combine_tdcb_maps.npy", np.arange(item_count, dtype=np.int64), allow_pickle=True)
    np.save(out_dir / f"{dataset_alias}.emb-{args.sem_encoder}-tdcb.npy", sem_embed, allow_pickle=True)
    with (out_dir / "SASRec_item_embed.pkl").open("wb") as fh:
        pickle.dump(item_embed, fh)

    manifest = {
        "dataset_alias": dataset_alias,
        "user_count": user_count,
        "item_count": item_count,
        "training_users": len(training),
        "testing_users": sum(1 for value in testing.values() if value),
        "cf_embedding_path": str(item_embed_path),
        "semantic_embedding_path": str(sem_embed_path),
        "official_data_dir": str(out_dir),
    }
    (out_dir / "same_candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_dir, manifest


def _import_official_model(repo_dir: Path) -> Any:
    code_dir = repo_dir / "code"
    if not (code_dir / "model_qwen.py").exists():
        raise FileNotFoundError(f"SETRec model_qwen.py not found under {code_dir}")
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    try:
        import builtins
        from transformers import Qwen2Config

        # The pinned SETRec Q_qwen.py annotates QQwen2Model.__init__ with
        # Qwen2Config but does not import that name. In Python versions that
        # eagerly evaluate annotations, importing the official module fails.
        if not hasattr(builtins, "Qwen2Config"):
            setattr(builtins, "Qwen2Config", Qwen2Config)
    except Exception:
        pass
    module = __import__("model_qwen", fromlist=["Qwen4Rec"])
    return getattr(module, "Qwen4Rec")


def _make_official_args(
    *,
    args: argparse.Namespace,
    dataset_path: Path,
    checkpoint_dir: Path,
    torch: Any,
) -> dict[str, Any]:
    layers = [int(item.strip()) for item in args.ae_layers.split(",") if item.strip()]
    lora_target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    return {
        "base_model": str(Path(args.backbone_path).expanduser()),
        "input_embeds": torch.FloatTensor([]),
        "cache_dir": "",
        "device_map": "auto",
        "input_dim": 0,
        "instruction_text": ("You are a recommender system.", "Predict the next item identifiers."),
        "user_embeds": None,
        "m_item": 0,
        "n_query": int(args.n_sem) + int(args.n_cf),
        "n_cf": int(args.n_cf),
        "n_sem": int(args.n_sem),
        "alpha": float(args.alpha),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_target_modules": lora_target_modules,
        "data_path": str(dataset_path) + os.sep,
        "output_dir": str(checkpoint_dir),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "micro_batch_size": int(args.micro_batch_size),
        "num_epochs": int(args.epochs),
        "learning_rate": float(args.lr),
        "val_set_size": int(args.val_set_size),
        "lr_scheduler": args.lr_scheduler,
        "warmup_steps": int(args.warmup_steps),
        "model_class": args.model_class,
        "in_dim": 0,
        "layers": layers,
        "dropout_prob": float(args.dropout_prob),
        "bn": bool(args.bn),
        "loss_type": args.loss_type,
        "sem_encoder": args.sem_encoder,
    }


def _train_model_with_official_class(
    *,
    repo_dir: Path,
    dataset_path: Path,
    checkpoint_dir: Path,
    args: argparse.Namespace,
    np: Any,
    torch: Any,
    transformers: Any,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    if not torch.cuda.is_available() and args.device == "auto":
        raise RuntimeError("Pinned SETRec Qwen4Rec calls .cuda(); run this official adapter on a CUDA device.")
    code_dir = repo_dir / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    from utils.data_utils import SequentialCollator, SequentialDataset  # type: ignore

    item_embed = _load_pickle_matrix(dataset_path / "SASRec_item_embed.pkl", np)
    item_feature = torch.FloatTensor(
        _load_feature_matrix(dataset_path / f"{dataset_path.name}.emb-{args.sem_encoder}-tdcb.npy", np)
    )
    model_kwargs = _make_official_args(args=args, dataset_path=dataset_path, checkpoint_dir=checkpoint_dir, torch=torch)
    model_kwargs.update(
        {
            "input_embeds": torch.FloatTensor(item_embed),
            "input_dim": int(item_embed.shape[1]),
            "m_item": int(item_embed.shape[0]),
            "in_dim": int(item_feature.shape[-1]),
            "item_feature": item_feature,
        }
    )
    Model = _import_official_model(repo_dir)
    dataset = SequentialDataset(str(dataset_path) + os.sep, args.max_len, model_kwargs["n_query"], args.n_sem)
    collator = SequentialCollator()
    model = Model(**model_kwargs)
    gradient_accumulation_steps = max(1, int(args.batch_size) // max(1, int(args.micro_batch_size)))
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=int(args.micro_batch_size),
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=int(args.warmup_steps),
        num_train_epochs=int(args.epochs),
        learning_rate=float(args.lr),
        bf16=True,
        logging_steps=20,
        optim="adamw_torch",
        evaluation_strategy="no",
        save_strategy="no",
        lr_scheduler_type=args.lr_scheduler,
        output_dir=str(checkpoint_dir),
        report_to="none",
        save_safetensors=False,
    )
    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=collator,
    )
    train_output = trainer.train()
    logs = [
        {
            "epoch": row.get("epoch", ""),
            "train_loss": row.get("loss", ""),
            "learning_rate": row.get("learning_rate", ""),
        }
        for row in trainer.state.log_history
        if "loss" in row
    ]
    summary = {
        "train_runtime": getattr(train_output.metrics, "get", lambda *_: "")("train_runtime", "")
        if hasattr(train_output, "metrics")
        else "",
        "train_samples": len(dataset),
    }
    return model, logs, summary


def _score_candidates(
    *,
    model: Any,
    candidate_events: list[dict[str, Any]],
    sequences: dict[int, list[int]],
    output_scores_path: Path,
    args: argparse.Namespace,
    torch: Any,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    model.eval()
    model.beta = torch.nn.Parameter(
        torch.tensor([1.0 - float(args.beta)] + [float(args.beta)] * int(args.n_sem), dtype=torch.float, requires_grad=False)
    )
    idx_tensor = torch.arange(model.m_item, dtype=torch.long).cuda()
    model.all_cf = model.input_proj(model.input_embeds[0](idx_tensor + 1)).unsqueeze(0)
    if model.n_sem:
        model.recon_all = model.tokenize_all()

    with torch.no_grad():
        for start in range(0, len(candidate_events), max(1, int(args.score_batch_size))):
            batch_events = candidate_events[start : start + max(1, int(args.score_batch_size))]
            encoded: list[list[int]] = []
            max_len = 2
            for event in batch_events:
                seq = [idx + 1 for idx in sequences.get(int(event["setrec_user_idx"]), [])][-int(args.max_len) :]
                encoded.append(seq)
                max_len = max(max_len, len(seq))
            inputs = [[0] * (max_len - len(seq)) + seq for seq in encoded]
            inputs_tensor = torch.LongTensor(inputs).cuda()
            inputs_mask = torch.ones((inputs_tensor.shape[0], inputs_tensor.shape[1] * int(model.n_query))).cuda()
            _outputs, ratings, _x, _sem, _x_hat = model.predict(inputs_tensor, inputs_mask, inference=True)
            if ratings.ndim == 1:
                ratings = ratings.unsqueeze(0)
            for offset, event in enumerate(batch_events):
                candidate_indices = [int(row["setrec_item_idx"]) for row in event["candidate_rows"]]
                score_tensor = ratings[offset].index_select(0, torch.LongTensor(candidate_indices).to(ratings.device))
                if not torch.isfinite(score_tensor).all():
                    raise RuntimeError(f"Non-finite SETRec score for source_event_id={event['source_event_id']}")
                for row, score in zip(event["candidate_rows"], score_tensor.detach().cpu().tolist()):
                    rows.append(
                        {
                            "source_event_id": event["source_event_id"],
                            "user_id": event["user_id"],
                            "item_id": row["item_id"],
                            "score": float(score),
                        }
                    )

    _write_csv(rows, output_scores_path, ["source_event_id", "user_id", "item_id", "score"])
    candidate_rows = sum(len(event["candidate_rows"]) for event in candidate_events)
    return {
        "candidate_events": len(candidate_events),
        "candidate_rows": candidate_rows,
        "scored_rows": len(rows),
        "score_coverage_rate": float(len(rows) / candidate_rows) if candidate_rows else 0.0,
        "output_scores_path": str(output_scores_path),
    }


def train_and_score(args: argparse.Namespace) -> dict[str, Any]:
    np, torch, transformers, _DataLoader, _AdamW, _nn = _import_runtime()
    _set_seed(args.seed, torch, np)
    adapter_dir = Path(args.adapter_dir).expanduser().resolve()
    repo_dir = Path(args.setrec_repo_dir).expanduser().resolve()
    dataset_alias = adapter_dir.name.replace("_setrec_official_adapter", "_setrec")
    output_scores_path = Path(args.output_scores_path).expanduser().resolve() if args.output_scores_path else adapter_dir / "setrec_official_scores.csv"
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir else adapter_dir / "setrec_official_checkpoint"
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve() if args.checkpoint_path else checkpoint_dir / "adapter.pth"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset_path, dataset_manifest = _write_official_dataset(
        adapter_dir=adapter_dir,
        setrec_repo_dir=repo_dir,
        dataset_alias=dataset_alias,
        args=args,
        np=np,
    )
    sequences = _load_interactions(adapter_dir / "llm_esr" / "handled" / "inter.txt")
    candidate_events = _candidate_groups(adapter_dir / "candidate_items_mapped.csv")
    with _pushd(repo_dir / "code"):
        model, epoch_logs, train_summary = _train_model_with_official_class(
            repo_dir=repo_dir,
            dataset_path=dataset_path,
            checkpoint_dir=checkpoint_dir,
            args=args,
            np=np,
            torch=torch,
            transformers=transformers,
        )
        score_summary = _score_candidates(
            model=model,
            candidate_events=candidate_events,
            sequences=sequences,
            output_scores_path=output_scores_path,
            args=args,
            torch=torch,
        )
        model.qwen_model.save_pretrained(str(checkpoint_dir))
        model.qwen_tokenizer.save_pretrained(str(checkpoint_dir))
        tokenizer_state = model.tokenizer.state_dict() if model.n_sem else None
        torch.save(
            {
                "input_embeds": model.input_embeds.state_dict(),
                "input_proj": model.input_proj.state_dict(),
                "tokenizer": tokenizer_state,
                "query": model.query.state_dict(),
                "args": vars(args),
                "score_summary": score_summary,
            },
            checkpoint_path,
        )

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    summary = {
        "adapter_dir": str(adapter_dir),
        "baseline_name": "setrec_official_qwen3base_identifier",
        "artifact_class": "official_setrec_same_candidate_score",
        "official_result_gate": "provenance_coverage_and_import_required",
        "upstream_repo": "https://github.com/Linxyhaha/SETRec",
        "official_model_class": "code.model_qwen.Qwen4Rec",
        "official_data_dir": str(dataset_path),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_path": str(checkpoint_path),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "lr": args.lr,
        "n_query": int(args.n_sem) + int(args.n_cf),
        "n_cf": args.n_cf,
        "n_sem": args.n_sem,
        "alpha": args.alpha,
        "beta": args.beta,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_target_modules": args.lora_target_modules,
        "seed": args.seed,
        "trainable_params": trainable_params,
        **dataset_manifest,
        **train_summary,
        **score_summary,
        "note": (
            "Scores use pinned SETRec Qwen4Rec with official query-guided "
            "simultaneous decoding, LoRA, CF token projection, and semantic AE "
            "tokenizer. Local code supplies same-candidate data conversion, "
            "Qwen3 item semantic features, and exact candidate-score export."
        ),
    }
    _write_csv([summary], adapter_dir / "setrec_upstream_score_summary.csv", list(summary.keys()))
    _write_csv(epoch_logs, adapter_dir / "setrec_upstream_epoch_log.csv", ["epoch", "train_loss", "learning_rate"])
    (adapter_dir / "setrec_upstream_score_metadata.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not checkpoint_path.exists() and checkpoint_dir.exists():
        adapter_file = checkpoint_dir / "adapter.pth"
        if adapter_file.exists():
            shutil.copy2(adapter_file, checkpoint_path)
    return summary


def main() -> None:
    summary = train_and_score(parse_args())
    for key, value in summary.items():
        print(f"{key}={value}", flush=True)


if __name__ == "__main__":
    main()
