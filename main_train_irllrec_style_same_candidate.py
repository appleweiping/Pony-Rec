from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an IRLLRec-style intent representation baseline using "
            "precomputed LLM item embeddings and emit exact same-candidate "
            "scores. This is a same-schema adapter baseline, not an official "
            "IRLLRec reproduction."
        )
    )
    parser.add_argument("--task_dir", required=True)
    parser.add_argument("--embedding_path", required=True)
    parser.add_argument("--item_map_path", required=True)
    parser.add_argument("--item_index_col", default="auto")
    parser.add_argument("--output_scores_path", default=None)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_intents", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-5)
    parser.add_argument("--finetune_item_embeddings", action="store_true")
    parser.add_argument("--allow_missing_embeddings", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_checkpoint", action="store_true")
    return parser.parse_args()


def _import_torch():
    try:
        import numpy as np
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        return np, torch, nn, DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError("main_train_irllrec_style_same_candidate.py requires numpy and PyTorch.") from exc


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def _load_train_sequences(path: Path) -> dict[str, list[str]]:
    rows = _read_csv(path)
    grouped: dict[str, list[tuple[float, int, str]]] = defaultdict(list)
    for row in rows:
        user_id = _text(row.get("user_id"))
        item_id = _text(row.get("item_id"))
        if not user_id or not item_id:
            continue
        try:
            timestamp = float(row.get("timestamp", 0.0))
        except Exception:
            timestamp = 0.0
        try:
            sequence_index = int(float(row.get("sequence_index", 0)))
        except Exception:
            sequence_index = 0
        grouped[user_id].append((timestamp, sequence_index, item_id))
    return {
        user_id: [item_id for _, _, item_id in sorted(events, key=lambda item: (item[1], item[0]))]
        for user_id, events in grouped.items()
    }


def _load_candidate_groups(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        source_event_id = _text(row.get("source_event_id"))
        user_id = _text(row.get("user_id"))
        item_id = _text(row.get("item_id"))
        if not source_event_id or not user_id or not item_id:
            continue
        key = (source_event_id, user_id)
        if key not in groups:
            groups[key] = {
                "source_event_id": source_event_id,
                "user_id": user_id,
                "timestamp": row.get("timestamp", ""),
                "split_name": row.get("split_name", ""),
                "candidate_rows": [],
            }
        try:
            candidate_index = int(float(row.get("candidate_index", 0)))
        except Exception:
            candidate_index = 0
        groups[key]["candidate_rows"].append({"candidate_index": candidate_index, "item_id": item_id})
    events = list(groups.values())
    for event in events:
        event["candidate_rows"].sort(key=lambda row: row["candidate_index"])
    events.sort(key=lambda row: (row["user_id"], row["source_event_id"]))
    return events


def _build_item_vocab(train_sequences: dict[str, list[str]], candidate_events: list[dict[str, Any]]) -> dict[str, int]:
    item_ids: set[str] = set()
    for seq in train_sequences.values():
        item_ids.update(seq)
    for event in candidate_events:
        for row in event["candidate_rows"]:
            item_ids.add(row["item_id"])
    return {item_id: idx for idx, item_id in enumerate(sorted(item_ids), start=1)}


def _auto_item_index_col(rows: list[dict[str, str]], requested: str) -> str:
    if requested != "auto":
        return requested
    columns = set(rows[0]) if rows else set()
    for column in [
        "irllrec_item_idx",
        "llmesr_item_idx",
        "llm2rec_item_idx",
        "item_idx",
        "mapped_item_idx",
    ]:
        if column in columns:
            return column
    raise ValueError(f"Could not infer item index column from {sorted(columns)}")


def _load_embedding_matrix(path: Path, np: Any) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    if path.suffix.lower() == ".npy":
        matrix = np.load(path)
    else:
        with path.open("rb") as fh:
            matrix = pickle.load(fh)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D item embeddings, got shape={matrix.shape}")
    return matrix


def _mapped_index_to_row(mapped_index: int, matrix_rows: int) -> int:
    if 1 <= mapped_index <= matrix_rows:
        return mapped_index - 1
    if 0 <= mapped_index < matrix_rows:
        return mapped_index
    raise ValueError(f"Mapped item index {mapped_index} outside embedding rows={matrix_rows}")


def _build_pretrained_item_matrix(
    *,
    item_to_idx: dict[str, int],
    item_map_path: Path,
    item_index_col: str,
    embedding_path: Path,
    allow_missing_embeddings: bool,
    np: Any,
    torch: Any,
) -> tuple[Any, dict[str, Any]]:
    map_rows = _read_csv(item_map_path)
    index_col = _auto_item_index_col(map_rows, item_index_col)
    mapped_by_item: dict[str, int] = {}
    for row in map_rows:
        item_id = _text(row.get("item_id"))
        value = _text(row.get(index_col))
        if item_id and value:
            mapped_by_item[item_id] = int(float(value))

    source_matrix = _load_embedding_matrix(embedding_path, np)
    out = np.zeros((len(item_to_idx) + 1, source_matrix.shape[1]), dtype=np.float32)
    missing: list[str] = []
    for item_id, local_idx in item_to_idx.items():
        mapped_idx = mapped_by_item.get(item_id)
        if mapped_idx is None:
            missing.append(item_id)
            continue
        out[local_idx] = source_matrix[_mapped_index_to_row(mapped_idx, source_matrix.shape[0])]
    if missing and not allow_missing_embeddings:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"Missing embeddings for {len(missing)} local items. "
            f"First missing item_ids: {preview}. Pass --allow_missing_embeddings only for diagnostics."
        )
    return torch.tensor(out, dtype=torch.float32), {
        "embedding_path": str(embedding_path),
        "item_map_path": str(item_map_path),
        "item_index_col": index_col,
        "source_embedding_rows": int(source_matrix.shape[0]),
        "source_embedding_dim": int(source_matrix.shape[1]),
        "local_embedding_rows": int(out.shape[0]),
        "missing_embedding_items": len(missing),
    }


def _make_prefix_training_tensors(
    train_sequences: dict[str, list[str]],
    item_to_idx: dict[str, int],
    *,
    max_seq_len: int,
    torch: Any,
) -> tuple[Any, Any]:
    inputs: list[list[int]] = []
    targets: list[int] = []
    for seq in train_sequences.values():
        encoded = [item_to_idx[item_id] for item_id in seq if item_id in item_to_idx]
        for target_pos in range(1, len(encoded)):
            src = encoded[:target_pos][-max_seq_len:]
            pad_len = max_seq_len - len(src)
            inputs.append([0] * pad_len + src)
            targets.append(encoded[target_pos])
    if not inputs:
        raise ValueError("No trainable prefix examples found; need at least one sequence with length >= 2.")
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def _resolve_device(device_arg: str, torch: Any) -> Any:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _set_seed(seed: int, torch: Any, np: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model_class(torch: Any, nn: Any):
    class IRLLRecIntentModel(nn.Module):
        def __init__(
            self,
            *,
            pretrained_item_matrix: Any,
            hidden_size: int,
            num_intents: int,
            dropout: float,
            finetune_item_embeddings: bool,
        ) -> None:
            super().__init__()
            self.item_embedding = nn.Embedding.from_pretrained(
                pretrained_item_matrix,
                freeze=not finetune_item_embeddings,
                padding_idx=0,
            )
            self.adapter = nn.Sequential(
                nn.Linear(pretrained_item_matrix.shape[1], hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
            )
            self.intent_queries = nn.Parameter(torch.empty(num_intents, hidden_size))
            nn.init.xavier_uniform_(self.intent_queries)
            self.intent_gate = nn.Linear(hidden_size, 1)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(hidden_size)

        def item_representations(self) -> Any:
            reps = self.adapter(self.item_embedding.weight)
            reps = reps.clone()
            reps[0] = 0.0
            return self.layer_norm(reps)

        def encode_intents(self, input_ids: Any) -> Any:
            hidden = self.adapter(self.item_embedding(input_ids))
            hidden = self.dropout(hidden)
            mask = input_ids.ne(0)
            scores = torch.einsum("bsh,kh->bks", hidden, self.intent_queries)
            scores = scores.masked_fill(~mask.unsqueeze(1), -1.0e9)
            weights = torch.softmax(scores, dim=-1)
            intents = torch.einsum("bks,bsh->bkh", weights, hidden)
            intents = self.layer_norm(intents)
            return intents

        def score_all_items(self, input_ids: Any) -> Any:
            intents = self.encode_intents(input_ids)
            item_reps = self.item_representations()
            scores = torch.einsum("bkh,vh->bkv", intents, item_reps)
            gates = torch.softmax(self.intent_gate(intents).squeeze(-1), dim=-1).unsqueeze(-1)
            return (scores * gates).sum(dim=1)

    return IRLLRecIntentModel


def _train_model(
    *,
    model: Any,
    train_inputs: Any,
    train_targets: Any,
    args: argparse.Namespace,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    device: Any,
) -> list[dict[str, Any]]:
    dataset = TensorDataset(train_inputs, train_targets)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    logs: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.score_all_items(batch_inputs)
            loss = loss_fn(logits, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            steps += 1
        mean_loss = total_loss / max(steps, 1)
        logs.append({"epoch": epoch, "train_loss": mean_loss})
        if args.log_every > 0 and (epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0):
            print(f"[irllrec-style] epoch={epoch} train_loss={mean_loss:.6f}", flush=True)
    return logs


def _context_tensor(seq: list[str], item_to_idx: dict[str, int], *, max_seq_len: int, torch: Any, device: Any) -> Any:
    encoded = [item_to_idx[item_id] for item_id in seq if item_id in item_to_idx][-max_seq_len:]
    pad_len = max_seq_len - len(encoded)
    return torch.tensor([[0] * pad_len + encoded], dtype=torch.long, device=device)


def _score_candidates(
    *,
    model: Any,
    train_sequences: dict[str, list[str]],
    candidate_events: list[dict[str, Any]],
    item_to_idx: dict[str, int],
    output_scores_path: Path,
    max_seq_len: int,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    output_scores_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    scored_rows = 0
    model.eval()
    with output_scores_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["source_event_id", "user_id", "item_id", "score"])
        writer.writeheader()
        with torch.no_grad():
            for event in candidate_events:
                user_id = event["user_id"]
                candidate_rows = event["candidate_rows"]
                total_rows += len(candidate_rows)
                input_ids = _context_tensor(
                    train_sequences.get(user_id, []),
                    item_to_idx,
                    max_seq_len=max_seq_len,
                    torch=torch,
                    device=device,
                )
                logits = model.score_all_items(input_ids).squeeze(0)
                candidate_indices = [item_to_idx[row["item_id"]] for row in candidate_rows]
                candidate_tensor = torch.tensor(candidate_indices, dtype=torch.long, device=device)
                score_tensor = logits.index_select(0, candidate_tensor)
                if not torch.isfinite(score_tensor).all():
                    raise RuntimeError("IRLLRec-style model produced non-finite candidate scores.")
                for row, score in zip(candidate_rows, score_tensor.detach().cpu().tolist()):
                    score_value = float(score)
                    if not math.isfinite(score_value):
                        raise RuntimeError(f"Non-finite score for source_event_id={event['source_event_id']}")
                    writer.writerow(
                        {
                            "source_event_id": event["source_event_id"],
                            "user_id": user_id,
                            "item_id": row["item_id"],
                            "score": score_value,
                        }
                    )
                    scored_rows += 1
    return {
        "output_scores_path": str(output_scores_path),
        "candidate_events": len(candidate_events),
        "candidate_rows": total_rows,
        "scored_rows": scored_rows,
        "score_coverage_rate": float(scored_rows / total_rows) if total_rows else 0.0,
    }


def main() -> None:
    args = parse_args()
    np, torch, nn, DataLoader, TensorDataset = _import_torch()
    _set_seed(args.seed, torch, np)

    task_dir = Path(args.task_dir).expanduser()
    train_path = task_dir / "train_interactions.csv"
    candidate_path = task_dir / "candidate_items.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"train_interactions.csv not found: {train_path}")
    if not candidate_path.exists():
        raise FileNotFoundError(f"candidate_items.csv not found: {candidate_path}")

    output_scores_path = Path(args.output_scores_path).expanduser() if args.output_scores_path else task_dir / "irllrec_style_scores.csv"
    summary_path = output_scores_path.with_name(output_scores_path.stem + "_summary.csv")
    epoch_log_path = output_scores_path.with_name(output_scores_path.stem + "_epoch_log.csv")
    metadata_path = output_scores_path.with_name(output_scores_path.stem + "_metadata.json")
    checkpoint_path = output_scores_path.with_name(output_scores_path.stem + "_model.pt")

    train_sequences = _load_train_sequences(train_path)
    candidate_events = _load_candidate_groups(candidate_path)
    item_to_idx = _build_item_vocab(train_sequences, candidate_events)
    train_inputs, train_targets = _make_prefix_training_tensors(
        train_sequences,
        item_to_idx,
        max_seq_len=args.max_seq_len,
        torch=torch,
    )
    pretrained_item_matrix, embedding_info = _build_pretrained_item_matrix(
        item_to_idx=item_to_idx,
        item_map_path=Path(args.item_map_path).expanduser(),
        item_index_col=args.item_index_col,
        embedding_path=Path(args.embedding_path).expanduser(),
        allow_missing_embeddings=args.allow_missing_embeddings,
        np=np,
        torch=torch,
    )

    device = _resolve_device(args.device, torch)
    IRLLRecIntentModel = _build_model_class(torch, nn)
    model = IRLLRecIntentModel(
        pretrained_item_matrix=pretrained_item_matrix,
        hidden_size=args.hidden_size,
        num_intents=args.num_intents,
        dropout=args.dropout,
        finetune_item_embeddings=args.finetune_item_embeddings,
    ).to(device)

    print(
        f"[irllrec-style] users={len(train_sequences)} train_examples={len(train_inputs)} "
        f"items={len(item_to_idx)} emb_dim={pretrained_item_matrix.shape[1]} "
        f"intents={args.num_intents} candidate_events={len(candidate_events)} device={device}",
        flush=True,
    )
    epoch_logs = _train_model(
        model=model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        args=args,
        torch=torch,
        nn=nn,
        DataLoader=DataLoader,
        TensorDataset=TensorDataset,
        device=device,
    )
    score_summary = _score_candidates(
        model=model,
        train_sequences=train_sequences,
        candidate_events=candidate_events,
        item_to_idx=item_to_idx,
        output_scores_path=output_scores_path,
        max_seq_len=args.max_seq_len,
        torch=torch,
        device=device,
    )

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    summary = {
        "baseline_name": "irllrec_style_qwen3_intent",
        "artifact_class": "irllrec_style_same_candidate_score",
        "paper_result_ready": score_summary["score_coverage_rate"] == 1.0,
        "upstream_repo": "https://github.com/wangyu0627/IRLLRec",
        "task_dir": str(task_dir),
        "train_users": len(train_sequences),
        "train_examples": int(train_inputs.shape[0]),
        "item_vocab_size": len(item_to_idx),
        "hidden_size": args.hidden_size,
        "num_intents": args.num_intents,
        "dropout": args.dropout,
        "max_seq_len": args.max_seq_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "finetune_item_embeddings": args.finetune_item_embeddings,
        "device": str(device),
        "seed": args.seed,
        "trainable_params": trainable_params,
        "final_train_loss": epoch_logs[-1]["train_loss"] if epoch_logs else "",
        "checkpoint_path": str(checkpoint_path) if args.save_checkpoint else "",
        **embedding_info,
        **score_summary,
        "note": (
            "Scores are exact same-candidate logits from a local IRLLRec-style "
            "intent representation adapter with Qwen item embeddings. Label as "
            "IRLLRec-style unless the official IRLLRec pipeline is reproduced."
        ),
    }
    _write_csv([summary], summary_path, list(summary.keys()))
    _write_csv(epoch_logs, epoch_log_path, ["epoch", "train_loss"])
    metadata_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.save_checkpoint:
        torch.save(model.state_dict(), checkpoint_path)

    for key, value in summary.items():
        print(f"{key}={value}", flush=True)


if __name__ == "__main__":
    main()
