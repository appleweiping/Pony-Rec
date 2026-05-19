from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight SASRec baseline and score exact candidate rows."
    )
    parser.add_argument("--task_dir", required=True, help="Directory exported by main_export_same_candidate_baseline_task.py.")
    parser.add_argument("--output_scores_path", default=None)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_checkpoint", action="store_true")
    return parser.parse_args()


def _import_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        return torch, nn, DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError("main_train_sasrec_same_candidate.py requires PyTorch.") from exc


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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
        groups[key]["candidate_rows"].append(
            {
                "candidate_index": candidate_index,
                "item_id": item_id,
            }
        )

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


def _make_training_tensors(
    train_sequences: dict[str, list[str]],
    item_to_idx: dict[str, int],
    *,
    max_seq_len: int,
    torch: Any,
) -> tuple[Any, Any]:
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for seq in train_sequences.values():
        encoded = [item_to_idx[item_id] for item_id in seq if item_id in item_to_idx]
        if len(encoded) < 2:
            continue
        src = encoded[:-1][-max_seq_len:]
        tgt = encoded[1:][-max_seq_len:]
        pad_len = max_seq_len - len(src)
        inputs.append([0] * pad_len + src)
        targets.append([0] * pad_len + tgt)
    if not inputs:
        raise ValueError("No trainable SASRec sequences found; need at least one user sequence with length >= 2.")
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def _resolve_device(device_arg: str, torch: Any) -> Any:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model_class(torch: Any, nn: Any):
    class SASRecModel(nn.Module):
        def __init__(
            self,
            *,
            num_items: int,
            hidden_size: int,
            max_seq_len: int,
            num_heads: int,
            num_layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
            self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.layer_norm = nn.LayerNorm(hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.max_seq_len = max_seq_len

        def forward(self, input_ids: Any) -> Any:
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
            hidden = self.item_embedding(input_ids) + self.position_embedding(positions)
            nonpad_mask = input_ids.ne(0).unsqueeze(-1)
            hidden = hidden * nonpad_mask
            hidden = self.dropout(hidden)
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
                diagonal=1,
            )
            encoded = self.encoder(hidden, mask=causal_mask)
            return self.layer_norm(encoded) * nonpad_mask

    return SASRecModel


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
    model.train()
    logs: list[dict[str, Any]] = []
    num_items = model.item_embedding.num_embeddings

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        steps = 0
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            hidden = model(batch_inputs)
            logits = hidden @ model.item_embedding.weight.t()
            loss = loss_fn(logits.reshape(-1, num_items), batch_targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            steps += 1
        mean_loss = total_loss / max(steps, 1)
        logs.append({"epoch": epoch, "train_loss": mean_loss})
        if args.log_every > 0 and (epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0):
            print(f"[sasrec] epoch={epoch} train_loss={mean_loss:.6f}")
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
                seq = train_sequences.get(user_id, [])
                candidate_rows = event["candidate_rows"]
                total_rows += len(candidate_rows)

                if not seq:
                    for row in candidate_rows:
                        writer.writerow(
                            {
                                "source_event_id": event["source_event_id"],
                                "user_id": user_id,
                                "item_id": row["item_id"],
                                "score": 0.0,
                            }
                        )
                        scored_rows += 1
                    continue

                input_ids = _context_tensor(seq, item_to_idx, max_seq_len=max_seq_len, torch=torch, device=device)
                hidden = model(input_ids)
                user_repr = hidden[:, -1, :]
                candidate_indices = [item_to_idx[row["item_id"]] for row in candidate_rows]
                candidate_tensor = torch.tensor(candidate_indices, dtype=torch.long, device=device)
                candidate_emb = model.item_embedding(candidate_tensor)
                score_tensor = (candidate_emb * user_repr.squeeze(0)).sum(dim=-1)
                if not torch.isfinite(score_tensor).all():
                    raise RuntimeError(
                        "SASRec produced non-finite candidate scores. "
                        "This run is invalid; rerun after checking model stability."
                    )
                scores = score_tensor.detach().cpu().tolist()
                for row, score in zip(candidate_rows, scores):
                    writer.writerow(
                        {
                            "source_event_id": event["source_event_id"],
                            "user_id": user_id,
                            "item_id": row["item_id"],
                            "score": float(score),
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


def _write_rows(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    args = parse_args()
    torch, nn, DataLoader, TensorDataset = _import_torch()
    _set_seed(args.seed, torch)

    task_dir = Path(args.task_dir).expanduser()
    train_path = task_dir / "train_interactions.csv"
    candidate_path = task_dir / "candidate_items.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"train_interactions.csv not found: {train_path}")
    if not candidate_path.exists():
        raise FileNotFoundError(f"candidate_items.csv not found: {candidate_path}")

    output_scores_path = Path(args.output_scores_path).expanduser() if args.output_scores_path else task_dir / "sasrec_scores.csv"
    summary_path = task_dir / "sasrec_train_summary.csv"
    epoch_log_path = task_dir / "sasrec_epoch_log.csv"
    metadata_path = task_dir / "sasrec_model_metadata.json"
    checkpoint_path = task_dir / "sasrec_model.pt"

    train_sequences = _load_train_sequences(train_path)
    candidate_events = _load_candidate_groups(candidate_path)
    item_to_idx = _build_item_vocab(train_sequences, candidate_events)
    train_inputs, train_targets = _make_training_tensors(
        train_sequences,
        item_to_idx,
        max_seq_len=args.max_seq_len,
        torch=torch,
    )

    device = _resolve_device(args.device, torch)
    SASRecModel = _build_model_class(torch, nn)
    model = SASRecModel(
        num_items=len(item_to_idx),
        hidden_size=args.hidden_size,
        max_seq_len=args.max_seq_len,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    print(
        f"[sasrec] users={len(train_sequences)} train_sequences={len(train_inputs)} "
        f"items={len(item_to_idx)} candidate_events={len(candidate_events)} device={device}"
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
        "baseline_name": "sasrec",
        "task_dir": str(task_dir),
        "train_users": len(train_sequences),
        "train_sequences": int(train_inputs.shape[0]),
        "item_vocab_size": len(item_to_idx),
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "max_seq_len": args.max_seq_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "trainable_params": trainable_params,
        "final_train_loss": epoch_logs[-1]["train_loss"] if epoch_logs else math.nan,
        **score_summary,
    }
    _write_rows([summary], summary_path, list(summary.keys()))
    _write_rows(epoch_logs, epoch_log_path, ["epoch", "train_loss"])
    metadata_path.write_text(
        json.dumps(
            {
                "item_to_idx_size": len(item_to_idx),
                "args": vars(args),
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.save_checkpoint:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "item_to_idx": item_to_idx,
                "summary": summary,
            },
            checkpoint_path,
        )

    print(f"[sasrec] Saved scores: {output_scores_path}")
    print(f"[sasrec] Saved summary: {summary_path}")
    print(f"[sasrec] score_coverage_rate={score_summary['score_coverage_rate']:.6f}")


if __name__ == "__main__":
    main()
