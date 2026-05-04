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
        description="Train a lightweight LightGCN baseline and score exact candidate rows."
    )
    parser.add_argument("--task_dir", required=True, help="Directory exported by main_export_same_candidate_baseline_task.py.")
    parser.add_argument("--output_scores_path", default=None)
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-5)
    parser.add_argument("--reg_weight", type=float, default=1.0e-4)
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
        raise RuntimeError("main_train_lightgcn_same_candidate.py requires PyTorch.") from exc


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


def _build_user_vocab(train_sequences: dict[str, list[str]], candidate_events: list[dict[str, Any]]) -> dict[str, int]:
    user_ids = set(train_sequences)
    for event in candidate_events:
        user_ids.add(event["user_id"])
    return {user_id: idx for idx, user_id in enumerate(sorted(user_ids))}


def _build_item_vocab(train_sequences: dict[str, list[str]], candidate_events: list[dict[str, Any]]) -> dict[str, int]:
    item_ids: set[str] = set()
    for seq in train_sequences.values():
        item_ids.update(seq)
    for event in candidate_events:
        for row in event["candidate_rows"]:
            item_ids.add(row["item_id"])
    return {item_id: idx for idx, item_id in enumerate(sorted(item_ids))}


def _make_positive_pairs(
    train_sequences: dict[str, list[str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
) -> tuple[list[tuple[int, int]], dict[int, set[int]]]:
    pairs: set[tuple[int, int]] = set()
    positives_by_user: dict[int, set[int]] = defaultdict(set)
    for user_id, seq in train_sequences.items():
        if user_id not in user_to_idx:
            continue
        user_idx = user_to_idx[user_id]
        for item_id in seq:
            if item_id not in item_to_idx:
                continue
            item_idx = item_to_idx[item_id]
            pairs.add((user_idx, item_idx))
            positives_by_user[user_idx].add(item_idx)
    if not pairs:
        raise ValueError("No trainable LightGCN user-item edges found.")
    return sorted(pairs), positives_by_user


def _resolve_device(device_arg: str, torch: Any) -> Any:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_normalized_adj(
    pairs: list[tuple[int, int]],
    *,
    num_users: int,
    num_items: int,
    torch: Any,
    device: Any,
) -> Any:
    num_nodes = num_users + num_items
    rows: list[int] = []
    cols: list[int] = []
    for user_idx, item_idx in pairs:
        item_node = num_users + item_idx
        rows.extend([user_idx, item_node])
        cols.extend([item_node, user_idx])

    row_tensor = torch.tensor(rows, dtype=torch.long)
    col_tensor = torch.tensor(cols, dtype=torch.long)
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    deg.index_add_(0, row_tensor, torch.ones_like(row_tensor, dtype=torch.float32))
    values = deg[row_tensor].clamp(min=1.0).pow(-0.5) * deg[col_tensor].clamp(min=1.0).pow(-0.5)
    indices = torch.stack([row_tensor, col_tensor], dim=0)
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce().to(device)


def _build_model_class(torch: Any, nn: Any):
    class LightGCNModel(nn.Module):
        def __init__(
            self,
            *,
            num_users: int,
            num_items: int,
            embedding_size: int,
            num_layers: int,
        ) -> None:
            super().__init__()
            self.num_users = num_users
            self.num_items = num_items
            self.num_layers = num_layers
            self.user_embedding = nn.Embedding(num_users, embedding_size)
            self.item_embedding = nn.Embedding(num_items, embedding_size)
            nn.init.normal_(self.user_embedding.weight, std=0.1)
            nn.init.normal_(self.item_embedding.weight, std=0.1)

        def propagate(self, normalized_adj: Any) -> tuple[Any, Any]:
            all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
            layer_embeddings = [all_embeddings]
            for _ in range(self.num_layers):
                all_embeddings = torch.sparse.mm(normalized_adj, all_embeddings)
                layer_embeddings.append(all_embeddings)
            final_embeddings = torch.stack(layer_embeddings, dim=0).mean(dim=0)
            return final_embeddings[: self.num_users], final_embeddings[self.num_users :]

    return LightGCNModel


def _sample_negatives(
    user_indices: list[int],
    *,
    positives_by_user: dict[int, set[int]],
    num_items: int,
    torch: Any,
    device: Any,
) -> Any:
    negatives: list[int] = []
    for user_idx in user_indices:
        positives = positives_by_user.get(user_idx, set())
        if len(positives) >= num_items:
            negatives.append(random.randrange(num_items))
            continue
        while True:
            candidate = random.randrange(num_items)
            if candidate not in positives:
                negatives.append(candidate)
                break
    return torch.tensor(negatives, dtype=torch.long, device=device)


def _train_model(
    *,
    model: Any,
    normalized_adj: Any,
    train_pairs: list[tuple[int, int]],
    positives_by_user: dict[int, set[int]],
    args: argparse.Namespace,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    device: Any,
) -> list[dict[str, Any]]:
    user_tensor = torch.tensor([user_idx for user_idx, _ in train_pairs], dtype=torch.long)
    item_tensor = torch.tensor([item_idx for _, item_idx in train_pairs], dtype=torch.long)
    dataset = TensorDataset(user_tensor, item_tensor)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    logs: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_bpr = 0.0
        steps = 0
        for batch_users, batch_pos_items in loader:
            batch_users = batch_users.to(device)
            batch_pos_items = batch_pos_items.to(device)
            batch_neg_items = _sample_negatives(
                [int(value) for value in batch_users.detach().cpu().tolist()],
                positives_by_user=positives_by_user,
                num_items=model.num_items,
                torch=torch,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            user_embeddings, item_embeddings = model.propagate(normalized_adj)
            batch_user_embeddings = user_embeddings[batch_users]
            pos_embeddings = item_embeddings[batch_pos_items]
            neg_embeddings = item_embeddings[batch_neg_items]
            pos_scores = (batch_user_embeddings * pos_embeddings).sum(dim=-1)
            neg_scores = (batch_user_embeddings * neg_embeddings).sum(dim=-1)
            bpr_loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
            reg_loss = (
                model.user_embedding(batch_users).pow(2).sum()
                + model.item_embedding(batch_pos_items).pow(2).sum()
                + model.item_embedding(batch_neg_items).pow(2).sum()
            ) / max(int(batch_users.numel()), 1)
            loss = bpr_loss + args.reg_weight * reg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_bpr += float(bpr_loss.detach().cpu())
            steps += 1

        mean_loss = total_loss / max(steps, 1)
        mean_bpr = total_bpr / max(steps, 1)
        logs.append({"epoch": epoch, "train_loss": mean_loss, "bpr_loss": mean_bpr})
        if args.log_every > 0 and (epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0):
            print(f"[lightgcn] epoch={epoch} train_loss={mean_loss:.6f} bpr_loss={mean_bpr:.6f}")
    return logs


def _score_candidates(
    *,
    model: Any,
    normalized_adj: Any,
    train_sequences: dict[str, list[str]],
    candidate_events: list[dict[str, Any]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    output_scores_path: Path,
    torch: Any,
) -> dict[str, Any]:
    output_scores_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    scored_rows = 0
    model.eval()
    with output_scores_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["source_event_id", "user_id", "item_id", "score"])
        writer.writeheader()
        with torch.no_grad():
            user_embeddings, item_embeddings = model.propagate(normalized_adj)
            for event in candidate_events:
                user_id = event["user_id"]
                candidate_rows = event["candidate_rows"]
                total_rows += len(candidate_rows)

                if user_id not in user_to_idx or not train_sequences.get(user_id):
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

                user_idx = user_to_idx[user_id]
                candidate_indices = [item_to_idx[row["item_id"]] for row in candidate_rows]
                candidate_tensor = torch.tensor(candidate_indices, dtype=torch.long, device=user_embeddings.device)
                score_tensor = (item_embeddings[candidate_tensor] * user_embeddings[user_idx]).sum(dim=-1)
                if not torch.isfinite(score_tensor).all():
                    raise RuntimeError(
                        "LightGCN produced non-finite candidate scores. "
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

    output_scores_path = Path(args.output_scores_path).expanduser() if args.output_scores_path else task_dir / "lightgcn_scores.csv"
    summary_path = task_dir / "lightgcn_train_summary.csv"
    epoch_log_path = task_dir / "lightgcn_epoch_log.csv"
    metadata_path = task_dir / "lightgcn_model_metadata.json"
    checkpoint_path = task_dir / "lightgcn_model.pt"

    train_sequences = _load_train_sequences(train_path)
    candidate_events = _load_candidate_groups(candidate_path)
    user_to_idx = _build_user_vocab(train_sequences, candidate_events)
    item_to_idx = _build_item_vocab(train_sequences, candidate_events)
    train_pairs, positives_by_user = _make_positive_pairs(train_sequences, user_to_idx, item_to_idx)

    device = _resolve_device(args.device, torch)
    normalized_adj = _build_normalized_adj(
        train_pairs,
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
        torch=torch,
        device=device,
    )
    LightGCNModel = _build_model_class(torch, nn)
    model = LightGCNModel(
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
        embedding_size=args.embedding_size,
        num_layers=args.num_layers,
    ).to(device)

    print(
        f"[lightgcn] users={len(user_to_idx)} train_edges={len(train_pairs)} "
        f"items={len(item_to_idx)} candidate_events={len(candidate_events)} device={device}"
    )
    epoch_logs = _train_model(
        model=model,
        normalized_adj=normalized_adj,
        train_pairs=train_pairs,
        positives_by_user=positives_by_user,
        args=args,
        torch=torch,
        DataLoader=DataLoader,
        TensorDataset=TensorDataset,
        device=device,
    )
    score_summary = _score_candidates(
        model=model,
        normalized_adj=normalized_adj,
        train_sequences=train_sequences,
        candidate_events=candidate_events,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        output_scores_path=output_scores_path,
        torch=torch,
    )

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    summary = {
        "baseline_name": "lightgcn",
        "task_dir": str(task_dir),
        "train_users": len(train_sequences),
        "graph_users": len(user_to_idx),
        "train_edges": len(train_pairs),
        "item_vocab_size": len(item_to_idx),
        "embedding_size": args.embedding_size,
        "num_layers": args.num_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "reg_weight": args.reg_weight,
        "seed": args.seed,
        "device": str(device),
        "trainable_params": trainable_params,
        "final_train_loss": epoch_logs[-1]["train_loss"] if epoch_logs else math.nan,
        "final_bpr_loss": epoch_logs[-1]["bpr_loss"] if epoch_logs else math.nan,
        **score_summary,
    }
    _write_rows([summary], summary_path, list(summary.keys()))
    _write_rows(epoch_logs, epoch_log_path, ["epoch", "train_loss", "bpr_loss"])
    metadata_path.write_text(
        json.dumps(
            {
                "user_to_idx_size": len(user_to_idx),
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
                "user_to_idx": user_to_idx,
                "item_to_idx": item_to_idx,
                "summary": summary,
            },
            checkpoint_path,
        )

    print(f"[lightgcn] Saved scores: {output_scores_path}")
    print(f"[lightgcn] Saved summary: {summary_path}")
    print(f"[lightgcn] score_coverage_rate={score_summary['score_coverage_rate']:.6f}")


if __name__ == "__main__":
    main()
