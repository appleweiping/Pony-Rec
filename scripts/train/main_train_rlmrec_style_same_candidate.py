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
            "Train an RLMRec-style graph/representation baseline with optional "
            "LLM item semantic embeddings and emit exact same-candidate scores. "
            "This is a same-schema style adapter, not an official RLMRec reproduction."
        )
    )
    parser.add_argument("--task_dir", required=True, help="Directory exported by main_export_same_candidate_baseline_task.py.")
    parser.add_argument("--embedding_path", default=None, help="Optional item embedding .pkl or .npy file for Qwen semantic item features.")
    parser.add_argument("--item_map_path", default=None, help="CSV with item_id and mapped item index columns.")
    parser.add_argument("--item_index_col", default="auto")
    parser.add_argument("--output_scores_path", default=None)
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-5)
    parser.add_argument("--reg_weight", type=float, default=1.0e-4)
    parser.add_argument("--cl_weight", type=float, default=0.05)
    parser.add_argument("--semantic_weight", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.2)
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
        from torch.nn import functional as F
        from torch.utils.data import DataLoader, TensorDataset

        return np, torch, nn, F, DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError("main_train_rlmrec_style_same_candidate.py requires numpy and PyTorch.") from exc


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


def _build_vocabs(
    train_sequences: dict[str, list[str]],
    candidate_events: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    user_ids = set(train_sequences)
    item_ids: set[str] = set()
    for user_id, seq in train_sequences.items():
        user_ids.add(user_id)
        item_ids.update(seq)
    for event in candidate_events:
        user_ids.add(event["user_id"])
        for row in event["candidate_rows"]:
            item_ids.add(row["item_id"])
    user_to_idx = {user_id: idx for idx, user_id in enumerate(sorted(user_ids))}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(sorted(item_ids))}
    return user_to_idx, item_to_idx


def _make_positive_pairs(
    train_sequences: dict[str, list[str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
) -> tuple[list[tuple[int, int]], dict[int, set[int]]]:
    pairs: set[tuple[int, int]] = set()
    positives_by_user: dict[int, set[int]] = defaultdict(set)
    for user_id, seq in train_sequences.items():
        user_idx = user_to_idx[user_id]
        for item_id in seq:
            if item_id not in item_to_idx:
                continue
            item_idx = item_to_idx[item_id]
            pairs.add((user_idx, item_idx))
            positives_by_user[user_idx].add(item_idx)
    if not pairs:
        raise ValueError("No trainable user-item edges found.")
    return sorted(pairs), positives_by_user


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


def _auto_item_index_col(rows: list[dict[str, str]], requested: str) -> str:
    if requested != "auto":
        return requested
    columns = set(rows[0]) if rows else set()
    for column in ["rlmrec_item_idx", "llmesr_item_idx", "llm2rec_item_idx", "item_idx", "mapped_item_idx"]:
        if column in columns:
            return column
    raise ValueError(f"Could not infer item index column from {sorted(columns)}")


def _load_embedding_matrix(path: Path, np: Any) -> Any:
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


def _build_semantic_item_matrix(
    *,
    item_to_idx: dict[str, int],
    embedding_path: str | None,
    item_map_path: str | None,
    item_index_col: str,
    allow_missing_embeddings: bool,
    np: Any,
    torch: Any,
) -> tuple[Any | None, dict[str, Any]]:
    if not embedding_path:
        return None, {"semantic_embedding_status": "not_used"}
    if not item_map_path:
        raise ValueError("--item_map_path is required when --embedding_path is provided.")
    map_rows = _read_csv(Path(item_map_path).expanduser())
    index_col = _auto_item_index_col(map_rows, item_index_col)
    mapped_by_item: dict[str, int] = {}
    for row in map_rows:
        item_id = _text(row.get("item_id"))
        value = _text(row.get(index_col))
        if item_id and value:
            mapped_by_item[item_id] = int(float(value))
    source_matrix = _load_embedding_matrix(Path(embedding_path).expanduser(), np)
    out = np.zeros((len(item_to_idx), source_matrix.shape[1]), dtype=np.float32)
    missing: list[str] = []
    for item_id, item_idx in item_to_idx.items():
        mapped_idx = mapped_by_item.get(item_id)
        if mapped_idx is None:
            missing.append(item_id)
            continue
        out[item_idx] = source_matrix[_mapped_index_to_row(mapped_idx, source_matrix.shape[0])]
    if missing and not allow_missing_embeddings:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"Missing semantic embeddings for {len(missing)} items. "
            f"First missing item_ids: {preview}. Pass --allow_missing_embeddings only for diagnostics."
        )
    return torch.tensor(out, dtype=torch.float32), {
        "semantic_embedding_status": "used",
        "embedding_path": str(Path(embedding_path).expanduser()),
        "item_map_path": str(Path(item_map_path).expanduser()),
        "item_index_col": index_col,
        "source_embedding_rows": int(source_matrix.shape[0]),
        "source_embedding_dim": int(source_matrix.shape[1]),
        "missing_embedding_items": len(missing),
    }


def _build_model_class(torch: Any, nn: Any, F: Any):
    class RLMRecStyleGraphCL(nn.Module):
        def __init__(
            self,
            *,
            num_users: int,
            num_items: int,
            embedding_size: int,
            num_layers: int,
            dropout: float,
            semantic_item_matrix: Any | None,
        ) -> None:
            super().__init__()
            self.num_users = num_users
            self.num_items = num_items
            self.num_layers = num_layers
            self.dropout = dropout
            self.user_embedding = nn.Embedding(num_users, embedding_size)
            self.item_embedding = nn.Embedding(num_items, embedding_size)
            nn.init.normal_(self.user_embedding.weight, std=0.1)
            nn.init.normal_(self.item_embedding.weight, std=0.1)
            if semantic_item_matrix is not None:
                self.register_buffer("semantic_item_matrix", semantic_item_matrix)
                self.semantic_adapter = nn.Sequential(
                    nn.Linear(semantic_item_matrix.shape[1], embedding_size),
                    nn.LayerNorm(embedding_size),
                    nn.GELU(),
                )
            else:
                self.semantic_item_matrix = None
                self.semantic_adapter = None

        def semantic_item_embedding(self) -> Any | None:
            if self.semantic_adapter is None or self.semantic_item_matrix is None:
                return None
            return self.semantic_adapter(self.semantic_item_matrix)

        def initial_embeddings(self, *, augment: bool = False) -> tuple[Any, Any]:
            users = self.user_embedding.weight
            items = self.item_embedding.weight
            semantic = self.semantic_item_embedding()
            if semantic is not None:
                items = items + semantic
            if augment and self.dropout > 0:
                users = F.dropout(users, p=self.dropout, training=True)
                items = F.dropout(items, p=self.dropout, training=True)
            return users, items

        def propagate(self, normalized_adj: Any, *, augment: bool = False) -> tuple[Any, Any]:
            users, items = self.initial_embeddings(augment=augment)
            all_embeddings = torch.cat([users, items], dim=0)
            layer_embeddings = [all_embeddings]
            for _ in range(self.num_layers):
                all_embeddings = torch.sparse.mm(normalized_adj, all_embeddings)
                layer_embeddings.append(all_embeddings)
            final = torch.stack(layer_embeddings, dim=0).mean(dim=0)
            return final[: self.num_users], final[self.num_users :]

    return RLMRecStyleGraphCL


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


def _info_nce(z1: Any, z2: Any, *, temperature: float, torch: Any, F: Any) -> Any:
    if z1.size(0) <= 1:
        return z1.new_tensor(0.0)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) * 0.5


def _train_model(
    *,
    model: Any,
    normalized_adj: Any,
    train_pairs: list[tuple[int, int]],
    positives_by_user: dict[int, set[int]],
    args: argparse.Namespace,
    torch: Any,
    F: Any,
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
    logs: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_bpr = 0.0
        total_cl = 0.0
        steps = 0
        for batch_users, batch_pos in loader:
            batch_users = batch_users.to(device)
            batch_pos = batch_pos.to(device)
            batch_neg = _sample_negatives(
                batch_users.detach().cpu().tolist(),
                positives_by_user=positives_by_user,
                num_items=model.num_items,
                torch=torch,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            user_emb, item_emb = model.propagate(normalized_adj, augment=False)
            pos_scores = (user_emb[batch_users] * item_emb[batch_pos]).sum(dim=-1)
            neg_scores = (user_emb[batch_users] * item_emb[batch_neg]).sum(dim=-1)
            bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            reg_loss = (
                model.user_embedding(batch_users).pow(2).sum(dim=-1).mean()
                + model.item_embedding(batch_pos).pow(2).sum(dim=-1).mean()
                + model.item_embedding(batch_neg).pow(2).sum(dim=-1).mean()
            ) * args.reg_weight

            user_aug1, item_aug1 = model.propagate(normalized_adj, augment=True)
            user_aug2, item_aug2 = model.propagate(normalized_adj, augment=True)
            unique_users = torch.unique(batch_users)
            unique_items = torch.unique(batch_pos)
            cl_loss = _info_nce(user_aug1[unique_users], user_aug2[unique_users], temperature=args.temperature, torch=torch, F=F)
            cl_loss = cl_loss + _info_nce(item_aug1[unique_items], item_aug2[unique_items], temperature=args.temperature, torch=torch, F=F)

            semantic_loss = item_emb.new_tensor(0.0)
            semantic = model.semantic_item_embedding()
            if semantic is not None and args.semantic_weight > 0:
                semantic_loss = 1.0 - F.cosine_similarity(item_emb[unique_items], semantic[unique_items], dim=-1).mean()

            loss = bpr_loss + reg_loss + args.cl_weight * cl_loss + args.semantic_weight * semantic_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_bpr += float(bpr_loss.detach().cpu())
            total_cl += float(cl_loss.detach().cpu())
            steps += 1
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(steps, 1),
            "bpr_loss": total_bpr / max(steps, 1),
            "cl_loss": total_cl / max(steps, 1),
        }
        logs.append(row)
        if args.log_every > 0 and (epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0):
            print(
                f"[rlmrec-style] epoch={epoch} train_loss={row['train_loss']:.6f} "
                f"bpr={row['bpr_loss']:.6f} cl={row['cl_loss']:.6f}",
                flush=True,
            )
    return logs


def _score_candidates(
    *,
    model: Any,
    normalized_adj: Any,
    candidate_events: list[dict[str, Any]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    output_scores_path: Path,
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
            user_emb, item_emb = model.propagate(normalized_adj, augment=False)
            for event in candidate_events:
                user_id = event["user_id"]
                candidate_rows = event["candidate_rows"]
                total_rows += len(candidate_rows)
                user_idx = user_to_idx[user_id]
                candidate_indices = [item_to_idx[row["item_id"]] for row in candidate_rows]
                candidate_tensor = torch.tensor(candidate_indices, dtype=torch.long, device=device)
                score_tensor = item_emb.index_select(0, candidate_tensor).matmul(user_emb[user_idx])
                if not torch.isfinite(score_tensor).all():
                    raise RuntimeError("RLMRec-style model produced non-finite candidate scores.")
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
    np, torch, nn, F, DataLoader, TensorDataset = _import_torch()
    _set_seed(args.seed, torch, np)

    task_dir = Path(args.task_dir).expanduser()
    train_path = task_dir / "train_interactions.csv"
    candidate_path = task_dir / "candidate_items.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"train_interactions.csv not found: {train_path}")
    if not candidate_path.exists():
        raise FileNotFoundError(f"candidate_items.csv not found: {candidate_path}")

    output_scores_path = Path(args.output_scores_path).expanduser() if args.output_scores_path else task_dir / "rlmrec_style_scores.csv"
    summary_path = output_scores_path.with_name(output_scores_path.stem + "_summary.csv")
    epoch_log_path = output_scores_path.with_name(output_scores_path.stem + "_epoch_log.csv")
    metadata_path = output_scores_path.with_name(output_scores_path.stem + "_metadata.json")
    checkpoint_path = output_scores_path.with_name(output_scores_path.stem + "_model.pt")

    train_sequences = _load_train_sequences(train_path)
    candidate_events = _load_candidate_groups(candidate_path)
    user_to_idx, item_to_idx = _build_vocabs(train_sequences, candidate_events)
    train_pairs, positives_by_user = _make_positive_pairs(train_sequences, user_to_idx, item_to_idx)
    semantic_item_matrix, semantic_info = _build_semantic_item_matrix(
        item_to_idx=item_to_idx,
        embedding_path=args.embedding_path,
        item_map_path=args.item_map_path,
        item_index_col=args.item_index_col,
        allow_missing_embeddings=args.allow_missing_embeddings,
        np=np,
        torch=torch,
    )

    device = _resolve_device(args.device, torch)
    normalized_adj = _build_normalized_adj(
        train_pairs,
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
        torch=torch,
        device=device,
    )
    semantic_item_matrix = semantic_item_matrix.to(device) if semantic_item_matrix is not None else None
    RLMRecStyleGraphCL = _build_model_class(torch, nn, F)
    model = RLMRecStyleGraphCL(
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
        embedding_size=args.embedding_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        semantic_item_matrix=semantic_item_matrix,
    ).to(device)

    print(
        f"[rlmrec-style] users={len(user_to_idx)} items={len(item_to_idx)} "
        f"train_edges={len(train_pairs)} candidate_events={len(candidate_events)} "
        f"semantic={semantic_info['semantic_embedding_status']} device={device}",
        flush=True,
    )
    epoch_logs = _train_model(
        model=model,
        normalized_adj=normalized_adj,
        train_pairs=train_pairs,
        positives_by_user=positives_by_user,
        args=args,
        torch=torch,
        F=F,
        DataLoader=DataLoader,
        TensorDataset=TensorDataset,
        device=device,
    )
    score_summary = _score_candidates(
        model=model,
        normalized_adj=normalized_adj,
        candidate_events=candidate_events,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        output_scores_path=output_scores_path,
        torch=torch,
        device=device,
    )

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    summary = {
        "baseline_name": "rlmrec_style_qwen3_graphcl",
        "artifact_class": "rlmrec_style_same_candidate_score",
        "paper_result_ready": score_summary["score_coverage_rate"] == 1.0,
        "upstream_repo": "https://github.com/HKUDS/RLMRec",
        "task_dir": str(task_dir),
        "train_users": len(user_to_idx),
        "item_vocab_size": len(item_to_idx),
        "train_edges": len(train_pairs),
        "embedding_size": args.embedding_size,
        "num_layers": args.num_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "reg_weight": args.reg_weight,
        "cl_weight": args.cl_weight,
        "semantic_weight": args.semantic_weight,
        "dropout": args.dropout,
        "temperature": args.temperature,
        "device": str(device),
        "seed": args.seed,
        "trainable_params": trainable_params,
        "final_train_loss": epoch_logs[-1]["train_loss"] if epoch_logs else "",
        "checkpoint_path": str(checkpoint_path) if args.save_checkpoint else "",
        **semantic_info,
        **score_summary,
        "note": (
            "Scores are exact same-candidate logits from a local RLMRec-style "
            "graph contrastive representation adapter with optional Qwen item "
            "semantic embeddings. Label as RLMRec-style unless the official "
            "RLMRec pipeline is reproduced."
        ),
    }
    _write_csv([summary], summary_path, list(summary.keys()))
    _write_csv(epoch_logs, epoch_log_path, ["epoch", "train_loss", "bpr_loss", "cl_loss"])
    metadata_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.save_checkpoint:
        torch.save(model.state_dict(), checkpoint_path)

    for key, value in summary.items():
        print(f"{key}={value}", flush=True)


if __name__ == "__main__":
    main()
