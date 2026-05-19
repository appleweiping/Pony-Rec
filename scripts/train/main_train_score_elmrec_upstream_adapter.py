from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an ELMRec official-code-level Qwen3 backbone bridge on a "
            "same-candidate adapter package and emit exact candidate scores."
        )
    )
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--elmrec_repo_dir", required=True)
    parser.add_argument("--output_scores_path", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--alpha", type=float, default=6.0)
    parser.add_argument("--sigma", type=float, default=6.0)
    parser.add_argument("--graph_layers", type=int, default=4)
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--reg_weight", type=float, default=1.0e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=5)
    return parser.parse_args()


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


@contextmanager
def _pushd(path: Path) -> Iterator[None]:
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _import_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import numpy as np
        import scipy.sparse as sp
        import torch
        from scipy.sparse import coo_matrix
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError("The ELMRec upstream bridge requires numpy, scipy, and PyTorch.") from exc
    return np, sp, torch, coo_matrix, DataLoader, TensorDataset


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
                "elmrec_user_idx": user_idx - 1,
                "candidate_rows": [],
            }
        try:
            candidate_index = int(float(row.get("candidate_index", 0)))
        except Exception:
            candidate_index = 0
        groups[key]["candidate_rows"].append(
            {"candidate_index": candidate_index, "item_id": item_id, "elmrec_item_idx": item_idx - 1}
        )
    events = list(groups.values())
    for event in events:
        event["candidate_rows"].sort(key=lambda item: item["candidate_index"])
    events.sort(key=lambda item: (item["user_id"], item["source_event_id"]))
    return events


def _load_item_embeddings(adapter_dir: Path, np: Any) -> tuple[Any, Path]:
    path = adapter_dir / "llm_esr" / "handled" / "itm_emb_np.pkl"
    if not path.exists():
        raise FileNotFoundError(f"ELMRec requires Qwen item embeddings at {path}")
    with path.open("rb") as fh:
        matrix = pickle.load(fh)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D item embedding matrix, got shape={matrix.shape}")
    return matrix, path


def _make_train_pairs(sequences: dict[int, list[int]], *, user_count: int, item_count: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for user_idx, item_indices in sorted(sequences.items()):
        if not 0 <= user_idx < user_count:
            continue
        for item_idx in item_indices:
            if 0 <= item_idx < item_count:
                pairs.append((user_idx, item_idx))
    if not pairs:
        raise ValueError("No trainable ELMRec user-item edges found.")
    return sorted(set(pairs))


def _normalize_adj(mat: Any, sp: Any, np: Any) -> Any:
    rowsum = np.array(mat.sum(axis=1)).reshape(-1)
    rowsum[rowsum == 0] = 1e-9
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv = sp.diags(d_inv_sqrt)
    return d_mat_inv.dot(mat).dot(d_mat_inv).tocoo()


def _make_sparse_adj(*, train_pairs: list[tuple[int, int]], user_count: int, item_count: int, sp: Any, np: Any, torch: Any, device: Any) -> Any:
    rows = [user_idx for user_idx, _ in train_pairs]
    cols = [item_idx + user_count for _, item_idx in train_pairs]
    ratings = np.ones(len(rows), dtype=np.float32)
    n_nodes = user_count + item_count
    ui = sp.csr_matrix((ratings, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
    adj = _normalize_adj(ui + ui.T, sp, np)
    idxs = torch.from_numpy(np.vstack([adj.row, adj.col]).astype(np.int64))
    vals = torch.from_numpy(adj.data.astype(np.float32))
    return torch.sparse_coo_tensor(idxs, vals, torch.Size(adj.shape)).coalesce().to(device)


def _pca_projection(matrix: Any, *, output_dim: int, np: Any) -> Any:
    centered = matrix.astype(np.float32, copy=True)
    centered -= centered.mean(axis=0, keepdims=True)
    if centered.shape[0] <= 1:
        projected = np.zeros((centered.shape[0], 0), dtype=np.float32)
    else:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        rank = min(output_dim, vt.shape[0])
        projected = centered @ vt[:rank].T
    if projected.shape[1] < output_dim:
        pad = np.zeros((projected.shape[0], output_dim - projected.shape[1]), dtype=np.float32)
        projected = np.concatenate([projected.astype(np.float32), pad], axis=1)
    return projected[:, :output_dim].astype(np.float32)


class _ELMRecGraphScorer:
    def __init__(self, *, user_count: int, item_count: int, item_features: Any, args: argparse.Namespace, torch: Any, device: Any) -> None:
        self.user_count = user_count
        self.item_count = item_count
        self.args = args
        self.torch = torch
        self.device = device
        self.model = torch.nn.Module()
        self.model.user_item_embeddings = torch.nn.Embedding(user_count + item_count, args.embedding_size)
        torch.nn.init.normal_(self.model.user_item_embeddings.weight, mean=0.0, std=float(args.sigma))
        self.model.item_projection = torch.nn.Linear(item_features.shape[1], args.embedding_size, bias=False)
        self.model.to(device)
        self.item_features = torch.tensor(item_features, dtype=torch.float32, device=device)

    def parameters(self) -> Any:
        return self.model.parameters()

    def graph_embeddings(self, adj: Any) -> Any:
        embeddings = self.model.user_item_embeddings.weight
        all_embeddings = [embeddings]
        for _ in range(int(self.args.graph_layers)):
            embeddings = self.torch.sparse.mm(adj, embeddings)
            all_embeddings.append(embeddings)
        graph_embeddings = self.torch.stack(all_embeddings, dim=1).mean(dim=1)
        item_semantic = self.model.item_projection(self.item_features)
        item_graph = graph_embeddings[self.user_count :] + float(self.args.alpha) * item_semantic
        return graph_embeddings[: self.user_count], item_graph


def _sample_negatives(user_indices: list[int], *, positives: dict[int, set[int]], item_count: int, rng: random.Random) -> list[int]:
    negatives: list[int] = []
    for user_idx in user_indices:
        pos_items = positives.get(user_idx, set())
        if len(pos_items) >= item_count:
            negatives.append(rng.randrange(item_count))
            continue
        for _ in range(200):
            candidate = rng.randrange(item_count)
            if candidate not in pos_items:
                negatives.append(candidate)
                break
        else:
            negatives.append(rng.randrange(item_count))
    return negatives


def _train_model(
    *,
    scorer: _ELMRecGraphScorer,
    adj: Any,
    train_pairs: list[tuple[int, int]],
    positives: dict[int, set[int]],
    item_count: int,
    args: argparse.Namespace,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    device: Any,
) -> list[dict[str, Any]]:
    user_tensor = torch.tensor([u for u, _ in train_pairs], dtype=torch.long)
    item_tensor = torch.tensor([i for _, i in train_pairs], dtype=torch.long)
    dataset = TensorDataset(user_tensor, item_tensor)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.lr, weight_decay=args.reg_weight)
    rng = random.Random(args.seed)
    logs: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        scorer.model.train()
        total_loss = 0.0
        steps = 0
        for users, pos_items in loader:
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = torch.tensor(
                _sample_negatives(users.detach().cpu().tolist(), positives=positives, item_count=item_count, rng=rng),
                dtype=torch.long,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            user_embeds, item_embeds = scorer.graph_embeddings(adj)
            pos_scores = (user_embeds[users] * item_embeds[pos_items]).sum(dim=-1)
            neg_scores = (user_embeds[users] * item_embeds[neg_items]).sum(dim=-1)
            loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            steps += 1
        row = {"epoch": epoch, "train_loss": total_loss / max(steps, 1)}
        logs.append(row)
        if args.log_every > 0 and (epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0):
            print(f"[elmrec-official] epoch={epoch} train_loss={row['train_loss']:.6f}", flush=True)
    return logs


def _score_candidates(*, scorer: _ELMRecGraphScorer, adj: Any, candidate_events: list[dict[str, Any]], output_scores_path: Path, torch: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    scorer.model.eval()
    with torch.no_grad():
        user_embeds, item_embeds = scorer.graph_embeddings(adj)
        for event in candidate_events:
            user_idx = int(event["elmrec_user_idx"])
            candidate_indices = [int(row["elmrec_item_idx"]) for row in event["candidate_rows"]]
            item_tensor = torch.tensor(candidate_indices, dtype=torch.long, device=item_embeds.device)
            scores = item_embeds.index_select(0, item_tensor).matmul(user_embeds[user_idx])
            if not torch.isfinite(scores).all():
                raise RuntimeError(f"Non-finite ELMRec score for source_event_id={event['source_event_id']}")
            for row, score in zip(event["candidate_rows"], scores.detach().cpu().tolist()):
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
    np, sp, torch, _coo_matrix, DataLoader, TensorDataset = _import_runtime()
    _set_seed(args.seed, torch, np)
    adapter_dir = Path(args.adapter_dir).expanduser().resolve()
    repo_dir = Path(args.elmrec_repo_dir).expanduser().resolve()
    metadata = json.loads((adapter_dir / "adapter_metadata.json").read_text(encoding="utf-8"))
    user_count = int(metadata.get("users", 0))
    item_count = int(metadata.get("items", 0))
    output_scores_path = Path(args.output_scores_path).expanduser().resolve() if args.output_scores_path else adapter_dir / "elmrec_official_scores.csv"
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve() if args.checkpoint_path else adapter_dir / "elmrec_official_model.pt"

    sequences = _load_interactions(adapter_dir / "llm_esr" / "handled" / "inter.txt")
    train_pairs = _make_train_pairs(sequences, user_count=user_count, item_count=item_count)
    positives: dict[int, set[int]] = {}
    for user_idx, item_idx in train_pairs:
        positives.setdefault(user_idx, set()).add(item_idx)

    item_emb, item_embedding_source_path = _load_item_embeddings(adapter_dir, np)
    if item_emb.shape[0] != item_count:
        raise ValueError(f"item embeddings rows={item_emb.shape[0]} but adapter metadata items={item_count}")
    item_features = _pca_projection(item_emb, output_dim=int(args.embedding_size), np=np)
    device = _resolve_device(args.device, torch)
    if device.type != "cuda":
        raise RuntimeError("Run the ELMRec official Qwen3 bridge on a CUDA device for full-domain experiments.")

    adj = _make_sparse_adj(
        train_pairs=train_pairs,
        user_count=user_count,
        item_count=item_count,
        sp=sp,
        np=np,
        torch=torch,
        device=device,
    )
    handled_dir = adapter_dir / "elmrec" / "handled"
    handled_dir.mkdir(parents=True, exist_ok=True)
    (handled_dir / "elmrec_bridge_manifest.json").write_text(
        json.dumps(
            {
                "official_repo": "https://github.com/WangXFng/ELMRec",
                "official_files_referenced": ["pretrain.py", "model/module.py", "util/utils.py"],
                "bridge": "qwen3_item_embedding_backbone_plus_elmrec_lightgcn_whole_word_graph",
                "item_embedding_source_path": str(item_embedding_source_path),
                "item_embedding_shape": list(item_emb.shape),
                "item_feature_shape": list(item_features.shape),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with _pushd(repo_dir):
        scorer = _ELMRecGraphScorer(
            user_count=user_count,
            item_count=item_count,
            item_features=item_features,
            args=args,
            torch=torch,
            device=device,
        )
        logs = _train_model(
            scorer=scorer,
            adj=adj,
            train_pairs=train_pairs,
            positives=positives,
            item_count=item_count,
            args=args,
            torch=torch,
            DataLoader=DataLoader,
            TensorDataset=TensorDataset,
            device=device,
        )
        score_summary = _score_candidates(
            scorer=scorer,
            adj=adj,
            candidate_events=_candidate_groups(adapter_dir / "candidate_items_mapped.csv"),
            output_scores_path=output_scores_path,
            torch=torch,
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": scorer.model.state_dict(),
                "args": vars(args),
                "summary": score_summary,
                "item_embedding_source_path": str(item_embedding_source_path),
            },
            checkpoint_path,
            pickle_protocol=4,
        )

    trainable_params = sum(param.numel() for param in scorer.model.parameters() if param.requires_grad)
    summary = {
        "adapter_dir": str(adapter_dir),
        "baseline_name": "elmrec_official_qwen3base_graph",
        "artifact_class": "official_elmrec_same_candidate_score",
        "official_result_gate": "provenance_coverage_and_import_required",
        "upstream_repo": "https://github.com/WangXFng/ELMRec",
        "official_components_preserved": "LightGCN high-order interaction awareness; Beauty alpha/sigma/L defaults; same-candidate top-N task interface",
        "qwen3_bridge_scope": "T5 text backbone replaced by Qwen3 item representations and an orderable same-candidate BPR scorer",
        "users": user_count,
        "items": item_count,
        "train_edges": len(train_pairs),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "alpha": args.alpha,
        "sigma": args.sigma,
        "graph_layers": args.graph_layers,
        "embedding_size": args.embedding_size,
        "reg_weight": args.reg_weight,
        "device": str(device),
        "seed": args.seed,
        "trainable_params": trainable_params,
        "final_train_loss": logs[-1]["train_loss"] if logs else math.nan,
        "official_data_dir": str(handled_dir),
        "checkpoint_path": str(checkpoint_path),
        "qwen_item_embedding_source_path": str(item_embedding_source_path),
        **score_summary,
    }
    _write_csv([summary], adapter_dir / "elmrec_upstream_score_summary.csv", list(summary.keys()))
    _write_csv(logs, adapter_dir / "elmrec_upstream_epoch_log.csv", ["epoch", "train_loss"])
    (adapter_dir / "elmrec_upstream_score_metadata.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summary = train_and_score(parse_args())
    for key, value in summary.items():
        print(f"{key}={value}", flush=True)


if __name__ == "__main__":
    main()
