from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train pinned upstream RLMRec SimGCL_plus on a same-candidate "
            "adapter package and emit exact candidate scores."
        )
    )
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--rlmrec_repo_dir", required=True)
    parser.add_argument("--output_scores_path", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--model_name", default="simgcl_plus")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument("--layer_num", type=int, default=3)
    parser.add_argument("--reg_weight", type=float, default=1.0e-5)
    parser.add_argument("--cl_weight", type=float, default=1.0e-1)
    parser.add_argument("--cl_temperature", type=float, default=0.2)
    parser.add_argument("--kd_weight", type=float, default=1.0e-2)
    parser.add_argument("--kd_temperature", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=0.9)
    parser.add_argument("--keep_rate", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--log_every", type=int, default=10)
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


def _import_runtime() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        import numpy as np
        import scipy.sparse as sp
        import torch
        from scipy.sparse import coo_matrix
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError("The RLMRec upstream wrapper requires numpy, scipy, and PyTorch.") from exc
    return np, sp, torch, coo_matrix, DataLoader, TensorDataset, torch.optim.Adam


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
                "rlmrec_user_idx": user_idx - 1,
                "candidate_rows": [],
            }
        try:
            candidate_index = int(float(row.get("candidate_index", 0)))
        except Exception:
            candidate_index = 0
        groups[key]["candidate_rows"].append(
            {"candidate_index": candidate_index, "item_id": item_id, "rlmrec_item_idx": item_idx - 1}
        )
    events = list(groups.values())
    for event in events:
        event["candidate_rows"].sort(key=lambda item: item["candidate_index"])
    events.sort(key=lambda item: (item["user_id"], item["source_event_id"]))
    return events


def _load_item_embeddings(adapter_dir: Path, np: Any) -> tuple[Any, Path]:
    path = adapter_dir / "llm_esr" / "handled" / "itm_emb_np.pkl"
    if not path.exists():
        raise FileNotFoundError(f"RLMRec requires Qwen item embeddings at {path}")
    with path.open("rb") as fh:
        matrix = pickle.load(fh)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D item embedding matrix, got shape={matrix.shape}")
    return matrix, path


def _build_user_embeddings(*, sequences: dict[int, list[int]], item_embeddings: Any, user_count: int, np: Any) -> tuple[Any, dict[str, Any]]:
    fallback = item_embeddings.mean(axis=0) if item_embeddings.shape[0] else np.zeros((item_embeddings.shape[1],), dtype=np.float32)
    user_embeddings = np.zeros((user_count, item_embeddings.shape[1]), dtype=np.float32)
    missing = 0
    for user_idx in range(user_count):
        item_indices = [idx for idx in sequences.get(user_idx, []) if 0 <= idx < item_embeddings.shape[0]]
        if item_indices:
            user_embeddings[user_idx] = item_embeddings[item_indices].mean(axis=0)
        else:
            user_embeddings[user_idx] = fallback
            missing += 1
    return user_embeddings, {
        "user_embedding_source": "train_history_mean_of_qwen_item_embeddings",
        "missing_history_users": missing,
        "user_embedding_dim": int(user_embeddings.shape[1]),
    }


def _make_sparse_mats(
    *,
    sequences: dict[int, list[int]],
    user_count: int,
    item_count: int,
    coo_matrix: Any,
) -> tuple[Any, Any, Any, list[tuple[int, int]]]:
    rows: list[int] = []
    cols: list[int] = []
    pairs: list[tuple[int, int]] = []
    for user_idx, item_indices in sorted(sequences.items()):
        for item_idx in item_indices:
            if 0 <= user_idx < user_count and 0 <= item_idx < item_count:
                rows.append(user_idx)
                cols.append(item_idx)
                pairs.append((user_idx, item_idx))
    if not pairs:
        raise ValueError("No trainable RLMRec user-item edges found.")
    data = [1.0] * len(rows)
    trn_mat = coo_matrix((data, (rows, cols)), shape=(user_count, item_count), dtype="float32")
    empty = coo_matrix((user_count, item_count), dtype="float32")
    return trn_mat, empty, empty, sorted(set(pairs))


def _write_official_data_artifacts(
    adapter_dir: Path,
    *,
    trn_mat: Any,
    val_mat: Any,
    tst_mat: Any,
    user_emb: Any,
    item_emb: Any,
    item_embedding_source_path: Path,
) -> Path:
    out_dir = adapter_dir / "rlmrec" / "handled"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in {
        "trn_mat.pkl": trn_mat,
        "val_mat.pkl": val_mat,
        "tst_mat.pkl": tst_mat,
    }.items():
        with (out_dir / name).open("wb") as fh:
            pickle.dump(payload, fh)
    manifest = {
        "storage_policy": "semantic_embeddings_loaded_via_config_not_duplicated",
        "reason": "large-domain Qwen item embeddings can be several GB; duplicating them can exhaust server storage",
        "user_embedding_source": "train_history_mean_of_qwen_item_embeddings",
        "user_embedding_shape": list(user_emb.shape),
        "item_embedding_source_path": str(item_embedding_source_path),
        "item_embedding_shape": list(item_emb.shape),
    }
    (out_dir / "semantic_embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_dir


def _normalize_adj(mat: Any, sp: Any, np: Any) -> Any:
    degree = np.array(mat.sum(axis=-1))
    d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
    return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()


def _make_torch_adj(trn_mat: Any, *, user_count: int, item_count: int, sp: Any, np: Any, torch: Any, device: Any) -> Any:
    a = sp.csr_matrix((user_count, user_count))
    b = sp.csr_matrix((item_count, item_count))
    mat = sp.vstack([sp.hstack([a, trn_mat]), sp.hstack([trn_mat.transpose(), b])])
    mat = (mat != 0) * 1.0
    mat = _normalize_adj(mat, sp, np)
    idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
    vals = torch.from_numpy(mat.data.astype(np.float32))
    shape = torch.Size(mat.shape)
    return torch.sparse_coo_tensor(idxs, vals, shape).coalesce().to(device)


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


def _official_configs(args: argparse.Namespace, *, user_count: int, item_count: int, user_emb: Any, item_emb: Any, device: Any) -> dict[str, Any]:
    hyper = {
        "layer_num": args.layer_num,
        "reg_weight": args.reg_weight,
        "cl_weight": args.cl_weight,
        "cl_temperature": args.cl_temperature,
        "kd_weight": args.kd_weight,
        "kd_temperature": args.kd_temperature,
        "eps": args.eps,
    }
    return {
        "device": str(device),
        "usrprf_embeds": user_emb,
        "itmprf_embeds": item_emb,
        "optimizer": {"name": "adam", "lr": args.lr, "weight_decay": args.weight_decay},
        "train": {
            "epoch": args.epochs,
            "batch_size": args.batch_size,
            "save_model": True,
            "loss": "pairwise",
            "test_step": 3,
            "reproducible": True,
            "seed": args.seed,
            "patience": 5,
            "log_loss": False,
        },
        "test": {"metrics": ["recall", "ndcg"], "k": [5, 10, 20], "batch_size": 1024},
        "data": {"type": "general_cf", "name": "pony", "user_num": user_count, "item_num": item_count},
        "model": {
            "name": args.model_name,
            "keep_rate": args.keep_rate,
            "embedding_size": args.embedding_size,
            "pony": hyper,
            **hyper,
        },
        "tune": {"enable": False},
    }


def _import_official_model(repo_dir: Path, configs: dict[str, Any]) -> Any:
    encoder_dir = repo_dir / "encoder"
    if not encoder_dir.exists():
        raise FileNotFoundError(f"RLMRec encoder directory not found: {encoder_dir}")
    config_pkg = types.ModuleType("config")
    config_pkg.__path__ = [str(encoder_dir / "config")]
    config_mod = types.ModuleType("config.configurator")
    config_mod.configs = configs
    sys.modules["config"] = config_pkg
    sys.modules["config.configurator"] = config_mod
    sys.path.insert(0, str(encoder_dir))
    module_name = "models.general_cf.simgcl_plus"
    module = __import__(module_name, fromlist=["SimGCL_plus"])
    return getattr(module, "SimGCL_plus")


class _DataHandler:
    def __init__(self, torch_adj: Any) -> None:
        self.torch_adj = torch_adj


def _train_official_model(
    *,
    model: Any,
    train_pairs: list[tuple[int, int]],
    positives: dict[int, set[int]],
    item_count: int,
    args: argparse.Namespace,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    Adam: Any,
    device: Any,
) -> list[dict[str, Any]]:
    user_tensor = torch.tensor([u for u, _ in train_pairs], dtype=torch.long)
    item_tensor = torch.tensor([i for _, i in train_pairs], dtype=torch.long)
    dataset = TensorDataset(user_tensor, item_tensor)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    logs: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
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
            loss, _losses = model.cal_loss([users, pos_items, neg_items])
            loss.backward()
            optimizer.step()
            if hasattr(model, "final_embeds"):
                model.final_embeds = None
            total_loss += float(loss.detach().cpu())
            steps += 1
        row = {"epoch": epoch, "train_loss": total_loss / max(steps, 1)}
        logs.append(row)
        if args.log_every > 0 and (epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0):
            print(f"[rlmrec-official] epoch={epoch} train_loss={row['train_loss']:.6f}", flush=True)
    return logs


def _score_candidates(
    *,
    model: Any,
    candidate_events: list[dict[str, Any]],
    output_scores_path: Path,
    torch: Any,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    model.eval()
    if hasattr(model, "is_training"):
        model.is_training = False
    with torch.no_grad():
        user_embeds, item_embeds = model.forward(model.adj, False)
        for event in candidate_events:
            user_idx = int(event["rlmrec_user_idx"])
            candidate_indices = [int(row["rlmrec_item_idx"]) for row in event["candidate_rows"]]
            item_tensor = torch.tensor(candidate_indices, dtype=torch.long, device=item_embeds.device)
            scores = item_embeds.index_select(0, item_tensor).matmul(user_embeds[user_idx])
            if not torch.isfinite(scores).all():
                raise RuntimeError(f"Non-finite RLMRec score for source_event_id={event['source_event_id']}")
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
    np, sp, torch, coo_matrix, DataLoader, TensorDataset, Adam = _import_runtime()
    _set_seed(args.seed, torch, np)
    adapter_dir = Path(args.adapter_dir).expanduser().resolve()
    repo_dir = Path(args.rlmrec_repo_dir).expanduser().resolve()
    metadata = json.loads((adapter_dir / "adapter_metadata.json").read_text(encoding="utf-8"))
    user_count = int(metadata.get("users", 0))
    item_count = int(metadata.get("items", 0))
    output_scores_path = Path(args.output_scores_path).expanduser().resolve() if args.output_scores_path else adapter_dir / "rlmrec_official_scores.csv"
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve() if args.checkpoint_path else adapter_dir / "rlmrec_official_model.pt"

    sequences = _load_interactions(adapter_dir / "llm_esr" / "handled" / "inter.txt")
    item_emb, item_embedding_source_path = _load_item_embeddings(adapter_dir, np)
    if item_emb.shape[0] != item_count:
        raise ValueError(f"item embeddings rows={item_emb.shape[0]} but adapter metadata items={item_count}")
    user_emb, user_embedding_summary = _build_user_embeddings(sequences=sequences, item_embeddings=item_emb, user_count=user_count, np=np)
    trn_mat, val_mat, tst_mat, train_pairs = _make_sparse_mats(
        sequences=sequences,
        user_count=user_count,
        item_count=item_count,
        coo_matrix=coo_matrix,
    )
    positives: dict[int, set[int]] = {}
    for user_idx, item_idx in train_pairs:
        positives.setdefault(user_idx, set()).add(item_idx)
    handled_dir = _write_official_data_artifacts(
        adapter_dir,
        trn_mat=trn_mat,
        val_mat=val_mat,
        tst_mat=tst_mat,
        user_emb=user_emb,
        item_emb=item_emb,
        item_embedding_source_path=item_embedding_source_path,
    )
    device = _resolve_device(args.device, torch)
    if device.type != "cuda":
        raise RuntimeError("Pinned RLMRec plus models call .cuda(); run this official adapter on a CUDA device.")
    torch_adj = _make_torch_adj(trn_mat, user_count=user_count, item_count=item_count, sp=sp, np=np, torch=torch, device=device)

    configs = _official_configs(args, user_count=user_count, item_count=item_count, user_emb=user_emb, item_emb=item_emb, device=device)
    SimGCLPlus = _import_official_model(repo_dir, configs)
    with _pushd(repo_dir):
        model = SimGCLPlus(_DataHandler(torch_adj)).to(device)
        logs = _train_official_model(
            model=model,
            train_pairs=train_pairs,
            positives=positives,
            item_count=item_count,
            args=args,
            torch=torch,
            DataLoader=DataLoader,
            TensorDataset=TensorDataset,
            Adam=Adam,
            device=device,
        )
        score_summary = _score_candidates(
            model=model,
            candidate_events=_candidate_groups(adapter_dir / "candidate_items_mapped.csv"),
            output_scores_path=output_scores_path,
            torch=torch,
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "args": vars(args),
                "configs": configs,
                "summary": score_summary,
            },
            checkpoint_path,
        )

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    summary = {
        "adapter_dir": str(adapter_dir),
        "baseline_name": "rlmrec_official_qwen3base_graphcl",
        "artifact_class": "official_rlmrec_same_candidate_score",
        "official_result_gate": "provenance_coverage_and_import_required",
        "upstream_repo": "https://github.com/HKUDS/RLMRec",
        "official_model_class": "encoder.models.general_cf.simgcl_plus.SimGCL_plus",
        "users": user_count,
        "items": item_count,
        "train_edges": len(train_pairs),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "embedding_size": args.embedding_size,
        "layer_num": args.layer_num,
        "cl_weight": args.cl_weight,
        "kd_weight": args.kd_weight,
        "device": str(device),
        "seed": args.seed,
        "trainable_params": trainable_params,
        "final_train_loss": logs[-1]["train_loss"] if logs else math.nan,
        "official_data_dir": str(handled_dir),
        "checkpoint_path": str(checkpoint_path),
        **user_embedding_summary,
        **score_summary,
        "note": (
            "Scores use pinned RLMRec SimGCL_plus with official BPR, graph contrastive, "
            "and semantic alignment losses. Local code supplies same-candidate graph data, "
            "Qwen3 item embeddings, train-history-derived user semantic embeddings, and "
            "exact candidate-score export."
        ),
    }
    _write_csv([summary], adapter_dir / "rlmrec_upstream_score_summary.csv", list(summary.keys()))
    _write_csv(logs, adapter_dir / "rlmrec_upstream_epoch_log.csv", ["epoch", "train_loss"])
    (adapter_dir / "rlmrec_upstream_score_metadata.json").write_text(
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
