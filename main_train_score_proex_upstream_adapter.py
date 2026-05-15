from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import re
import shutil
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train pinned upstream ProRec LightGCN_proex on a same-candidate "
            "adapter package and emit exact candidate scores."
        )
    )
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--prorec_repo_dir", required=True)
    parser.add_argument("--output_scores_path", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument("--layer_num", type=int, default=3)
    parser.add_argument("--reg_weight", type=float, default=1.0e-6)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--inv_weight", type=float, default=10.0)
    parser.add_argument("--ex_weight", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--keep_rate", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2025)
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
        raise RuntimeError("The ProEx upstream wrapper requires numpy, scipy, and PyTorch.") from exc
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


def _sanitize_dataset_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    return cleaned[:80] or "pony_proex"


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
                "proex_user_idx": user_idx - 1,
                "candidate_rows": [],
            }
        try:
            candidate_index = int(float(row.get("candidate_index", 0)))
        except Exception:
            candidate_index = 0
        groups[key]["candidate_rows"].append(
            {"candidate_index": candidate_index, "item_id": item_id, "proex_item_idx": item_idx - 1}
        )
    events = list(groups.values())
    for event in events:
        event["candidate_rows"].sort(key=lambda item: item["candidate_index"])
    events.sort(key=lambda item: (item["user_id"], item["source_event_id"]))
    return events


def _load_item_embeddings(adapter_dir: Path, np: Any) -> tuple[Any, Path]:
    path = adapter_dir / "llm_esr" / "handled" / "itm_emb_np.pkl"
    if not path.exists():
        raise FileNotFoundError(f"ProEx requires Qwen item embeddings at {path}")
    with path.open("rb") as fh:
        matrix = pickle.load(fh)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D item embedding matrix, got shape={matrix.shape}")
    return matrix, path


def _l2_normalize(matrix: Any, np: Any) -> Any:
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(denom, 1.0e-8)).astype(np.float32)


def _build_user_profile(*, sequences: dict[int, list[int]], item_profile: Any, user_count: int, np: Any) -> tuple[Any, int]:
    fallback = item_profile.mean(axis=0) if item_profile.shape[0] else np.zeros((item_profile.shape[1],), dtype=np.float32)
    user_profile = np.zeros((user_count, item_profile.shape[1]), dtype=np.float32)
    missing = 0
    for user_idx in range(user_count):
        item_indices = [idx for idx in sequences.get(user_idx, []) if 0 <= idx < item_profile.shape[0]]
        if item_indices:
            user_profile[user_idx] = item_profile[item_indices].mean(axis=0)
        else:
            user_profile[user_idx] = fallback
            missing += 1
    return user_profile.astype(np.float32), missing


def _profile_views(
    *,
    sequences: dict[int, list[int]],
    item_embeddings: Any,
    user_count: int,
    np: Any,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    item_count = int(item_embeddings.shape[0])
    base_items = np.asarray(item_embeddings, dtype=np.float32)
    base_users, missing_base = _build_user_profile(sequences=sequences, item_profile=base_items, user_count=user_count, np=np)

    item_pop = np.zeros((item_count,), dtype=np.float32)
    user_base_for_smoothing = base_users
    item_neighbor_sum = np.zeros_like(base_items, dtype=np.float32)
    item_neighbor_count = np.zeros((item_count, 1), dtype=np.float32)
    for user_idx, item_indices in sequences.items():
        if not (0 <= user_idx < user_count):
            continue
        for item_idx in item_indices:
            if 0 <= item_idx < item_count:
                item_pop[item_idx] += 1.0
                item_neighbor_sum[item_idx] += user_base_for_smoothing[user_idx]
                item_neighbor_count[item_idx, 0] += 1.0

    global_item = base_items.mean(axis=0, keepdims=True) if item_count else np.zeros((1, base_items.shape[1]), dtype=np.float32)
    neighbor_items = np.where(item_neighbor_count > 0, item_neighbor_sum / np.maximum(item_neighbor_count, 1.0), global_item)
    graph_context_items = (0.5 * base_items + 0.5 * neighbor_items).astype(np.float32)

    centered_items = (base_items - global_item).astype(np.float32)
    log_pop = np.log1p(item_pop)
    pop_scale = (log_pop - log_pop.mean()) / (log_pop.std() + 1.0e-6)
    popularity_residual_items = (centered_items * (1.0 + 0.10 * pop_scale[:, None])).astype(np.float32)

    inv_pop = 1.0 / np.sqrt(1.0 + item_pop)
    inv_scale = inv_pop / (inv_pop.mean() + 1.0e-6)
    tail_view_items = (_l2_normalize(base_items, np) * inv_scale[:, None]).astype(np.float32)

    item_profiles = [base_items, graph_context_items, popularity_residual_items, tail_view_items]
    user_profiles: list[Any] = []
    missing_users: list[int] = []
    for item_profile in item_profiles:
        user_profile, missing = _build_user_profile(sequences=sequences, item_profile=item_profile, user_count=user_count, np=np)
        user_profiles.append(user_profile)
        missing_users.append(missing)

    manifest = {
        "profile_bridge": "qwen3_item_representation_plus_train_graph_profile_views",
        "profile_count": 4,
        "profile_names": [
            "base_qwen3_item_profile",
            "graph_context_smoothed_profile",
            "popularity_residual_profile",
            "tail_emphasis_normalized_profile",
        ],
        "user_profile_source": "train_history_mean_of_corresponding_item_profile",
        "missing_history_users_per_profile": missing_users,
        "missing_base_history_users": missing_base,
        "item_popularity_nonzero": int((item_pop > 0).sum()),
        "item_embedding_dim": int(base_items.shape[1]),
        "note": (
            "ProRec's official LightGCN_proex model and loss are preserved. "
            "The original public release expects downloaded LLM profile arrays; "
            "this bridge replaces that profile source with deterministic Qwen3-8B "
            "item-representation and train-graph profile views under the unified "
            "backbone policy."
        ),
    }
    return user_profiles, item_profiles, manifest


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
        raise ValueError("No trainable ProEx user-item edges found.")
    data = [1.0] * len(rows)
    trn_mat = coo_matrix((data, (rows, cols)), shape=(user_count, item_count), dtype="float32")
    empty = coo_matrix((user_count, item_count), dtype="float32")
    return trn_mat, empty, empty, sorted(set(pairs))


def _write_profile_data(
    *,
    adapter_dir: Path,
    repo_dir: Path,
    dataset_name: str,
    trn_mat: Any,
    val_mat: Any,
    tst_mat: Any,
    user_profiles: list[Any],
    item_profiles: list[Any],
    item_embedding_source_path: Path,
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    handled_dir = adapter_dir / "proex" / "handled"
    handled_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in {
        "trn_mat.pkl": trn_mat,
        "val_mat.pkl": val_mat,
        "tst_mat.pkl": tst_mat,
        "usr_emb_np.pkl": user_profiles[0],
        "itm_emb_np.pkl": item_profiles[0],
        "usr_emb_np_1.pkl": user_profiles[1],
        "itm_emb_np_1.pkl": item_profiles[1],
        "usr_emb_np_2.pkl": user_profiles[2],
        "itm_emb_np_2.pkl": item_profiles[2],
        "usr_emb_np_3.pkl": user_profiles[3],
        "itm_emb_np_3.pkl": item_profiles[3],
    }.items():
        with (handled_dir / name).open("wb") as fh:
            pickle.dump(payload, fh)

    manifest = {
        **manifest,
        "adapter_profile_dir": str(handled_dir),
        "item_embedding_source_path": str(item_embedding_source_path),
        "user_profile_shapes": [list(value.shape) for value in user_profiles],
        "item_profile_shapes": [list(value.shape) for value in item_profiles],
    }
    (handled_dir / "proex_profile_bridge_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    official_data_dir = repo_dir / "data" / dataset_name
    if official_data_dir.exists() or official_data_dir.is_symlink():
        resolved = official_data_dir.resolve() if official_data_dir.exists() else official_data_dir
        if repo_dir.resolve() not in resolved.parents and resolved != handled_dir.resolve():
            raise RuntimeError(f"Refusing to replace unexpected ProRec data directory: {official_data_dir}")
        if official_data_dir.is_symlink() or official_data_dir.is_file():
            official_data_dir.unlink()
        else:
            shutil.rmtree(official_data_dir)
    official_data_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        official_data_dir.symlink_to(handled_dir.resolve(), target_is_directory=True)
    except Exception:
        shutil.copytree(handled_dir, official_data_dir)
    return handled_dir, official_data_dir


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


def _official_configs(
    args: argparse.Namespace,
    *,
    dataset_name: str,
    user_count: int,
    item_count: int,
    user_profiles: list[Any],
    item_profiles: list[Any],
    device: Any,
) -> dict[str, Any]:
    hyper = {
        "layer_num": args.layer_num,
        "reg_weight": args.reg_weight,
        "num_envs": args.num_envs,
        "inv_weight": args.inv_weight,
        "ex_weight": args.ex_weight,
        "alpha": args.alpha,
    }
    return {
        "device": str(device),
        "usrprf_embeds": user_profiles[0],
        "itmprf_embeds": item_profiles[0],
        "optimizer": {"name": "adam", "lr": args.lr, "weight_decay": args.weight_decay},
        "train": {
            "epoch": args.epochs,
            "batch_size": args.batch_size,
            "save_model": True,
            "loss": "pairwise",
            "test_step": 3,
            "reproducible": True,
            "seed": args.seed,
            "patience": 20,
            "log_loss": False,
        },
        "test": {"metrics": ["recall", "ndcg"], "k": [10, 20], "batch_size": 1024},
        "data": {"type": "general_cf", "name": dataset_name, "user_num": user_count, "item_num": item_count},
        "model": {
            "name": "lightgcn_proex",
            "type": "discriminative",
            "keep_rate": args.keep_rate,
            "profiles": 4,
            "embedding_size": args.embedding_size,
            dataset_name: hyper,
            **hyper,
        },
        "tune": {"enable": False},
    }


def _checkpoint_config_summary(configs: dict[str, Any], *, profile_manifest_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in configs.items():
        if key in {"usrprf_embeds", "itmprf_embeds"}:
            summary[key] = {
                "externalized": True,
                "shape": list(getattr(value, "shape", [])),
                "dtype": str(getattr(value, "dtype", "")),
            }
        else:
            summary[key] = value
    summary["profile_bridge_manifest_path"] = str(profile_manifest_path)
    return summary


def _import_official_model(repo_dir: Path, configs: dict[str, Any]) -> Any:
    encoder_dir = repo_dir / "encoder"
    if not encoder_dir.exists():
        raise FileNotFoundError(f"ProRec encoder directory not found: {encoder_dir}")
    for name in list(sys.modules):
        if name == "config" or name.startswith("config.") or name.startswith("models."):
            del sys.modules[name]
    config_pkg = types.ModuleType("config")
    config_pkg.__path__ = [str(encoder_dir / "config")]
    config_mod = types.ModuleType("config.configurator")
    config_mod.configs = configs
    sys.modules["config"] = config_pkg
    sys.modules["config.configurator"] = config_mod
    sys.path.insert(0, str(encoder_dir))
    module = __import__("models.general_cf.lightgcn_proex", fromlist=["LightGCN_proex"])
    return getattr(module, "LightGCN_proex")


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
            print(f"[proex-official] epoch={epoch} train_loss={row['train_loss']:.6f}", flush=True)
    return logs


def _profile_adjusted_embeddings(model: Any, torch: Any) -> tuple[Any, Any]:
    if hasattr(model, "is_training"):
        model.is_training = False
    user_embeds, item_embeds = model.forward(model.adj, 1.0)
    usr_profile_embeds = []
    itm_profile_embeds = []
    for env in range(model.num_profiles):
        usr_profile_embeds.append(model.mlp(model.usrprf_embeds[env]))
        itm_profile_embeds.append(model.mlp(model.itmprf_embeds[env]))
    user_profile_mean = torch.stack(usr_profile_embeds, dim=1).mean(dim=1)
    item_profile_mean = torch.stack(itm_profile_embeds, dim=1).mean(dim=1)
    return user_embeds + user_profile_mean, item_embeds + item_profile_mean


def _score_candidates(
    *,
    model: Any,
    candidate_events: list[dict[str, Any]],
    output_scores_path: Path,
    torch: Any,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        user_embeds, item_embeds = _profile_adjusted_embeddings(model, torch)
        for event in candidate_events:
            user_idx = int(event["proex_user_idx"])
            candidate_indices = [int(row["proex_item_idx"]) for row in event["candidate_rows"]]
            item_tensor = torch.tensor(candidate_indices, dtype=torch.long, device=item_embeds.device)
            scores = item_embeds.index_select(0, item_tensor).matmul(user_embeds[user_idx])
            if not torch.isfinite(scores).all():
                raise RuntimeError(f"Non-finite ProEx score for source_event_id={event['source_event_id']}")
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
    repo_dir = Path(args.prorec_repo_dir).expanduser().resolve()
    metadata = json.loads((adapter_dir / "adapter_metadata.json").read_text(encoding="utf-8"))
    user_count = int(metadata.get("users", 0))
    item_count = int(metadata.get("items", 0))
    dataset_name = _sanitize_dataset_name(args.dataset_name or f"pony_proex_{adapter_dir.name}")
    output_scores_path = Path(args.output_scores_path).expanduser().resolve() if args.output_scores_path else adapter_dir / "proex_official_scores.csv"
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve() if args.checkpoint_path else adapter_dir / "proex_official_model.pt"

    sequences = _load_interactions(adapter_dir / "llm_esr" / "handled" / "inter.txt")
    item_emb, item_embedding_source_path = _load_item_embeddings(adapter_dir, np)
    if item_emb.shape[0] != item_count:
        raise ValueError(f"item embeddings rows={item_emb.shape[0]} but adapter metadata items={item_count}")
    user_profiles, item_profiles, profile_manifest = _profile_views(
        sequences=sequences,
        item_embeddings=item_emb,
        user_count=user_count,
        np=np,
    )
    trn_mat, val_mat, tst_mat, train_pairs = _make_sparse_mats(
        sequences=sequences,
        user_count=user_count,
        item_count=item_count,
        coo_matrix=coo_matrix,
    )
    positives: dict[int, set[int]] = {}
    for user_idx, item_idx in train_pairs:
        positives.setdefault(user_idx, set()).add(item_idx)
    handled_dir, official_data_dir = _write_profile_data(
        adapter_dir=adapter_dir,
        repo_dir=repo_dir,
        dataset_name=dataset_name,
        trn_mat=trn_mat,
        val_mat=val_mat,
        tst_mat=tst_mat,
        user_profiles=user_profiles,
        item_profiles=item_profiles,
        item_embedding_source_path=item_embedding_source_path,
        manifest=profile_manifest,
    )
    profile_manifest_path = handled_dir / "proex_profile_bridge_manifest.json"
    device = _resolve_device(args.device, torch)
    if device.type != "cuda":
        raise RuntimeError("Pinned ProEx LightGCN_proex calls .cuda(); run this official adapter on a CUDA device.")
    torch_adj = _make_torch_adj(trn_mat, user_count=user_count, item_count=item_count, sp=sp, np=np, torch=torch, device=device)

    configs = _official_configs(
        args,
        dataset_name=dataset_name,
        user_count=user_count,
        item_count=item_count,
        user_profiles=user_profiles,
        item_profiles=item_profiles,
        device=device,
    )
    LightGCNProEx = _import_official_model(repo_dir, configs)
    with _pushd(repo_dir / "encoder"):
        model = LightGCNProEx(_DataHandler(torch_adj)).to(device)
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
                "configs": _checkpoint_config_summary(configs, profile_manifest_path=profile_manifest_path),
                "summary": score_summary,
            },
            checkpoint_path,
            pickle_protocol=4,
        )

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    summary = {
        "adapter_dir": str(adapter_dir),
        "baseline_name": "proex_official_qwen3base_profile",
        "artifact_class": "official_proex_same_candidate_score",
        "official_result_gate": "provenance_coverage_and_import_required",
        "upstream_repo": "https://github.com/BlueGhostYi/ProRec",
        "official_model_class": "encoder.models.general_cf.lightgcn_proex.LightGCN_proex",
        "users": user_count,
        "items": item_count,
        "train_edges": len(train_pairs),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "embedding_size": args.embedding_size,
        "layer_num": args.layer_num,
        "num_envs": args.num_envs,
        "inv_weight": args.inv_weight,
        "ex_weight": args.ex_weight,
        "alpha": args.alpha,
        "device": str(device),
        "seed": args.seed,
        "trainable_params": trainable_params,
        "final_train_loss": logs[-1]["train_loss"] if logs else math.nan,
        "official_data_dir": str(official_data_dir),
        "adapter_profile_dir": str(handled_dir),
        "profile_bridge_manifest_path": str(profile_manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_storage_decision": "profile_arrays_externalized_and_manifest_recorded",
        **profile_manifest,
        **score_summary,
        "note": (
            "Scores use pinned ProRec LightGCN_proex with official BPR, invariant, "
            "and profile extrapolation losses. Local code supplies same-candidate "
            "graph data, Qwen3-derived profile arrays, and exact candidate-score export."
        ),
    }
    _write_csv([summary], adapter_dir / "proex_upstream_score_summary.csv", list(summary.keys()))
    _write_csv(logs, adapter_dir / "proex_upstream_epoch_log.csv", ["epoch", "train_loss"])
    (adapter_dir / "proex_upstream_score_metadata.json").write_text(
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
