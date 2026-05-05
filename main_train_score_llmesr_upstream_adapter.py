from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an upstream LLM-ESR model class on a same-candidate adapter "
            "package and emit exact candidate scores. This is an LLM-ESR-style "
            "same-schema wrapper, not an official paper reproduction."
        )
    )
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--llmesr_repo_dir", required=True)
    parser.add_argument("--dataset_alias", default=None)
    parser.add_argument("--model_name", default="llmesr_sasrec", choices=["llmesr_sasrec"])
    parser.add_argument("--output_scores_path", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--trm_num", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--train_neg", type=int, default=1)
    parser.add_argument("--sim_user_num", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--user_sim_func", default="kd", choices=["kd", "cl"])
    parser.add_argument("--item_reg", action="store_true")
    parser.add_argument("--use_cross_att", action="store_true")
    parser.add_argument("--freeze", action="store_true")
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


def _as_int(value: Any) -> int | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _import_torch() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise RuntimeError("This LLM-ESR upstream wrapper requires PyTorch.") from exc
    return torch, DataLoader, Dataset, torch.optim.Adam


@contextmanager
def _pushd(path: Path) -> Iterator[None]:
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_arg: str, torch: Any) -> Any:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _load_inter_sequences(path: Path, *, user_count: int) -> dict[int, list[int]]:
    sequences = {idx: [] for idx in range(1, user_count + 1)}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            user_idx = _as_int(parts[0])
            item_idx = _as_int(parts[1])
            if user_idx is None or item_idx is None:
                continue
            sequences.setdefault(user_idx, []).append(item_idx)
    return sequences


def _copy_required_handled_files(adapter_dir: Path, llmesr_repo_dir: Path, dataset_alias: str) -> Path:
    source = adapter_dir / "llm_esr" / "handled"
    target = llmesr_repo_dir / "data" / dataset_alias / "handled"
    target.mkdir(parents=True, exist_ok=True)
    for name in ["inter.txt", "sim_user_100.pkl", "itm_emb_np.pkl", "pca64_itm_emb_np.pkl"]:
        source_path = source / name
        if not source_path.exists():
            raise FileNotFoundError(f"Required LLM-ESR handled file missing: {source_path}")
        shutil.copy2(source_path, target / name)
    return target


def _sequence_tensor(seq: list[int], *, max_len: int) -> tuple[list[int], list[int]]:
    trimmed = seq[-max_len:]
    pad_len = max_len - len(trimmed)
    values = [0] * pad_len + trimmed
    positions = [0] * pad_len + list(range(1, len(trimmed) + 1))
    return values, positions


def _seq2seq_arrays(
    seq: list[int],
    *,
    item_count: int,
    max_len: int,
    rng: random.Random,
) -> tuple[list[int], list[int], list[int], list[int]]:
    non_neg = set(seq)
    src = [0] * max_len
    pos = [0] * max_len
    neg = [0] * max_len
    nxt = seq[-1]
    idx = max_len - 1
    for item_idx in reversed(seq[:-1]):
        src[idx] = item_idx
        pos[idx] = nxt
        neg[idx] = _sample_negative(item_count, non_neg, rng)
        nxt = item_idx
        idx -= 1
        if idx < 0:
            break

    if len(seq) > max_len:
        positions = list(range(1, max_len + 1))
    else:
        active = max(0, len(seq) - 1)
        positions = [0] * (max_len - active) + list(range(1, active + 1))
    return src, pos, neg, positions[-max_len:]


def _sample_negative(item_count: int, non_neg: set[int], rng: random.Random) -> int:
    if item_count <= 0:
        return 0
    if len(non_neg) >= item_count:
        return rng.randint(1, item_count)
    for _ in range(100):
        candidate = rng.randint(1, item_count)
        if candidate not in non_neg:
            return candidate
    candidates = [idx for idx in range(1, item_count + 1) if idx not in non_neg]
    return candidates[0] if candidates else rng.randint(1, item_count)


class _LLMESRTrainDataset:
    def __init__(
        self,
        *,
        sequences: dict[int, list[int]],
        sim_users: list[list[int]],
        user_indices: list[int],
        item_count: int,
        max_len: int,
        sim_user_num: int,
        seed: int,
        torch: Any,
        Dataset: Any,
    ) -> None:
        class TorchDataset(Dataset):  # type: ignore[misc, valid-type]
            def __len__(inner_self: Any) -> int:
                return len(user_indices)

            def __getitem__(inner_self: Any, index: int) -> tuple[Any, ...]:
                rng = random.Random(seed + index)
                user_idx = user_indices[index]
                seq = sequences[user_idx]
                src, pos, neg, positions = _seq2seq_arrays(
                    seq,
                    item_count=item_count,
                    max_len=max_len,
                    rng=rng,
                )
                sim_seq_rows: list[list[int]] = []
                sim_pos_rows: list[list[int]] = []
                sim_dataset_indices = sim_users[user_idx - 1][:sim_user_num] if user_idx - 1 < len(sim_users) else []
                if not sim_dataset_indices:
                    sim_dataset_indices = [user_idx - 1]
                while len(sim_dataset_indices) < sim_user_num:
                    sim_dataset_indices.extend(sim_dataset_indices)
                for sim_dataset_idx in sim_dataset_indices[:sim_user_num]:
                    sim_user_idx = int(sim_dataset_idx) + 1
                    sim_values, sim_positions = _sequence_tensor(
                        sequences.get(sim_user_idx, []),
                        max_len=max_len,
                    )
                    sim_seq_rows.append(sim_values)
                    sim_pos_rows.append(sim_positions)
                return (
                    torch.tensor(src, dtype=torch.long),
                    torch.tensor(pos, dtype=torch.long),
                    torch.tensor(neg, dtype=torch.long),
                    torch.tensor(positions, dtype=torch.long),
                    torch.tensor(user_idx - 1, dtype=torch.long),
                    torch.tensor(sim_seq_rows, dtype=torch.long),
                    torch.tensor(sim_pos_rows, dtype=torch.long),
                )

        self.dataset = TorchDataset()


def _make_model_args(args: argparse.Namespace, dataset_alias: str) -> argparse.Namespace:
    if args.hidden_size != 64:
        raise ValueError("LLM-ESR pca64 item embeddings require --hidden_size 64 for the native SASRec wrapper.")
    return argparse.Namespace(
        dataset=dataset_alias,
        model_name=args.model_name,
        hidden_size=args.hidden_size,
        trm_num=args.trm_num,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        max_len=args.max_len,
        alpha=args.alpha,
        beta=args.beta,
        user_sim_func=args.user_sim_func,
        item_reg=args.item_reg,
        freeze=args.freeze,
        use_cross_att=args.use_cross_att,
    )


def _train_model(
    *,
    model: Any,
    loader: Any,
    optimizer: Any,
    epochs: int,
    device: Any,
    log_every: int,
) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            seq, pos, neg, positions, user_id, sim_seq, sim_positions = [tensor.to(device) for tensor in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = model(
                seq=seq,
                pos=pos,
                neg=neg,
                positions=positions,
                user_id=user_id,
                sim_seq=sim_seq,
                sim_positions=sim_positions,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            steps += 1
        mean_loss = total_loss / max(steps, 1)
        logs.append({"epoch": epoch, "train_loss": mean_loss})
        if log_every > 0 and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
            print(f"[llmesr] epoch={epoch} train_loss={mean_loss:.6f}", flush=True)
    return logs


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
                "llmesr_user_idx": user_idx,
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
                "llmesr_item_idx": item_idx,
            }
        )
    events = list(groups.values())
    for event in events:
        event["candidate_rows"].sort(key=lambda item: item["candidate_index"])
    events.sort(key=lambda item: (item["user_id"], item["source_event_id"]))
    return events


def _score_candidates(
    *,
    model: Any,
    sequences: dict[int, list[int]],
    candidate_events: list[dict[str, Any]],
    output_scores_path: Path,
    max_len: int,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    output_scores_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for event in candidate_events:
            user_idx = int(event["llmesr_user_idx"])
            seq_values, positions = _sequence_tensor(sequences.get(user_idx, []), max_len=max_len)
            seq_tensor = torch.tensor([seq_values], dtype=torch.long, device=device)
            position_tensor = torch.tensor([positions], dtype=torch.long, device=device)
            item_indices = [int(row["llmesr_item_idx"]) for row in event["candidate_rows"]]
            item_tensor = torch.tensor([item_indices], dtype=torch.long, device=device)
            logits = model.predict(seq=seq_tensor, item_indices=item_tensor, positions=position_tensor)
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"Non-finite score for source_event_id={event['source_event_id']}")
            scores = logits.squeeze(0).detach().cpu().tolist()
            for row, score in zip(event["candidate_rows"], scores):
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
    torch, DataLoader, Dataset, Adam = _import_torch()
    _set_seed(args.seed, torch)
    adapter_dir = Path(args.adapter_dir).expanduser()
    llmesr_repo_dir = Path(args.llmesr_repo_dir).expanduser()
    metadata_path = adapter_dir / "adapter_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"adapter_metadata.json not found: {metadata_path}")
    if not llmesr_repo_dir.exists():
        raise FileNotFoundError(f"LLM-ESR repo dir not found: {llmesr_repo_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    dataset_alias = args.dataset_alias or str(metadata.get("exp_name") or adapter_dir.name).replace("_adapter", "")
    _copy_required_handled_files(adapter_dir, llmesr_repo_dir, dataset_alias)
    sequences = _load_inter_sequences(
        adapter_dir / "llm_esr" / "handled" / "inter.txt",
        user_count=int(metadata.get("users", 0)),
    )
    train_user_indices = [user_idx for user_idx, seq in sorted(sequences.items()) if len(seq) >= 2]
    if not train_user_indices:
        raise ValueError("No trainable LLM-ESR sequences found; need at least one user with sequence length >= 2.")

    import pickle

    with (adapter_dir / "llm_esr" / "handled" / "sim_user_100.pkl").open("rb") as fh:
        sim_users = pickle.load(fh)
    if not isinstance(sim_users, list):
        raise ValueError("sim_user_100.pkl must be a list of dataset-index neighbor rows.")

    device = _resolve_device(args.device, torch)
    dataset = _LLMESRTrainDataset(
        sequences=sequences,
        sim_users=sim_users,
        user_indices=train_user_indices,
        item_count=int(metadata.get("items", 0)),
        max_len=args.max_len,
        sim_user_num=args.sim_user_num,
        seed=args.seed,
        torch=torch,
        Dataset=Dataset,
    ).dataset
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)

    sys.path.insert(0, str(llmesr_repo_dir))
    with _pushd(llmesr_repo_dir):
        from models.LLMESR import LLMESR_SASRec

        model_args = _make_model_args(args, dataset_alias)
        model = LLMESR_SASRec(
            int(metadata.get("users", 0)),
            int(metadata.get("items", 0)),
            device,
            model_args,
        ).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        logs = _train_model(
            model=model,
            loader=loader,
            optimizer=optimizer,
            epochs=args.epochs,
            device=device,
            log_every=args.log_every,
        )
        candidate_events = _candidate_groups(adapter_dir / "candidate_items_mapped.csv")
        output_scores_path = (
            Path(args.output_scores_path).expanduser()
            if args.output_scores_path
            else adapter_dir / "llmesr_upstream_same_candidate_scores.csv"
        )
        score_summary = _score_candidates(
            model=model,
            sequences=sequences,
            candidate_events=candidate_events,
            output_scores_path=output_scores_path,
            max_len=args.max_len,
            torch=torch,
            device=device,
        )
        checkpoint_path = (
            Path(args.checkpoint_path).expanduser()
            if args.checkpoint_path
            else adapter_dir / "llmesr_upstream_model.pt"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "dataset_alias": dataset_alias,
                "summary": score_summary,
            },
            checkpoint_path,
        )

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    summary = {
        "adapter_dir": str(adapter_dir),
        "baseline_name": "llmesr_style_qwen3_sasrec",
        "artifact_class": "upstream_llmesr_style_same_candidate_score",
        "paper_result_ready": True,
        "upstream_repo": "https://github.com/liuqidong07/LLM-ESR",
        "dataset_alias": dataset_alias,
        "model_name": args.model_name,
        "users": int(metadata.get("users", 0)),
        "items": int(metadata.get("items", 0)),
        "train_users": len(train_user_indices),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "l2": args.l2,
        "max_len": args.max_len,
        "hidden_size": args.hidden_size,
        "sim_user_num": args.sim_user_num,
        "alpha": args.alpha,
        "user_sim_func": args.user_sim_func,
        "item_reg": args.item_reg,
        "use_cross_att": args.use_cross_att,
        "device": str(device),
        "seed": args.seed,
        "trainable_params": trainable_params,
        "final_train_loss": logs[-1]["train_loss"] if logs else math.nan,
        "checkpoint_path": str(checkpoint_path),
        **score_summary,
        "note": (
            "Scores are exact same-candidate logits from the upstream LLM-ESR "
            "LLMESR_SASRec model class trained on this adapter's Qwen/item "
            "embeddings. Label as LLM-ESR-style unless the full official "
            "LLM-ESR preprocessing and experiment pipeline is reproduced."
        ),
    }
    _write_csv([summary], adapter_dir / "llmesr_upstream_score_summary.csv", list(summary.keys()))
    _write_csv(logs, adapter_dir / "llmesr_upstream_epoch_log.csv", ["epoch", "train_loss"])
    (adapter_dir / "llmesr_upstream_score_metadata.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summary = train_and_score(parse_args())
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
