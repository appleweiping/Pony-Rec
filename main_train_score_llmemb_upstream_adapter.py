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
            "Train the pinned upstream LLMEmb SASRec path on a same-candidate "
            "adapter package and emit exact candidate scores."
        )
    )
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--llmemb_repo_dir", required=True)
    parser.add_argument("--dataset_alias", default=None)
    parser.add_argument("--output_scores_path", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--sasrec_embedding_path", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--sasrec_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--trm_num", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--freeze_emb", action="store_true")
    parser.add_argument("--no_freeze_emb", dest="freeze_emb", action="store_false")
    parser.set_defaults(freeze_emb=True)
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


def _import_torch() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise RuntimeError("This LLMEmb upstream wrapper requires PyTorch.") from exc
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


def _copy_handled_files(adapter_dir: Path, repo_dir: Path, dataset_alias: str) -> Path:
    source = adapter_dir / "llm_esr" / "handled"
    target = repo_dir / "data" / dataset_alias / "handled"
    target.mkdir(parents=True, exist_ok=True)
    for name in ["inter.txt", "itm_emb_np.pkl"]:
        source_path = source / name
        if not source_path.exists():
            raise FileNotFoundError(f"Required handled file missing: {source_path}")
        shutil.copy2(source_path, target / name)
    return target


def _load_sequences(path: Path, *, user_count: int) -> dict[int, list[int]]:
    sequences = {idx: [] for idx in range(1, user_count + 1)}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            user_idx = _as_int(parts[0])
            item_idx = _as_int(parts[1])
            if user_idx is not None and item_idx is not None:
                sequences.setdefault(user_idx, []).append(item_idx)
    return sequences


def _sample_negative(item_count: int, positives: set[int], rng: random.Random) -> int:
    if item_count <= 0:
        return 0
    if len(positives) >= item_count:
        return rng.randint(1, item_count)
    for _ in range(100):
        candidate = rng.randint(1, item_count)
        if candidate not in positives:
            return candidate
    for candidate in range(1, item_count + 1):
        if candidate not in positives:
            return candidate
    return rng.randint(1, item_count)


def _seq2seq_arrays(
    seq: list[int],
    *,
    item_count: int,
    max_len: int,
    rng: random.Random,
) -> tuple[list[int], list[int], list[int], list[int]]:
    positives = set(seq)
    src = [0] * max_len
    pos = [0] * max_len
    neg = [0] * max_len
    nxt = seq[-1]
    idx = max_len - 1
    for item_idx in reversed(seq[:-1]):
        src[idx] = item_idx
        pos[idx] = nxt
        neg[idx] = _sample_negative(item_count, positives, rng)
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


def _sequence_tensor(seq: list[int], *, max_len: int) -> tuple[list[int], list[int]]:
    trimmed = seq[-max_len:]
    pad_len = max_len - len(trimmed)
    return [0] * pad_len + trimmed, [0] * pad_len + list(range(1, len(trimmed) + 1))


def _make_train_dataset(
    *,
    sequences: dict[int, list[int]],
    user_indices: list[int],
    item_count: int,
    max_len: int,
    seed: int,
    torch: Any,
    Dataset: Any,
) -> Any:
    class TrainDataset(Dataset):  # type: ignore[misc, valid-type]
        def __len__(self) -> int:
            return len(user_indices)

        def __getitem__(self, index: int) -> tuple[Any, ...]:
            rng = random.Random(seed + index)
            user_idx = user_indices[index]
            src, pos, neg, positions = _seq2seq_arrays(
                sequences[user_idx],
                item_count=item_count,
                max_len=max_len,
                rng=rng,
            )
            return (
                torch.tensor(src, dtype=torch.long),
                torch.tensor(pos, dtype=torch.long),
                torch.tensor(neg, dtype=torch.long),
                torch.tensor(positions, dtype=torch.long),
            )

    return TrainDataset()


def _model_args(args: argparse.Namespace, dataset_alias: str, *, model_name: str) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=dataset_alias,
        model_name=model_name,
        llm_emb_path=str(Path("data") / dataset_alias / "handled" / "itm_emb_np.pkl"),
        freeze_emb=args.freeze_emb,
        hidden_size=args.hidden_size,
        trm_num=args.trm_num,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        max_len=args.max_len,
        train_neg=1,
        alpha=args.alpha,
        tau=args.tau,
    )


def _train_seq_model(
    *,
    model: Any,
    loader: Any,
    optimizer: Any,
    epochs: int,
    device: Any,
    log_prefix: str,
    log_every: int,
) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            seq, pos, neg, positions = [tensor.to(device) for tensor in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = model(seq=seq, pos=pos, neg=neg, positions=positions)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            steps += 1
        mean_loss = total_loss / max(steps, 1)
        logs.append({"epoch": epoch, "train_loss": mean_loss})
        if log_every > 0 and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
            print(f"[{log_prefix}] epoch={epoch} train_loss={mean_loss:.6f}", flush=True)
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
            {"candidate_index": candidate_index, "item_id": item_id, "llmesr_item_idx": item_idx}
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
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for event in candidate_events:
            user_idx = int(event["llmesr_user_idx"])
            seq_values, positions = _sequence_tensor(sequences.get(user_idx, []), max_len=max_len)
            seq_tensor = torch.tensor([seq_values], dtype=torch.long, device=device)
            pos_tensor = torch.tensor([positions], dtype=torch.long, device=device)
            item_indices = [int(row["llmesr_item_idx"]) for row in event["candidate_rows"]]
            item_tensor = torch.tensor([item_indices], dtype=torch.long, device=device)
            logits = model.predict(seq=seq_tensor, item_indices=item_tensor, positions=pos_tensor)
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"Non-finite score for source_event_id={event['source_event_id']}")
            for row, score in zip(event["candidate_rows"], logits.squeeze(0).detach().cpu().tolist()):
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
    adapter_dir = Path(args.adapter_dir).expanduser().resolve()
    repo_dir = Path(args.llmemb_repo_dir).expanduser().resolve()
    metadata_path = adapter_dir / "adapter_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"adapter_metadata.json not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset_alias = args.dataset_alias or str(metadata.get("exp_name") or adapter_dir.name).replace("_adapter", "")
    handled_dir = _copy_handled_files(adapter_dir, repo_dir, dataset_alias)

    sequences = _load_sequences(
        adapter_dir / "llm_esr" / "handled" / "inter.txt",
        user_count=int(metadata.get("users", 0)),
    )
    train_user_indices = [user_idx for user_idx, seq in sorted(sequences.items()) if len(seq) >= 2]
    if not train_user_indices:
        raise ValueError("No trainable LLMEmb sequences found; need at least one user with length >= 2.")

    device = _resolve_device(args.device, torch)
    dataset = _make_train_dataset(
        sequences=sequences,
        user_indices=train_user_indices,
        item_count=int(metadata.get("items", 0)),
        max_len=args.max_len,
        seed=args.seed,
        torch=torch,
        Dataset=Dataset,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)

    output_scores_path = (
        Path(args.output_scores_path).expanduser().resolve()
        if args.output_scores_path
        else adapter_dir / "llmemb_upstream_same_candidate_scores.csv"
    )
    checkpoint_path = (
        Path(args.checkpoint_path).expanduser().resolve()
        if args.checkpoint_path
        else adapter_dir / "llmemb_upstream_model.pt"
    )
    sasrec_embedding_path = (
        Path(args.sasrec_embedding_path).expanduser().resolve()
        if args.sasrec_embedding_path
        else handled_dir / "itm_emb_sasrec.pkl"
    )

    sys.path.insert(0, str(repo_dir))
    with _pushd(repo_dir):
        import pickle

        from models.LLMEmb import LLMEmbSASRec
        from models.SASRec import SASRec_seq

        sasrec_model = SASRec_seq(
            int(metadata.get("users", 0)),
            int(metadata.get("items", 0)),
            device,
            _model_args(args, dataset_alias, model_name="sasrec_seq"),
        ).to(device)
        sasrec_optimizer = Adam(sasrec_model.parameters(), lr=args.lr, weight_decay=args.l2)
        sasrec_logs = _train_seq_model(
            model=sasrec_model,
            loader=loader,
            optimizer=sasrec_optimizer,
            epochs=args.sasrec_epochs,
            device=device,
            log_prefix="llmemb-sasrec",
            log_every=args.log_every,
        )
        all_items = torch.arange(start=1, end=int(metadata.get("items", 0)) + 1, device=device)
        sasrec_item_emb = sasrec_model._get_embedding(all_items).detach().cpu().numpy()
        with sasrec_embedding_path.open("wb") as fh:
            pickle.dump(sasrec_item_emb, fh)

        llmemb_model = LLMEmbSASRec(
            int(metadata.get("users", 0)),
            int(metadata.get("items", 0)),
            device,
            _model_args(args, dataset_alias, model_name="llmemb_sasrec"),
        ).to(device)
        llmemb_optimizer = Adam(llmemb_model.parameters(), lr=args.lr, weight_decay=args.l2)
        llmemb_logs = _train_seq_model(
            model=llmemb_model,
            loader=loader,
            optimizer=llmemb_optimizer,
            epochs=args.epochs,
            device=device,
            log_prefix="llmemb",
            log_every=args.log_every,
        )
        score_summary = _score_candidates(
            model=llmemb_model,
            sequences=sequences,
            candidate_events=_candidate_groups(adapter_dir / "candidate_items_mapped.csv"),
            output_scores_path=output_scores_path,
            max_len=args.max_len,
            torch=torch,
            device=device,
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": llmemb_model.state_dict(),
                "args": vars(args),
                "dataset_alias": dataset_alias,
                "summary": score_summary,
            },
            checkpoint_path,
        )

    trainable_params = sum(param.numel() for param in llmemb_model.parameters() if param.requires_grad)
    summary = {
        "adapter_dir": str(adapter_dir),
        "baseline_name": "llmemb_official_qwen3base",
        "artifact_class": "official_llmemb_same_candidate_score",
        "official_result_gate": "provenance_coverage_and_import_required",
        "upstream_repo": "https://github.com/Applied-Machine-Learning-Lab/LLMEmb",
        "dataset_alias": dataset_alias,
        "users": int(metadata.get("users", 0)),
        "items": int(metadata.get("items", 0)),
        "train_users": len(train_user_indices),
        "sasrec_epochs": args.sasrec_epochs,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "l2": args.l2,
        "max_len": args.max_len,
        "hidden_size": args.hidden_size,
        "alpha": args.alpha,
        "tau": args.tau,
        "freeze_emb": args.freeze_emb,
        "device": str(device),
        "seed": args.seed,
        "trainable_params": trainable_params,
        "sasrec_final_train_loss": sasrec_logs[-1]["train_loss"] if sasrec_logs else math.nan,
        "llmemb_final_train_loss": llmemb_logs[-1]["train_loss"] if llmemb_logs else math.nan,
        "sasrec_embedding_path": str(sasrec_embedding_path),
        "checkpoint_path": str(checkpoint_path),
        **score_summary,
        "note": (
            "Scores use the pinned LLMEmb SASRec path: upstream SASRec_seq first "
            "produces itm_emb_sasrec.pkl, then upstream LLMEmbSASRec trains with "
            "the official alignment loss and exports exact same-candidate scores."
        ),
    }
    _write_csv([summary], adapter_dir / "llmemb_upstream_score_summary.csv", list(summary.keys()))
    _write_csv(sasrec_logs, adapter_dir / "llmemb_sasrec_epoch_log.csv", ["epoch", "train_loss"])
    _write_csv(llmemb_logs, adapter_dir / "llmemb_upstream_epoch_log.csv", ["epoch", "train_loss"])
    (adapter_dir / "llmemb_upstream_score_metadata.json").write_text(
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
