from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.candidate_sampling import sample_negative_items
from src.data.sample_builder import (
    build_item_lookup,
    build_popularity_lookup,
    deduplicate_user_sequences,
    sort_and_group_interactions,
)
from src.utils.io import save_jsonl


@dataclass(frozen=True)
class LargeScaleRuntimeConfig:
    processed_dir: Path
    domain: str
    dataset_name: str
    output_root: Path
    exp_prefix: str
    user_limit: int = 10000
    num_negatives: int = 100
    max_history_len: int = 50
    min_sequence_length: int = 3
    seed: int = 20260506
    shuffle_seed: int = 42
    splits: tuple[str, ...] = ("valid", "test")
    selection_strategy: str = "random"
    negative_sampling: str = "popularity"
    require_user_count: bool = False
    test_history_mode: str = "train_plus_valid"
    rating_value: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a large-scale leave-one-out same-candidate runtime. The "
            "recommended Week8 setting is 10k users with 100 sampled negatives "
            "plus one positive candidate per event."
        )
    )
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--exp_prefix", default=None)
    parser.add_argument("--user_limit", type=int, default=10000)
    parser.add_argument("--num_negatives", type=int, default=100)
    parser.add_argument("--max_history_len", type=int, default=50)
    parser.add_argument("--min_sequence_length", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--splits", default="valid,test")
    parser.add_argument("--selection_strategy", choices=["random", "lexical"], default="random")
    parser.add_argument(
        "--negative_sampling",
        choices=["uniform", "popularity"],
        default="popularity",
        help=(
            "Candidate negative sampler. popularity is closer to BERT4Rec-style "
            "sampled evaluation; uniform matches RecBole uni100."
        ),
    )
    parser.add_argument("--require_user_count", action="store_true")
    parser.add_argument(
        "--test_history_mode",
        choices=["train_only", "train_plus_valid"],
        default="train_plus_valid",
        help=(
            "train_plus_valid uses all pre-test interactions for final test "
            "training/scoring; valid always uses the train prefix before the "
            "validation target."
        ),
    )
    parser.add_argument("--rating_value", type=float, default=1.0)
    return parser.parse_args()


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _stable_rng(seed: int, *parts: str) -> random.Random:
    digest = hashlib.sha256("||".join([str(seed), *parts]).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _event_key(user_id: str, timestamp: Any) -> str:
    return f"{user_id}::{timestamp}"


def _history_item_ids(seq: list[dict[str, Any]], *, max_history_len: int) -> list[str]:
    ids = [_text(event.get("item_id")) for event in seq]
    ids = [item_id for item_id in ids if item_id]
    return ids[-max_history_len:]


def _history_texts(
    history_item_ids: list[str],
    item_lookup: dict[str, dict[str, str]],
) -> list[str]:
    texts: list[str] = []
    for item_id in history_item_ids:
        info = item_lookup.get(item_id, {})
        title = _text(info.get("candidate_title"))
        candidate_text = _text(info.get("candidate_text"))
        if title:
            texts.append(title)
        elif candidate_text:
            texts.append(candidate_text[:200])
        else:
            texts.append(item_id)
    return texts


def _item_info(item_id: str, item_lookup: dict[str, dict[str, str]]) -> dict[str, str]:
    info = item_lookup.get(item_id, {})
    return {
        "candidate_title": _text(info.get("candidate_title")),
        "candidate_text": _text(info.get("candidate_text")),
    }


def _candidate_order(
    *,
    positive_item_id: str,
    negative_item_ids: list[str],
    seed: int,
    user_id: str,
    timestamp: Any,
    split_name: str,
) -> list[str]:
    candidates = [positive_item_id, *negative_item_ids]
    rng = _stable_rng(seed, split_name, user_id, _text(timestamp))
    rng.shuffle(candidates)
    return candidates


def _split_parts(
    seq: list[dict[str, Any]],
    *,
    split_name: str,
    test_history_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if split_name == "valid":
        return seq[:-2], seq[-2]
    if split_name == "test":
        history = seq[:-1] if test_history_mode == "train_plus_valid" else seq[:-2]
        return history, seq[-1]
    raise ValueError(f"Unsupported split: {split_name}")


def _known_positive_items(seq: list[dict[str, Any]]) -> set[str]:
    return {_text(event.get("item_id")) for event in seq if _text(event.get("item_id"))}


def _can_build_split(
    seq: list[dict[str, Any]],
    *,
    split_name: str,
    item_lookup: dict[str, dict[str, str]],
    all_item_ids: list[str],
    num_negatives: int,
    test_history_mode: str,
) -> bool:
    if len(seq) < 3:
        return False
    history, target = _split_parts(seq, split_name=split_name, test_history_mode=test_history_mode)
    target_item_id = _text(target.get("item_id"))
    if not target_item_id or target_item_id not in item_lookup:
        return False
    history_ids = {_text(event.get("item_id")) for event in history if _text(event.get("item_id"))}
    if target_item_id in history_ids:
        return False
    excluded = _known_positive_items(seq)
    excluded.add(target_item_id)
    return len(all_item_ids) - len(excluded) >= num_negatives


def _popularity_weights(
    all_item_ids: list[str],
    interactions_df: pd.DataFrame,
) -> dict[str, float]:
    if "item_id" not in interactions_df.columns:
        return {item_id: 1.0 for item_id in all_item_ids}
    counts = interactions_df["item_id"].map(_text).value_counts().to_dict()
    return {item_id: float(counts.get(item_id, 0.0) + 1.0) for item_id in all_item_ids}


def _popularity_cum_weights(
    all_item_ids: list[str],
    *,
    weights_by_item: dict[str, float],
) -> list[float]:
    total = 0.0
    cum_weights: list[float] = []
    for item_id in all_item_ids:
        total += max(float(weights_by_item.get(item_id, 1.0)), 0.0)
        cum_weights.append(total)
    if total <= 0.0:
        return [float(idx + 1) for idx in range(len(all_item_ids))]
    return cum_weights


def _popularity_rejection_sample(
    *,
    excluded_items: set[str],
    all_item_ids: list[str],
    cum_weights: list[float],
    num_samples: int,
    rng: random.Random,
) -> list[str]:
    selected: list[str] = []
    selected_set: set[str] = set()
    max_attempts = max(1000, num_samples * 50)
    attempts = 0
    while len(selected) < num_samples and attempts < max_attempts:
        draw_count = min(num_samples * 2, max_attempts - attempts)
        attempts += draw_count
        for item_id in rng.choices(all_item_ids, cum_weights=cum_weights, k=draw_count):
            if item_id in excluded_items or item_id in selected_set:
                continue
            selected.append(item_id)
            selected_set.add(item_id)
            if len(selected) >= num_samples:
                break
    if len(selected) < num_samples:
        fallback = [
            item_id
            for item_id in all_item_ids
            if item_id not in excluded_items and item_id not in selected_set
        ]
        rng.shuffle(fallback)
        selected.extend(fallback[: num_samples - len(selected)])
    return selected


def _sample_negative_candidates(
    *,
    excluded_items: set[str],
    all_item_ids: list[str],
    num_negatives: int,
    rng: random.Random,
    negative_sampling: str,
    popularity_cum_weights: list[float],
) -> list[str]:
    if negative_sampling == "uniform":
        return sample_negative_items(
            user_seen_items=excluded_items,
            all_item_ids=all_item_ids,
            num_negatives=num_negatives,
            rng=rng,
        )
    return _popularity_rejection_sample(
        excluded_items=excluded_items,
        all_item_ids=all_item_ids,
        cum_weights=popularity_cum_weights,
        num_samples=num_negatives,
        rng=rng,
    )


def _eligible_users(
    user_sequences: dict[str, list[dict[str, Any]]],
    *,
    item_lookup: dict[str, dict[str, str]],
    all_item_ids: list[str],
    cfg: LargeScaleRuntimeConfig,
) -> list[str]:
    eligible: list[str] = []
    for user_id, seq in user_sequences.items():
        if len(seq) < cfg.min_sequence_length:
            continue
        if all(
            _can_build_split(
                seq,
                split_name=split_name,
                item_lookup=item_lookup,
                all_item_ids=all_item_ids,
                num_negatives=cfg.num_negatives,
                test_history_mode=cfg.test_history_mode,
            )
            for split_name in cfg.splits
        ):
            eligible.append(user_id)

    eligible = sorted(eligible)
    if cfg.selection_strategy == "random":
        rng = random.Random(cfg.seed)
        rng.shuffle(eligible)
    return eligible


def _select_users(eligible_users: list[str], *, cfg: LargeScaleRuntimeConfig) -> list[str]:
    selected = eligible_users[: cfg.user_limit] if cfg.user_limit > 0 else eligible_users
    if cfg.require_user_count and cfg.user_limit > 0 and len(selected) < cfg.user_limit:
        raise ValueError(
            f"Only {len(selected)} eligible users available, but --user_limit={cfg.user_limit} "
            "and --require_user_count was set."
        )
    return sorted(selected)


def _build_ranking_samples_for_split(
    user_sequences: dict[str, list[dict[str, Any]]],
    selected_users: list[str],
    *,
    split_name: str,
    item_lookup: dict[str, dict[str, str]],
    popularity_lookup: dict[str, str],
    all_item_ids: list[str],
    popularity_cum_weights: list[float],
    cfg: LargeScaleRuntimeConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for user_id in selected_users:
        seq = user_sequences[user_id]
        history, target = _split_parts(seq, split_name=split_name, test_history_mode=cfg.test_history_mode)
        target_item_id = _text(target.get("item_id"))
        timestamp = target.get("timestamp")
        excluded_items = _known_positive_items(seq)
        excluded_items.add(target_item_id)
        rng = _stable_rng(cfg.seed, "negatives", split_name, user_id, _text(timestamp))
        negative_item_ids = _sample_negative_candidates(
            excluded_items=excluded_items,
            all_item_ids=all_item_ids,
            num_negatives=cfg.num_negatives,
            rng=rng,
            negative_sampling=cfg.negative_sampling,
            popularity_cum_weights=popularity_cum_weights,
        )
        if len(negative_item_ids) != cfg.num_negatives:
            raise ValueError(
                f"{cfg.domain} {split_name} user_id={user_id!r} has only "
                f"{len(negative_item_ids)} sampled negatives; expected {cfg.num_negatives}."
            )

        ordered_candidates = _candidate_order(
            positive_item_id=target_item_id,
            negative_item_ids=negative_item_ids,
            seed=cfg.shuffle_seed,
            user_id=user_id,
            timestamp=timestamp,
            split_name=split_name,
        )
        history_ids = _history_item_ids(history, max_history_len=cfg.max_history_len)
        candidate_titles: list[str] = []
        candidate_texts: list[str] = []
        candidate_groups: list[str] = []
        candidate_labels: list[int] = []
        for item_id in ordered_candidates:
            info = _item_info(item_id, item_lookup)
            candidate_titles.append(info["candidate_title"])
            candidate_texts.append(info["candidate_text"])
            candidate_groups.append(popularity_lookup.get(item_id, "mid"))
            candidate_labels.append(1 if item_id == target_item_id else 0)

        positive_index = ordered_candidates.index(target_item_id)
        positive_info = _item_info(target_item_id, item_lookup)
        rows.append(
            {
                "source_event_id": _event_key(user_id, timestamp),
                "user_id": user_id,
                "history": _history_texts(history_ids, item_lookup),
                "history_item_ids": history_ids,
                "candidate_item_ids": ordered_candidates,
                "candidate_titles": candidate_titles,
                "candidate_texts": candidate_texts,
                "candidate_popularity_groups": candidate_groups,
                "candidate_labels": candidate_labels,
                "positive_item_id": target_item_id,
                "positive_item_title": positive_info["candidate_title"],
                "positive_item_text": positive_info["candidate_text"],
                "positive_item_index": positive_index,
                "timestamp": timestamp,
                "split_name": split_name,
                "num_candidates": len(ordered_candidates),
                "source_pointwise_size": len(ordered_candidates),
                "negative_sampling": f"{cfg.negative_sampling}_uninteracted_{cfg.num_negatives}",
            }
        )
    return rows


def _train_rows_for_split(
    user_sequences: dict[str, list[dict[str, Any]]],
    selected_users: list[str],
    *,
    split_name: str,
    cfg: LargeScaleRuntimeConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for user_id in selected_users:
        seq = user_sequences[user_id]
        history, _ = _split_parts(seq, split_name=split_name, test_history_mode=cfg.test_history_mode)
        for idx, event in enumerate(history):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": _text(event.get("item_id")),
                    "timestamp": event.get("timestamp"),
                    "sequence_index": idx,
                }
            )
    return rows


def _candidate_rows(ranking_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in ranking_samples:
        candidate_ids = sample["candidate_item_ids"]
        labels = sample["candidate_labels"]
        groups = sample["candidate_popularity_groups"]
        titles = sample["candidate_titles"]
        texts = sample["candidate_texts"]
        positive_item_id = _text(sample.get("positive_item_id"))
        for idx, item_id in enumerate(candidate_ids):
            rows.append(
                {
                    "source_event_id": sample["source_event_id"],
                    "user_id": sample["user_id"],
                    "timestamp": sample["timestamp"],
                    "split_name": sample["split_name"],
                    "candidate_index": idx,
                    "item_id": item_id,
                    "label": int(labels[idx]),
                    "is_positive": int(item_id == positive_item_id),
                    "popularity_group": groups[idx],
                    "candidate_title": titles[idx],
                    "candidate_text": texts[idx],
                }
            )
    return rows


def _item_metadata_rows(
    item_ids: set[str],
    *,
    item_lookup: dict[str, dict[str, str]],
    popularity_lookup: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item_id in sorted(item_ids):
        info = _item_info(item_id, item_lookup)
        rows.append(
            {
                "item_id": item_id,
                "candidate_title": info["candidate_title"],
                "candidate_text": info["candidate_text"],
                "popularity_group": popularity_lookup.get(item_id, "mid"),
            }
        )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_recbole_inter(rows: list[dict[str, Any]], path: Path, *, rating_value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["user_id:token", "item_id:token", "timestamp:float", "rating:float"])
        for row in rows:
            writer.writerow([row["user_id"], row["item_id"], row["timestamp"], rating_value])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _collect_task_item_ids(train_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> set[str]:
    item_ids = {_text(row.get("item_id")) for row in train_rows}
    item_ids.update(_text(row.get("item_id")) for row in candidate_rows)
    item_ids.discard("")
    return item_ids


def _write_task_package(
    *,
    split_name: str,
    ranking_samples: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    selected_users: list[str],
    item_lookup: dict[str, dict[str, str]],
    popularity_lookup: dict[str, str],
    cfg: LargeScaleRuntimeConfig,
) -> dict[str, Any]:
    task_dir = cfg.output_root / "baselines" / "external_tasks" / f"{cfg.exp_prefix}_{split_name}_same_candidate"
    ranking_path = task_dir / f"ranking_{split_name}.jsonl"
    candidates_path = task_dir / "candidate_items.csv"
    train_path = task_dir / "train_interactions.csv"
    item_metadata_path = task_dir / "item_metadata.csv"
    recbole_path = task_dir / "recbole" / f"{cfg.dataset_name}.inter"
    metadata_path = task_dir / "metadata.json"
    selected_users_path = task_dir / "selected_users.csv"

    candidate_rows = _candidate_rows(ranking_samples)
    task_item_ids = _collect_task_item_ids(train_rows, candidate_rows)
    item_metadata = _item_metadata_rows(
        task_item_ids,
        item_lookup=item_lookup,
        popularity_lookup=popularity_lookup,
    )

    save_jsonl(ranking_samples, ranking_path)
    _write_csv(
        train_rows,
        train_path,
        ["user_id", "item_id", "timestamp", "sequence_index"],
    )
    _write_csv(
        candidate_rows,
        candidates_path,
        [
            "source_event_id",
            "user_id",
            "timestamp",
            "split_name",
            "candidate_index",
            "item_id",
            "label",
            "is_positive",
            "popularity_group",
            "candidate_title",
            "candidate_text",
        ],
    )
    _write_csv(
        item_metadata,
        item_metadata_path,
        ["item_id", "candidate_title", "candidate_text", "popularity_group"],
    )
    _write_csv(
        [{"user_id": user_id, "selection_index": idx} for idx, user_id in enumerate(selected_users)],
        selected_users_path,
        ["user_id", "selection_index"],
    )
    _write_recbole_inter(train_rows, recbole_path, rating_value=cfg.rating_value)

    metadata = {
        "exp_name": f"{cfg.exp_prefix}_{split_name}_same_candidate",
        "domain": cfg.domain,
        "dataset_name": cfg.dataset_name,
        "processed_dir": str(cfg.processed_dir),
        "split_name": split_name,
        "ranking_input_path": str(ranking_path),
        "train_interactions_path": str(train_path),
        "candidate_items_path": str(candidates_path),
        "item_metadata_path": str(item_metadata_path),
        "recbole_inter_path": str(recbole_path),
        "selected_users_path": str(selected_users_path),
        "selected_users": len(selected_users),
        "user_limit": cfg.user_limit,
        "num_negatives": cfg.num_negatives,
        "num_candidates": cfg.num_negatives + 1,
        "candidate_events": len(ranking_samples),
        "candidate_rows": len(candidate_rows),
        "train_interactions": len(train_rows),
        "task_items": len(task_item_ids),
        "max_history_len": cfg.max_history_len,
        "min_sequence_length": cfg.min_sequence_length,
        "selection_strategy": cfg.selection_strategy,
        "seed": cfg.seed,
        "shuffle_seed": cfg.shuffle_seed,
        "test_history_mode": cfg.test_history_mode,
        "negative_sampling": f"{cfg.negative_sampling} over items not interacted by the selected user",
        "protocol": "large_scale_leave_one_out_same_candidate_sampled_ranking",
        "required_score_schema": ["source_event_id", "user_id", "item_id", "score"],
        "status_label_after_import": "same_schema_external_baseline",
        "paper_scope_note": (
            "This large-scale sampled-ranking protocol is separate from the "
            "Week7.7 six-candidate LLM replay table and should not be mixed "
            "as a direct row-level comparison without stating the candidate "
            "set difference."
        ),
    }
    _write_json(metadata_path, metadata)
    return metadata


def build_large_scale_runtime(cfg: LargeScaleRuntimeConfig) -> dict[str, Any]:
    interactions_path = cfg.processed_dir / "interactions.csv"
    items_path = cfg.processed_dir / "items.csv"
    popularity_path = cfg.processed_dir / "popularity_stats.csv"
    for path in [interactions_path, items_path, popularity_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required processed file not found: {path}")

    interactions_df = pd.read_csv(interactions_path)
    items_df = pd.read_csv(items_path)
    popularity_df = pd.read_csv(popularity_path)

    item_lookup = build_item_lookup(items_df)
    popularity_lookup = build_popularity_lookup(popularity_df)
    all_item_ids = sorted(item_lookup)
    weights_by_item = _popularity_weights(all_item_ids, interactions_df)
    popularity_cum_weights = _popularity_cum_weights(all_item_ids, weights_by_item=weights_by_item)

    user_sequences = sort_and_group_interactions(interactions_df)
    raw_users = len(user_sequences)
    user_sequences = deduplicate_user_sequences(user_sequences)
    deduped_users = len(user_sequences)
    user_sequences = {
        user_id: seq
        for user_id, seq in user_sequences.items()
        if len(seq) >= cfg.min_sequence_length
    }
    min_len_users = len(user_sequences)

    eligible = _eligible_users(
        user_sequences,
        item_lookup=item_lookup,
        all_item_ids=all_item_ids,
        cfg=cfg,
    )
    selected_users = _select_users(eligible, cfg=cfg)
    selected_sequences = {user_id: user_sequences[user_id] for user_id in selected_users}

    manifest_rows = []
    for split_name in cfg.splits:
        ranking_samples = _build_ranking_samples_for_split(
            selected_sequences,
            selected_users,
            split_name=split_name,
            item_lookup=item_lookup,
            popularity_lookup=popularity_lookup,
            all_item_ids=all_item_ids,
            popularity_cum_weights=popularity_cum_weights,
            cfg=cfg,
        )
        train_rows = _train_rows_for_split(
            selected_sequences,
            selected_users,
            split_name=split_name,
            cfg=cfg,
        )
        manifest_rows.append(
            _write_task_package(
                split_name=split_name,
                ranking_samples=ranking_samples,
                train_rows=train_rows,
                selected_users=selected_users,
                item_lookup=item_lookup,
                popularity_lookup=popularity_lookup,
                cfg=cfg,
            )
        )

    summary = {
        "domain": cfg.domain,
        "dataset_name": cfg.dataset_name,
        "processed_dir": str(cfg.processed_dir),
        "raw_users": raw_users,
        "deduped_users": deduped_users,
        "users_after_min_sequence_length": min_len_users,
        "eligible_users": len(eligible),
        "selected_users": len(selected_users),
        "user_limit": cfg.user_limit,
        "num_negatives": cfg.num_negatives,
        "negative_sampling": cfg.negative_sampling,
        "splits": list(cfg.splits),
        "task_manifests": manifest_rows,
    }
    summary_path = cfg.output_root / "summary" / f"{cfg.exp_prefix}_runtime_summary.json"
    _write_json(summary_path, summary)
    return summary


def _parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(item.strip() for item in value.split(",") if item.strip())
    allowed = {"valid", "test"}
    invalid = [split for split in splits if split not in allowed]
    if invalid:
        raise ValueError(f"Unsupported splits: {invalid}. Allowed: {sorted(allowed)}")
    if not splits:
        raise ValueError("--splits must contain at least one split")
    return splits


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir).expanduser()
    dataset_name = args.dataset_name or processed_dir.name
    exp_prefix = args.exp_prefix or f"{args.domain}_large{args.user_limit}_100neg"
    if args.num_negatives != 100 and args.exp_prefix is None:
        exp_prefix = f"{args.domain}_large{args.user_limit}_{args.num_negatives}neg"
    cfg = LargeScaleRuntimeConfig(
        processed_dir=processed_dir,
        domain=args.domain,
        dataset_name=dataset_name,
        output_root=Path(args.output_root).expanduser(),
        exp_prefix=exp_prefix,
        user_limit=args.user_limit,
        num_negatives=args.num_negatives,
        max_history_len=args.max_history_len,
        min_sequence_length=args.min_sequence_length,
        seed=args.seed,
        shuffle_seed=args.shuffle_seed,
        splits=_parse_splits(args.splits),
        selection_strategy=args.selection_strategy,
        negative_sampling=args.negative_sampling,
        require_user_count=args.require_user_count,
        test_history_mode=args.test_history_mode,
        rating_value=args.rating_value,
    )
    summary = build_large_scale_runtime(cfg)
    print(
        f"[{cfg.domain}] selected_users={summary['selected_users']} "
        f"eligible_users={summary['eligible_users']} num_negatives={cfg.num_negatives}"
    )
    for manifest in summary["task_manifests"]:
        print(
            f"Saved {manifest['split_name']} task: "
            f"{Path(manifest['candidate_items_path']).parent} "
            f"events={manifest['candidate_events']} candidates={manifest['candidate_rows']}"
        )


if __name__ == "__main__":
    main()
