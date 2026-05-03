from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.processed_loader import ProcessedDatasetLoader, SUPPORTED_PROCESSED_DOMAINS
from src.data.protocol import popularity_buckets, write_json, write_jsonl
from src.utils.research_artifacts import config_hash, git_commit_or_unknown, stable_int_hash, utc_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reprocess audited processed CSV sources into clean experiment inputs.")
    parser.add_argument("--source_root", default="data/processed")
    parser.add_argument("--output_dir", default="outputs/reprocessed_processed_source")
    parser.add_argument("--domains", nargs="*", default=sorted(SUPPORTED_PROCESSED_DOMAINS))
    parser.add_argument("--max_users_per_domain", type=int, default=None)
    parser.add_argument("--candidate_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_sequence_length", type=int, default=3)
    return parser.parse_args()


def _first_present(columns: list[str], names: list[str]) -> str:
    for name in names:
        if name in columns:
            return name
    raise ValueError(f"Missing required semantic column from {columns}; expected one of {names}")


def _read_item_ids(items_path: Path) -> list[str]:
    columns = list(pd.read_csv(items_path, nrows=0).columns)
    item_col = _first_present(columns, ["item_id", "parent_asin", "asin", "movieId", "iid"])
    item_ids: list[str] = []
    for chunk in pd.read_csv(items_path, usecols=[item_col], chunksize=500_000):
        item_ids.extend(str(x).strip() for x in chunk[item_col].tolist() if str(x).strip())
    return sorted(dict.fromkeys(item_ids))


def _read_limited_interactions(
    interactions_path: Path,
    *,
    max_users: int | None,
    min_sequence_length: int,
) -> pd.DataFrame:
    columns = list(pd.read_csv(interactions_path, nrows=0).columns)
    user_col = _first_present(columns, ["user_id", "reviewerID", "userId", "uid"])
    item_col = _first_present(columns, ["item_id", "parent_asin", "asin", "movieId", "iid"])
    rating_col = _first_present(columns, ["rating", "label", "feedback", "overall", "score"])
    time_col = _first_present(columns, ["timestamp", "time", "unixReviewTime"])
    usecols = [user_col, item_col, rating_col, time_col]
    if max_users is None:
        df = pd.read_csv(interactions_path, usecols=usecols)
    else:
        selected: set[str] = set()
        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(interactions_path, usecols=usecols, chunksize=100_000):
            chunk[user_col] = chunk[user_col].astype(str).str.strip()
            chunk[item_col] = chunk[item_col].astype(str).str.strip()
            counts = chunk.groupby(user_col)[item_col].nunique()
            for user_id in counts[counts >= min_sequence_length].index:
                selected.add(str(user_id))
                if len(selected) >= max_users:
                    break
            if selected:
                chunks.append(chunk[chunk[user_col].isin(selected)].copy())
            if len(selected) >= max_users:
                break
        if not chunks:
            raise ValueError(f"No users with at least {min_sequence_length} interactions found in {interactions_path}")
        df = pd.concat(chunks, ignore_index=True)
        keep_users = sorted(selected)[:max_users]
        df = df[df[user_col].isin(keep_users)].copy()
    out = df[[user_col, item_col, rating_col, time_col]].copy()
    out.columns = ["user_id", "item_id", "rating", "timestamp"]
    out["user_id"] = out["user_id"].astype(str).str.strip()
    out["item_id"] = out["item_id"].astype(str).str.strip()
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["user_id", "item_id", "rating", "timestamp"])
    out["timestamp"] = out["timestamp"].astype("int64")
    return out.reset_index(drop=True)


def _temporal_split(interactions: pd.DataFrame, *, min_sequence_length: int) -> dict[str, list[dict[str, Any]]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    ordered = interactions.sort_values(["user_id", "timestamp", "item_id"]).drop_duplicates(["user_id", "item_id"], keep="first")
    for user_id, group in ordered.groupby("user_id", sort=True):
        seq = [
            {"item_id": str(row.item_id), "timestamp": int(row.timestamp), "rating": float(row.rating)}
            for row in group.itertuples(index=False)
        ]
        if len(seq) < max(3, min_sequence_length):
            continue
        train_history = seq[:-2]
        valid_target = seq[-2]
        test_target = seq[-1]
        positives = [x["item_id"] for x in seq]
        rows_by_split["train"].append(
            {
                "user_id": str(user_id),
                "history_item_ids": [x["item_id"] for x in train_history[:-1]],
                "target_item_id": train_history[-1]["item_id"],
                "timestamp": train_history[-1]["timestamp"],
                "full_positive_item_ids": positives,
                "train_item_ids": [x["item_id"] for x in train_history],
            }
        )
        rows_by_split["valid"].append(
            {
                "user_id": str(user_id),
                "history_item_ids": [x["item_id"] for x in train_history],
                "target_item_id": valid_target["item_id"],
                "timestamp": valid_target["timestamp"],
                "full_positive_item_ids": positives,
                "train_item_ids": [x["item_id"] for x in train_history],
            }
        )
        rows_by_split["test"].append(
            {
                "user_id": str(user_id),
                "history_item_ids": [x["item_id"] for x in train_history + [valid_target]],
                "target_item_id": test_target["item_id"],
                "timestamp": test_target["timestamp"],
                "full_positive_item_ids": positives,
                "train_item_ids": [x["item_id"] for x in train_history],
            }
        )
    return rows_by_split


def _train_popularity(rows_by_split: dict[str, list[dict[str, Any]]]) -> tuple[pd.DataFrame, dict[str, int], dict[str, str]]:
    counts: Counter[str] = Counter()
    for row in rows_by_split["train"]:
        counts.update(str(item_id) for item_id in row.get("train_item_ids", []))
    buckets = popularity_buckets(counts, head_ratio=0.2, mid_ratio=0.6)
    frame = pd.DataFrame(
        [
            {"item_id": item_id, "interaction_count": int(count), "popularity_bucket": buckets.get(item_id, "tail")}
            for item_id, count in sorted(counts.items())
        ]
    )
    return frame, dict(counts), buckets


def _build_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    all_item_ids: list[str],
    candidate_size: int,
    seed: int,
    split: str,
    popularity_counts: dict[str, int],
    popularity_buckets_lookup: dict[str, str],
) -> list[dict[str, Any]]:
    negative_count = candidate_size - 1
    out = []
    for row in rows:
        target = str(row["target_item_id"])
        forbidden = set(str(x) for x in row.get("full_positive_item_ids", []))
        forbidden.update(str(x) for x in row.get("history_item_ids", []))
        forbidden.add(target)
        pool = [item_id for item_id in all_item_ids if item_id not in forbidden]
        if len(pool) < negative_count:
            raise ValueError(f"Not enough negative items for user={row['user_id']} split={split}")
        rng = random.Random(stable_int_hash(seed, split, row["user_id"], target, candidate_size))
        candidate_item_ids = [target] + rng.sample(pool, negative_count)
        rng.shuffle(candidate_item_ids)
        out.append(
            {
                **row,
                "candidate_item_ids": candidate_item_ids,
                "candidate_popularity_counts": [int(popularity_counts.get(item_id, 0)) for item_id in candidate_item_ids],
                "candidate_popularity_buckets": [popularity_buckets_lookup.get(item_id, "tail") for item_id in candidate_item_ids],
                "target_popularity_count": int(popularity_counts.get(target, 0)),
                "target_popularity_bucket": popularity_buckets_lookup.get(target, "tail"),
                "seed": seed,
                "candidate_size": candidate_size,
            }
        )
    return out


def _load_item_texts(items_path: Path, item_ids: set[str]) -> dict[str, dict[str, str]]:
    columns = list(pd.read_csv(items_path, nrows=0).columns)
    item_col = _first_present(columns, ["item_id", "parent_asin", "asin", "movieId", "iid"])
    wanted = [item_col]
    for col in ["title", "candidate_text"]:
        if col in columns:
            wanted.append(col)
    lookup: dict[str, dict[str, str]] = {}
    for chunk in pd.read_csv(items_path, usecols=wanted, chunksize=500_000):
        chunk = chunk.fillna("")
        chunk[item_col] = chunk[item_col].astype(str).str.strip()
        sub = chunk[chunk[item_col].isin(item_ids)]
        for row in sub.itertuples(index=False):
            data = row._asdict()
            item_id = str(data[item_col])
            title = str(data.get("title", ""))
            text = str(data.get("candidate_text", "")) or (f"Title: {title}" if title else f"Item ID: {item_id}")
            lookup[item_id] = {"title": title, "candidate_text": text}
    return lookup


def _attach_text(rows_by_split: dict[str, list[dict[str, Any]]], item_texts: dict[str, dict[str, str]]) -> None:
    for rows in rows_by_split.values():
        for row in rows:
            ids = [str(item_id) for item_id in row["candidate_item_ids"]]
            row["candidate_titles"] = [item_texts.get(item_id, {}).get("title", "") for item_id in ids]
            row["candidate_texts"] = [item_texts.get(item_id, {}).get("candidate_text", f"Item ID: {item_id}") for item_id in ids]


def _leakage_report(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    violations = []
    for split, rows in rows_by_split.items():
        for row in rows:
            target = str(row["target_item_id"])
            history = set(str(x) for x in row.get("history_item_ids", []))
            positives = set(str(x) for x in row.get("full_positive_item_ids", []))
            candidates = [str(x) for x in row.get("candidate_item_ids", [])]
            negatives = set(candidates) - {target}
            if target in history:
                violations.append({"split": split, "user_id": row["user_id"], "type": "target_in_history"})
            if target in negatives:
                violations.append({"split": split, "user_id": row["user_id"], "type": "target_in_negatives"})
            leaked = sorted(negatives & (history | positives))
            if leaked:
                violations.append({"split": split, "user_id": row["user_id"], "type": "negative_seen_item", "items": leaked[:5]})
    return {
        "passed": not violations,
        "violation_count": len(violations),
        "violations": violations[:50],
        "popularity_source": "train_only",
    }


def _stats(rows_by_split: dict[str, list[dict[str, Any]]], candidate_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "split_rows": {split: len(rows) for split, rows in rows_by_split.items()},
        "candidate_rows": {split: len(rows) for split, rows in candidate_rows.items()},
        "candidate_size": {
            split: sorted({len(row.get("candidate_item_ids", [])) for row in rows}) for split, rows in candidate_rows.items()
        },
    }


def _reprocess_domain(args: argparse.Namespace, domain: str) -> dict[str, Any]:
    source_dir = Path(args.source_root) / domain
    loader = ProcessedDatasetLoader(source_dir)
    loader.validate_source_layout()
    output_dir = Path(args.output_dir) / domain
    output_dir.mkdir(parents=True, exist_ok=True)
    interactions = _read_limited_interactions(
        source_dir / "interactions.csv",
        max_users=args.max_users_per_domain,
        min_sequence_length=args.min_sequence_length,
    )
    rows_by_split = _temporal_split(interactions, min_sequence_length=args.min_sequence_length)
    all_item_ids = _read_item_ids(source_dir / "items.csv")
    train_popularity, popularity_counts, popularity_bucket_lookup = _train_popularity(rows_by_split)
    candidate_rows = {
        split: _build_candidate_rows(
            rows,
            all_item_ids=all_item_ids,
            candidate_size=args.candidate_size,
            seed=args.seed,
            split=split,
            popularity_counts=popularity_counts,
            popularity_buckets_lookup=popularity_bucket_lookup,
        )
        for split, rows in rows_by_split.items()
    }
    candidate_ids = {item_id for rows in candidate_rows.values() for row in rows for item_id in row["candidate_item_ids"]}
    item_texts = _load_item_texts(source_dir / "items.csv", candidate_ids)
    _attach_text(candidate_rows, item_texts)
    for split, rows in rows_by_split.items():
        write_jsonl(rows, output_dir / f"{split}.jsonl")
    for split, rows in candidate_rows.items():
        write_jsonl(rows, output_dir / f"{split}_candidates.jsonl")
    train_popularity.to_csv(output_dir / "train_popularity.csv", index=False)
    leakage = _leakage_report(candidate_rows)
    stats = _stats(rows_by_split, candidate_rows)
    source_paths = {
        name: str(source_dir / f"{name}.csv")
        for name in ["interactions", "items", "popularity_stats", "users"]
    }
    manifest = {
        "created_at": utc_timestamp(),
        "domain": domain,
        "source_root": str(args.source_root),
        "source_table_paths": source_paths,
        "output_dir": str(output_dir),
        "split_protocol": "per_user_temporal_leave_one_out",
        "candidate_protocol": "target_plus_seeded_uniform_negatives",
        "candidate_size": int(args.candidate_size),
        "negative_count": int(args.candidate_size - 1),
        "seed": int(args.seed),
        "min_sequence_length": int(args.min_sequence_length),
        "max_users_per_domain": args.max_users_per_domain,
        "popularity_source": "train_only",
        "config_hash": config_hash(
            {
                "source_paths": source_paths,
                "domain": domain,
                "candidate_size": args.candidate_size,
                "seed": args.seed,
                "min_sequence_length": args.min_sequence_length,
                "max_users_per_domain": args.max_users_per_domain,
            }
        ),
        "git_commit": git_commit_or_unknown("."),
        "outputs": {
            "train": str(output_dir / "train.jsonl"),
            "valid": str(output_dir / "valid.jsonl"),
            "test": str(output_dir / "test.jsonl"),
            "train_candidates": str(output_dir / "train_candidates.jsonl"),
            "valid_candidates": str(output_dir / "valid_candidates.jsonl"),
            "test_candidates": str(output_dir / "test_candidates.jsonl"),
            "train_popularity": str(output_dir / "train_popularity.csv"),
            "split_statistics": str(output_dir / "split_statistics.json"),
            "leakage_report": str(output_dir / "leakage_report.json"),
        },
        "eligibility": {"smoke": True, "pilot": False, "paper_result": False},
    }
    write_json(stats, output_dir / "split_statistics.json")
    write_json(leakage, output_dir / "leakage_report.json")
    write_json(manifest, output_dir / "manifest.json")
    return {"domain": domain, "stats": stats, "leakage_passed": leakage["passed"], "output_dir": str(output_dir)}


def main() -> None:
    args = parse_args()
    if args.candidate_size < 2:
        raise ValueError("--candidate_size must be at least 2")
    summary = {
        "created_at": utc_timestamp(),
        "command": "python -m src.cli.reprocess_processed_source",
        "source_root": args.source_root,
        "output_dir": args.output_dir,
        "domains": [],
    }
    for domain in args.domains:
        result = _reprocess_domain(args, domain)
        summary["domains"].append(result)
        print(f"[reprocess_processed_source] {domain}: {result['stats']} leakage_passed={result['leakage_passed']}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(summary, out / "summary_manifest.json")
    print(f"[reprocess_processed_source] wrote {out}")


if __name__ == "__main__":
    main()
