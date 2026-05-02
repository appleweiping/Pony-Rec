from __future__ import annotations

import csv
import gzip
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.utils.research_artifacts import config_hash, stable_int_hash, utc_timestamp


NORMALIZED_INTERACTION_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
NORMALIZED_ITEM_COLUMNS = ["item_id", "title", "categories", "description", "candidate_text"]
SPLIT_NAMES = ("train", "valid", "test")


@dataclass(frozen=True)
class DataProtocolConfig:
    dataset: str
    domain: str
    raw_interactions_path: Path
    raw_items_path: Path | None
    processed_dir: Path
    raw_format: str = "amazon"
    rating_threshold: float = 4.0
    k_core: int = 5
    min_sequence_length: int = 3
    seed: int = 42
    negative_count: int = 99
    negative_strategy: str = "uniform"
    head_ratio: float = 0.2
    mid_ratio: float = 0.6
    max_description_chars: int = 600


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def protocol_config_from_dict(config: dict[str, Any]) -> DataProtocolConfig:
    raw = config.get("raw", {}) or {}
    filters = config.get("filter", config.get("filters", {})) or {}
    split = config.get("split", {}) or {}
    candidates = config.get("candidates", config.get("sampling", {})) or {}
    popularity = config.get("popularity", {}) or {}
    text = config.get("text", {}) or {}
    return DataProtocolConfig(
        dataset=str(config.get("dataset", config.get("dataset_name", "unknown"))),
        domain=str(config.get("domain", config.get("domain_name", config.get("dataset", "unknown")))),
        raw_interactions_path=Path(str(raw.get("interactions_path") or raw.get("review_path"))),
        raw_items_path=Path(str(raw["items_path"] if "items_path" in raw else raw["meta_path"]))
        if raw.get("items_path") or raw.get("meta_path")
        else None,
        processed_dir=Path(str(config.get("processed_dir", f"data/processed/{config.get('dataset', 'unknown')}"))),
        raw_format=str(raw.get("format", config.get("raw_format", "amazon"))),
        rating_threshold=float(filters.get("rating_threshold", 4.0)),
        k_core=int(filters.get("k_core", filters.get("min_user_interactions", 5))),
        min_sequence_length=int(split.get("min_sequence_length", 3)),
        seed=int(config.get("seed", candidates.get("seed", 42))),
        negative_count=int(candidates.get("negative_count", candidates.get("num_negatives", 99))),
        negative_strategy=str(candidates.get("strategy", candidates.get("negative_strategy", "uniform"))),
        head_ratio=float(popularity.get("head_ratio", 0.2)),
        mid_ratio=float(popularity.get("mid_ratio", 0.6)),
        max_description_chars=int(text.get("max_description_chars", text.get("max_desc_len", 600))),
    )


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _first_present(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _normalize_categories(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, list):
        flattened: list[str] = []
        for entry in value:
            if isinstance(entry, list):
                flattened.extend(str(x) for x in entry if str(x).strip())
            elif str(entry).strip():
                flattened.append(str(entry))
        return " > ".join(flattened)
    return str(value).strip()


def _normalize_description(value: Any, max_chars: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, list):
        text = " ".join(str(x) for x in value if str(x).strip())
    else:
        text = str(value)
    return " ".join(text.split())[:max_chars]


def _candidate_text(title: str, categories: str, description: str) -> str:
    fields = []
    if title:
        fields.append(f"Title: {title}")
    if categories:
        fields.append(f"Categories: {categories}")
    if description:
        fields.append(f"Description: {description}")
    return " | ".join(fields) if fields else "Unknown item"


def load_raw_interactions(config: DataProtocolConfig) -> pd.DataFrame:
    if config.raw_format.lower() in {"amazon", "amazon_jsonl", "jsonl"}:
        records = _read_jsonl(config.raw_interactions_path)
        rows = []
        for row in records:
            user_id = _first_present(row, ["user_id", "reviewerID", "userId", "uid"])
            item_id = _first_present(row, ["item_id", "parent_asin", "asin", "movieId", "iid"])
            rating = _first_present(row, ["rating", "overall", "score"])
            timestamp = _first_present(row, ["timestamp", "unixReviewTime", "time"])
            rows.append({"user_id": user_id, "item_id": item_id, "rating": rating, "timestamp": timestamp})
        df = pd.DataFrame(rows)
    elif config.raw_format.lower() in {"movie", "csv", "movielens"}:
        df = pd.read_csv(config.raw_interactions_path)
        rename = {
            "userId": "user_id",
            "movieId": "item_id",
            "rating": "rating",
            "timestamp": "timestamp",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        df = df[NORMALIZED_INTERACTION_COLUMNS]
    else:
        raise ValueError(f"Unsupported raw format: {config.raw_format}")

    for col in NORMALIZED_INTERACTION_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Raw interactions missing normalized column: {col}")
    out = df[NORMALIZED_INTERACTION_COLUMNS].copy()
    out["user_id"] = out["user_id"].astype(str).str.strip()
    out["item_id"] = out["item_id"].astype(str).str.strip()
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    out = out.dropna(subset=NORMALIZED_INTERACTION_COLUMNS)
    out = out[(out["user_id"] != "") & (out["item_id"] != "")]
    out["timestamp"] = out["timestamp"].astype("int64")
    out = out.drop_duplicates(subset=["user_id", "item_id", "timestamp"]).reset_index(drop=True)
    return out


def load_raw_items(config: DataProtocolConfig, observed_item_ids: set[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if config.raw_items_path is not None and config.raw_items_path.exists():
        if config.raw_items_path.suffix.lower() == ".csv":
            raw_rows = pd.read_csv(config.raw_items_path).to_dict(orient="records")
        else:
            raw_rows = _read_jsonl(config.raw_items_path)
        for row in raw_rows:
            item_id = _first_present(row, ["item_id", "parent_asin", "asin", "movieId", "iid"])
            title = str(_first_present(row, ["title", "name", "movie_title"]) or "").strip()
            categories = _normalize_categories(_first_present(row, ["categories", "category", "genres"]))
            description = _normalize_description(
                _first_present(row, ["description", "desc", "overview"]),
                config.max_description_chars,
            )
            if item_id is not None:
                records.append(
                    {
                        "item_id": str(item_id).strip(),
                        "title": title,
                        "categories": categories,
                        "description": description,
                    }
                )
    existing = {str(row["item_id"]) for row in records if str(row.get("item_id", "")).strip()}
    for item_id in sorted(observed_item_ids - existing):
        records.append({"item_id": item_id, "title": item_id, "categories": "", "description": ""})
    items = pd.DataFrame(records).drop_duplicates(subset=["item_id"]).reset_index(drop=True)
    if items.empty:
        items = pd.DataFrame({"item_id": sorted(observed_item_ids), "title": sorted(observed_item_ids)})
        items["categories"] = ""
        items["description"] = ""
    items["candidate_text"] = [
        _candidate_text(str(row.title), str(row.categories), str(row.description))
        for row in items.itertuples(index=False)
    ]
    return items[NORMALIZED_ITEM_COLUMNS]


def iterative_k_core_filter(
    interactions: pd.DataFrame,
    *,
    k_core: int,
) -> tuple[pd.DataFrame, list[dict[str, int]]]:
    out = interactions.copy()
    trace: list[dict[str, int]] = []
    iteration = 0
    while True:
        before = len(out)
        user_counts = out["user_id"].value_counts()
        out = out[out["user_id"].isin(user_counts[user_counts >= k_core].index)].copy()
        item_counts = out["item_id"].value_counts()
        out = out[out["item_id"].isin(item_counts[item_counts >= k_core].index)].copy()
        after = len(out)
        trace.append(
            {
                "iteration": iteration,
                "interactions_before": int(before),
                "interactions_after": int(after),
                "users": int(out["user_id"].nunique()) if after else 0,
                "items": int(out["item_id"].nunique()) if after else 0,
            }
        )
        iteration += 1
        if after == before:
            break
    return out.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True), trace


def build_user_sequences(interactions: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    sequences: dict[str, list[dict[str, Any]]] = {}
    ordered = interactions.sort_values(["user_id", "timestamp", "item_id"])
    for user_id, group in ordered.groupby("user_id"):
        seen: set[str] = set()
        seq = []
        for row in group.itertuples(index=False):
            item_id = str(row.item_id)
            if item_id in seen:
                continue
            seen.add(item_id)
            seq.append({"item_id": item_id, "rating": float(row.rating), "timestamp": int(row.timestamp)})
        if seq:
            sequences[str(user_id)] = seq
    return sequences


def temporal_leave_one_out(
    sequences: dict[str, list[dict[str, Any]]],
    *,
    min_sequence_length: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_NAMES}
    for user_id in sorted(sequences):
        seq = sequences[user_id]
        if len(seq) < max(3, min_sequence_length):
            continue
        train_history = seq[:-2]
        valid_target = seq[-2]
        test_target = seq[-1]
        rows_by_split["train"].append(
            {
                "user_id": user_id,
                "history_item_ids": [],
                "target_item_id": train_history[-1]["item_id"],
                "timestamp": train_history[-1]["timestamp"],
                "full_positive_item_ids": [x["item_id"] for x in seq],
                "train_item_ids": [x["item_id"] for x in train_history],
            }
        )
        rows_by_split["valid"].append(
            {
                "user_id": user_id,
                "history_item_ids": [x["item_id"] for x in train_history],
                "target_item_id": valid_target["item_id"],
                "timestamp": valid_target["timestamp"],
                "full_positive_item_ids": [x["item_id"] for x in seq],
                "train_item_ids": [x["item_id"] for x in train_history],
            }
        )
        rows_by_split["test"].append(
            {
                "user_id": user_id,
                "history_item_ids": [x["item_id"] for x in train_history + [valid_target]],
                "target_item_id": test_target["item_id"],
                "timestamp": test_target["timestamp"],
                "full_positive_item_ids": [x["item_id"] for x in seq],
                "train_item_ids": [x["item_id"] for x in train_history],
            }
        )
    return rows_by_split


def item_popularity_from_train(rows_by_split: dict[str, list[dict[str, Any]]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows_by_split["train"]:
        counts.update(row.get("train_item_ids", []))
    return counts


def popularity_buckets(counts: Counter[str], *, head_ratio: float, mid_ratio: float) -> dict[str, str]:
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    n_items = len(ordered)
    n_head = max(1, int(round(n_items * head_ratio))) if n_items else 0
    n_mid = max(0, int(round(n_items * mid_ratio))) if n_items else 0
    buckets = {}
    for idx, (item_id, _) in enumerate(ordered):
        if idx < n_head:
            bucket = "head"
        elif idx < n_head + n_mid:
            bucket = "mid"
        else:
            bucket = "tail"
        buckets[item_id] = bucket
    return buckets


def dataset_statistics(
    *,
    raw_interactions: pd.DataFrame,
    positive_interactions: pd.DataFrame,
    filtered_interactions: pd.DataFrame,
    sequences: dict[str, list[dict[str, Any]]],
    rows_by_split: dict[str, list[dict[str, Any]]],
    popularity_counts: Counter[str],
    popularity_bucket_lookup: dict[str, str],
    k_core_trace: list[dict[str, int]],
    config: DataProtocolConfig,
) -> dict[str, Any]:
    seq_lengths = [len(seq) for seq in sequences.values()]
    bucket_counts = Counter(popularity_bucket_lookup.values())
    users = int(filtered_interactions["user_id"].nunique()) if len(filtered_interactions) else 0
    items = int(filtered_interactions["item_id"].nunique()) if len(filtered_interactions) else 0
    density = float(len(filtered_interactions) / (users * items)) if users and items else 0.0
    config_for_hash = {
        "dataset": config.dataset,
        "domain": config.domain,
        "raw_interactions_path": str(config.raw_interactions_path),
        "raw_items_path": str(config.raw_items_path) if config.raw_items_path else None,
        "processed_dir": str(config.processed_dir),
        "raw_format": config.raw_format,
        "rating_threshold": config.rating_threshold,
        "k_core": config.k_core,
        "min_sequence_length": config.min_sequence_length,
        "seed": config.seed,
        "negative_count": config.negative_count,
        "negative_strategy": config.negative_strategy,
    }
    return {
        "dataset": config.dataset,
        "domain": config.domain,
        "seed": config.seed,
        "rating_threshold": config.rating_threshold,
        "k_core": config.k_core,
        "raw_interactions": int(len(raw_interactions)),
        "positive_interactions": int(len(positive_interactions)),
        "filtered_interactions": int(len(filtered_interactions)),
        "raw_users": int(raw_interactions["user_id"].nunique()) if len(raw_interactions) else 0,
        "raw_items": int(raw_interactions["item_id"].nunique()) if len(raw_interactions) else 0,
        "filtered_users": users,
        "filtered_items": items,
        "density": density,
        "average_sequence_length": float(sum(seq_lengths) / len(seq_lengths)) if seq_lengths else 0.0,
        "min_sequence_length": int(min(seq_lengths)) if seq_lengths else 0,
        "max_sequence_length": int(max(seq_lengths)) if seq_lengths else 0,
        "popularity_bucket_counts": dict(bucket_counts),
        "split_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "k_core_trace": k_core_trace,
        "config_hash": config_hash({"data_protocol": config_for_hash}),
        "timestamp": utc_timestamp(),
    }


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def write_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def preprocess_from_config(config: DataProtocolConfig) -> dict[str, Any]:
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    raw_interactions = load_raw_interactions(config)
    positive = raw_interactions[raw_interactions["rating"] >= config.rating_threshold].copy()
    filtered, k_core_trace = iterative_k_core_filter(positive, k_core=config.k_core)
    sequences = build_user_sequences(filtered)
    rows_by_split = temporal_leave_one_out(sequences, min_sequence_length=config.min_sequence_length)
    popularity_counts = item_popularity_from_train(rows_by_split)
    popularity_bucket_lookup = popularity_buckets(
        popularity_counts,
        head_ratio=config.head_ratio,
        mid_ratio=config.mid_ratio,
    )
    items = load_raw_items(config, set(filtered["item_id"].astype(str).unique()))
    items["item_popularity_count"] = items["item_id"].map(lambda item_id: int(popularity_counts.get(str(item_id), 0)))
    items["item_popularity_bucket"] = items["item_id"].map(
        lambda item_id: popularity_bucket_lookup.get(str(item_id), "tail")
    )
    stats = dataset_statistics(
        raw_interactions=raw_interactions,
        positive_interactions=positive,
        filtered_interactions=filtered,
        sequences=sequences,
        rows_by_split=rows_by_split,
        popularity_counts=popularity_counts,
        popularity_bucket_lookup=popularity_bucket_lookup,
        k_core_trace=k_core_trace,
        config=config,
    )
    filtered.to_csv(config.processed_dir / "interactions.csv", index=False)
    items.to_csv(config.processed_dir / "items.csv", index=False)
    write_json(stats, config.processed_dir / "data_stats.json")
    write_jsonl(
        (
            {
                "user_id": user_id,
                "sequence": seq,
                "item_ids": [x["item_id"] for x in seq],
            }
            for user_id, seq in sorted(sequences.items())
        ),
        config.processed_dir / "sequences.jsonl",
    )
    for split, rows in rows_by_split.items():
        write_jsonl(rows, config.processed_dir / f"{split}.jsonl")
    with (config.processed_dir / "popularity_stats.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "interaction_count", "popularity_bucket"])
        writer.writeheader()
        for item_id in sorted(set(items["item_id"].astype(str))):
            writer.writerow(
                {
                    "item_id": item_id,
                    "interaction_count": int(popularity_counts.get(item_id, 0)),
                    "popularity_bucket": popularity_bucket_lookup.get(item_id, "tail"),
                }
            )
    return stats


def _history_bucket(length: int) -> str:
    if length <= 2:
        return "short"
    if length <= 10:
        return "medium"
    return "long"


def _sample_negatives(
    *,
    row: dict[str, Any],
    all_item_ids: list[str],
    popularity_bucket_lookup: dict[str, str],
    negative_count: int,
    seed: int,
    split: str,
    strategy: str,
) -> list[str]:
    forbidden = set(str(x) for x in row.get("full_positive_item_ids", []))
    target = str(row["target_item_id"])
    forbidden.add(target)
    pool = [item_id for item_id in all_item_ids if item_id not in forbidden]
    rng = random.Random(stable_int_hash(seed, split, row["user_id"], target, negative_count, strategy))
    if strategy == "popularity_stratified":
        by_bucket: dict[str, list[str]] = defaultdict(list)
        for item_id in pool:
            by_bucket[popularity_bucket_lookup.get(item_id, "tail")].append(item_id)
        negatives: list[str] = []
        per_bucket = max(1, negative_count // 3)
        for bucket in ["head", "mid", "tail"]:
            values = by_bucket.get(bucket, [])
            rng.shuffle(values)
            negatives.extend(values[:per_bucket])
        remaining = [item_id for item_id in pool if item_id not in set(negatives)]
        rng.shuffle(remaining)
        negatives.extend(remaining[: max(0, negative_count - len(negatives))])
        return negatives[:negative_count]
    rng.shuffle(pool)
    return pool[:negative_count]


def build_candidates_from_processed(
    processed_dir: str | Path,
    *,
    seed: int,
    negative_count: int,
    strategy: str = "uniform",
) -> dict[str, int]:
    processed_dir = Path(processed_dir)
    items = pd.read_csv(processed_dir / "items.csv").fillna("")
    item_lookup = {str(row.item_id): row._asdict() for row in items.itertuples(index=False)}
    all_item_ids = sorted(item_lookup)
    popularity_bucket_lookup = {
        item_id: str(info.get("item_popularity_bucket") or info.get("popularity_bucket") or "tail")
        for item_id, info in item_lookup.items()
    }
    counts: dict[str, int] = {}
    for split in SPLIT_NAMES:
        split_rows = read_jsonl(processed_dir / f"{split}.jsonl")
        out_rows = []
        for row in split_rows:
            target = str(row["target_item_id"])
            candidate_ids = [target] + _sample_negatives(
                row=row,
                all_item_ids=all_item_ids,
                popularity_bucket_lookup=popularity_bucket_lookup,
                negative_count=negative_count,
                seed=seed,
                split=split,
                strategy=strategy,
            )
            rng = random.Random(stable_int_hash(seed, split, row["user_id"], target, "candidate_order"))
            rng.shuffle(candidate_ids)
            history_len = len(row.get("history_item_ids", []))
            out_rows.append(
                {
                    "dataset_split": split,
                    "user_id": row["user_id"],
                    "history_item_ids": row.get("history_item_ids", []),
                    "history_length": history_len,
                    "history_length_bucket": _history_bucket(history_len),
                    "target_item_id": target,
                    "candidate_item_ids": candidate_ids,
                    "candidate_titles": [str(item_lookup[item_id].get("title", "")) for item_id in candidate_ids],
                    "candidate_texts": [str(item_lookup[item_id].get("candidate_text", "")) for item_id in candidate_ids],
                    "candidate_popularity_counts": [
                        int(item_lookup[item_id].get("item_popularity_count", 0) or 0) for item_id in candidate_ids
                    ],
                    "candidate_popularity_buckets": [
                        popularity_bucket_lookup.get(item_id, "tail") for item_id in candidate_ids
                    ],
                    "target_popularity_count": int(item_lookup[target].get("item_popularity_count", 0) or 0),
                    "target_popularity_bucket": popularity_bucket_lookup.get(target, "tail"),
                    "timestamp": row.get("timestamp"),
                    "seed": seed,
                    "negative_count": negative_count,
                    "negative_strategy": strategy,
                }
            )
        write_jsonl(out_rows, processed_dir / f"{split}_candidates.jsonl")
        counts[split] = len(out_rows)
    return counts
