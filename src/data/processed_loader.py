from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.protocol import NORMALIZED_INTERACTION_COLUMNS, popularity_buckets
from src.utils.research_artifacts import stable_int_hash


SUPPORTED_PROCESSED_DOMAINS = {
    "amazon_beauty",
    "amazon_books",
    "amazon_electronics",
    "amazon_movies",
}
REJECTED_SOURCE_MARKERS = {"srpd", "small", "noisy", "outputs", "predictions", "repaired"}
CLEAN_PROCESSED_ROOT_NAME = "processed"
ALLOWED_PROCESSED_FILENAMES = {"interactions.csv", "items.csv", "users.csv", "popularity_stats.csv"}


@dataclass(frozen=True)
class ProcessedDataset:
    domain: str
    root: Path
    interactions: pd.DataFrame
    items: pd.DataFrame
    users: pd.DataFrame
    popularity_stats: pd.DataFrame


def _first_present(columns: list[str], candidates: list[str]) -> str | None:
    return next((name for name in candidates if name in columns), None)


def _candidate_text(row: pd.Series) -> str:
    fields = []
    title = str(row.get("title", "") or "").strip()
    categories = str(row.get("categories", "") or "").strip()
    description = str(row.get("description", "") or "").strip()
    brand = str(row.get("brand", "") or "").strip()
    if title:
        fields.append(f"Title: {title}")
    if categories:
        fields.append(f"Categories: {categories}")
    if brand:
        fields.append(f"Brand: {brand}")
    if description:
        fields.append(f"Description: {description}")
    return " | ".join(fields) if fields else f"Item ID: {row['item_id']}"


class ProcessedDatasetLoader:
    """Load audited processed CSV data without trusting old splits or predictions."""

    def __init__(
        self,
        root: str | Path,
        *,
        allowed_domains: set[str] | None = None,
        reject_markers: set[str] | None = None,
        require_clean_root: bool = True,
    ) -> None:
        self.root = Path(root)
        self.allowed_domains = allowed_domains or SUPPORTED_PROCESSED_DOMAINS
        self.reject_markers = reject_markers or REJECTED_SOURCE_MARKERS
        self.require_clean_root = require_clean_root

    def _validate_source_root(self) -> None:
        name_parts = [part.lower() for part in self.root.parts]
        if any(self._has_rejected_marker(part) for part in name_parts):
            raise ValueError(f"Rejected processed dataset source: {self.root}")
        if self.require_clean_root and self.root.parent.name != CLEAN_PROCESSED_ROOT_NAME:
            raise ValueError(f"Processed dataset source must be under {CLEAN_PROCESSED_ROOT_NAME}: {self.root}")
        domain = self.root.name
        if self.allowed_domains is not None and domain not in self.allowed_domains:
            raise ValueError(f"Unsupported processed domain: {domain}")
        self._validate_allowed_entries()

    def _has_rejected_marker(self, part: str) -> bool:
        return any(part == marker or part.endswith(f"_{marker}") or f"_{marker}_" in part for marker in self.reject_markers)

    def _validate_allowed_entries(self) -> None:
        if not self.root.exists():
            return
        extras = []
        for entry in self.root.iterdir():
            name = entry.name
            if name in ALLOWED_PROCESSED_FILENAMES:
                continue
            extras.append(name)
        if extras:
            raise ValueError(f"Rejected legacy artifacts in processed dataset source: {sorted(extras)}")

    def load(self) -> ProcessedDataset:
        self.validate_source_layout()
        paths = {
            "interactions": self.root / "interactions.csv",
            "items": self.root / "items.csv",
            "users": self.root / "users.csv",
            "popularity_stats": self.root / "popularity_stats.csv",
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing processed CSV files: {missing}")
        interactions = self._load_interactions(paths["interactions"])
        items = self._load_items(paths["items"])
        users = self._load_users(paths["users"])
        popularity_stats = self._load_popularity(paths["popularity_stats"])
        self._validate_consistency(interactions=interactions, items=items, users=users, popularity_stats=popularity_stats)
        return ProcessedDataset(self.root.name, self.root, interactions, items, users, popularity_stats)

    def validate_source_layout(self) -> None:
        self._validate_source_root()

    def _load_interactions(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        user_col = _first_present(list(df.columns), ["user_id", "reviewerID", "userId", "uid"])
        item_col = _first_present(list(df.columns), ["item_id", "parent_asin", "asin", "movieId", "iid"])
        rating_col = _first_present(list(df.columns), ["rating", "label", "feedback", "overall", "score"])
        time_col = _first_present(list(df.columns), ["timestamp", "time", "unixReviewTime"])
        missing = [
            name
            for name, value in {
                "user_id": user_col,
                "item_id": item_col,
                "rating": rating_col,
                "timestamp": time_col,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(f"interactions.csv missing required semantic columns: {missing}")
        out = df[[user_col, item_col, rating_col, time_col]].copy()
        out.columns = NORMALIZED_INTERACTION_COLUMNS
        out["user_id"] = out["user_id"].astype(str).str.strip()
        out["item_id"] = out["item_id"].astype(str).str.strip()
        out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
        out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
        out = out.dropna(subset=NORMALIZED_INTERACTION_COLUMNS)
        out = out[(out["user_id"] != "") & (out["item_id"] != "")]
        out["timestamp"] = out["timestamp"].astype("int64")
        return out.reset_index(drop=True)

    def _load_items(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path).fillna("")
        item_col = _first_present(list(df.columns), ["item_id", "parent_asin", "asin", "movieId", "iid"])
        if item_col is None:
            raise ValueError("items.csv missing required semantic column: item_id")
        out = df.rename(columns={item_col: "item_id"}).copy()
        out["item_id"] = out["item_id"].astype(str).str.strip()
        if out["item_id"].duplicated().any():
            raise ValueError("items.csv contains duplicate item_id values")
        for col in ["title", "categories", "description", "brand"]:
            if col not in out.columns:
                out[col] = ""
        if "candidate_text" not in out.columns:
            out["candidate_text"] = out.apply(_candidate_text, axis=1)
        out["candidate_text"] = out["candidate_text"].astype(str).str.strip()
        return out

    def _load_users(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path).fillna("")
        user_col = _first_present(list(df.columns), ["user_id", "reviewerID", "userId", "uid"])
        if user_col is None:
            raise ValueError("users.csv missing required semantic column: user_id")
        out = df.rename(columns={user_col: "user_id"}).copy()
        out["user_id"] = out["user_id"].astype(str).str.strip()
        if out["user_id"].duplicated().any():
            raise ValueError("users.csv contains duplicate user_id values")
        return out

    def _load_popularity(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path).fillna("")
        item_col = _first_present(list(df.columns), ["item_id", "parent_asin", "asin", "movieId", "iid"])
        count_col = _first_present(list(df.columns), ["interaction_count", "count", "popularity_count", "item_popularity_count"])
        bucket_col = _first_present(list(df.columns), ["popularity_group", "popularity_bucket", "bucket", "group"])
        if item_col is None or count_col is None:
            raise ValueError("popularity_stats.csv missing item_id or interaction_count semantic column")
        rename = {item_col: "item_id", count_col: "interaction_count"}
        if bucket_col is not None:
            rename[bucket_col] = "popularity_bucket"
        out = df.rename(columns=rename).copy()
        out["item_id"] = out["item_id"].astype(str).str.strip()
        out["interaction_count"] = pd.to_numeric(out["interaction_count"], errors="coerce").fillna(0).astype(int)
        if "popularity_bucket" not in out.columns:
            out["popularity_bucket"] = ""
        if out["item_id"].duplicated().any():
            raise ValueError("popularity_stats.csv contains duplicate item_id values")
        return out[["item_id", "interaction_count", "popularity_bucket"]]

    @staticmethod
    def _validate_consistency(
        *,
        interactions: pd.DataFrame,
        items: pd.DataFrame,
        users: pd.DataFrame,
        popularity_stats: pd.DataFrame,
    ) -> None:
        missing_items = set(interactions["item_id"]) - set(items["item_id"])
        missing_users = set(interactions["user_id"]) - set(users["user_id"])
        missing_popularity = set(interactions["item_id"]) - set(popularity_stats["item_id"])
        if missing_items:
            raise ValueError(f"items.csv missing {len(missing_items)} interaction item_ids")
        if missing_users:
            raise ValueError(f"users.csv missing {len(missing_users)} interaction user_ids")
        if missing_popularity:
            raise ValueError(f"popularity_stats.csv missing {len(missing_popularity)} interaction item_ids")

    @staticmethod
    def recompute_popularity(
        interactions: pd.DataFrame,
        *,
        head_ratio: float = 0.2,
        mid_ratio: float = 0.6,
    ) -> pd.DataFrame:
        counts = Counter(str(item_id) for item_id in interactions["item_id"])
        buckets = popularity_buckets(counts, head_ratio=head_ratio, mid_ratio=mid_ratio)
        return pd.DataFrame(
            [
                {
                    "item_id": item_id,
                    "interaction_count": int(count),
                    "popularity_bucket": buckets.get(item_id, "tail"),
                }
                for item_id, count in sorted(counts.items())
            ]
        )

    @staticmethod
    def compare_popularity(popularity_stats: pd.DataFrame, recomputed: pd.DataFrame) -> dict[str, int | bool]:
        old = popularity_stats.set_index("item_id")["interaction_count"].astype(int)
        new = recomputed.set_index("item_id")["interaction_count"].astype(int)
        common = old.index.intersection(new.index)
        mismatched = int((old.loc[common] != new.loc[common]).sum())
        missing = int(len(new.index.difference(old.index)))
        extra = int(len(old.index.difference(new.index)))
        return {
            "matches": mismatched == 0 and missing == 0,
            "mismatched_items": mismatched,
            "missing_items": missing,
            "extra_items": extra,
        }

    @staticmethod
    def temporal_leave_one_out_split(
        interactions: pd.DataFrame,
        *,
        min_sequence_length: int = 3,
    ) -> dict[str, list[dict[str, Any]]]:
        rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
        ordered = interactions.sort_values(["user_id", "timestamp", "item_id"]).drop_duplicates(
            ["user_id", "item_id"],
            keep="first",
        )
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

    @staticmethod
    def build_candidates(
        rows_by_split: dict[str, list[dict[str, Any]]],
        items: pd.DataFrame,
        *,
        seed: int = 42,
        negative_count: int = 99,
    ) -> dict[str, list[dict[str, Any]]]:
        item_lookup = {str(row.item_id): row._asdict() for row in items.itertuples(index=False)}
        all_item_ids = sorted(item_lookup)
        out: dict[str, list[dict[str, Any]]] = {}
        for split, rows in rows_by_split.items():
            split_rows = []
            for row in rows:
                target = str(row["target_item_id"])
                forbidden = set(str(x) for x in row.get("full_positive_item_ids", []))
                forbidden.update(str(x) for x in row.get("history_item_ids", []))
                forbidden.add(target)
                pool = [item_id for item_id in all_item_ids if item_id not in forbidden]
                rng = random.Random(stable_int_hash(seed, split, row["user_id"], target, negative_count))
                if len(pool) < negative_count:
                    raise ValueError(f"Not enough negatives for user={row['user_id']} split={split}")
                negatives = rng.sample(pool, negative_count)
                candidate_ids = [target] + negatives
                rng.shuffle(candidate_ids)
                split_rows.append(
                    {
                        **row,
                        "candidate_item_ids": candidate_ids,
                        "candidate_texts": [str(item_lookup[item_id].get("candidate_text", "")) for item_id in candidate_ids],
                    }
                )
            out[split] = split_rows
        return out
