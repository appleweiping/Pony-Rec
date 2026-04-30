from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def _normalize_item_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass
    return [text]


def _normalize_int_list(value: Any) -> list[int]:
    out: list[int] = []
    for item in _normalize_item_list(value):
        try:
            out.append(int(float(item)))
        except Exception:
            out.append(0)
    return out


def _normalize_title(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _json_counter(counter: Counter[str]) -> str:
    total = sum(counter.values())
    if total <= 0:
        return "{}"
    payload = {key: counter[key] / total for key in sorted(counter)}
    return json.dumps(payload, sort_keys=True)


def _event_key_columns(df: pd.DataFrame) -> list[str]:
    if "source_event_id" in df.columns:
        return ["source_event_id"]
    if {"user_id", "timestamp"}.issubset(df.columns):
        return ["user_id", "timestamp"]
    if "user_id" in df.columns:
        return ["user_id"]
    return []


def build_candidate_event_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per candidate event for listwise or pointwise schemas."""

    if "candidate_item_ids" in df.columns:
        rows: list[dict[str, Any]] = []
        for idx, record in enumerate(df.to_dict(orient="records")):
            candidate_ids = _normalize_item_list(record.get("candidate_item_ids"))
            labels = _normalize_int_list(record.get("candidate_labels"))
            if not labels and record.get("positive_item_id"):
                positive = str(record.get("positive_item_id")).strip()
                labels = [1 if item_id == positive else 0 for item_id in candidate_ids]
            titles = _normalize_item_list(record.get("candidate_titles"))
            groups = _normalize_item_list(record.get("candidate_popularity_groups"))
            rows.append(
                {
                    "event_id": str(record.get("source_event_id", idx)),
                    "user_id": record.get("user_id"),
                    "candidate_item_ids": candidate_ids,
                    "candidate_titles": titles,
                    "candidate_popularity_groups": groups,
                    "candidate_labels": labels,
                    "positive_item_id": str(record.get("positive_item_id", "")).strip(),
                }
            )
        return pd.DataFrame(rows)

    if "candidate_item_id" not in df.columns:
        return pd.DataFrame(
            columns=[
                "event_id",
                "user_id",
                "candidate_item_ids",
                "candidate_titles",
                "candidate_popularity_groups",
                "candidate_labels",
                "positive_item_id",
            ]
        )

    key_cols = _event_key_columns(df)
    if not key_cols:
        df = df.copy()
        df["_event_row_id"] = range(len(df))
        key_cols = ["_event_row_id"]

    rows = []
    for group_key, group in df.groupby(key_cols, dropna=False):
        labels = group.get("label", pd.Series([0] * len(group))).astype(int).tolist()
        positive_ids = group.loc[group.get("label", pd.Series([0] * len(group))).astype(int) == 1, "candidate_item_id"].astype(str).tolist()
        event_id = group_key if isinstance(group_key, str) else "::".join(str(item) for item in (group_key if isinstance(group_key, tuple) else (group_key,)))
        rows.append(
            {
                "event_id": event_id,
                "user_id": group["user_id"].iloc[0] if "user_id" in group.columns else "",
                "candidate_item_ids": group["candidate_item_id"].astype(str).tolist(),
                "candidate_titles": group["candidate_title"].astype(str).tolist() if "candidate_title" in group.columns else [],
                "candidate_popularity_groups": group["target_popularity_group"].astype(str).tolist()
                if "target_popularity_group" in group.columns
                else [],
                "candidate_labels": labels,
                "positive_item_id": positive_ids[0] if positive_ids else "",
            }
        )
    return pd.DataFrame(rows)


def collect_users(df: pd.DataFrame) -> set[str]:
    if "user_id" not in df.columns:
        return set()
    return {str(value) for value in df["user_id"].dropna().tolist() if str(value).strip()}


def collect_items_from_events(events_df: pd.DataFrame) -> set[str]:
    items: set[str] = set()
    if events_df.empty or "candidate_item_ids" not in events_df.columns:
        return items
    for value in events_df["candidate_item_ids"].tolist():
        items.update(_normalize_item_list(value))
    return items


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_json(path, lines=True)


def audit_candidate_split(
    df: pd.DataFrame,
    *,
    domain: str,
    split: str,
    negative_sampling_strategy: str,
    hard_negative_groups: set[str] | None = None,
    item_catalog_size: int | None = None,
) -> dict[str, Any]:
    hard_negative_groups = hard_negative_groups or {"head"}
    events = build_candidate_event_frame(df)
    candidate_sizes: list[int] = []
    positives_per_event: list[int] = []
    hard_negative_count = 0
    negative_count = 0
    popularity_counter: Counter[str] = Counter()
    duplicate_title_events = 0
    duplicate_title_count = 0
    title_count = 0

    for record in events.to_dict(orient="records"):
        candidate_ids = _normalize_item_list(record.get("candidate_item_ids"))
        labels = _normalize_int_list(record.get("candidate_labels"))
        titles = [_normalize_title(title) for title in _normalize_item_list(record.get("candidate_titles"))]
        groups = [group.strip().lower() for group in _normalize_item_list(record.get("candidate_popularity_groups"))]

        candidate_sizes.append(len(candidate_ids))
        positives = sum(labels[: len(candidate_ids)]) if labels else (1 if record.get("positive_item_id") else 0)
        positives_per_event.append(int(positives))

        seen_titles = [title for title in titles if title]
        title_count += len(seen_titles)
        unique_title_count = len(set(seen_titles))
        duplicate_title_count += max(0, len(seen_titles) - unique_title_count)
        if unique_title_count < len(seen_titles):
            duplicate_title_events += 1

        for idx, _item_id in enumerate(candidate_ids):
            group = groups[idx] if idx < len(groups) and groups[idx] else "unknown"
            popularity_counter[group] += 1
            label = labels[idx] if idx < len(labels) else 0
            if int(label) != 1:
                negative_count += 1
                if group in hard_negative_groups:
                    hard_negative_count += 1

    candidate_size_series = pd.Series(candidate_sizes, dtype="float64")
    positives_series = pd.Series(positives_per_event, dtype="float64")
    one_positive = bool(len(positives_series) > 0 and (positives_series == 1).all())
    full_catalog_available = bool(
        item_catalog_size
        and len(candidate_size_series) > 0
        and candidate_size_series.min() == item_catalog_size
        and candidate_size_series.max() == item_catalog_size
    )

    return {
        "domain": domain,
        "split": split,
        "status_label": "completed_result",
        "num_users": int(events["user_id"].nunique()) if "user_id" in events.columns else 0,
        "num_events": int(len(events)),
        "candidate_set_size_mean": float(candidate_size_series.mean()) if len(candidate_size_series) else 0.0,
        "candidate_size_mean": float(candidate_size_series.mean()) if len(candidate_size_series) else 0.0,
        "candidate_set_size_min": int(candidate_size_series.min()) if len(candidate_size_series) else 0,
        "candidate_size_min": int(candidate_size_series.min()) if len(candidate_size_series) else 0,
        "candidate_set_size_max": int(candidate_size_series.max()) if len(candidate_size_series) else 0,
        "candidate_size_max": int(candidate_size_series.max()) if len(candidate_size_series) else 0,
        "positives_per_event": float(positives_series.mean()) if len(positives_series) else 0.0,
        "negative_sampling_strategy": negative_sampling_strategy,
        "hard_negative_ratio": float(hard_negative_count / negative_count) if negative_count else 0.0,
        "popularity_bin_distribution": _json_counter(popularity_counter),
        "duplicate_title_rate": float(duplicate_title_count / title_count) if title_count else 0.0,
        "title_overlap_or_duplicate_rate": float(duplicate_title_events / len(events)) if len(events) else 0.0,
        "one_positive_setting": one_positive,
        "hr_recall_equivalent_flag": one_positive,
        "hr_recall_equivalent": one_positive,
        "full_catalog_eval_available_flag": full_catalog_available,
    }


def audit_candidate_protocol(
    split_frames: dict[str, pd.DataFrame],
    *,
    domain: str,
    negative_sampling_strategy: str = "sampled_candidates_unspecified",
    hard_negative_groups: set[str] | None = None,
    train_df: pd.DataFrame | None = None,
    item_catalog_size: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    event_frames = {split: build_candidate_event_frame(df) for split, df in split_frames.items()}
    valid_users = collect_users(event_frames.get("valid", pd.DataFrame()))
    test_users = collect_users(event_frames.get("test", pd.DataFrame()))
    test_items = collect_items_from_events(event_frames.get("test", pd.DataFrame()))

    train_items: set[str] = set()
    if train_df is not None:
        train_items = collect_items_from_events(build_candidate_event_frame(train_df))

    user_overlap_valid_test = len(valid_users & test_users)
    item_overlap_train_test = len(train_items & test_items) if train_items and test_items else 0

    for split, df in split_frames.items():
        row = audit_candidate_split(
            df,
            domain=domain,
            split=split,
            negative_sampling_strategy=negative_sampling_strategy,
            hard_negative_groups=hard_negative_groups,
            item_catalog_size=item_catalog_size,
        )
        row["user_overlap_valid_test"] = int(user_overlap_valid_test)
        row["item_overlap_train_test"] = int(item_overlap_train_test)
        rows.append(row)
    return pd.DataFrame(rows)
