from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.sample_builder import (
    deduplicate_user_sequences,
    sort_and_group_interactions,
    split_user_sequence_leave_one_out,
)
from src.utils.io import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a same-candidate external-baseline task package for SASRec/BERT4Rec/GRU4Rec/LightGCN-style methods."
    )
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--ranking_input_path", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--min_sequence_length", type=int, default=3)
    parser.add_argument("--rating_value", type=float, default=1.0)
    return parser.parse_args()


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _train_rows_from_interactions(interactions_df: pd.DataFrame, *, min_sequence_length: int) -> list[dict[str, Any]]:
    user_sequences = sort_and_group_interactions(interactions_df)
    user_sequences = deduplicate_user_sequences(user_sequences)
    user_sequences = {
        user_id: seq
        for user_id, seq in user_sequences.items()
        if len(seq) >= min_sequence_length
    }
    train_histories, _, _ = split_user_sequence_leave_one_out(user_sequences)

    rows: list[dict[str, Any]] = []
    for user_id, seq in train_histories.items():
        for idx, event in enumerate(seq):
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
        candidate_ids = [_text(item_id) for item_id in sample.get("candidate_item_ids", [])]
        labels = sample.get("candidate_labels", [])
        groups = sample.get("candidate_popularity_groups", [])
        titles = sample.get("candidate_titles", [])
        positive_item_id = _text(sample.get("positive_item_id"))
        for idx, item_id in enumerate(candidate_ids):
            label = labels[idx] if idx < len(labels) else int(item_id == positive_item_id)
            rows.append(
                {
                    "source_event_id": sample.get("source_event_id"),
                    "user_id": sample.get("user_id"),
                    "timestamp": sample.get("timestamp"),
                    "split_name": sample.get("split_name"),
                    "candidate_index": idx,
                    "item_id": item_id,
                    "label": int(label),
                    "is_positive": int(item_id == positive_item_id),
                    "popularity_group": groups[idx] if idx < len(groups) else "unknown",
                    "candidate_title": titles[idx] if idx < len(titles) else "",
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


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir).expanduser()
    ranking_input_path = Path(args.ranking_input_path).expanduser()
    dataset_name = args.dataset_name or processed_dir.name
    output_dir = Path(args.output_root).expanduser() / "baselines" / "external_tasks" / args.exp_name

    interactions_path = processed_dir / "interactions.csv"
    if not interactions_path.exists():
        raise FileNotFoundError(f"interactions.csv not found: {interactions_path}")
    if not ranking_input_path.exists():
        raise FileNotFoundError(f"ranking input not found: {ranking_input_path}")

    interactions_df = pd.read_csv(interactions_path)
    train_rows = _train_rows_from_interactions(interactions_df, min_sequence_length=args.min_sequence_length)
    ranking_samples = load_jsonl(ranking_input_path)
    candidate_rows = _candidate_rows(ranking_samples)

    train_path = output_dir / "train_interactions.csv"
    candidates_path = output_dir / "candidate_items.csv"
    recbole_path = output_dir / "recbole" / f"{dataset_name}.inter"
    metadata_path = output_dir / "metadata.json"

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
        ],
    )
    _write_recbole_inter(train_rows, recbole_path, rating_value=args.rating_value)

    metadata = {
        "exp_name": args.exp_name,
        "dataset_name": dataset_name,
        "processed_dir": str(processed_dir),
        "ranking_input_path": str(ranking_input_path),
        "train_interactions_path": str(train_path),
        "candidate_items_path": str(candidates_path),
        "recbole_inter_path": str(recbole_path),
        "train_interactions": len(train_rows),
        "ranking_events": len(ranking_samples),
        "candidate_rows": len(candidate_rows),
        "protocol": "leave_one_out_train_prefix_and_exact_candidate_scoring",
        "required_score_schema": ["source_event_id", "user_id", "item_id", "score"],
        "status_label_after_import": "same_schema_external_baseline",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved train interactions: {train_path}")
    print(f"Saved candidate items: {candidates_path}")
    print(f"Saved RecBole .inter: {recbole_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"train_interactions={len(train_rows)} ranking_events={len(ranking_samples)} candidate_rows={len(candidate_rows)}")


if __name__ == "__main__":
    main()
