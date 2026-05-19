from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _save_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def build_verification_rows(
    ranking_records: list[dict[str, Any]],
    *,
    domain: str,
    split_name: str,
    max_events: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    records = ranking_records[:max_events] if max_events else ranking_records
    for event_idx, record in enumerate(records):
        candidate_ids = [str(item).strip() for item in _list(record.get("candidate_item_ids"))]
        candidate_titles = [str(item or "").strip() for item in _list(record.get("candidate_titles"))]
        candidate_texts = [str(item or "").strip() for item in _list(record.get("candidate_texts"))]
        candidate_labels = _list(record.get("candidate_labels"))
        positive_item_id = str(record.get("positive_item_id", "")).strip()
        source_event_id = str(record.get("source_event_id") or f"{domain}_{split_name}_{event_idx}")

        for idx, item_id in enumerate(candidate_ids):
            title = candidate_titles[idx] if idx < len(candidate_titles) else item_id
            text = candidate_texts[idx] if idx < len(candidate_texts) else title
            raw_label = candidate_labels[idx] if idx < len(candidate_labels) else int(item_id == positive_item_id)
            try:
                label = int(raw_label)
            except (TypeError, ValueError):
                label = int(item_id == positive_item_id)
            rows.append(
                {
                    "domain": domain,
                    "split_name": split_name,
                    "source_event_id": source_event_id,
                    "user_id": str(record.get("user_id", "")),
                    "history": record.get("history", []),
                    "history_items": record.get("history_items", record.get("history", [])),
                    "generated_title": title,
                    "catalog_item_id": item_id,
                    "catalog_title": title,
                    "catalog_text": text,
                    "verification_label": label,
                    "verification_target": "supported_by_user_history_and_catalog",
                    "scaffold_note": (
                        "Catalog-title proxy scaffold. Replace generated_title with model-generated "
                        "titles before using as a final generative recommendation result."
                    ),
                    "protocol": "large_scale_generated_title_verification_scaffold",
                }
            )
    return rows


def _write_summary(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    events = {row["source_event_id"] for row in rows}
    positives = sum(int(row.get("verification_label", 0)) for row in rows)
    summary = {
        "rows": len(rows),
        "events": len(events),
        "positive_rows": positives,
        "negative_rows": len(rows) - positives,
        "protocol": "large_scale_generated_title_verification_scaffold",
    }
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a catalog-grounded generated-title verification scaffold from same-candidate rows."
    )
    parser.add_argument("--ranking_input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--split_name", required=True, choices=["valid", "test", "train"])
    parser.add_argument("--max_events", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_jsonl(args.ranking_input_path)
    rows = build_verification_rows(
        records,
        domain=args.domain,
        split_name=args.split_name,
        max_events=args.max_events,
    )
    output_path = Path(args.output_path)
    _save_jsonl(rows, output_path)
    summary_path = Path(args.summary_path) if args.summary_path else output_path.with_suffix(".summary.csv")
    _write_summary(rows, summary_path)
    print(f"Saved generated-title verification scaffold: {output_path}")
    print(f"rows={len(rows)}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
