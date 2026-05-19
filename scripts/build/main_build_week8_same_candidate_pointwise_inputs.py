from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _save_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalise_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _candidate_text(title: str, text: str) -> str:
    title = title.strip()
    text = text.strip()
    if text:
        return text
    if title:
        return f"Title: {title}"
    return ""


def build_pointwise_rows(
    ranking_records: list[dict[str, Any]],
    *,
    domain: str,
    split_name: str,
    task_name: str,
    max_events: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_records = ranking_records[:max_events] if max_events else ranking_records

    for event_idx, record in enumerate(selected_records):
        candidate_ids = [str(item).strip() for item in _normalise_list(record.get("candidate_item_ids"))]
        candidate_titles = [str(item or "").strip() for item in _normalise_list(record.get("candidate_titles"))]
        candidate_texts = [str(item or "").strip() for item in _normalise_list(record.get("candidate_texts"))]
        candidate_groups = [str(item or "unknown").strip().lower() for item in _normalise_list(record.get("candidate_popularity_groups"))]
        candidate_labels = _normalise_list(record.get("candidate_labels"))
        positive_item_id = str(record.get("positive_item_id", "")).strip()

        if not candidate_ids:
            raise ValueError(f"Ranking event {event_idx} is missing candidate_item_ids.")

        for idx, item_id in enumerate(candidate_ids):
            title = candidate_titles[idx] if idx < len(candidate_titles) else ""
            text = candidate_texts[idx] if idx < len(candidate_texts) else ""
            label_value = candidate_labels[idx] if idx < len(candidate_labels) else int(item_id == positive_item_id)
            try:
                label = int(label_value)
            except (TypeError, ValueError):
                label = int(item_id == positive_item_id)

            rows.append(
                {
                    "task_name": task_name,
                    "domain": domain,
                    "split_name": split_name,
                    "source_event_id": str(record.get("source_event_id") or f"{domain}_{split_name}_{event_idx}"),
                    "user_id": str(record.get("user_id", "")),
                    "timestamp": record.get("timestamp"),
                    "history": record.get("history", []),
                    "history_items": record.get("history_items", record.get("history", [])),
                    "candidate_item_id": item_id,
                    "candidate_title": title,
                    "candidate_text": _candidate_text(title, text),
                    "candidate_meta": _candidate_text(title, text),
                    "candidate_popularity_group": candidate_groups[idx] if idx < len(candidate_groups) else "unknown",
                    "candidate_position": idx + 1,
                    "candidate_count": len(candidate_ids),
                    "num_candidates": len(candidate_ids),
                    "positive_item_id": positive_item_id,
                    "label": label,
                    "is_positive": bool(label),
                    "protocol": "large_scale_leave_one_out_same_candidate_sampled_ranking",
                }
            )
    return rows


def _summary_row(rows: list[dict[str, Any]], *, input_path: Path, output_path: Path) -> dict[str, Any]:
    events = {row["source_event_id"] for row in rows}
    positives = sum(int(row.get("label", 0)) for row in rows)
    candidate_counts = [int(row.get("candidate_count", 0)) for row in rows]
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "pointwise_rows": len(rows),
        "events": len(events),
        "positives": positives,
        "avg_candidates": (sum(candidate_counts) / len(candidate_counts)) if candidate_counts else 0.0,
        "protocol": "large_scale_leave_one_out_same_candidate_sampled_ranking",
    }


def _write_summary(row: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert same-candidate ranking JSONL into pointwise rows for shadow/light signals."
    )
    parser.add_argument("--ranking_input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--split_name", required=True, choices=["valid", "test", "train"])
    parser.add_argument("--task_name", default=None)
    parser.add_argument("--max_events", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.ranking_input_path)
    output_path = Path(args.output_path)
    records = _load_jsonl(input_path)
    task_name = args.task_name or input_path.parent.name
    rows = build_pointwise_rows(
        records,
        domain=args.domain,
        split_name=args.split_name,
        task_name=task_name,
        max_events=args.max_events,
    )
    _save_jsonl(rows, output_path)
    summary_path = Path(args.summary_path) if args.summary_path else output_path.with_suffix(".summary.csv")
    summary = _summary_row(rows, input_path=input_path, output_path=output_path)
    _write_summary(summary, summary_path)
    print(f"Saved pointwise rows: {output_path}")
    print(f"rows={summary['pointwise_rows']} events={summary['events']} positives={summary['positives']}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
