from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "adapter_dir",
    "status",
    "dataset_alias",
    "users",
    "items",
    "candidate_events",
    "candidate_rows",
    "train_sequence_rows",
    "valid_sequence_rows",
    "test_sequence_rows",
    "data_rows",
    "missing_mapped_candidate_count",
    "user_map_valid",
    "item_map_valid",
    "item_titles_valid",
    "sequence_files_valid",
    "same_candidate_events_valid",
    "item_text_rows",
    "title_seed_coverage",
    "non_id_embedding_text_coverage",
    "ready_for_embedding_generation",
    "ready_for_upstream_wrapper",
    "diagnosis",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an LLM2Rec same-candidate adapter package.")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_path", default=None)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})


def _as_int(value: Any) -> int | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def _contiguous_one_based(values: list[int], expected_count: int) -> bool:
    return sorted(values) == list(range(1, expected_count + 1))


def _map_status(path: Path, id_col: str, idx_col: str, expected_count: int) -> tuple[bool, int]:
    rows = _read_csv(path)
    ids = [str(row.get(id_col, "")).strip() for row in rows]
    indices = [_as_int(row.get(idx_col)) for row in rows]
    valid_indices = [idx for idx in indices if idx is not None]
    is_valid = (
        len(rows) == expected_count
        and len(set(ids)) == expected_count
        and len(valid_indices) == expected_count
        and _contiguous_one_based(valid_indices, expected_count)
    )
    return is_valid, len(rows)


def _candidate_status(path: Path, expected_count: int) -> tuple[int, int]:
    rows = _read_csv(path)
    missing = 0
    for row in rows:
        if _as_int(row.get("llm2rec_user_idx")) is None:
            missing += 1
        if _as_int(row.get("llm2rec_item_idx")) is None:
            missing += 1
        if _as_int(row.get("llm2rec_test_row_idx")) is None:
            missing += 1
    if expected_count and len(rows) != expected_count:
        missing += abs(len(rows) - expected_count)
    return len(rows), missing


def _line_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def _sequence_file_status(path: Path, expected_rows: int, *, max_item_idx: int, require_label: bool) -> tuple[bool, int]:
    if not path.exists():
        return False, 0
    count = 0
    valid = True
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            parts = [_as_int(part) for part in stripped.split()]
            if any(part is None for part in parts):
                valid = False
                continue
            indices = [part for part in parts if part is not None]
            if require_label and len(indices) < 2:
                valid = False
            if any(item <= 0 or item > max_item_idx for item in indices):
                valid = False
            count += 1
    if count != expected_rows:
        valid = False
    return valid, count


def _item_titles_status(path: Path, *, expected_items: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    keys = [_as_int(key) for key in payload.keys()]
    valid_keys = [key for key in keys if key is not None]
    if not _contiguous_one_based(valid_keys, expected_items):
        return False
    return all(str(payload.get(str(idx), "")).strip() for idx in range(1, expected_items + 1))


def _same_candidate_events_status(path: Path, *, expected_events: int, max_item_idx: int) -> bool:
    if not path.exists():
        return False
    rows = _read_csv(path)
    if len(rows) != expected_events:
        return False
    seen_indices: list[int] = []
    for row in rows:
        event_idx = _as_int(row.get("llm2rec_test_row_idx"))
        positive_idx = _as_int(row.get("llm2rec_positive_item_idx"))
        seq = [part for part in str(row.get("sequence_with_label", "")).split() if part.strip()]
        seq_indices = [_as_int(part) for part in seq]
        if event_idx is None or positive_idx is None:
            return False
        if positive_idx <= 0 or positive_idx > max_item_idx:
            return False
        if not seq_indices or any(part is None for part in seq_indices):
            return False
        if seq_indices[-1] != positive_idx:
            return False
        seen_indices.append(event_idx)
    return sorted(seen_indices) == list(range(expected_events))


def _item_text_status(path: Path, *, item_count: int) -> tuple[int, float, float]:
    rows = _read_csv(path)
    covered = 0
    non_id_text = 0
    for row in rows:
        title = str(row.get("candidate_title", "")).strip()
        if title:
            covered += 1
        embedding_text = str(row.get("embedding_text", "")).strip()
        item_id = str(row.get("item_id", "")).strip()
        if embedding_text and embedding_text != item_id and embedding_text.lower() != f"item id: {item_id}".lower():
            non_id_text += 1
    return (
        len(rows),
        float(covered / item_count) if item_count else 0.0,
        float(non_id_text / item_count) if item_count else 0.0,
    )


def audit(adapter_dir: Path) -> dict[str, Any]:
    metadata_path = adapter_dir / "adapter_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"adapter_metadata.json not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    user_count = int(metadata.get("users", 0))
    item_count = int(metadata.get("items", 0))
    expected_candidates = int(metadata.get("candidate_rows", 0))
    expected_events = int(metadata.get("candidate_events", 0))
    expected_train_sequences = int(metadata.get("train_sequence_rows", 0))
    expected_valid_sequences = int(metadata.get("valid_sequence_rows", 0))
    expected_test_sequences = int(metadata.get("test_sequence_rows", 0))
    expected_data_rows = int(metadata.get("data_rows", 0))

    candidate_path = adapter_dir / "candidate_items_mapped.csv"
    user_map_path = adapter_dir / "user_id_map.csv"
    item_map_path = adapter_dir / "item_id_map.csv"
    item_text_path = adapter_dir / "item_text_seed.csv"
    events_path = adapter_dir / "same_candidate_events.csv"
    item_titles_path = Path(metadata.get("item_titles_path", ""))
    data_path = Path(metadata.get("data_path", ""))
    train_data_path = Path(metadata.get("train_data_path", ""))
    val_data_path = Path(metadata.get("val_data_path", ""))
    test_data_path = Path(metadata.get("test_data_path", ""))

    candidate_rows, missing_mapped = _candidate_status(candidate_path, expected_candidates)
    user_map_valid, _ = _map_status(user_map_path, "user_id", "llm2rec_user_idx", user_count)
    item_map_valid, _ = _map_status(item_map_path, "item_id", "llm2rec_item_idx", item_count)
    item_titles_valid = _item_titles_status(item_titles_path, expected_items=item_count)
    data_valid, data_rows = _sequence_file_status(data_path, expected_data_rows, max_item_idx=item_count, require_label=False)
    train_valid, train_rows = _sequence_file_status(
        train_data_path,
        expected_train_sequences,
        max_item_idx=item_count,
        require_label=True,
    )
    val_valid, val_rows = _sequence_file_status(val_data_path, expected_valid_sequences, max_item_idx=item_count, require_label=True)
    test_valid, test_rows = _sequence_file_status(test_data_path, expected_test_sequences, max_item_idx=item_count, require_label=True)
    sequence_files_valid = data_valid and train_valid and val_valid and test_valid
    events_valid = _same_candidate_events_status(events_path, expected_events=expected_events, max_item_idx=item_count)
    item_text_rows, title_seed_coverage, non_id_embedding_text_coverage = _item_text_status(item_text_path, item_count=item_count)

    core_ready = (
        metadata.get("status") == "adapter_package_only"
        and user_map_valid
        and item_map_valid
        and item_titles_valid
        and sequence_files_valid
        and events_valid
        and candidate_rows == expected_candidates
        and missing_mapped == 0
        and item_text_rows == item_count
    )
    if core_ready:
        diagnosis = "ready_for_llm2rec_upstream_wrapper"
    else:
        diagnosis = "adapter_core_incomplete"

    return {
        "adapter_dir": str(adapter_dir),
        "status": metadata.get("status", ""),
        "dataset_alias": metadata.get("dataset_alias", ""),
        "users": user_count,
        "items": item_count,
        "candidate_events": expected_events,
        "candidate_rows": candidate_rows,
        "train_sequence_rows": train_rows,
        "valid_sequence_rows": val_rows,
        "test_sequence_rows": test_rows,
        "data_rows": data_rows,
        "missing_mapped_candidate_count": missing_mapped,
        "user_map_valid": user_map_valid,
        "item_map_valid": item_map_valid,
        "item_titles_valid": item_titles_valid,
        "sequence_files_valid": sequence_files_valid,
        "same_candidate_events_valid": events_valid,
        "item_text_rows": item_text_rows,
        "title_seed_coverage": f"{title_seed_coverage:.6f}",
        "non_id_embedding_text_coverage": f"{non_id_embedding_text_coverage:.6f}",
        "ready_for_embedding_generation": core_ready,
        "ready_for_upstream_wrapper": core_ready,
        "diagnosis": diagnosis,
    }


def main() -> None:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir).expanduser()
    row = audit(adapter_dir)
    output_path = Path(args.output_path).expanduser() if args.output_path else adapter_dir / "adapter_audit.csv"
    _write_csv(row, output_path)
    for field in FIELDS:
        print(f"{field}={row.get(field, '')}")
    print(f"Saved audit: {output_path}")


if __name__ == "__main__":
    main()
