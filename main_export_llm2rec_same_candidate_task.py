from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a same-candidate task package in the closest LLM2Rec-style "
            "sequence and item-title format. This prepares adapter inputs only; "
            "it does not produce a completed LLM2Rec result."
        )
    )
    parser.add_argument("--task_dir", required=True, help="Directory exported by main_export_same_candidate_baseline_task.py.")
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--dataset_alias", default="PonySameCandidate")
    return parser.parse_args()


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sequence_sort_key(row: dict[str, str]) -> tuple[float, int, str]:
    try:
        timestamp = float(row.get("timestamp", 0.0))
    except Exception:
        timestamp = 0.0
    try:
        sequence_index = int(float(row.get("sequence_index", 0)))
    except Exception:
        sequence_index = 0
    return timestamp, sequence_index, _text(row.get("item_id"))


def _is_positive(row: dict[str, str]) -> bool:
    for key in ("is_positive", "label"):
        value = _text(row.get(key)).lower()
        if value in {"1", "true", "yes"}:
            return True
    return False


def _source_event_id(row: dict[str, str]) -> str:
    source_event_id = _text(row.get("source_event_id"))
    if source_event_id:
        return source_event_id
    return f"{_text(row.get('user_id'))}::{_text(row.get('timestamp'))}"


def _build_user_item_maps(
    train_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
) -> tuple[dict[str, int], dict[str, int]]:
    user_ids = {_text(row.get("user_id")) for row in train_rows}
    user_ids.update(_text(row.get("user_id")) for row in candidate_rows)
    item_ids = {_text(row.get("item_id")) for row in train_rows}
    item_ids.update(_text(row.get("item_id")) for row in candidate_rows)

    user_ids.discard("")
    item_ids.discard("")
    user_to_idx = {user_id: idx for idx, user_id in enumerate(sorted(user_ids), start=1)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(sorted(item_ids), start=1)}
    return user_to_idx, item_to_idx


def _group_train_sequences(
    train_rows: list[dict[str, str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
) -> dict[int, list[int]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        user_id = _text(row.get("user_id"))
        item_id = _text(row.get("item_id"))
        if user_id in user_to_idx and item_id in item_to_idx:
            grouped[user_id].append(row)

    mapped: dict[int, list[int]] = {}
    for user_id, rows in grouped.items():
        user_idx = user_to_idx[user_id]
        mapped[user_idx] = [item_to_idx[_text(row.get("item_id"))] for row in sorted(rows, key=_sequence_sort_key)]
    return mapped


def _candidate_event_rows(
    candidate_rows: list[dict[str, str]],
    train_sequences: dict[int, list[int]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[_source_event_id(row)].append(row)

    event_rows: list[dict[str, Any]] = []
    event_index_by_source: dict[str, int] = {}
    for event_index, source_event_id in enumerate(sorted(grouped), start=0):
        rows = grouped[source_event_id]
        positive_rows = [row for row in rows if _is_positive(row)]
        if len(positive_rows) != 1:
            raise ValueError(f"source_event_id={source_event_id!r} has {len(positive_rows)} positive rows; expected exactly 1")
        positive = positive_rows[0]
        user_id = _text(positive.get("user_id"))
        item_id = _text(positive.get("item_id"))
        user_idx = user_to_idx[user_id]
        item_idx = item_to_idx[item_id]
        history = train_sequences.get(user_idx, [])
        if not history:
            raise ValueError(f"source_event_id={source_event_id!r} has empty mapped history for user_id={user_id!r}")
        event_index_by_source[source_event_id] = event_index
        event_rows.append(
            {
                "llm2rec_test_row_idx": event_index,
                "source_event_id": source_event_id,
                "user_id": user_id,
                "llm2rec_user_idx": user_idx,
                "positive_item_id": item_id,
                "llm2rec_positive_item_idx": item_idx,
                "history_item_indices": " ".join(str(item) for item in history),
                "sequence_with_label": " ".join(str(item) for item in [*history, item_idx]),
                "candidate_count": len(rows),
            }
        )
    return event_rows, event_index_by_source


def _candidate_mapped_rows(
    candidate_rows: list[dict[str, str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    event_index_by_source: dict[str, int],
) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for row in candidate_rows:
        user_id = _text(row.get("user_id"))
        item_id = _text(row.get("item_id"))
        source_event_id = _source_event_id(row)
        mapped.append(
            {
                **row,
                "source_event_id": source_event_id,
                "llm2rec_user_idx": user_to_idx.get(user_id, ""),
                "llm2rec_item_idx": item_to_idx.get(item_id, ""),
                "llm2rec_test_row_idx": event_index_by_source.get(source_event_id, ""),
            }
        )
    return mapped


def _title_by_item(candidate_rows: list[dict[str, str]]) -> dict[str, str]:
    titles: dict[str, str] = {}
    for row in candidate_rows:
        item_id = _text(row.get("item_id"))
        title = _text(row.get("candidate_title"))
        if item_id and title and item_id not in titles:
            titles[item_id] = title
    return titles


def _item_text_seed_rows(candidate_rows: list[dict[str, str]], item_to_idx: dict[str, int]) -> list[dict[str, Any]]:
    titles = _title_by_item(candidate_rows)
    rows = []
    for item_id, item_idx in sorted(item_to_idx.items(), key=lambda pair: pair[1]):
        title = titles.get(item_id, "")
        rows.append(
            {
                "item_id": item_id,
                "llm2rec_item_idx": item_idx,
                "candidate_title": title,
                "embedding_text": title or item_id,
                "title_source": "candidate_title" if title else "item_id_fallback",
            }
        )
    return rows


def _item_titles_payload(candidate_rows: list[dict[str, str]], item_to_idx: dict[str, int]) -> dict[str, str]:
    titles = _title_by_item(candidate_rows)
    return {
        str(item_idx): titles.get(item_id) or item_id
        for item_id, item_idx in sorted(item_to_idx.items(), key=lambda pair: pair[1])
    }


def _user_map_rows(user_to_idx: dict[str, int]) -> list[dict[str, Any]]:
    return [{"user_id": user_id, "llm2rec_user_idx": idx} for user_id, idx in sorted(user_to_idx.items(), key=lambda pair: pair[1])]


def _item_map_rows(item_to_idx: dict[str, int]) -> list[dict[str, Any]]:
    return [{"item_id": item_id, "llm2rec_item_idx": idx} for item_id, idx in sorted(item_to_idx.items(), key=lambda pair: pair[1])]


def _write_sequence_lines(sequences: list[list[int]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        for seq in sequences:
            if not seq:
                continue
            fh.write(" ".join(str(item) for item in seq) + "\n")
            count += 1
    return count


def export_llm2rec_package(
    task_dir: Path,
    *,
    exp_name: str,
    output_root: Path,
    dataset_alias: str = "PonySameCandidate",
) -> dict[str, Any]:
    train_path = task_dir / "train_interactions.csv"
    candidate_path = task_dir / "candidate_items.csv"
    metadata_path = task_dir / "metadata.json"
    if not train_path.exists():
        raise FileNotFoundError(f"train_interactions.csv not found: {train_path}")
    if not candidate_path.exists():
        raise FileNotFoundError(f"candidate_items.csv not found: {candidate_path}")

    output_dir = output_root / "baselines" / "paper_adapters" / exp_name
    llm2rec_data_dir = output_dir / "llm2rec" / "data" / dataset_alias / "downstream"

    train_rows = _read_csv(train_path)
    candidate_rows = _read_csv(candidate_path)
    user_to_idx, item_to_idx = _build_user_item_maps(train_rows, candidate_rows)
    train_sequences = _group_train_sequences(train_rows, user_to_idx, item_to_idx)
    event_rows, event_index_by_source = _candidate_event_rows(candidate_rows, train_sequences, user_to_idx, item_to_idx)
    mapped_candidates = _candidate_mapped_rows(candidate_rows, user_to_idx, item_to_idx, event_index_by_source)

    train_seq_rows = [seq for _, seq in sorted(train_sequences.items()) if len(seq) >= 2]
    test_seq_rows = [
        [int(part) for part in row["sequence_with_label"].split()]
        for row in sorted(event_rows, key=lambda item: int(item["llm2rec_test_row_idx"]))
    ]
    all_item_marker_rows = [[item_idx] for item_idx in range(1, len(item_to_idx) + 1)]
    data_rows = [*train_seq_rows, *test_seq_rows, *all_item_marker_rows]

    data_count = _write_sequence_lines(data_rows, llm2rec_data_dir / "data.txt")
    train_count = _write_sequence_lines(train_seq_rows, llm2rec_data_dir / "train_data.txt")
    valid_count = _write_sequence_lines(test_seq_rows, llm2rec_data_dir / "val_data.txt")
    test_count = _write_sequence_lines(test_seq_rows, llm2rec_data_dir / "test_data.txt")
    _write_json(_item_titles_payload(candidate_rows, item_to_idx), llm2rec_data_dir / "item_titles.json")

    candidate_fieldnames = list(mapped_candidates[0].keys()) if mapped_candidates else [
        "source_event_id",
        "user_id",
        "item_id",
        "llm2rec_user_idx",
        "llm2rec_item_idx",
        "llm2rec_test_row_idx",
    ]
    _write_csv(mapped_candidates, output_dir / "candidate_items_mapped.csv", candidate_fieldnames)
    _write_csv(event_rows, output_dir / "same_candidate_events.csv", list(event_rows[0].keys()) if event_rows else [])
    _write_csv(_user_map_rows(user_to_idx), output_dir / "user_id_map.csv", ["user_id", "llm2rec_user_idx"])
    _write_csv(_item_map_rows(item_to_idx), output_dir / "item_id_map.csv", ["item_id", "llm2rec_item_idx"])
    _write_csv(
        _item_text_seed_rows(candidate_rows, item_to_idx),
        output_dir / "item_text_seed.csv",
        ["item_id", "llm2rec_item_idx", "candidate_title", "embedding_text", "title_source"],
    )

    source_metadata = {}
    if metadata_path.exists():
        try:
            source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            source_metadata = {"metadata_parse_error": str(metadata_path)}

    metadata = {
        "exp_name": exp_name,
        "adapter_name": "llm2rec_same_candidate",
        "status": "adapter_package_only",
        "artifact_class": "adapter_package",
        "upstream_repo": "https://github.com/HappyPointer/LLM2Rec",
        "source_task_dir": str(task_dir),
        "source_metadata": source_metadata,
        "dataset_alias": dataset_alias,
        "llm2rec_data_dir": str(llm2rec_data_dir),
        "item_titles_path": str(llm2rec_data_dir / "item_titles.json"),
        "data_path": str(llm2rec_data_dir / "data.txt"),
        "train_data_path": str(llm2rec_data_dir / "train_data.txt"),
        "val_data_path": str(llm2rec_data_dir / "val_data.txt"),
        "test_data_path": str(llm2rec_data_dir / "test_data.txt"),
        "candidate_items_mapped_path": str(output_dir / "candidate_items_mapped.csv"),
        "same_candidate_events_path": str(output_dir / "same_candidate_events.csv"),
        "user_id_map_path": str(output_dir / "user_id_map.csv"),
        "item_id_map_path": str(output_dir / "item_id_map.csv"),
        "item_text_seed_path": str(output_dir / "item_text_seed.csv"),
        "users": len(user_to_idx),
        "items": len(item_to_idx),
        "train_interaction_rows": len(train_rows),
        "train_sequence_rows": train_count,
        "data_rows": data_count,
        "valid_sequence_rows": valid_count,
        "test_sequence_rows": test_count,
        "candidate_events": len(event_rows),
        "candidate_rows": len(candidate_rows),
        "required_score_schema": ["source_event_id", "user_id", "item_id", "score"],
        "upstream_adapter_blockers": [
            "LLM2Rec hard-codes dataset aliases and source_dict paths; wrapper must register dataset_alias and llm2rec_data_dir.",
            "Native seqrec evaluation ranks against the full item pool, not exact same-candidate rows.",
            "A same-candidate scorer must gather candidate scores using same_candidate_events.csv and candidate_items_mapped.csv.",
            "Completed-result status requires an upstream-compatible LLM2Rec extraction/training/scoring run with full candidate coverage.",
        ],
    }
    _write_json(metadata, output_dir / "adapter_metadata.json")
    (output_dir / "README.md").write_text(
        "# LLM2Rec Same-Candidate Adapter Package\n\n"
        "This package maps the project's same-candidate task into the closest LLM2Rec-style files:\n\n"
        "- `llm2rec/data/<dataset_alias>/downstream/item_titles.json`\n"
        "- `llm2rec/data/<dataset_alias>/downstream/data.txt`\n"
        "- `llm2rec/data/<dataset_alias>/downstream/train_data.txt`\n"
        "- `llm2rec/data/<dataset_alias>/downstream/val_data.txt`\n"
        "- `llm2rec/data/<dataset_alias>/downstream/test_data.txt`\n\n"
        "Status is `adapter_package_only`. The upstream LLM2Rec repo still needs a small wrapper or patch for "
        "dataset registration and exact same-candidate scoring before any completed-result claim.\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    args = parse_args()
    task_dir = Path(args.task_dir).expanduser()
    exp_name = args.exp_name or f"{task_dir.name}_llm2rec_adapter"
    metadata = export_llm2rec_package(
        task_dir,
        exp_name=exp_name,
        output_root=Path(args.output_root).expanduser(),
        dataset_alias=args.dataset_alias,
    )
    print(f"Saved LLM2Rec adapter package: {Path(metadata['candidate_items_mapped_path']).parent}")
    print(
        f"users={metadata['users']} items={metadata['items']} "
        f"train_rows={metadata['train_interaction_rows']} candidate_events={metadata['candidate_events']} "
        f"candidate_rows={metadata['candidate_rows']}"
    )
    print("status=adapter_package_only")


if __name__ == "__main__":
    main()
