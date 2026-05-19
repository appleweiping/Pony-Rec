from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a same-candidate task package in the id conventions needed "
            "for an LLM-ESR adapter. This prepares adapter inputs only; it does "
            "not produce a completed baseline result."
        )
    )
    parser.add_argument("--task_dir", required=True, help="Directory exported by main_export_same_candidate_baseline_task.py.")
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--top_sim_users", type=int, default=100)
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


def _write_json(payload: dict[str, Any], path: Path) -> None:
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


def _write_llmesr_inter(train_sequences: dict[int, list[int]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        for user_idx in sorted(train_sequences):
            for item_idx in train_sequences[user_idx]:
                fh.write(f"{user_idx} {item_idx}\n")
                count += 1
    return count


def _candidate_mapped_rows(
    candidate_rows: list[dict[str, str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        user_id = _text(row.get("user_id"))
        item_id = _text(row.get("item_id"))
        rows.append(
            {
                **row,
                "llmesr_user_idx": user_to_idx.get(user_id, ""),
                "llmesr_item_idx": item_to_idx.get(item_id, ""),
            }
        )
    return rows


def _item_catalog_from_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    catalog: dict[str, dict[str, str]] = {}
    for row in rows:
        item_id = _text(row.get("item_id"))
        if not item_id:
            continue
        current = catalog.setdefault(item_id, {"candidate_title": "", "candidate_text": ""})
        title = _text(row.get("candidate_title") or row.get("title"))
        text = _text(row.get("candidate_text") or row.get("embedding_text"))
        if title and not current["candidate_title"]:
            current["candidate_title"] = title
        if text and not current["candidate_text"]:
            current["candidate_text"] = text
    return catalog


def _load_item_metadata(task_dir: Path) -> dict[str, dict[str, str]]:
    path = task_dir / "item_metadata.csv"
    if not path.exists():
        return {}
    return _item_catalog_from_rows(_read_csv(path))


def _item_text_seed_rows(
    candidate_rows: list[dict[str, str]],
    item_to_idx: dict[str, int],
    *,
    item_metadata: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    catalog = _item_catalog_from_rows(candidate_rows)
    if item_metadata:
        for item_id, info in item_metadata.items():
            current = catalog.setdefault(item_id, {"candidate_title": "", "candidate_text": ""})
            if info.get("candidate_title") and not current["candidate_title"]:
                current["candidate_title"] = info["candidate_title"]
            if info.get("candidate_text") and not current["candidate_text"]:
                current["candidate_text"] = info["candidate_text"]

    rows = []
    for item_id, item_idx in sorted(item_to_idx.items(), key=lambda pair: pair[1]):
        info = catalog.get(item_id, {})
        title = _text(info.get("candidate_title"))
        embedding_text = _text(info.get("candidate_text")) or title or item_id
        source = "item_metadata" if item_metadata and item_id in item_metadata else "candidate_title"
        if embedding_text == item_id:
            source = "item_id_fallback"
        rows.append(
            {
                "item_id": item_id,
                "llmesr_item_idx": item_idx,
                "candidate_title": title,
                "embedding_text": embedding_text,
                "title_source": source,
            }
        )
    return rows


def _user_map_rows(user_to_idx: dict[str, int]) -> list[dict[str, Any]]:
    return [{"user_id": user_id, "llmesr_user_idx": idx} for user_id, idx in sorted(user_to_idx.items(), key=lambda pair: pair[1])]


def _item_map_rows(item_to_idx: dict[str, int]) -> list[dict[str, Any]]:
    return [{"item_id": item_id, "llmesr_item_idx": idx} for item_id, idx in sorted(item_to_idx.items(), key=lambda pair: pair[1])]


def _similar_users_by_jaccard(train_sequences: dict[int, list[int]], *, top_k: int) -> dict[int, list[int]]:
    item_sets = {user_idx: set(seq) for user_idx, seq in train_sequences.items()}
    user_indices = sorted(item_sets)
    dataset_index_by_user_idx = {user_idx: dataset_idx for dataset_idx, user_idx in enumerate(user_indices)}
    similar: dict[int, list[int]] = {}
    for user_idx in user_indices:
        current = item_sets[user_idx]
        scored: list[tuple[float, int]] = []
        for other_idx in user_indices:
            if other_idx == user_idx:
                continue
            other = item_sets[other_idx]
            union = current | other
            score = float(len(current & other) / len(union)) if union else 0.0
            scored.append((score, other_idx))
        ranked = [other_idx for _, other_idx in sorted(scored, key=lambda pair: (-pair[0], pair[1]))]
        if not ranked:
            ranked = [user_idx]
        while len(ranked) < top_k:
            ranked.extend(ranked)
        similar[user_idx] = [dataset_index_by_user_idx[other_idx] for other_idx in ranked[:top_k]]
    return similar


def _write_sim_users(similar: dict[int, list[int]], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_order = [similar[user_idx] for user_idx in sorted(similar)]

    dataset_order_path = output_dir / "sim_user_100.pkl"
    with dataset_order_path.open("wb") as fh:
        pickle.dump(dataset_order, fh)

    rows = []
    for user_idx, neighbors in sorted(similar.items()):
        rows.append(
            {
                "llmesr_user_idx": user_idx,
                "llmesr_dataset_index": user_idx - 1,
                "similar_dataset_indices": " ".join(str(idx) for idx in neighbors),
            }
        )
    _write_csv(rows, output_dir / "sim_user_100.csv", ["llmesr_user_idx", "llmesr_dataset_index", "similar_dataset_indices"])
    return {
        "dataset_order_pickle": str(dataset_order_path),
        "csv": str(output_dir / "sim_user_100.csv"),
    }


def _write_readme(path: Path, metadata: dict[str, Any]) -> None:
    text = f"""# LLM-ESR Same-Candidate Adapter Package

This package maps the repository's same-candidate task into LLM-ESR's 1-based
user/item id conventions.

Status: `adapter_package_only`

It is not a completed baseline result. Do not import it into the unified method
matrix until an LLM-ESR run emits `{metadata["required_score_schema"]}` with
full candidate coverage.

Important files:

- `handled/inter.txt`: mapped train-only interactions, one `user item` pair per line.
- `candidate_items_mapped.csv`: exact candidate rows with mapped ids.
- `item_text_seed.csv`: item text seeds for generating `itm_emb_np.pkl`.
- `handled/sim_user_100.pkl`: deterministic Jaccard similar-user fallback.
- `adapter_metadata.json`: protocol notes and remaining blockers.

Remaining blockers:

- Generate or provide LLM-ESR-compatible `itm_emb_np.pkl` and `pca64_itm_emb_np.pkl`.
- Patch or wrap LLM-ESR evaluation so it scores `candidate_items_mapped.csv`
  instead of using its sampled negative test loader.
- Keep native LLM-ESR reported metrics out of the paper table unless they pass
  the same-candidate import/audit step.
"""
    path.write_text(text, encoding="utf-8")


def export_llmesr_package(
    task_dir: Path,
    *,
    exp_name: str | None = None,
    output_root: Path | str = "outputs",
    top_sim_users: int = 100,
) -> dict[str, Any]:
    task_dir = Path(task_dir).expanduser()
    train_path = task_dir / "train_interactions.csv"
    candidate_path = task_dir / "candidate_items.csv"
    metadata_path = task_dir / "metadata.json"
    if not train_path.exists():
        raise FileNotFoundError(f"train_interactions.csv not found: {train_path}")
    if not candidate_path.exists():
        raise FileNotFoundError(f"candidate_items.csv not found: {candidate_path}")

    exp_name = exp_name or f"{task_dir.name}_llmesr_adapter"
    output_dir = Path(output_root).expanduser() / "baselines" / "paper_adapters" / exp_name
    handled_dir = output_dir / "llm_esr" / "handled"

    train_rows = _read_csv(train_path)
    candidate_rows = _read_csv(candidate_path)
    item_metadata = _load_item_metadata(task_dir)
    user_to_idx, item_to_idx = _build_user_item_maps(train_rows, candidate_rows)
    train_sequences = _group_train_sequences(train_rows, user_to_idx, item_to_idx)

    inter_rows = _write_llmesr_inter(train_sequences, handled_dir / "inter.txt")
    mapped_candidates = _candidate_mapped_rows(candidate_rows, user_to_idx, item_to_idx)
    candidate_fieldnames = list(mapped_candidates[0].keys()) if mapped_candidates else [
        "source_event_id",
        "user_id",
        "item_id",
        "llmesr_user_idx",
        "llmesr_item_idx",
    ]
    _write_csv(mapped_candidates, output_dir / "candidate_items_mapped.csv", candidate_fieldnames)
    _write_csv(_user_map_rows(user_to_idx), output_dir / "user_id_map.csv", ["user_id", "llmesr_user_idx"])
    _write_csv(_item_map_rows(item_to_idx), output_dir / "item_id_map.csv", ["item_id", "llmesr_item_idx"])
    _write_csv(
        _item_text_seed_rows(candidate_rows, item_to_idx, item_metadata=item_metadata),
        output_dir / "item_text_seed.csv",
        ["item_id", "llmesr_item_idx", "candidate_title", "embedding_text", "title_source"],
    )
    sim_paths = _write_sim_users(
        _similar_users_by_jaccard(train_sequences, top_k=top_sim_users),
        handled_dir,
    )

    source_metadata = {}
    if metadata_path.exists():
        try:
            source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            source_metadata = {"metadata_parse_error": str(metadata_path)}

    metadata = {
        "exp_name": exp_name,
        "adapter_name": "llm_esr_same_candidate",
        "status": "adapter_package_only",
        "source_task_dir": str(task_dir),
        "source_metadata": source_metadata,
        "llmesr_handled_dir": str(handled_dir),
        "llmesr_inter_path": str(handled_dir / "inter.txt"),
        "candidate_items_mapped_path": str(output_dir / "candidate_items_mapped.csv"),
        "user_id_map_path": str(output_dir / "user_id_map.csv"),
        "item_id_map_path": str(output_dir / "item_id_map.csv"),
        "item_text_seed_path": str(output_dir / "item_text_seed.csv"),
        "item_metadata_path": str(task_dir / "item_metadata.csv") if item_metadata else "",
        "sim_user_paths": sim_paths,
        "users": len(user_to_idx),
        "items": len(item_to_idx),
        "train_interaction_rows": len(train_rows),
        "llmesr_inter_rows": inter_rows,
        "candidate_rows": len(candidate_rows),
        "required_score_schema": ["source_event_id", "user_id", "item_id", "score"],
        "remaining_blockers": [
            "Generate LLM-ESR-compatible itm_emb_np.pkl for every mapped item index.",
            "Generate LLM-ESR-compatible pca64_itm_emb_np.pkl for every mapped item index.",
            "Patch or wrap LLM-ESR inference to score candidate_items_mapped.csv exactly.",
            "Import resulting scores with main_import_same_candidate_baseline_scores.py before any result claim.",
        ],
        "native_llmesr_warning": (
            "The upstream generator performs its own leave-last-two split and sampled-negative evaluation. "
            "Use this package with a same-candidate scorer wrapper rather than native reported metrics."
        ),
    }
    _write_json(metadata, output_dir / "adapter_metadata.json")
    _write_readme(output_dir / "README.md", metadata)
    return metadata


def main() -> None:
    args = parse_args()
    metadata = export_llmesr_package(
        Path(args.task_dir).expanduser(),
        exp_name=args.exp_name,
        output_root=args.output_root,
        top_sim_users=args.top_sim_users,
    )
    output_dir = Path(metadata["llmesr_handled_dir"]).parent.parent

    print(f"Saved LLM-ESR adapter package: {output_dir}")
    print(
        "users={users} items={items} train_rows={train_rows} candidate_rows={candidate_rows}".format(
            users=metadata["users"],
            items=metadata["items"],
            train_rows=metadata["train_interaction_rows"],
            candidate_rows=metadata["candidate_rows"],
        )
    )
    print(f"status=adapter_package_only")


if __name__ == "__main__":
    main()
