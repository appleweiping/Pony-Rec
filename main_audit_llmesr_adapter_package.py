from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any


FIELDS = [
    "adapter_dir",
    "status",
    "users",
    "items",
    "candidate_rows",
    "inter_rows",
    "missing_mapped_candidate_count",
    "user_map_valid",
    "item_map_valid",
    "inter_valid",
    "sim_user_status",
    "item_text_rows",
    "title_seed_coverage",
    "itm_emb_status",
    "pca64_emb_status",
    "ready_for_embedding_generation",
    "ready_for_scoring",
    "diagnosis",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit an LLM-ESR same-candidate adapter package before scorer work."
    )
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
        if _as_int(row.get("llmesr_user_idx")) is None or _as_int(row.get("llmesr_item_idx")) is None:
            missing += 1
    if expected_count and len(rows) != expected_count:
        missing += abs(len(rows) - expected_count)
    return len(rows), missing


def _inter_status(path: Path, expected_rows: int, *, user_count: int, item_count: int) -> tuple[bool, int]:
    count = 0
    valid = True
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 2:
                valid = False
                continue
            user_idx = _as_int(parts[0])
            item_idx = _as_int(parts[1])
            if user_idx is None or item_idx is None:
                valid = False
            elif not (1 <= user_idx <= user_count and 1 <= item_idx <= item_count):
                valid = False
            count += 1
    if count != expected_rows:
        valid = False
    return valid, count


def _sim_user_status(path: Path, *, user_count: int) -> str:
    if not path.exists():
        return "missing"
    try:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
    except Exception as exc:
        return f"unreadable:{type(exc).__name__}"

    if not isinstance(payload, list):
        return "invalid_type"
    if len(payload) not in {user_count, user_count + 1}:
        return f"unexpected_length:{len(payload)}"

    rows = payload[1:] if len(payload) == user_count + 1 else payload
    for row in rows:
        if not isinstance(row, list) or not row:
            return "invalid_row"
        for value in row:
            idx = _as_int(value)
            if idx is None or not (1 <= idx <= user_count):
                return "out_of_range"
    return "ready"


def _item_text_status(path: Path, *, item_count: int) -> tuple[int, float]:
    rows = _read_csv(path)
    covered = 0
    for row in rows:
        title = str(row.get("candidate_title", "")).strip()
        if title:
            covered += 1
    return len(rows), float(covered / item_count) if item_count else 0.0


def _shape_of_pickle(path: Path) -> tuple[str, int | None]:
    if not path.exists():
        return "missing", None
    try:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
    except Exception as exc:
        return f"unreadable:{type(exc).__name__}", None

    shape = getattr(payload, "shape", None)
    if shape is not None:
        try:
            first_dim = int(shape[0])
        except Exception:
            first_dim = None
        return f"present_shape:{tuple(int(dim) for dim in shape)}", first_dim

    try:
        first_dim = len(payload)
    except Exception:
        first_dim = None
    return f"present_len:{first_dim}", first_dim


def _embedding_status(path: Path, *, expected_items: int) -> tuple[str, bool]:
    status, first_dim = _shape_of_pickle(path)
    if status == "missing":
        return status, False
    if first_dim != expected_items:
        return f"{status}:expected_first_dim_{expected_items}", False
    return status, True


def audit(adapter_dir: Path) -> dict[str, Any]:
    metadata_path = adapter_dir / "adapter_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"adapter_metadata.json not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    user_count = int(metadata.get("users", 0))
    item_count = int(metadata.get("items", 0))
    expected_candidates = int(metadata.get("candidate_rows", 0))
    expected_inter = int(metadata.get("llmesr_inter_rows", 0))

    candidate_path = adapter_dir / "candidate_items_mapped.csv"
    user_map_path = adapter_dir / "user_id_map.csv"
    item_map_path = adapter_dir / "item_id_map.csv"
    item_text_path = adapter_dir / "item_text_seed.csv"
    handled_dir = adapter_dir / "llm_esr" / "handled"
    inter_path = handled_dir / "inter.txt"
    sim_path = handled_dir / "sim_user_100.pkl"
    itm_emb_path = handled_dir / "itm_emb_np.pkl"
    pca64_path = handled_dir / "pca64_itm_emb_np.pkl"

    candidate_rows, missing_mapped = _candidate_status(candidate_path, expected_candidates)
    user_map_valid, _ = _map_status(user_map_path, "user_id", "llmesr_user_idx", user_count)
    item_map_valid, _ = _map_status(item_map_path, "item_id", "llmesr_item_idx", item_count)
    inter_valid, inter_rows = _inter_status(inter_path, expected_inter, user_count=user_count, item_count=item_count)
    sim_status = _sim_user_status(sim_path, user_count=user_count)
    item_text_rows, title_seed_coverage = _item_text_status(item_text_path, item_count=item_count)
    itm_status, itm_ready = _embedding_status(itm_emb_path, expected_items=item_count)
    pca_status, pca_ready = _embedding_status(pca64_path, expected_items=item_count)

    core_ready = (
        metadata.get("status") == "adapter_package_only"
        and candidate_rows == expected_candidates
        and missing_mapped == 0
        and user_map_valid
        and item_map_valid
        and inter_valid
        and sim_status == "ready"
        and item_text_rows == item_count
    )
    scoring_ready = core_ready and itm_ready and pca_ready
    if scoring_ready:
        diagnosis = "ready_for_llmesr_scorer_wrapper"
    elif core_ready:
        diagnosis = "adapter_core_ready_embeddings_missing_or_invalid"
    else:
        diagnosis = "adapter_core_incomplete"

    return {
        "adapter_dir": str(adapter_dir),
        "status": metadata.get("status", ""),
        "users": user_count,
        "items": item_count,
        "candidate_rows": candidate_rows,
        "inter_rows": inter_rows,
        "missing_mapped_candidate_count": missing_mapped,
        "user_map_valid": user_map_valid,
        "item_map_valid": item_map_valid,
        "inter_valid": inter_valid,
        "sim_user_status": sim_status,
        "item_text_rows": item_text_rows,
        "title_seed_coverage": f"{title_seed_coverage:.6f}",
        "itm_emb_status": itm_status,
        "pca64_emb_status": pca_status,
        "ready_for_embedding_generation": core_ready,
        "ready_for_scoring": scoring_ready,
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
