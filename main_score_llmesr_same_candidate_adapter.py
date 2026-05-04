from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score exact same-candidate rows from an LLM-ESR adapter package. "
            "This is an adapter scaffold scorer unless the package embeddings "
            "come from true upstream-compatible LLM item embeddings."
        )
    )
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_scores_path", default=None)
    parser.add_argument("--embedding_source", choices=["pca64", "itm"], default="pca64")
    parser.add_argument("--similar_user_weight", type=float, default=0.15)
    parser.add_argument("--max_seq_len", type=int, default=200)
    return parser.parse_args()


def _numpy():
    import numpy as np

    return np


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


def _as_int(value: Any) -> int | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _load_inter_sequences(path: Path) -> dict[int, list[int]]:
    sequences: dict[int, list[int]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            user_idx = _as_int(parts[0])
            item_idx = _as_int(parts[1])
            if user_idx is None or item_idx is None:
                continue
            sequences.setdefault(user_idx, []).append(item_idx)
    return sequences


def _normalize_rows(matrix: Any) -> Any:
    np = _numpy()
    matrix = matrix.astype(np.float32, copy=True)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, np.maximum(norms, 1.0e-12), out=np.zeros_like(matrix), where=norms > 0)


def _sequence_vector(item_indices: list[int], item_embeddings: Any, *, max_seq_len: int) -> Any:
    np = _numpy()
    selected = [idx for idx in item_indices[-max_seq_len:] if 1 <= idx <= item_embeddings.shape[0]]
    if not selected:
        return np.zeros(item_embeddings.shape[1], dtype=np.float32)
    return item_embeddings[[idx - 1 for idx in selected]].mean(axis=0).astype(np.float32)


def _user_vector(
    *,
    user_idx: int,
    sequences: dict[int, list[int]],
    sim_users: list[list[int]],
    item_embeddings: Any,
    max_seq_len: int,
    similar_user_weight: float,
) -> Any:
    np = _numpy()
    own = _sequence_vector(sequences.get(user_idx, []), item_embeddings, max_seq_len=max_seq_len)
    if similar_user_weight <= 0.0 or user_idx - 1 >= len(sim_users):
        vec = own
    else:
        sim_vectors = []
        for sim_dataset_idx in sim_users[user_idx - 1]:
            sim_user_idx = _as_int(sim_dataset_idx)
            if sim_user_idx is None:
                continue
            sim_user_idx += 1
            if sim_user_idx == user_idx:
                continue
            sim_vec = _sequence_vector(sequences.get(sim_user_idx, []), item_embeddings, max_seq_len=max_seq_len)
            if float(np.linalg.norm(sim_vec)) > 0.0:
                sim_vectors.append(sim_vec)
        if sim_vectors:
            sim_mean = np.vstack(sim_vectors).mean(axis=0)
            vec = (1.0 - similar_user_weight) * own + similar_user_weight * sim_mean
        else:
            vec = own

    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        return (vec / norm).astype(np.float32)
    return vec.astype(np.float32)


def _embedding_path(adapter_dir: Path, source: str) -> Path:
    handled_dir = adapter_dir / "llm_esr" / "handled"
    if source == "pca64":
        return handled_dir / "pca64_itm_emb_np.pkl"
    return handled_dir / "itm_emb_np.pkl"


def score_adapter(
    adapter_dir: Path,
    *,
    output_scores_path: Path | None = None,
    embedding_source: str = "pca64",
    similar_user_weight: float = 0.15,
    max_seq_len: int = 200,
) -> dict[str, Any]:
    np = _numpy()
    if not (0.0 <= similar_user_weight <= 1.0):
        raise ValueError("--similar_user_weight must be in [0, 1]")

    metadata_path = adapter_dir / "adapter_metadata.json"
    embedding_metadata_path = adapter_dir / "llmesr_embedding_metadata.json"
    candidate_path = adapter_dir / "candidate_items_mapped.csv"
    inter_path = adapter_dir / "llm_esr" / "handled" / "inter.txt"
    sim_user_path = adapter_dir / "llm_esr" / "handled" / "sim_user_100.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"adapter_metadata.json not found: {metadata_path}")
    if not candidate_path.exists():
        raise FileNotFoundError(f"candidate_items_mapped.csv not found: {candidate_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    embedding_metadata = {}
    if embedding_metadata_path.exists():
        embedding_metadata = json.loads(embedding_metadata_path.read_text(encoding="utf-8"))

    item_embeddings = _normalize_rows(np.asarray(_load_pickle(_embedding_path(adapter_dir, embedding_source))))
    expected_items = int(metadata.get("items", 0))
    if item_embeddings.ndim != 2 or item_embeddings.shape[0] != expected_items:
        raise ValueError(
            f"{embedding_source} embeddings shape={getattr(item_embeddings, 'shape', None)} "
            f"does not match adapter items={expected_items}"
        )

    sequences = _load_inter_sequences(inter_path)
    sim_users = _load_pickle(sim_user_path)
    if not isinstance(sim_users, list) or len(sim_users) != int(metadata.get("users", 0)):
        raise ValueError("sim_user_100.pkl must be a dataset-order list with one row per user")

    rows = []
    candidate_rows = _read_csv(candidate_path)
    user_vector_cache: dict[int, Any] = {}
    for row in candidate_rows:
        user_idx = _as_int(row.get("llmesr_user_idx"))
        item_idx = _as_int(row.get("llmesr_item_idx"))
        if user_idx is None or item_idx is None:
            raise ValueError("candidate_items_mapped.csv contains an unmapped user or item")
        if not (1 <= item_idx <= item_embeddings.shape[0]):
            raise ValueError(f"Candidate item index out of range: {item_idx}")

        if user_idx not in user_vector_cache:
            user_vector_cache[user_idx] = _user_vector(
                user_idx=user_idx,
                sequences=sequences,
                sim_users=sim_users,
                item_embeddings=item_embeddings,
                max_seq_len=max_seq_len,
                similar_user_weight=similar_user_weight,
            )
        score = float(np.dot(user_vector_cache[user_idx], item_embeddings[item_idx - 1]))
        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite score for source_event_id={row.get('source_event_id')}")
        rows.append(
            {
                "source_event_id": _text(row.get("source_event_id")),
                "user_id": _text(row.get("user_id")),
                "item_id": _text(row.get("item_id")),
                "score": score,
            }
        )

    if output_scores_path is None:
        output_scores_path = adapter_dir / "llmesr_scaffold_scores.csv"
    _write_csv(rows, output_scores_path, ["source_event_id", "user_id", "item_id", "score"])

    embedding_artifact_class = str(embedding_metadata.get("artifact_class", "")).strip()
    artifact_class = (
        "adapter_scaffold_score"
        if embedding_artifact_class == "adapter_scaffold_embedding"
        else "paper_adapter_score"
    )
    summary = {
        "adapter_dir": str(adapter_dir),
        "baseline_name": "llmesr_same_candidate_adapter",
        "artifact_class": artifact_class,
        "paper_result_ready": artifact_class == "paper_adapter_score",
        "embedding_source": embedding_source,
        "embedding_artifact_class": embedding_artifact_class,
        "similar_user_weight": similar_user_weight,
        "max_seq_len": max_seq_len,
        "candidate_rows": len(candidate_rows),
        "scored_rows": len(rows),
        "score_coverage_rate": float(len(rows) / len(candidate_rows)) if candidate_rows else 0.0,
        "output_scores_path": str(output_scores_path),
        "note": (
            "This scorer uses the LLM-ESR adapter package and exact same-candidate rows. "
            "Rows produced from deterministic scaffold embeddings are for protocol smoke "
            "testing only and must not be promoted to completed_result."
        ),
    }
    _write_csv([summary], adapter_dir / "llmesr_scaffold_score_summary.csv", list(summary.keys()))
    (adapter_dir / "llmesr_scaffold_score_metadata.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    output_scores_path = Path(args.output_scores_path).expanduser() if args.output_scores_path else None
    summary = score_adapter(
        Path(args.adapter_dir).expanduser(),
        output_scores_path=output_scores_path,
        embedding_source=args.embedding_source,
        similar_user_weight=args.similar_user_weight,
        max_seq_len=args.max_seq_len,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
