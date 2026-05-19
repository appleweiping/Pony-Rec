from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate LLM-ESR-compatible item embedding pickle files from an "
            "adapter package's item_text_seed.csv. The default backend is a "
            "deterministic text-hash scaffold; replace these files with true "
            "LLM embeddings before claiming a native paper-project result."
        )
    )
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--embedding_dim", type=int, default=384)
    parser.add_argument("--pca_dim", type=int, default=64)
    parser.add_argument("--backend", default="deterministic_text_hash")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _hash_unit_interval(text: str, salt: str) -> float:
    digest = hashlib.blake2b(f"{salt}\t{text}".encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="little", signed=False)
    return float(value / ((1 << 64) - 1))


def _signed_hash_bucket(text: str, *, dim: int, salt: str) -> tuple[int, float]:
    digest = hashlib.blake2b(f"{salt}\t{text}".encode("utf-8"), digest_size=16).digest()
    bucket = int.from_bytes(digest[:8], byteorder="little", signed=False) % dim
    sign = 1.0 if digest[8] & 1 else -1.0
    return bucket, sign


def _token_features(text: str) -> list[str]:
    lowered = text.lower()
    tokens = TOKEN_RE.findall(lowered)
    features: list[str] = []
    for token in tokens:
        features.append(f"tok:{token}")
        if len(token) >= 3:
            for idx in range(len(token) - 2):
                features.append(f"tri:{token[idx:idx + 3]}")
    if not features and lowered.strip():
        features.append(f"raw:{lowered.strip()}")
    return features


def _numpy():
    import numpy as np

    return np


def _text_hash_embedding(text: str, *, dim: int) -> Any:
    np = _numpy()
    vec = np.zeros(dim, dtype=np.float32)
    features = _token_features(text)
    if not features:
        features = ["empty"]

    for feature in features:
        bucket, sign = _signed_hash_bucket(feature, dim=dim, salt="llmesr_text")
        # A tiny deterministic magnitude jitter breaks ties while preserving scale.
        magnitude = 0.75 + 0.5 * _hash_unit_interval(feature, "llmesr_weight")
        vec[bucket] += np.float32(sign * magnitude)

    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= np.float32(norm)
    return vec


def _pca_projection(embeddings: Any, *, output_dim: int) -> Any:
    np = _numpy()
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape={embeddings.shape}")
    centered = embeddings.astype(np.float32, copy=True)
    centered -= centered.mean(axis=0, keepdims=True)

    if centered.shape[0] <= 1:
        projected = np.zeros((centered.shape[0], 0), dtype=np.float32)
    else:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        rank = min(output_dim, vt.shape[0])
        projected = centered @ vt[:rank].T

    if projected.shape[1] < output_dim:
        pad = np.zeros((projected.shape[0], output_dim - projected.shape[1]), dtype=np.float32)
        projected = np.concatenate([projected.astype(np.float32), pad], axis=1)
    elif projected.shape[1] > output_dim:
        projected = projected[:, :output_dim]

    row_norms = np.linalg.norm(projected, axis=1, keepdims=True)
    projected = np.divide(projected, np.maximum(row_norms, 1.0e-12), out=np.zeros_like(projected), where=row_norms > 0)
    return projected.astype(np.float32)


def generate_embeddings(adapter_dir: Path, *, embedding_dim: int, pca_dim: int, backend: str) -> dict[str, Any]:
    np = _numpy()
    if embedding_dim < 2:
        raise ValueError("--embedding_dim must be at least 2")
    if pca_dim != 64:
        raise ValueError("LLM-ESR expects pca64_itm_emb_np.pkl to have exactly 64 columns; keep --pca_dim=64.")
    if backend != "deterministic_text_hash":
        raise ValueError(f"Unsupported backend: {backend}")

    metadata_path = adapter_dir / "adapter_metadata.json"
    item_text_path = adapter_dir / "item_text_seed.csv"
    handled_dir = adapter_dir / "llm_esr" / "handled"
    if not metadata_path.exists():
        raise FileNotFoundError(f"adapter_metadata.json not found: {metadata_path}")
    if not item_text_path.exists():
        raise FileNotFoundError(f"item_text_seed.csv not found: {item_text_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_items = int(metadata.get("items", 0))
    rows = _read_csv(item_text_path)
    if len(rows) != expected_items:
        raise ValueError(f"item_text_seed.csv rows={len(rows)} but adapter metadata items={expected_items}")

    sorted_rows = sorted(rows, key=lambda row: int(float(row["llmesr_item_idx"])))
    embeddings = []
    covered_titles = 0
    for expected_idx, row in enumerate(sorted_rows, start=1):
        item_idx = int(float(row["llmesr_item_idx"]))
        if item_idx != expected_idx:
            raise ValueError(f"item_text_seed.csv is not contiguous at row {expected_idx}: found {item_idx}")
        text = str(row.get("embedding_text") or row.get("candidate_title") or row.get("item_id") or "").strip()
        if str(row.get("candidate_title", "")).strip():
            covered_titles += 1
        embeddings.append(_text_hash_embedding(text, dim=embedding_dim))

    itm_emb = np.vstack(embeddings).astype(np.float32) if embeddings else np.zeros((0, embedding_dim), dtype=np.float32)
    pca64_emb = _pca_projection(itm_emb, output_dim=pca_dim)

    handled_dir.mkdir(parents=True, exist_ok=True)
    itm_path = handled_dir / "itm_emb_np.pkl"
    pca_path = handled_dir / "pca64_itm_emb_np.pkl"
    with itm_path.open("wb") as fh:
        pickle.dump(itm_emb, fh)
    with pca_path.open("wb") as fh:
        pickle.dump(pca64_emb, fh)

    summary = {
        "adapter_dir": str(adapter_dir),
        "backend": backend,
        "artifact_class": "adapter_scaffold_embedding",
        "paper_result_ready": False,
        "items": expected_items,
        "embedding_dim": embedding_dim,
        "pca_dim": pca_dim,
        "title_seed_coverage": float(covered_titles / expected_items) if expected_items else 0.0,
        "itm_emb_path": str(itm_path),
        "pca64_emb_path": str(pca_path),
        "note": (
            "Deterministic text-hash embeddings make the LLM-ESR adapter runnable "
            "for protocol/scorer checks. Replace with upstream-compatible LLM item "
            "embeddings before marking a paper-project baseline as completed_result."
        ),
    }
    (adapter_dir / "llmesr_embedding_metadata.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = generate_embeddings(
        Path(args.adapter_dir).expanduser(),
        embedding_dim=args.embedding_dim,
        pca_dim=args.pca_dim,
        backend=args.backend,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
