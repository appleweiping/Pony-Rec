from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate LLM-ESR item embedding pickle files with a real "
            "sentence-transformers backend. The scorer remains an adapter "
            "protocol scorer unless paired with an upstream-compatible LLM-ESR run."
        )
    )
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pca_dim", type=int, default=64)
    parser.add_argument("--max_text_chars", type=int, default=1200)
    return parser.parse_args()


def _numpy():
    import numpy as np

    return np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _pca_projection(embeddings: Any, *, output_dim: int) -> Any:
    np = _numpy()
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
    return projected[:, :output_dim].astype(np.float32)


def _resolve_device(device: str) -> str | None:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return None


def _load_model(model_name: str, device: str | None) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            "sentence-transformers is required for true text embeddings. Install it "
            "or pass a Python environment where it is already available."
        ) from exc
    kwargs = {"device": device} if device else {}
    return SentenceTransformer(model_name, **kwargs)


def generate_sentence_embeddings(
    adapter_dir: Path,
    *,
    model_name: str,
    batch_size: int,
    device: str = "auto",
    pca_dim: int = 64,
    max_text_chars: int = 1200,
) -> dict[str, Any]:
    np = _numpy()
    if pca_dim != 64:
        raise ValueError("LLM-ESR expects pca64_itm_emb_np.pkl to have exactly 64 columns; keep --pca_dim=64.")
    item_text_path = adapter_dir / "item_text_seed.csv"
    metadata_path = adapter_dir / "adapter_metadata.json"
    handled_dir = adapter_dir / "llm_esr" / "handled"
    if not item_text_path.exists():
        raise FileNotFoundError(f"item_text_seed.csv not found: {item_text_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    rows = sorted(_read_csv(item_text_path), key=lambda row: int(float(row["llmesr_item_idx"])))
    expected_items = int(metadata.get("items", len(rows)))
    if len(rows) != expected_items:
        raise ValueError(f"item_text_seed.csv rows={len(rows)} but adapter metadata items={expected_items}")
    texts = [
        str(row.get("embedding_text") or row.get("candidate_text") or row.get("candidate_title") or row.get("item_id") or "")[
            :max_text_chars
        ]
        for row in rows
    ]
    resolved_device = _resolve_device(device)
    model = _load_model(model_name, resolved_device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    itm_emb = np.asarray(embeddings, dtype=np.float32)
    pca64_emb = _pca_projection(itm_emb, output_dim=pca_dim)

    handled_dir.mkdir(parents=True, exist_ok=True)
    itm_path = handled_dir / "itm_emb_np.pkl"
    pca_path = handled_dir / "pca64_itm_emb_np.pkl"
    with itm_path.open("wb") as fh:
        pickle.dump(itm_emb, fh)
    with pca_path.open("wb") as fh:
        pickle.dump(pca64_emb, fh)

    title_coverage = sum(1 for row in rows if str(row.get("candidate_title", "")).strip()) / len(rows) if rows else 0.0
    text_coverage = sum(1 for text in texts if text.strip()) / len(texts) if texts else 0.0
    summary = {
        "adapter_dir": str(adapter_dir),
        "backend": "sentence_transformers",
        "model_name": model_name,
        "device": resolved_device or "",
        "artifact_class": "adapter_text_embedding",
        "paper_result_ready": False,
        "items": len(rows),
        "embedding_dim": int(itm_emb.shape[1]) if itm_emb.ndim == 2 else "",
        "pca_dim": pca_dim,
        "title_seed_coverage": title_coverage,
        "embedding_text_coverage": text_coverage,
        "itm_emb_path": str(itm_path),
        "pca64_emb_path": str(pca_path),
        "note": (
            "These are real text embeddings for the LLM-ESR adapter files, but "
            "they are not by themselves a completed upstream LLM-ESR paper result."
        ),
    }
    (adapter_dir / "llmesr_embedding_metadata.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = generate_sentence_embeddings(
        Path(args.adapter_dir).expanduser(),
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        pca_dim=args.pca_dim,
        max_text_chars=args.max_text_chars,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
