from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse a completed LLM-ESR item embedding matrix for the matching "
            "LLM2Rec adapter package. Both adapters sort the same item ids "
            "lexically, so this avoids a second expensive Qwen encoding pass."
        )
    )
    parser.add_argument("--llmesr_adapter_dir", required=True)
    parser.add_argument("--llm2rec_adapter_dir", required=True)
    parser.add_argument("--llm2rec_repo_dir", default="")
    parser.add_argument("--save_info", default="pony_qwen3_8b")
    parser.add_argument("--output_path", default=None)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _validate_item_order(llmesr_adapter_dir: Path, llm2rec_adapter_dir: Path) -> tuple[list[str], dict[str, Any]]:
    llmesr_map = _read_csv(llmesr_adapter_dir / "item_id_map.csv")
    llm2rec_map = _read_csv(llm2rec_adapter_dir / "item_id_map.csv")
    llmesr_items = [row["item_id"] for row in sorted(llmesr_map, key=lambda row: int(float(row["llmesr_item_idx"])))]
    llm2rec_items = [row["item_id"] for row in sorted(llm2rec_map, key=lambda row: int(float(row["llm2rec_item_idx"])))]
    if llmesr_items != llm2rec_items:
        raise ValueError("LLM-ESR and LLM2Rec adapter item order differs; cannot safely reuse embeddings.")

    metadata_path = llm2rec_adapter_dir / "adapter_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_items = int(metadata.get("items", len(llm2rec_items)))
    if len(llm2rec_items) != expected_items:
        raise ValueError(f"LLM2Rec item map rows={len(llm2rec_items)} but metadata items={expected_items}")
    return llm2rec_items, metadata


def reuse_embeddings(
    llmesr_adapter_dir: Path,
    llm2rec_adapter_dir: Path,
    *,
    llm2rec_repo_dir: Path | None = None,
    save_info: str = "pony_qwen3_8b",
    output_path: Path | None = None,
) -> dict[str, Any]:
    import numpy as np

    items, metadata = _validate_item_order(llmesr_adapter_dir, llm2rec_adapter_dir)
    source_path = llmesr_adapter_dir / "llm_esr" / "handled" / "itm_emb_np.pkl"
    if not source_path.exists():
        raise FileNotFoundError(f"LLM-ESR embedding pickle not found: {source_path}")

    item_embeddings = np.asarray(_load_pickle(source_path), dtype=np.float32)
    if item_embeddings.ndim != 2:
        raise ValueError(f"Expected 2D LLM-ESR embeddings, got shape={item_embeddings.shape}")
    if item_embeddings.shape[0] != len(items):
        raise ValueError(f"Embedding rows={item_embeddings.shape[0]} but item map rows={len(items)}")

    padding = np.zeros((1, item_embeddings.shape[1]), dtype=np.float32)
    padded_embeddings = np.concatenate([padding, item_embeddings], axis=0)

    output_path = output_path or (llm2rec_adapter_dir / "llm2rec_item_embeddings.npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, padded_embeddings)

    dataset_alias = str(metadata.get("dataset_alias", "")).strip()
    upstream_item_embedding_path = ""
    if llm2rec_repo_dir is not None:
        target = llm2rec_repo_dir / "item_info" / dataset_alias / f"{save_info}_title_item_embs.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        if output_path.resolve() != target.resolve():
            shutil.copy2(output_path, target)
        upstream_item_embedding_path = str(target)

    summary = {
        "adapter_dir": str(llm2rec_adapter_dir),
        "dataset_alias": dataset_alias,
        "backend": "reuse_llmesr_hf_mean_pool",
        "source_llmesr_adapter_dir": str(llmesr_adapter_dir),
        "source_embedding_path": str(source_path),
        "artifact_class": "llm2rec_style_text_embedding",
        "paper_result_ready": False,
        "items": len(items),
        "embedding_rows": int(padded_embeddings.shape[0]),
        "embedding_dim": int(padded_embeddings.shape[1]),
        "padding_row_zero": True,
        "output_path": str(output_path),
        "upstream_item_embedding_path": upstream_item_embedding_path,
        "save_info": save_info,
        "note": (
            "This LLM2Rec-style embedding artifact reuses the same Qwen mean-pooled "
            "item vectors generated for the paired LLM-ESR adapter package."
        ),
    }
    _write_json(llm2rec_adapter_dir / "llm2rec_embedding_metadata.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = reuse_embeddings(
        Path(args.llmesr_adapter_dir).expanduser(),
        Path(args.llm2rec_adapter_dir).expanduser(),
        llm2rec_repo_dir=Path(args.llm2rec_repo_dir).expanduser() if args.llm2rec_repo_dir else None,
        save_info=args.save_info,
        output_path=Path(args.output_path).expanduser() if args.output_path else None,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
