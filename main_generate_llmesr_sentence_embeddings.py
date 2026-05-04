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
            "Generate LLM-ESR item embedding pickle files with real local text "
            "embedding backends. The scorer remains an adapter "
            "protocol scorer unless paired with an upstream-compatible LLM-ESR run."
        )
    )
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--backend", choices=["sentence_transformers", "hf_mean_pool"], default="sentence_transformers")
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pca_dim", type=int, default=64)
    parser.add_argument("--max_text_chars", type=int, default=1200)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--hf_device_map", default="")
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


def _load_sentence_transformer_model(model_name: str, device: str | None) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            "sentence-transformers is required for true text embeddings. Install it "
            "or pass a Python environment where it is already available."
        ) from exc
    kwargs = {"device": device} if device else {}
    return SentenceTransformer(model_name, **kwargs)


def _torch_dtype_arg(torch_dtype: str) -> Any:
    if torch_dtype == "auto":
        return "auto"
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[torch_dtype]


def _load_hf_model(
    model_name: str,
    *,
    device: str | None,
    trust_remote_code: bool,
    torch_dtype: str,
    hf_device_map: str,
) -> tuple[Any, Any]:
    try:
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "transformers and torch are required for --backend hf_mean_pool. "
            "Use a Python environment where the local Qwen/BGE/E5 model can be loaded."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": _torch_dtype_arg(torch_dtype),
    }
    if hf_device_map:
        load_kwargs["device_map"] = hf_device_map
    try:
        model = AutoModel.from_pretrained(model_name, **load_kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    if not hf_device_map and device:
        model.to(device)
    return tokenizer, model


def _model_input_device(model: Any) -> Any:
    for parameter in model.parameters():
        return parameter.device
    return "cpu"


def _encode_hf_mean_pool(
    texts: list[str],
    *,
    model_name: str,
    batch_size: int,
    device: str | None,
    max_length: int,
    trust_remote_code: bool,
    torch_dtype: str,
    hf_device_map: str,
) -> Any:
    import torch

    tokenizer, model = _load_hf_model(
        model_name,
        device=device,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        hf_device_map=hf_device_map,
    )
    input_device = _model_input_device(model)
    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            outputs = model(**encoded, output_hidden_states=True, return_dict=True)
            token_embeddings = getattr(outputs, "last_hidden_state", None)
            if token_embeddings is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if not hidden_states:
                    raise RuntimeError("HF model did not return last_hidden_state or hidden_states.")
                token_embeddings = hidden_states[-1]
            mask = encoded["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu())
            print(f"[hf_mean_pool] encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    if not vectors:
        np = _numpy()
        return np.zeros((0, 0), dtype=np.float32)
    return torch.cat(vectors, dim=0).numpy()


def generate_sentence_embeddings(
    adapter_dir: Path,
    *,
    backend: str = "sentence_transformers",
    model_name: str,
    batch_size: int,
    device: str = "auto",
    pca_dim: int = 64,
    max_text_chars: int = 1200,
    max_length: int = 256,
    trust_remote_code: bool = False,
    torch_dtype: str = "auto",
    hf_device_map: str = "",
) -> dict[str, Any]:
    np = _numpy()
    if pca_dim != 64:
        raise ValueError("LLM-ESR expects pca64_itm_emb_np.pkl to have exactly 64 columns; keep --pca_dim=64.")
    if batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
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
    if backend == "sentence_transformers":
        model = _load_sentence_transformer_model(model_name, resolved_device)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    elif backend == "hf_mean_pool":
        embeddings = _encode_hf_mean_pool(
            texts,
            model_name=model_name,
            batch_size=batch_size,
            device=resolved_device,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            hf_device_map=hf_device_map,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
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
        "backend": backend,
        "model_name": model_name,
        "device": resolved_device or "",
        "max_length": max_length if backend == "hf_mean_pool" else "",
        "hf_device_map": hf_device_map,
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
        backend=args.backend,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        pca_dim=args.pca_dim,
        max_text_chars=args.max_text_chars,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
        hf_device_map=args.hf_device_map,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
