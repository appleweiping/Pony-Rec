from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate LLM2Rec-compatible item embedding .npy files from an "
            "adapter package. This supports a same-backbone Qwen3-8B embedding "
            "baseline without requiring upstream llm2vec/flash-attn extraction."
        )
    )
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument(
        "--backend",
        choices=["hf_mean_pool", "sentence_transformers", "deterministic_text_hash"],
        default="hf_mean_pool",
    )
    parser.add_argument("--model_name", default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_text_chars", type=int, default=1200)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--hf_device_map", default="")
    parser.add_argument("--deterministic_dim", type=int, default=384)
    parser.add_argument("--llm2rec_repo_dir", default="")
    parser.add_argument("--save_info", default="pony_qwen3_8b")
    parser.add_argument("--output_path", default=None)
    return parser.parse_args()


def _numpy():
    import numpy as np

    return np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolve_device(device: str) -> str | None:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return None


def _torch_dtype_arg(torch_dtype: str) -> Any:
    if torch_dtype == "auto":
        return "auto"
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[torch_dtype]


def _model_input_device(model: Any) -> Any:
    for parameter in model.parameters():
        return parameter.device
    return "cpu"


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
            "Use the server environment where the local Qwen3-8B model can be loaded."
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
            vectors.append(pooled.float().cpu())
            print(f"[hf_mean_pool] encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    if not vectors:
        np = _numpy()
        return np.zeros((0, 0), dtype=np.float32)
    return torch.cat(vectors, dim=0).float().numpy()


def _load_sentence_transformer_model(model_name: str, device: str | None) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError("sentence-transformers is required for --backend sentence_transformers.") from exc
    kwargs = {"device": device} if device else {}
    return SentenceTransformer(model_name, **kwargs)


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
    return features or ["empty"]


def _text_hash_embedding(text: str, *, dim: int) -> Any:
    np = _numpy()
    vec = np.zeros(dim, dtype=np.float32)
    for feature in _token_features(text):
        bucket, sign = _signed_hash_bucket(feature, dim=dim, salt="llm2rec_text")
        magnitude = 0.75 + 0.5 * _hash_unit_interval(feature, "llm2rec_weight")
        vec[bucket] += np.float32(sign * magnitude)
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= np.float32(norm)
    return vec


def _embedding_texts(adapter_dir: Path, *, max_text_chars: int) -> tuple[list[str], list[dict[str, str]], dict[str, Any]]:
    item_text_path = adapter_dir / "item_text_seed.csv"
    metadata_path = adapter_dir / "adapter_metadata.json"
    if not item_text_path.exists():
        raise FileNotFoundError(f"item_text_seed.csv not found: {item_text_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"adapter_metadata.json not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = sorted(_read_csv(item_text_path), key=lambda row: int(float(row["llm2rec_item_idx"])))
    expected_items = int(metadata.get("items", len(rows)))
    if len(rows) != expected_items:
        raise ValueError(f"item_text_seed.csv rows={len(rows)} but adapter metadata items={expected_items}")
    texts = [
        str(row.get("embedding_text") or row.get("candidate_text") or row.get("candidate_title") or row.get("item_id") or "")[
            :max_text_chars
        ]
        for row in rows
    ]
    return texts, rows, metadata


def _default_output_path(adapter_dir: Path) -> Path:
    return adapter_dir / "llm2rec_item_embeddings.npy"


def _copy_to_upstream_item_info(
    output_path: Path,
    *,
    llm2rec_repo_dir: Path,
    dataset_alias: str,
    save_info: str,
) -> Path:
    target = llm2rec_repo_dir / "item_info" / dataset_alias / f"{save_info}_title_item_embs.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    if output_path.resolve() != target.resolve():
        shutil.copy2(output_path, target)
    return target


def generate_embeddings(
    adapter_dir: Path,
    *,
    backend: str = "hf_mean_pool",
    model_name: str = "",
    batch_size: int = 8,
    device: str = "auto",
    max_text_chars: int = 1200,
    max_length: int = 256,
    trust_remote_code: bool = False,
    torch_dtype: str = "auto",
    hf_device_map: str = "",
    deterministic_dim: int = 384,
    llm2rec_repo_dir: Path | None = None,
    save_info: str = "pony_qwen3_8b",
    output_path: Path | None = None,
) -> dict[str, Any]:
    np = _numpy()
    if batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    texts, rows, metadata = _embedding_texts(adapter_dir, max_text_chars=max_text_chars)
    if backend in {"hf_mean_pool", "sentence_transformers"} and not model_name:
        raise ValueError("--model_name is required for hf_mean_pool or sentence_transformers.")

    resolved_device = _resolve_device(device)
    if backend == "hf_mean_pool":
        item_embeddings = _encode_hf_mean_pool(
            texts,
            model_name=model_name,
            batch_size=batch_size,
            device=resolved_device,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            hf_device_map=hf_device_map,
        )
    elif backend == "sentence_transformers":
        model = _load_sentence_transformer_model(model_name, resolved_device)
        item_embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    elif backend == "deterministic_text_hash":
        if deterministic_dim < 2:
            raise ValueError("--deterministic_dim must be at least 2.")
        item_embeddings = np.vstack([_text_hash_embedding(text, dim=deterministic_dim) for text in texts]).astype(np.float32)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    item_embeddings = np.asarray(item_embeddings, dtype=np.float32)
    if item_embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape={item_embeddings.shape}")

    # Upstream LLM2Rec seqrec expects row 0 to be padding/null and item ids to be 1-based.
    padding = np.zeros((1, item_embeddings.shape[1]), dtype=np.float32)
    padded_embeddings = np.concatenate([padding, item_embeddings], axis=0)

    output_path = output_path or _default_output_path(adapter_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, padded_embeddings)

    dataset_alias = str(metadata.get("dataset_alias", "")).strip()
    upstream_item_embedding_path = ""
    if llm2rec_repo_dir is not None:
        upstream_path = _copy_to_upstream_item_info(
            output_path,
            llm2rec_repo_dir=llm2rec_repo_dir,
            dataset_alias=dataset_alias,
            save_info=save_info,
        )
        upstream_item_embedding_path = str(upstream_path)

    title_coverage = sum(1 for row in rows if str(row.get("candidate_title", "")).strip()) / len(rows) if rows else 0.0
    text_coverage = sum(1 for text in texts if text.strip()) / len(texts) if texts else 0.0
    summary = {
        "adapter_dir": str(adapter_dir),
        "dataset_alias": dataset_alias,
        "backend": backend,
        "model_name": model_name,
        "device": resolved_device or "",
        "artifact_class": "llm2rec_style_text_embedding",
        "paper_result_ready": False,
        "items": len(rows),
        "embedding_rows": int(padded_embeddings.shape[0]),
        "embedding_dim": int(padded_embeddings.shape[1]),
        "padding_row_zero": True,
        "title_seed_coverage": title_coverage,
        "embedding_text_coverage": text_coverage,
        "output_path": str(output_path),
        "upstream_item_embedding_path": upstream_item_embedding_path,
        "save_info": save_info,
        "note": (
            "This is a same-backbone LLM2Rec-style item embedding artifact. "
            "It supports Qwen3-8B + SASRec comparison without upstream CSFT/IEM; "
            "label results as LLM2Rec-style Qwen3-8B embedding baseline, not "
            "official LLM2Rec CSFT/IEM reproduction."
        ),
    }
    _write_json(summary, adapter_dir / "llm2rec_embedding_metadata.json")
    return summary


def main() -> None:
    args = parse_args()
    summary = generate_embeddings(
        Path(args.adapter_dir).expanduser(),
        backend=args.backend,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        max_text_chars=args.max_text_chars,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
        hf_device_map=args.hf_device_map,
        deterministic_dim=args.deterministic_dim,
        llm2rec_repo_dir=Path(args.llm2rec_repo_dir).expanduser() if args.llm2rec_repo_dir else None,
        save_info=args.save_info,
        output_path=Path(args.output_path).expanduser() if args.output_path else None,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
