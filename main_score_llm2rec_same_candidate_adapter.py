from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score exact same-candidate rows from an LLM2Rec adapter package "
            "using an upstream HappyPointer/LLM2Rec seqrec checkpoint. The "
            "output schema is source_event_id,user_id,item_id,score."
        )
    )
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--llm2rec_repo_dir", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--item_embedding_path", required=True)
    parser.add_argument("--model", choices=["SASRec", "GRU4Rec"], default="SASRec")
    parser.add_argument("--output_scores_path", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=10)
    parser.add_argument("--embedding_padding", choices=["auto", "already", "prepend"], default="auto")
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--allow_partial_checkpoint", action="store_true")
    return parser.parse_args()


def _numpy():
    import numpy as np

    return np


def _torch():
    import torch

    return torch


def _yaml():
    import yaml

    return yaml


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


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Upstream config file not found: {path}")
    payload = _yaml().safe_load(path.read_text(encoding="utf-8"))
    return dict(payload or {})


def _load_upstream_config(
    llm2rec_repo_dir: Path,
    *,
    model: str,
    item_count: int,
    max_seq_length: int,
    hidden_size: int | None,
    dropout: float | None,
) -> dict[str, Any]:
    seqrec_dir = llm2rec_repo_dir / "seqrec"
    config: dict[str, Any] = {}
    config.update(_load_yaml(seqrec_dir / "default.yaml"))
    config.update(_load_yaml(seqrec_dir / "models" / model / "config.yaml"))
    config.update(
        {
            "model": model,
            "dataset": "pony_same_candidate_adapter",
            "item_num": item_count,
            "select_pool": [1, item_count + 1],
            "eos_token": item_count + 1,
            "max_seq_length": max_seq_length,
            "use_ddp": False,
        }
    )
    if hidden_size is not None:
        config["hidden_size"] = hidden_size
    if dropout is not None:
        config["dropout"] = dropout
    return config


def _resolve_device(device: str) -> Any:
    torch = _torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_item_embeddings(path: Path, *, expected_items: int, padding_mode: str) -> Any:
    np = _numpy()
    if not path.exists():
        raise FileNotFoundError(f"LLM2Rec item embedding .npy not found: {path}")
    matrix = np.asarray(np.load(path), dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D item embeddings, got shape={matrix.shape}")
    if padding_mode in {"auto", "already"} and matrix.shape[0] == expected_items + 1:
        return matrix
    if padding_mode in {"auto", "prepend"} and matrix.shape[0] == expected_items:
        pad = np.zeros((1, matrix.shape[1]), dtype=np.float32)
        return np.concatenate([pad, matrix], axis=0)
    raise ValueError(
        f"Item embedding rows={matrix.shape[0]} do not match adapter items={expected_items}. "
        "Expected items+1 with padding row 0, or pass --embedding_padding prepend for item-only matrices."
    )


def _load_state_dict(checkpoint_path: Path, *, device: Any) -> dict[str, Any]:
    torch = _torch()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"LLM2Rec seqrec checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    cleaned = {}
    for key, value in payload.items():
        cleaned[key[7:] if str(key).startswith("module.") else key] = value
    return cleaned


def _load_state_dict_with_external_item_embedding(
    checkpoint_path: Path,
    *,
    device: Any,
    expected_items: int,
    item_embeddings: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_dict = _load_state_dict(checkpoint_path, device=device)
    missing_large_keys = [
        key
        for key in ("item_embedding.weight", "model.item_embedding.weight")
        if key not in state_dict
    ]
    injected_keys: list[str] = []
    if "item_embedding.weight" not in state_dict:
        torch = _torch()
        state_dict["item_embedding.weight"] = torch.tensor(item_embeddings, dtype=torch.float32, device=device)
        injected_keys.append("item_embedding.weight")
    return state_dict, {
        "external_item_embedding_injected_keys": injected_keys,
        "checkpoint_missing_externalized_item_embedding_keys": missing_large_keys,
        "external_item_embedding_rows": int(item_embeddings.shape[0]),
        "external_item_embedding_expected_rows": int(expected_items + 1),
    }


def _load_upstream_model(
    *,
    llm2rec_repo_dir: Path,
    model_name: str,
    config: dict[str, Any],
    item_embeddings: Any,
    checkpoint_path: Path,
    device: Any,
    allow_partial_checkpoint: bool,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    repo_text = str(llm2rec_repo_dir.resolve())
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    from seqrec.models import GRU4Rec, SASRec

    model_cls = {"SASRec": SASRec, "GRU4Rec": GRU4Rec}[model_name]
    pretrained = torch.tensor(item_embeddings, dtype=torch.float32, device=device)
    model = model_cls(config, pretrained).to(device)
    state_dict, external_info = _load_state_dict_with_external_item_embedding(
        checkpoint_path,
        device=device,
        expected_items=int(config["item_num"]),
        item_embeddings=item_embeddings,
    )
    load_result = model.load_state_dict(state_dict, strict=not allow_partial_checkpoint)
    model.eval()
    return model, {
        "missing_checkpoint_keys": list(getattr(load_result, "missing_keys", [])),
        "unexpected_checkpoint_keys": list(getattr(load_result, "unexpected_keys", [])),
        **external_info,
    }


def _event_sort_key(row: dict[str, str]) -> int:
    value = _as_int(row.get("llm2rec_test_row_idx"))
    if value is None:
        raise ValueError("same_candidate_events.csv has a missing llm2rec_test_row_idx")
    return value


def _candidate_sort_key(row: dict[str, str]) -> tuple[int, int]:
    event_idx = _as_int(row.get("llm2rec_test_row_idx"))
    candidate_idx = _as_int(row.get("candidate_index"))
    if event_idx is None:
        raise ValueError("candidate_items_mapped.csv has a missing llm2rec_test_row_idx")
    return event_idx, candidate_idx if candidate_idx is not None else 0


def _candidate_groups(candidate_rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    groups: dict[int, list[dict[str, str]]] = {}
    for row in sorted(candidate_rows, key=_candidate_sort_key):
        event_idx = _as_int(row.get("llm2rec_test_row_idx"))
        if event_idx is None:
            raise ValueError("candidate_items_mapped.csv contains an unmapped candidate event")
        groups.setdefault(event_idx, []).append(row)
    return groups


def _history_indices(event_row: dict[str, str], *, max_seq_length: int) -> list[int]:
    indices = [_as_int(part) for part in str(event_row.get("history_item_indices", "")).split()]
    history = [idx for idx in indices if idx is not None and idx > 0]
    if not history:
        raise ValueError(f"Empty history for source_event_id={event_row.get('source_event_id')!r}")
    return history[-max_seq_length:]


def _padded_batch(event_rows: list[dict[str, str]], *, max_seq_length: int, device: Any) -> dict[str, Any]:
    torch = _torch()
    padded = []
    lengths = []
    for row in event_rows:
        history = _history_indices(row, max_seq_length=max_seq_length)
        lengths.append(len(history))
        padded.append(history + [0] * (max_seq_length - len(history)))
    return {
        "item_seqs": torch.tensor(padded, dtype=torch.long, device=device),
        "seq_lengths": torch.tensor(lengths, dtype=torch.long, device=device),
    }


def _score_event_candidates(model: Any, batch: dict[str, Any], candidate_groups: dict[int, list[dict[str, str]]]) -> list[dict[str, Any]]:
    torch = _torch()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        state_hidden = model.get_representation(batch)
        if state_hidden.ndim == 1:
            state_hidden = state_hidden.unsqueeze(0)
        item_embeddings = model.get_all_embeddings(state_hidden.device)
        for batch_idx, event_idx in enumerate(batch["event_indices"]):
            candidates = candidate_groups.get(event_idx, [])
            if not candidates:
                raise ValueError(f"No candidates found for llm2rec_test_row_idx={event_idx}")
            item_indices = []
            for row in candidates:
                item_idx = _as_int(row.get("llm2rec_item_idx"))
                if item_idx is None:
                    raise ValueError("candidate_items_mapped.csv contains an unmapped item")
                item_indices.append(item_idx)
            index_tensor = torch.tensor(item_indices, dtype=torch.long, device=state_hidden.device)
            candidate_emb = item_embeddings.index_select(0, index_tensor)
            scores = torch.matmul(candidate_emb, state_hidden[batch_idx].view(-1, 1)).view(-1).detach().cpu().tolist()
            for row, score in zip(candidates, scores):
                score_value = float(score)
                if not math.isfinite(score_value):
                    raise RuntimeError(f"Non-finite score for source_event_id={row.get('source_event_id')}")
                rows.append(
                    {
                        "source_event_id": _text(row.get("source_event_id")),
                        "user_id": _text(row.get("user_id")),
                        "item_id": _text(row.get("item_id")),
                        "score": score_value,
                    }
                )
    return rows


def score_adapter(
    adapter_dir: Path,
    *,
    llm2rec_repo_dir: Path,
    checkpoint_path: Path,
    item_embedding_path: Path,
    model_name: str = "SASRec",
    output_scores_path: Path | None = None,
    device: str = "auto",
    batch_size: int = 128,
    max_seq_length: int = 10,
    embedding_padding: str = "auto",
    hidden_size: int | None = None,
    dropout: float | None = None,
    allow_partial_checkpoint: bool = False,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if max_seq_length <= 0:
        raise ValueError("--max_seq_length must be positive.")
    metadata_path = adapter_dir / "adapter_metadata.json"
    events_path = adapter_dir / "same_candidate_events.csv"
    candidate_path = adapter_dir / "candidate_items_mapped.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"adapter_metadata.json not found: {metadata_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"same_candidate_events.csv not found: {events_path}")
    if not candidate_path.exists():
        raise FileNotFoundError(f"candidate_items_mapped.csv not found: {candidate_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_items = int(metadata.get("items", 0))
    expected_candidates = int(metadata.get("candidate_rows", 0))
    events = sorted(_read_csv(events_path), key=_event_sort_key)
    candidates = _read_csv(candidate_path)
    groups = _candidate_groups(candidates)
    if len(candidates) != expected_candidates:
        raise ValueError(f"candidate rows={len(candidates)} but adapter metadata candidate_rows={expected_candidates}")

    resolved_device = _resolve_device(device)
    item_embeddings = _load_item_embeddings(item_embedding_path, expected_items=expected_items, padding_mode=embedding_padding)
    config = _load_upstream_config(
        llm2rec_repo_dir,
        model=model_name,
        item_count=expected_items,
        max_seq_length=max_seq_length,
        hidden_size=hidden_size,
        dropout=dropout,
    )
    model, checkpoint_info = _load_upstream_model(
        llm2rec_repo_dir=llm2rec_repo_dir,
        model_name=model_name,
        config=config,
        item_embeddings=item_embeddings,
        checkpoint_path=checkpoint_path,
        device=resolved_device,
        allow_partial_checkpoint=allow_partial_checkpoint,
    )

    scored_rows: list[dict[str, Any]] = []
    for start in range(0, len(events), batch_size):
        event_batch = events[start : start + batch_size]
        batch = _padded_batch(event_batch, max_seq_length=max_seq_length, device=resolved_device)
        batch["event_indices"] = [_event_sort_key(row) for row in event_batch]
        scored_rows.extend(_score_event_candidates(model, batch, groups))

    if output_scores_path is None:
        output_scores_path = adapter_dir / "llm2rec_same_candidate_scores.csv"
    _write_csv(scored_rows, output_scores_path, ["source_event_id", "user_id", "item_id", "score"])

    score_coverage_rate = float(len(scored_rows) / expected_candidates) if expected_candidates else 0.0
    summary = {
        "adapter_dir": str(adapter_dir),
        "baseline_name": "llm2rec",
        "artifact_class": "upstream_llm2rec_same_candidate_score",
        "paper_result_ready": score_coverage_rate == 1.0,
        "upstream_repo": "https://github.com/HappyPointer/LLM2Rec",
        "model": model_name,
        "checkpoint_path": str(checkpoint_path),
        "item_embedding_path": str(item_embedding_path),
        "embedding_rows_after_padding": int(item_embeddings.shape[0]),
        "embedding_dim": int(item_embeddings.shape[1]),
        "max_seq_length": max_seq_length,
        "candidate_rows": expected_candidates,
        "scored_rows": len(scored_rows),
        "score_coverage_rate": score_coverage_rate,
        "output_scores_path": str(output_scores_path),
        **checkpoint_info,
        "note": (
            "Scores are exact same-candidate logits from an upstream LLM2Rec "
            "seqrec model/checkpoint. Import as same_schema_external_baseline "
            "only after confirming the checkpoint and embeddings are produced by "
            "the intended LLM2Rec upstream run."
        ),
    }
    (adapter_dir / "llm2rec_same_candidate_score_metadata.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv([summary], adapter_dir / "llm2rec_same_candidate_score_summary.csv", list(summary.keys()))
    return summary


def main() -> None:
    args = parse_args()
    output_scores_path = Path(args.output_scores_path).expanduser() if args.output_scores_path else None
    summary = score_adapter(
        Path(args.adapter_dir).expanduser(),
        llm2rec_repo_dir=Path(args.llm2rec_repo_dir).expanduser(),
        checkpoint_path=Path(args.checkpoint_path).expanduser(),
        item_embedding_path=Path(args.item_embedding_path).expanduser(),
        model_name=args.model,
        output_scores_path=output_scores_path,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        embedding_padding=args.embedding_padding,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        allow_partial_checkpoint=args.allow_partial_checkpoint,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
