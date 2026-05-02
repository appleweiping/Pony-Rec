from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.backends import GenerationRequest, build_backend
from src.data.protocol import read_jsonl, write_jsonl
from src.prompts import candidate_block, get_prompt_template, history_block, parse_pointwise_output, parse_ranking_output
from src.uncertainty.interface import VerbalizedConfidenceEstimator
from src.utils.manifest import backend_type_from_name, build_manifest, is_paper_result, write_manifest
from src.utils.research_artifacts import config_hash, git_commit_or_unknown, utc_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM recommendation inference.")
    parser.add_argument("--config", required=True, help="YAML experiment config.")
    parser.add_argument("--split", default=None, choices=["train", "valid", "test"])
    parser.add_argument("--input_path", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_item_texts(processed_dir: Path) -> dict[str, str]:
    items_path = processed_dir / "items.csv"
    if not items_path.exists():
        return {}
    items = pd.read_csv(items_path).fillna("")
    return {str(row.item_id): str(row.candidate_text) for row in items.itertuples(index=False)}


def _build_prompt(sample: dict[str, Any], prompt_id: str, item_texts: dict[str, str], topk: int) -> str:
    template = get_prompt_template(prompt_id)
    history = history_block([str(x) for x in sample.get("history_item_ids", [])], item_texts)
    candidates = candidate_block([str(x) for x in sample.get("candidate_item_ids", [])], item_texts)
    if template.task == "pointwise":
        candidate_id = str(sample["candidate_item_ids"][0])
        return template.render(
            history_block=history,
            candidate_item_id=candidate_id,
            candidate_text=item_texts.get(candidate_id, ""),
        )
    return template.render(
        history_block=history,
        candidate_block=candidates,
        allowed_item_ids=", ".join(str(x) for x in sample.get("candidate_item_ids", [])),
        topk=topk,
    )


async def _run(config: dict[str, Any], args: argparse.Namespace) -> None:
    dataset_cfg = config.get("dataset", {}) or {}
    inference_cfg = config.get("inference", {}) or {}
    backend_cfg = config.get("backend", {}) or {}
    split = args.split or str(inference_cfg.get("split", "test"))
    processed_dir = Path(str(dataset_cfg.get("processed_dir", "data/processed/unknown")))
    input_path = Path(args.input_path or inference_cfg.get("input_path") or processed_dir / f"{split}_candidates.jsonl")
    output_path = Path(
        args.output_path
        or inference_cfg.get("output_path")
        or Path(str(config.get("output_dir", "outputs/smoke"))) / "predictions" / f"{split}_raw.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = read_jsonl(input_path)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    existing: list[dict[str, Any]] = []
    if args.resume and output_path.exists():
        existing = read_jsonl(output_path)
        samples = samples[len(existing) :]
    item_texts = _load_item_texts(processed_dir)
    prompt_id = str(inference_cfg.get("prompt_id", "listwise_ranking_v1"))
    topk = int(inference_cfg.get("topk", 10))
    requests = []
    for sample in samples:
        prompt = _build_prompt(sample, prompt_id, item_texts, topk)
        requests.append(GenerationRequest(prompt=prompt, request_id=f"{split}:{sample['user_id']}"))
    backend = build_backend(backend_cfg)
    backend_name = getattr(backend, "backend_name", str(backend_cfg.get("backend", "unknown")))
    backend_type = backend_type_from_name(backend_name)
    run_type = str(config.get("run_type") or ("smoke" if backend_type == "mock" else "pilot")).lower()
    responses = await backend.abatch_generate(requests)
    estimator = VerbalizedConfidenceEstimator()
    records = list(existing)
    meta_common = {
        "dataset": dataset_cfg.get("dataset", dataset_cfg.get("dataset_name", "unknown")),
        "domain": dataset_cfg.get("domain", dataset_cfg.get("domain_name", "unknown")),
        "split": split,
        "seed": int(config.get("seed", dataset_cfg.get("seed", 42))),
        "method": str(config.get("method", "llm_listwise")),
        "prompt_template_id": prompt_id,
        "config_hash": config_hash(config),
        "git_commit": git_commit_or_unknown("."),
        "run_type": run_type,
        "backend_type": backend_type,
        "is_paper_result": is_paper_result(run_type, backend_type),
    }
    for sample, response in zip(samples, responses):
        task = get_prompt_template(prompt_id).task
        if task == "pointwise":
            parsed = parse_pointwise_output(response.raw_text)
            ranking = [str(sample["candidate_item_ids"][0])] if parsed.recommend == "yes" else []
        else:
            parsed = parse_ranking_output(
                response.raw_text,
                allowed_item_ids=[str(x) for x in sample.get("candidate_item_ids", [])],
                topk=topk,
            )
            ranking = parsed.ranked_item_ids or []
        predicted = ranking[0] if ranking else ""
        raw_confidence = parsed.confidence if parsed.confidence is not None else 0.0
        estimate = estimator.estimate({"raw_confidence": raw_confidence})
        target = str(sample.get("target_item_id", ""))
        records.append(
            {
                **meta_common,
                "timestamp": utc_timestamp(),
                "user_id": sample.get("user_id"),
                "history_length": sample.get("history_length", len(sample.get("history_item_ids", []))),
                "history_length_bucket": sample.get("history_length_bucket"),
                "candidate_item_ids": sample.get("candidate_item_ids", []),
                "candidate_popularity_counts": sample.get("candidate_popularity_counts", []),
                "candidate_popularity_buckets": sample.get("candidate_popularity_buckets", []),
                "predicted_ranking": ranking,
                "predicted_item_id": predicted,
                "target_item_id": target,
                "correctness": bool(predicted == target),
                "raw_confidence": raw_confidence,
                "calibrated_confidence": None,
                "uncertainty_score": estimate.uncertainty_score,
                "uncertainty_estimator_name": estimate.estimator_name,
                "item_popularity_count": sample.get("target_popularity_count"),
                "item_popularity_bucket": sample.get("target_popularity_bucket"),
                "is_valid": parsed.is_valid,
                "hallucinated_item": parsed.hallucinated_item,
                "duplicate_item": parsed.duplicate_item,
                "missing_confidence": parsed.missing_confidence,
                "output_not_in_candidate_set": parsed.output_not_in_candidate_set,
                "backend": response.backend,
                "model": response.model,
                "raw_response": response.raw_text,
                "usage": response.usage,
                "latency_seconds": response.latency_seconds,
                "retry_count": response.retry_count,
                "cache_hit": response.cache_hit,
                "error": response.error,
            }
        )
    write_jsonl(records, output_path)
    api_key_env = (backend_cfg.get("connection", {}) or {}).get("api_key_env") or backend_cfg.get("api_key_env")
    write_manifest(
        output_path.parent.parent / "manifest.json",
        build_manifest(
            config=config,
            dataset=str(meta_common["dataset"]),
            domain=str(meta_common["domain"]),
            raw_data_paths=[],
            processed_data_paths=[str(processed_dir)],
            method=str(meta_common["method"]),
            backend=backend_name,
            model=getattr(backend, "model_name", str(backend_cfg.get("model", "unknown"))),
            prompt_template=prompt_id,
            seed=int(meta_common["seed"]),
            candidate_size=len(samples[0].get("candidate_item_ids", [])) if samples else None,
            calibration_source=None,
            api_key_env=str(api_key_env) if api_key_env else None,
            mock_data_used=backend_type == "mock",
        ),
    )
    print(f"[infer] saved={output_path} rows={len(records)}")


def main() -> None:
    args = parse_args()
    asyncio.run(_run(_load_yaml(args.config), args))


if __name__ == "__main__":
    main()
