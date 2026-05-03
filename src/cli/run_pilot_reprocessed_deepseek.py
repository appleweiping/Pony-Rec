"""DeepSeek API pilot on reprocess_processed_source outputs (httpx; no openai/pydantic)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
import pandas as pd
import yaml

from src.data.protocol import read_jsonl, write_jsonl
from src.prompts import candidate_block, get_prompt_template, history_block, parse_ranking_output
from src.uncertainty.interface import VerbalizedConfidenceEstimator
from src.utils.manifest import backend_type_from_name, build_manifest, is_paper_result, write_manifest
from src.utils.research_artifacts import config_hash, git_commit_or_unknown, utc_timestamp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run DeepSeek pilot inference on reprocessed candidate JSONL.")
    p.add_argument("--reprocess_dir", default="outputs/reprocessed_processed_source")
    p.add_argument("--output_root", default="outputs/pilots/deepseek_v4_flash_processed_20u_c19_seed42")
    p.add_argument("--backend_config", default="configs/backends/deepseek_v4_flash.yaml")
    p.add_argument(
        "--domains",
        nargs="*",
        default=["amazon_beauty", "amazon_books", "amazon_electronics", "amazon_movies"],
    )
    p.add_argument("--splits", nargs="*", default=["valid", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt_id", default="listwise_ranking_v1")
    p.add_argument("--run_type", default="pilot")
    p.add_argument("--method", default="llm_listwise")
    p.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Listwise slate size for parse_ranking_output. Default: infer from first row's candidate_item_ids length.",
    )
    return p.parse_args(argv)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_item_lookup(processed_dir: Path) -> dict[str, str]:
    path = processed_dir / "items.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path).fillna("")
    col = "item_id" if "item_id" in df.columns else None
    if col is None:
        for c in ["parent_asin", "asin", "movieId"]:
            if c in df.columns:
                col = c
                break
    if col is None:
        return {}
    text_col = "candidate_text" if "candidate_text" in df.columns else "title"
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        iid = str(row[col]).strip()
        out[iid] = str(row.get(text_col, "") or "").strip()
    return out


def _merge_item_texts(sample: dict[str, Any], base: dict[str, str]) -> dict[str, str]:
    out = dict(base)
    ids = [str(x) for x in sample.get("candidate_item_ids", [])]
    texts = sample.get("candidate_texts") or []
    titles = sample.get("candidate_titles") or []
    for idx, item_id in enumerate(ids):
        if item_id in out and str(out[item_id]).strip():
            continue
        if idx < len(texts) and str(texts[idx]).strip():
            out[item_id] = str(texts[idx])
        elif idx < len(titles) and str(titles[idx]).strip():
            out[item_id] = f"Title: {titles[idx]}"
    return out


def _build_ranking_prompt(sample: dict[str, Any], prompt_id: str, item_lookup: dict[str, str], topk: int) -> str:
    texts = _merge_item_texts(sample, item_lookup)
    template = get_prompt_template(prompt_id)
    history = history_block([str(x) for x in sample.get("history_item_ids", [])], texts)
    candidates = candidate_block([str(x) for x in sample.get("candidate_item_ids", [])], texts)
    allowed_list = [str(x) for x in sample.get("candidate_item_ids", [])]
    kwargs: dict[str, Any] = {
        "history_block": history,
        "candidate_block": candidates,
        "allowed_item_ids": ", ".join(allowed_list),
        "topk": topk,
    }
    if prompt_id == "listwise_ranking_json_lora":
        kwargs["allowed_item_ids_json"] = json.dumps(allowed_list, ensure_ascii=False)
    return template.render(**kwargs)


async def _one_request(
    client: httpx.AsyncClient,
    *,
    url: str,
    headers: dict[str, str],
    model: str,
    prompt: str,
    generation: dict[str, Any],
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None, float, str | None]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(generation.get("temperature", 0.0)),
        "top_p": float(generation.get("top_p", 1.0)),
        "max_tokens": int(generation.get("max_tokens", 800)),
    }
    if not bool(generation.get("thinking_mode", True)):
        payload["thinking"] = {"type": "disabled"}
    started = perf_counter()
    err: str | None = None
    usage = None
    raw_text = ""
    response_json = None
    try:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        response_json = resp.json()
        msg = (response_json.get("choices") or [{}])[0].get("message") or {}
        raw_text = str(msg.get("content") or "").strip()
        if not raw_text:
            raw_text = str(msg.get("reasoning_content") or "").strip()
        usage = response_json.get("usage")
    except Exception as exc:
        err = repr(exc)
    return raw_text, usage, response_json, perf_counter() - started, err


async def _run_split(
    *,
    samples: list[dict[str, Any]],
    backend_cfg: dict[str, Any],
    processed_dir: Path,
    prompt_id: str,
    topk: int,
    meta: dict[str, Any],
    concurrency: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    connection = backend_cfg.get("connection", {}) or {}
    generation = backend_cfg.get("generation", {}) or {}
    base_url = str(connection.get("base_url", "https://api.deepseek.com")).rstrip("/")
    timeout = float(connection.get("timeout", 60))
    api_key_env = str(connection.get("api_key_env", "DEEPSEEK_API_KEY"))
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing API key env: {api_key_env}")
    model = str(backend_cfg.get("model", "deepseek-v4-flash"))
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    item_lookup = _load_item_lookup(processed_dir)
    estimator = VerbalizedConfidenceEstimator()
    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async with httpx.AsyncClient(timeout=timeout) as client:

        async def bound(sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            prompt = _build_ranking_prompt(sample, prompt_id, item_lookup, topk)
            async with sem:
                raw_text, usage, response_json, latency, err = await _one_request(
                    client,
                    url=url,
                    headers=headers,
                    model=model,
                    prompt=prompt,
                    generation=generation,
                )
            allowed = [str(x) for x in sample.get("candidate_item_ids", [])]
            parsed = parse_ranking_output(raw_text, allowed_item_ids=allowed, topk=topk)
            ranking = list(parsed.ranked_item_ids or [])
            predicted = ranking[0] if ranking else ""
            raw_conf = float(parsed.confidence or 0.0)
            est = estimator.estimate({"raw_confidence": raw_conf})
            target = str(sample.get("target_item_id", ""))
            raw_row = {
                "timestamp": utc_timestamp(),
                "request_id": f"{meta['split']}:{sample.get('user_id')}",
                "raw_response": raw_text,
                "api_response": response_json,
                "token_usage": usage,
                "latency_seconds": latency,
                "error": err,
                "model_name": model,
            }
            parsed_row = {
                "timestamp": utc_timestamp(),
                "user_id": sample.get("user_id"),
                "is_valid": parsed.is_valid,
                "invalid_output": parsed.invalid_output,
                "hallucinated_item": parsed.hallucinated_item,
                "duplicate_item": parsed.duplicate_item,
                "missing_confidence": parsed.missing_confidence,
                "output_not_in_candidate_set": parsed.output_not_in_candidate_set,
                "malformed_json": parsed.malformed_json,
                "repaired_json": parsed.repaired_json,
                "ranked_item_ids": ranking,
                "confidence": parsed.confidence,
                "error": err,
            }
            pred_row = {
                **meta,
                "timestamp": utc_timestamp(),
                "user_id": sample.get("user_id"),
                "history_length": len(sample.get("history_item_ids", [])),
                "candidate_item_ids": sample.get("candidate_item_ids", []),
                "candidate_popularity_counts": sample.get("candidate_popularity_counts", []),
                "candidate_popularity_buckets": sample.get("candidate_popularity_buckets", []),
                "predicted_ranking": ranking,
                "predicted_item_id": predicted,
                "target_item_id": target,
                "correctness": bool(predicted == target),
                "raw_confidence": raw_conf,
                "calibrated_confidence": None,
                "uncertainty_score": est.uncertainty_score,
                "uncertainty_estimator_name": est.estimator_name,
                "item_popularity_count": sample.get("target_popularity_count"),
                "item_popularity_bucket": sample.get("target_popularity_bucket"),
                "is_valid": parsed.is_valid,
                "hallucinated_item": parsed.hallucinated_item,
                "duplicate_item": parsed.duplicate_item,
                "missing_confidence": parsed.missing_confidence,
                "output_not_in_candidate_set": parsed.output_not_in_candidate_set,
                "backend": "deepseek",
                "model": model,
                "raw_response": raw_text,
                "usage": usage,
                "latency_seconds": latency,
                "retry_count": 0,
                "cache_hit": False,
                "error": err,
            }
            return raw_row, parsed_row, pred_row

        tasks = [bound(s) for s in samples]
        out = await asyncio.gather(*tasks)

    raw_rows, parsed_rows, pred_rows = [], [], []
    for r, p, pr in out:
        raw_rows.append(r)
        parsed_rows.append(p)
        pred_rows.append(pr)
    return raw_rows, parsed_rows, pred_rows


def _flatten(prefix: str, data: dict[str, Any], out: dict[str, Any]) -> None:
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            _flatten(name, value, out)
        else:
            out[name] = value


def _aggregate_metrics(output_root: Path, output_csv: Path) -> None:
    rows = []
    for path in sorted(output_root.rglob("eval/metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {"path": str(path)}
        _flatten("", data, row)
        rows.append(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_csv.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for r in rows for k in r})
    import csv

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args(None)
    reprocess_dir = Path(args.reprocess_dir)
    output_root = Path(args.output_root)
    backend_cfg = _load_yaml(Path(args.backend_config))
    runtime = backend_cfg.get("runtime", {}) or {}
    concurrency = int(runtime.get("max_concurrency", 8))
    pilot_config = {
        "run_type": args.run_type,
        "seed": args.seed,
        "method": args.method,
        "backend": backend_cfg,
        "reprocess_dir": str(reprocess_dir),
        "output_root": str(output_root),
        "prompt_id": args.prompt_id,
        "domains": list(args.domains),
        "splits": list(args.splits),
    }
    summary: dict[str, Any] = {
        "created_at": utc_timestamp(),
        "git_commit": git_commit_or_unknown("."),
        "config_hash": config_hash(pilot_config),
        "domains": {},
    }
    for domain in args.domains:
        processed_dir = Path("data/processed") / domain
        summary["domains"][domain] = {}
        for split in args.splits:
            in_path = reprocess_dir / domain / f"{split}_candidates.jsonl"
            if not in_path.exists():
                raise FileNotFoundError(f"Missing pilot input: {in_path}")
            samples = read_jsonl(in_path)
            if args.topk is not None:
                inference_topk = int(args.topk)
            elif samples:
                inference_topk = len(samples[0].get("candidate_item_ids") or [])
            else:
                inference_topk = 0
            if inference_topk <= 0:
                raise ValueError(f"No samples or empty candidate_item_ids in {in_path}")
            out_dir = output_root / domain / split
            pred_dir = out_dir / "predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "dataset": domain,
                "domain": domain,
                "split": split,
                "seed": int(args.seed),
                "method": args.method,
                "prompt_template_id": args.prompt_id,
                "config_hash": config_hash(pilot_config),
                "git_commit": git_commit_or_unknown("."),
                "run_type": str(args.run_type).lower(),
                "backend_type": backend_type_from_name("deepseek"),
                "is_paper_result": is_paper_result(str(args.run_type).lower(), backend_type_from_name("deepseek")),
            }
            raw_rows, parsed_rows, pred_rows = asyncio.run(
                _run_split(
                    samples=samples,
                    backend_cfg=backend_cfg,
                    processed_dir=processed_dir,
                    prompt_id=args.prompt_id,
                    topk=inference_topk,
                    meta=meta,
                    concurrency=concurrency,
                )
            )
            write_jsonl(raw_rows, pred_dir / "raw_responses.jsonl")
            write_jsonl(parsed_rows, pred_dir / "parsed_responses.jsonl")
            pred_path = pred_dir / "rank_predictions.jsonl"
            write_jsonl(pred_rows, pred_path)
            api_key_env = str((backend_cfg.get("connection") or {}).get("api_key_env", "DEEPSEEK_API_KEY"))
            write_manifest(
                out_dir / "manifest.json",
                build_manifest(
                    config=pilot_config,
                    dataset=domain,
                    domain=domain,
                    raw_data_paths=[],
                    processed_data_paths=[str(processed_dir), str(in_path)],
                    method=args.method,
                    backend="deepseek",
                    model=str(backend_cfg.get("model", "deepseek-v4-flash")),
                    prompt_template=args.prompt_id,
                    seed=int(args.seed),
                    candidate_size=len(samples[0]["candidate_item_ids"]) if samples else None,
                    calibration_source=None,
                    command=sys.argv,
                    api_key_env=api_key_env,
                    mock_data_used=False,
                ),
            )
            n = len(parsed_rows)
            invalid = sum(1 for r in parsed_rows if r.get("invalid_output"))
            conf_avail = sum(1 for r in parsed_rows if r.get("confidence") is not None)
            summary["domains"][domain][split] = {
                "n": n,
                "invalid_output_rate": invalid / n if n else float("nan"),
                "confidence_availability_rate": conf_avail / n if n else float("nan"),
                "errors": sum(1 for r in raw_rows if r.get("error")),
            }
            eval_dir = out_dir / "eval"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.cli.evaluate",
                    "--predictions_path",
                    str(pred_path),
                    "--output_dir",
                    str(eval_dir),
                    "--candidates_source_path",
                    str(in_path),
                ],
                check=True,
                cwd=str(Path.cwd()),
            )
    agg_path = output_root / "pilot_metrics_aggregate.csv"
    _aggregate_metrics(output_root, agg_path)
    summary_path = output_root / "pilot_run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[pilot] done output_root={output_root} summary={summary_path} aggregate={agg_path}")


if __name__ == "__main__":
    main()
