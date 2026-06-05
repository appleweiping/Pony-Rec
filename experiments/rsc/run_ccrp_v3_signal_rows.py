"""Generate full-scale C-CRP signal rows from same-candidate ranking tasks.

This runner is intentionally separate from ``run_ccrp_v3_domain.py`` so the
paper-critical signal-row recovery path can emit recomputable uncertainty
inputs without changing the historical score-only runner surface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SIGNAL_SCHEMA_VERSION = "ccrp_v3_signal_rows.2026-06-06"
SIGNAL_NUMERIC_FIELDS = (
    "relevance_probability",
    "calibrated_relevance_probability",
    "evidence_support",
    "counterevidence_strength",
)


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    return max(0.0, min(1.0, parsed))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def build_signal_prompt(history: list[str], candidate_title: str, candidate_text: str = "") -> str:
    hist_block = "\n".join(f"- {item}" for item in history[-5:])
    meta = candidate_text[:240] if candidate_text else ""
    desc_line = f"\nDescription: {meta}" if meta else ""
    return (
        "You are an expert recommendation system.\n\n"
        f"User purchase history (most recent first):\n{hist_block}\n\n"
        f"Candidate item:\nTitle: {candidate_title}{desc_line}\n\n"
        "Estimate whether this candidate is the user's next purchase. Return calibrated, "
        "task-grounded signals only; do not rank the whole candidate set.\n\n"
        "Return ONLY JSON with numeric values in [0, 1]: "
        '{"relevance_probability": 0.0, '
        '"calibrated_relevance_probability": 0.0, '
        '"evidence_support": 0.0, '
        '"counterevidence_strength": 0.0, '
        '"reason": "one short sentence"}'
    )


def _json_object_from_text(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _regex_float(text: str, field: str) -> float | None:
    pattern = rf'"?{re.escape(field)}"?\s*[:=]\s*([01](?:\.\d+)?)'
    match = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
    if not match:
        return None
    return clamp01(match.group(1))


def parse_signal_response(text: str) -> dict[str, Any]:
    payload = _json_object_from_text(text)
    out: dict[str, Any] = {}
    parse_success = True
    for field in SIGNAL_NUMERIC_FIELDS:
        if field in payload:
            out[field] = clamp01(payload.get(field))
        else:
            fallback = _regex_float(text, field)
            if fallback is None:
                parse_success = False
                break
            out[field] = fallback

    if not parse_success:
        score = _regex_float(text, "relevance_probability")
        if score is None:
            generic = re.search(r"\b(0\.\d+|1\.0|0|1)\b", str(text or ""))
            score = clamp01(generic.group(1)) if generic else 0.0
        out = {
            "relevance_probability": score,
            "calibrated_relevance_probability": score,
            "evidence_support": 0.0,
            "counterevidence_strength": 1.0,
        }

    reason = payload.get("reason", "") if isinstance(payload, dict) else ""
    out["reason"] = str(reason).replace("\n", " ")[:240]
    out["parse_success"] = parse_success
    return out


def iter_task_prompts(records: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    prompts: list[str] = []
    meta_rows: list[dict[str, Any]] = []
    for rec in records:
        history = [str(item) for item in rec.get("history", [])]
        candidate_titles = rec.get("candidate_titles", [])
        candidate_texts = rec.get("candidate_texts") or [""] * len(candidate_titles)
        candidate_ids = rec.get("candidate_item_ids", [])
        if len(candidate_titles) != len(candidate_ids):
            raise ValueError(f"candidate title/id count mismatch for user {rec.get('user_id')}")
        for idx, (title, item_id) in enumerate(zip(candidate_titles, candidate_ids)):
            text = candidate_texts[idx] if idx < len(candidate_texts) else ""
            prompts.append(build_signal_prompt(history, str(title), str(text or "")))
            meta_rows.append(
                {
                    "source_event_id": str(rec.get("source_event_id", rec.get("user_id", ""))),
                    "user_id": str(rec.get("user_id", "")),
                    "candidate_item_id": str(item_id),
                    "item_id": str(item_id),
                    "candidate_idx": idx,
                }
            )
    return prompts, meta_rows


def validate_generation_count(*, expected_rows: int, prompt_count: int, meta_count: int, result_count: int) -> None:
    counts = {
        "expected_rows": int(expected_rows),
        "prompt_count": int(prompt_count),
        "meta_count": int(meta_count),
        "result_count": int(result_count),
    }
    if len(set(counts.values())) != 1:
        raise ValueError(f"signal-row generation count mismatch: {counts}")


def write_signal_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "source_event_id",
        "user_id",
        "candidate_item_id",
        "item_id",
        "candidate_idx",
        "relevance_probability",
        "calibrated_relevance_probability",
        "evidence_support",
        "counterevidence_strength",
        "reason",
        "parse_success",
        "signal_schema_version",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate C-CRP recomputable signal rows with vLLM.")
    parser.add_argument("--data", required=True, help="ranking_task.jsonl path for one split.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--domain", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--n_users", type=int, default=0)
    parser.add_argument("--model", default="/home/ajifang/models/Qwen/Qwen3-8B")
    parser.add_argument("--gpu_mem", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--chunk_users", type=int, default=5000)
    parser.add_argument("--expected_candidates_per_event", type=int, default=101)
    parser.add_argument("--max_parse_failure_rate", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    records = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.n_users and args.n_users < len(records):
        records = records[: args.n_users]
    if not records:
        raise ValueError("No ranking records loaded.")

    prompts, meta_rows = iter_task_prompts(records)
    expected_rows = len(records) * int(args.expected_candidates_per_event)
    if len(prompts) != expected_rows:
        raise ValueError(f"candidate-grid row mismatch: expected {expected_rows}, got {len(prompts)}")

    # Import lazily so unit tests can exercise parsing without requiring vLLM.
    from src.llm.vllm_backend import VLLMBackend

    backend = VLLMBackend(
        model_name_or_path=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        enable_prefix_caching=True,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    n_candidates = int(args.expected_candidates_per_event)
    chunk_size = max(1, int(args.chunk_users)) * n_candidates
    start = time.time()
    raw_results: list[dict[str, Any]] = []
    for idx in range(0, len(prompts), chunk_size):
        chunk = prompts[idx : idx + chunk_size]
        raw_results.extend(backend.batch_generate(chunk))
        done = min(idx + chunk_size, len(prompts))
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0.0
        print(f"[{done}/{len(prompts)}] {rate:.0f} prompts/s", flush=True)

    validate_generation_count(
        expected_rows=expected_rows,
        prompt_count=len(prompts),
        meta_count=len(meta_rows),
        result_count=len(raw_results),
    )

    rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for meta, result in zip(meta_rows, raw_results):
        parsed = parse_signal_response(str(result.get("raw_text", "")))
        row = {**meta, **parsed, "signal_schema_version": SIGNAL_SCHEMA_VERSION}
        rows.append(row)
        if not parsed["parse_success"]:
            failure_rows.append({**meta, "raw_text": str(result.get("raw_text", ""))[:1000]})

    parse_failure_rate = len(failure_rows) / len(rows) if rows else 1.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_path = output_dir / f"{args.split or 'split'}_ccrp_signal_rows.csv"
    write_signal_rows_csv(signal_path, rows)
    if failure_rows:
        with (output_dir / f"{args.split or 'split'}_parse_failures.jsonl").open("w", encoding="utf-8") as handle:
            for row in failure_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    provenance = {
        "status_label": "ccrp_v3_recomputable_signal_rows_generated",
        "artifact_class": "paper_critical_signal_rows",
        "schema_version": SIGNAL_SCHEMA_VERSION,
        "domain": args.domain,
        "split": args.split,
        "git_commit": git_commit(),
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "model": args.model,
        "gpu_mem": args.gpu_mem,
        "max_model_len": args.max_model_len,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "chunk_users": args.chunk_users,
        "expected_candidates_per_event": args.expected_candidates_per_event,
        "expected_signal_rows": expected_rows,
        "prompt_count": len(prompts),
        "meta_row_count": len(meta_rows),
        "raw_result_count": len(raw_results),
        "n_events": len(records),
        "n_signal_rows": len(rows),
        "signal_rows_path": str(signal_path),
        "signal_rows_sha256": sha256_file(signal_path),
        "parse_failure_count": len(failure_rows),
        "parse_failure_rate": parse_failure_rate,
        "max_parse_failure_rate": args.max_parse_failure_rate,
        "inference_time_s": time.time() - start,
        "leakage_guard": (
            "Prompt does not include positive_item_index or labels. Hyperparameter/model "
            "selection must be done on validation outputs only before test rows are used."
        ),
    }
    write_json(output_dir / f"{args.split or 'split'}_ccrp_signal_rows_provenance.json", provenance)
    print(json.dumps({"ok": parse_failure_rate <= args.max_parse_failure_rate, **provenance}, indent=2))
    if parse_failure_rate > args.max_parse_failure_rate:
        raise SystemExit(
            f"parse failure rate {parse_failure_rate:.6f} exceeds max {args.max_parse_failure_rate:.6f}"
        )


if __name__ == "__main__":
    main()
