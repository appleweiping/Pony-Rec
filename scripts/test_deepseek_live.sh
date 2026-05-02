#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is not set; skipping live DeepSeek test."
  exit 0
fi

OUT_DIR="${1:-outputs/deepseek_live_test}"
mkdir -p "$OUT_DIR"

$PYTHON_BIN - <<'PY'
import asyncio
import json
from pathlib import Path

import yaml

from src.backends import GenerationRequest, build_backend
from src.prompts.parsers import parse_pointwise_output
from src.utils.research_artifacts import config_hash, utc_timestamp

out_dir = Path("outputs/deepseek_live_test")
out_dir.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load(Path("configs/backends/deepseek_v4_flash.yaml").read_text())
backend = build_backend(cfg)
prompts = [
    "Return JSON: {\"recommend\":\"yes\",\"confidence\":0.7,\"reason\":\"tiny live test\"}",
    "For a user who likes science fiction, recommend yes/no for a space movie. Return JSON with recommend, confidence, reason.",
    "Return JSON only: recommend no with confidence 0.6.",
]

async def main():
    responses = await backend.abatch_generate([
        GenerationRequest(prompt=prompt, request_id=f"live_{idx}") for idx, prompt in enumerate(prompts)
    ])
    rows = []
    for response in responses:
        parsed = parse_pointwise_output(response.raw_text).to_dict()
        rows.append({
            "timestamp": utc_timestamp(),
            "raw_response": response.raw_text,
            "parsed_response": parsed,
            "token_usage": response.usage,
            "latency_seconds": response.latency_seconds,
            "model_name": response.model,
            "config_hash": config_hash(cfg),
            "retry_count": response.retry_count,
            "cache_hit": response.cache_hit,
            "error": response.error,
        })
    (out_dir / "deepseek_live_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} DeepSeek live responses to {out_dir / 'deepseek_live_results.json'}")

asyncio.run(main())
PY
