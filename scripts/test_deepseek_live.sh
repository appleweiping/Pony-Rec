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
import json
import os
from pathlib import Path
from time import perf_counter

import httpx
import yaml

from src.prompts.parsers import parse_pointwise_output
from src.utils.research_artifacts import config_hash, utc_timestamp

out_dir = Path("outputs/deepseek_live_test")
out_dir.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load(Path("configs/backends/deepseek_v4_flash.yaml").read_text())
model = str(cfg["model"])
connection = cfg.get("connection", {})
generation = cfg.get("generation", {})
base_url = str(connection.get("base_url", "https://api.deepseek.com")).rstrip("/")
timeout = float(connection.get("timeout", 60))
api_key = os.environ["DEEPSEEK_API_KEY"]
prompts = [
    "Return JSON: {\"recommend\":\"yes\",\"confidence\":0.7,\"reason\":\"tiny live test\"}",
    "For a user who likes science fiction, recommend yes/no for a space movie. Return JSON with recommend, confidence, reason.",
    "Return JSON only: recommend no with confidence 0.6.",
]

raw_path = out_dir / "raw_responses.jsonl"
parsed_path = out_dir / "parsed_responses.jsonl"
combined_path = out_dir / "deepseek_live_results.json"
manifest_path = out_dir / "manifest.json"

rows = []
raw_rows = []
parsed_rows = []
errors = []

with httpx.Client(timeout=timeout) as client:
    for idx, prompt in enumerate(prompts):
        request_id = f"live_{idx}"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(generation.get("temperature", 0.0)),
            "top_p": float(generation.get("top_p", 1.0)),
            "max_tokens": int(generation.get("max_tokens", 800)),
        }
        started = perf_counter()
        timestamp = utc_timestamp()
        error = None
        response_json = None
        raw_text = ""
        usage = None
        status_code = None
        try:
            response = client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            status_code = response.status_code
            response.raise_for_status()
            response_json = response.json()
            raw_text = response_json["choices"][0]["message"]["content"]
            usage = response_json.get("usage")
        except Exception as exc:
            error = repr(exc)
            errors.append({"request_id": request_id, "error": error, "status_code": status_code})
        latency_seconds = perf_counter() - started
        parsed = parse_pointwise_output(raw_text).to_dict() if raw_text else None
        row = {
            "timestamp": timestamp,
            "request_id": request_id,
            "raw_response": raw_text,
            "parsed_response": parsed,
            "token_usage": usage,
            "latency_seconds": latency_seconds,
            "model_name": model,
            "config_hash": config_hash(cfg),
            "retry_count": 0,
            "cache_hit": False,
            "status_code": status_code,
            "error": error,
        }
        rows.append(row)
        raw_rows.append(
            {
                "timestamp": timestamp,
                "request_id": request_id,
                "model_name": model,
                "status_code": status_code,
                "raw_response": raw_text,
                "api_response": response_json,
                "token_usage": usage,
                "latency_seconds": latency_seconds,
                "error": error,
            }
        )
        parsed_rows.append(
            {
                "timestamp": timestamp,
                "request_id": request_id,
                "model_name": model,
                "parsed_response": parsed,
                "token_usage": usage,
                "latency_seconds": latency_seconds,
                "error": error,
            }
        )

combined_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
raw_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in raw_rows) + "\n", encoding="utf-8")
parsed_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in parsed_rows) + "\n", encoding="utf-8")
manifest = {
    "timestamp": utc_timestamp(),
    "command": "PYTHON=.venv_lora/bin/python3.11 bash scripts/test_deepseek_live.sh",
    "python_path": os.environ.get("PYTHON", "python"),
    "backend": cfg.get("backend"),
    "model": model,
    "config_path": "configs/backends/deepseek_v4_flash.yaml",
    "config_hash": config_hash(cfg),
    "raw_response_output_path": str(raw_path),
    "parsed_response_output_path": str(parsed_path),
    "combined_output_path": str(combined_path),
    "manifest_path": str(manifest_path),
    "latency_logging": True,
    "token_logging": True,
    "success": not errors and all(row["parsed_response"] for row in rows),
    "errors": errors,
}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"DeepSeek live model: {model}")
print(f"Raw responses: {raw_path}")
print(f"Parsed responses: {parsed_path}")
print(f"Manifest: {manifest_path}")
print(f"Success: {manifest['success']}")
if errors:
    raise SystemExit(1)
PY
