from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from src.utils.research_artifacts import config_hash, git_commit_or_unknown, utc_timestamp


def backend_type_from_name(name: str) -> str:
    normalized = str(name or "").lower()
    if normalized == "mock":
        return "mock"
    if normalized in {"deepseek", "openai", "api", "openai_compatible"}:
        return "api"
    if normalized in {"hf", "local_hf", "hf_local", "transformers"}:
        return "local"
    if normalized in {"lora", "hf_lora", "peft"}:
        return "lora"
    return "unknown"


def is_paper_result(run_type: str, backend_type: str) -> bool:
    return str(run_type).lower() == "full" and str(backend_type).lower() in {"api", "local"}


def build_manifest(
    *,
    config: dict[str, Any],
    dataset: str = "unknown",
    domain: str = "unknown",
    raw_data_paths: list[str] | None = None,
    processed_data_paths: list[str] | None = None,
    method: str = "unknown",
    backend: str = "unknown",
    model: str = "unknown",
    prompt_template: str = "unknown",
    seed: int | None = None,
    candidate_size: int | None = None,
    calibration_source: str | None = None,
    command: list[str] | None = None,
    api_key_env: str | None = None,
    mock_data_used: bool | None = None,
) -> dict[str, Any]:
    backend_type = backend_type_from_name(backend)
    run_type = str(config.get("run_type") or ("smoke" if backend_type == "mock" else "pilot")).lower()
    return {
        "git_commit": git_commit_or_unknown("."),
        "config_hash": config_hash(config),
        "dataset": dataset,
        "domain": domain,
        "raw_data_paths": raw_data_paths or [],
        "processed_data_paths": processed_data_paths or [],
        "method": method,
        "backend": backend,
        "backend_type": backend_type,
        "model": model,
        "prompt_template": prompt_template,
        "seed": seed,
        "candidate_size": candidate_size,
        "calibration_source": calibration_source,
        "timestamp": utc_timestamp(),
        "command": command or sys.argv,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "api_key_used": bool(api_key_env and os.getenv(api_key_env)),
        "api_key_env": api_key_env,
        "mock_data_used": bool(mock_data_used) if mock_data_used is not None else backend_type == "mock",
        "run_type": run_type,
        "is_paper_result": is_paper_result(run_type, backend_type),
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
