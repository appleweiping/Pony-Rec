from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_dumps(config).encode("utf-8")).hexdigest()[:16]


def stable_int_hash(*parts: Any) -> int:
    payload = "||".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def git_commit_or_unknown(repo_root: str | Path = ".") -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def result_metadata(
    *,
    dataset: str,
    domain: str,
    split: str,
    seed: int,
    method: str,
    backend: str,
    model: str,
    prompt_id: str,
    config: dict[str, Any],
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "domain": domain,
        "split": split,
        "seed": int(seed),
        "method": method,
        "backend": backend,
        "model": model,
        "prompt_id": prompt_id,
        "config_hash": config_hash(config),
        "git_commit": git_commit_or_unknown(repo_root),
        "timestamp": utc_timestamp(),
    }
