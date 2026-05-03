from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.backends.base import GenerationRequest, GenerationResponse
from src.backends.deepseek_backend import DeepSeekBackend
from src.backends.hf_local_backend import HFLocalBackend
from src.backends.mock_backend import MockBackend
from src.backends.openai_compatible_backend import OpenAICompatibleBackend


def build_backend(config: dict[str, Any]):
    name = str(config.get("backend", config.get("backend_name", "mock"))).lower()
    generation = config.get("generation", {}) or {}
    runtime = config.get("runtime", {}) or {}
    connection = config.get("connection", {}) or {}
    cache = config.get("cache", {}) or {}
    if name == "mock":
        return MockBackend(latency_seconds=float(runtime.get("latency_seconds", 0.0)))
    if name == "deepseek":
        return DeepSeekBackend(
            model_name=str(config.get("model", config.get("model_name", "deepseek-v4-flash"))),
            base_url=str(connection.get("base_url", config.get("base_url", "https://api.deepseek.com"))),
            api_key_env=str(connection.get("api_key_env", config.get("api_key_env", "DEEPSEEK_API_KEY"))),
            timeout=float(connection.get("timeout", config.get("timeout", 60.0))),
            temperature=float(generation.get("temperature", 0.0)),
            max_tokens=int(generation.get("max_tokens", generation.get("max_new_tokens", 512))),
            top_p=float(generation.get("top_p", 1.0)),
            max_concurrency=int(runtime.get("max_concurrency", 8)),
            max_retries=int(runtime.get("max_retries", 6)),
            retry_backoff_seconds=float(runtime.get("retry_backoff_seconds", 1.0)),
            cache_dir=cache.get("cache_dir"),
            raw_response_dir=cache.get("raw_response_dir"),
            thinking_mode=generation.get("thinking_mode"),
        )
    if name in {"openai_compatible", "api"}:
        return OpenAICompatibleBackend(
            backend_name=str(config.get("provider", name)),
            model_name=str(config["model"]),
            base_url=connection.get("base_url", config.get("base_url")),
            api_key_env=str(connection.get("api_key_env", config.get("api_key_env"))),
            timeout=float(connection.get("timeout", config.get("timeout", 60.0))),
            temperature=float(generation.get("temperature", 0.0)),
            max_tokens=int(generation.get("max_tokens", generation.get("max_new_tokens", 512))),
            top_p=float(generation.get("top_p", 1.0)),
            max_concurrency=int(runtime.get("max_concurrency", 8)),
            cache_dir=cache.get("cache_dir"),
            raw_response_dir=cache.get("raw_response_dir"),
        )
    if name in {"hf_local", "local_hf", "hf"}:
        return HFLocalBackend(
            model_name_or_path=str(config["model_name_or_path"]),
            adapter_path=config.get("adapter_path"),
            model_name=config.get("model"),
            device_map=runtime.get("device_map", "auto"),
            dtype=str(runtime.get("dtype", "auto")),
            load_in_4bit=bool(runtime.get("load_in_4bit", False)),
            trust_remote_code=bool(runtime.get("trust_remote_code", False)),
            max_new_tokens=int(generation.get("max_new_tokens", 512)),
            temperature=float(generation.get("temperature", 0.0)),
            repetition_penalty=float(generation.get("repetition_penalty", 1.0)),
            no_repeat_ngram_size=int(generation.get("no_repeat_ngram_size", 0)),
            batch_size=int(runtime.get("batch_size", 1)),
            use_chat_template=bool(runtime.get("use_chat_template", False)),
            enable_thinking=runtime.get("enable_thinking"),
            stop_at_json_end=bool(generation.get("stop_at_json_end", False)),
            stop_strings=generation.get("stop_strings"),
        )
    raise ValueError(f"Unsupported backend: {name}")


def build_backend_from_yaml(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return build_backend(yaml.safe_load(f) or {})


__all__ = [
    "DeepSeekBackend",
    "GenerationRequest",
    "GenerationResponse",
    "HFLocalBackend",
    "MockBackend",
    "OpenAICompatibleBackend",
    "build_backend",
    "build_backend_from_yaml",
]
