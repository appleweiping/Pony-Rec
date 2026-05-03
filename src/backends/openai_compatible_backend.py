from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

try:
    from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
except ImportError:  # pragma: no cover - exercised when optional OpenAI deps are absent.
    AsyncOpenAI = None  # type: ignore[assignment]

    class APIConnectionError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    class RateLimitError(Exception):
        pass

from src.backends.base import GenerationRequest, GenerationResponse
from src.utils.research_artifacts import stable_json_dumps


def stable_cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def build_openai_chat_payload(
    *,
    prompt: str,
    model: str,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    response_format: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "top_p": float(top_p),
    }
    if response_format:
        payload["response_format"] = response_format
    if extra_body:
        payload["extra_body"] = extra_body
    return payload


def _usage_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}


def _response_to_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return None


class OpenAICompatibleBackend:
    def __init__(
        self,
        *,
        backend_name: str = "openai_compatible",
        model_name: str,
        api_key_env: str,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        timeout: float | None = 60.0,
        max_concurrency: int = 8,
        max_retries: int = 6,
        retry_backoff_seconds: float = 1.0,
        cache_dir: str | Path | None = None,
        raw_response_dir: str | Path | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Required API key env var is not set: {api_key_env}")
        self.backend_name = backend_name
        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.top_p = float(top_p)
        self.max_concurrency = max(1, int(max_concurrency))
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.raw_response_dir = Path(raw_response_dir) if raw_response_dir else None
        self.extra_body = extra_body or {}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.raw_response_dir:
            self.raw_response_dir.mkdir(parents=True, exist_ok=True)
        if AsyncOpenAI is None:
            raise ImportError("OpenAI-compatible backends require the optional openai dependency and its runtime deps")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def _request_payload(self, request: GenerationRequest, kwargs: dict[str, Any]) -> dict[str, Any]:
        response_format = kwargs.get("response_format") or request.response_format
        extra_body = dict(self.extra_body)
        if isinstance(kwargs.get("extra_body"), dict):
            extra_body.update(kwargs["extra_body"])
        return build_openai_chat_payload(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            model=str(kwargs.get("model_name", self.model_name)),
            temperature=float(kwargs.get("temperature", self.temperature)),
            max_tokens=int(kwargs.get("max_tokens", self.max_tokens)),
            top_p=float(kwargs.get("top_p", self.top_p)),
            response_format=response_format,
            extra_body=extra_body,
        )

    def _cache_path(self, payload: dict[str, Any]) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{stable_cache_key(payload)}.json"

    def _raw_path(self, request_id: str, payload: dict[str, Any]) -> Path | None:
        if not self.raw_response_dir:
            return None
        safe_id = request_id or stable_cache_key(payload)[:16]
        return self.raw_response_dir / f"{safe_id}.json"

    async def agenerate(self, request: GenerationRequest, **kwargs: Any) -> GenerationResponse:
        payload = self._request_payload(request, kwargs)
        cache_path = self._cache_path(payload)
        if cache_path and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached["cache_hit"] = True
            return GenerationResponse(**cached)

        retry_count = 0
        start = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.chat.completions.create(**payload)
                latency = time.perf_counter() - start
                choices = getattr(response, "choices", []) or []
                text = ""
                if choices:
                    message = getattr(choices[0], "message", None)
                    text = str(getattr(message, "content", "") or "")
                raw = _response_to_dict(response)
                result = GenerationResponse(
                    request_id=request.request_id,
                    prompt=request.prompt,
                    raw_text=text.strip(),
                    backend=self.backend_name,
                    model=str(getattr(response, "model", payload["model"])),
                    latency_seconds=latency,
                    usage=_usage_to_dict(getattr(response, "usage", None)),
                    raw_response=raw,
                    cache_hit=False,
                    retry_count=retry_count,
                )
                raw_path = self._raw_path(request.request_id, payload)
                if raw_path:
                    raw_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
                if cache_path:
                    cache_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")
                return result
            except Exception as exc:
                retryable = isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, TimeoutError))
                status_code = getattr(exc, "status_code", None)
                retryable = retryable or status_code == 429 or (
                    isinstance(status_code, int) and 500 <= status_code < 600
                )
                if not retryable or attempt >= self.max_retries:
                    return GenerationResponse(
                        request_id=request.request_id,
                        prompt=request.prompt,
                        raw_text="",
                        backend=self.backend_name,
                        model=payload["model"],
                        latency_seconds=time.perf_counter() - start,
                        usage={},
                        raw_response=None,
                        cache_hit=False,
                        retry_count=retry_count,
                        error=str(exc),
                    )
                retry_count += 1
                sleep_for = self.retry_backoff_seconds * (2**attempt) + random.random() * 0.25
                await asyncio.sleep(sleep_for)

        raise RuntimeError("unreachable")

    async def abatch_generate(
        self,
        requests: list[GenerationRequest],
        **kwargs: Any,
    ) -> list[GenerationResponse]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _guarded(request: GenerationRequest) -> GenerationResponse:
            async with semaphore:
                return await self.agenerate(request, **kwargs)

        return await asyncio.gather(*[_guarded(request) for request in requests])
