from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from typing import Any

from src.backends.base import GenerationRequest, GenerationResponse


class MockBackend:
    """Deterministic backend for tests and smoke runs only."""

    backend_name = "mock"
    model_name = "mock-deterministic"

    def __init__(self, *, latency_seconds: float = 0.0) -> None:
        self.latency_seconds = float(latency_seconds)

    @staticmethod
    def _candidate_ids(prompt: str) -> list[str]:
        ids = re.findall(r"item_id=([A-Za-z0-9_.:-]+)", prompt)
        if ids:
            return list(dict.fromkeys(ids))
        ids = re.findall(r"\bI[0-9A-Za-z_.:-]+\b", prompt)
        return list(dict.fromkeys(ids))

    @staticmethod
    def _confidence(prompt: str) -> float:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return 0.55 + (int(digest[:4], 16) % 40) / 100.0

    def _text(self, prompt: str) -> str:
        candidate_ids = self._candidate_ids(prompt)
        confidence = min(self._confidence(prompt), 0.99)
        if candidate_ids:
            ranked = sorted(candidate_ids, key=lambda item_id: hashlib.sha256((prompt + item_id).encode()).hexdigest())
            return json.dumps(
                {
                    "ranked_item_ids": ranked,
                    "topk_item_ids": ranked[: min(10, len(ranked))],
                    "confidence": round(confidence, 3),
                    "reason": "deterministic smoke response",
                }
            )
        recommend = "yes" if confidence >= 0.72 else "no"
        return json.dumps({"recommend": recommend, "confidence": round(confidence, 3), "reason": "mock"})

    async def agenerate(self, request: GenerationRequest, **kwargs: Any) -> GenerationResponse:
        start = time.perf_counter()
        if self.latency_seconds > 0:
            await asyncio.sleep(self.latency_seconds)
        return GenerationResponse(
            request_id=request.request_id,
            prompt=request.prompt,
            raw_text=self._text(request.prompt),
            backend=self.backend_name,
            model=self.model_name,
            latency_seconds=time.perf_counter() - start,
            usage={"prompt_tokens": len(request.prompt.split()), "completion_tokens": 16, "total_tokens": 16},
            raw_response={"mock": True},
        )

    async def abatch_generate(
        self,
        requests: list[GenerationRequest],
        **kwargs: Any,
    ) -> list[GenerationResponse]:
        return [await self.agenerate(request, **kwargs) for request in requests]
