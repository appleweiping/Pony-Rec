from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    request_id: str = ""
    system_prompt: str | None = None
    response_format: dict[str, Any] | None = None


@dataclass
class GenerationResponse:
    request_id: str
    prompt: str
    raw_text: str
    backend: str
    model: str
    latency_seconds: float
    usage: dict[str, Any]
    raw_response: dict[str, Any] | None = None
    cache_hit: bool = False
    retry_count: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["usage"] = data.get("usage") or {}
        return data


class LLMBackend(Protocol):
    backend_name: str
    model_name: str

    async def agenerate(self, request: GenerationRequest, **kwargs: Any) -> GenerationResponse:
        ...

    async def abatch_generate(
        self,
        requests: list[GenerationRequest],
        **kwargs: Any,
    ) -> list[GenerationResponse]:
        ...
