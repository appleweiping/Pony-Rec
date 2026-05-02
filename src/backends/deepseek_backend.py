from __future__ import annotations

from pathlib import Path
from typing import Any

from src.backends.openai_compatible_backend import OpenAICompatibleBackend


DEEPSEEK_MODEL_ALIASES = {
    "flash": "deepseek-v4-flash",
    "pro": "deepseek-v4-pro",
}


class DeepSeekBackend(OpenAICompatibleBackend):
    def __init__(
        self,
        *,
        model_name: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        api_key_env: str = "DEEPSEEK_API_KEY",
        thinking_mode: bool | None = None,
        cache_dir: str | Path | None = None,
        raw_response_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_model = DEEPSEEK_MODEL_ALIASES.get(model_name, model_name)
        extra_body = dict(kwargs.pop("extra_body", {}) or {})
        if thinking_mode is not None:
            # DeepSeek's OpenAI-compatible surface evolves; keep this configurable
            # instead of baking a deprecated model split into the backend.
            extra_body["thinking"] = {"enabled": bool(thinking_mode)}
        super().__init__(
            backend_name="deepseek",
            model_name=resolved_model,
            api_key_env=api_key_env,
            base_url=base_url,
            cache_dir=cache_dir,
            raw_response_dir=raw_response_dir,
            extra_body=extra_body,
            **kwargs,
        )
