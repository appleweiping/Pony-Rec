from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.backends.base import GenerationRequest, GenerationResponse


class HFLocalBackend:
    backend_name = "hf_local"

    def __init__(
        self,
        *,
        model_name_or_path: str,
        adapter_path: str | None = None,
        model_name: str | None = None,
        device_map: str | dict[str, Any] | None = "auto",
        dtype: str = "auto",
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        batch_size: int = 1,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.adapter_path = adapter_path
        self.model_name = model_name or Path(model_name_or_path).name
        self.device_map = device_map
        self.dtype = dtype
        self.load_in_4bit = bool(load_in_4bit)
        self.trust_remote_code = bool(trust_remote_code)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.batch_size = max(1, int(batch_size))
        self._model = None
        self._tokenizer = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("HFLocalBackend requires torch and transformers.") from exc
        self._torch = torch
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            use_fast=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.dtype != "auto":
            dtype_lookup = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16}
            model_kwargs["torch_dtype"] = dtype_lookup.get(self.dtype, torch.float32)
        if self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **model_kwargs)
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    async def agenerate(self, request: GenerationRequest, **kwargs: Any) -> GenerationResponse:
        return (await self.abatch_generate([request], **kwargs))[0]

    async def abatch_generate(
        self,
        requests: list[GenerationRequest],
        **kwargs: Any,
    ) -> list[GenerationResponse]:
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._torch is not None
        max_new_tokens = int(kwargs.get("max_new_tokens", self.max_new_tokens))
        temperature = float(kwargs.get("temperature", self.temperature))
        results: list[GenerationResponse] = []
        for start_idx in range(0, len(requests), self.batch_size):
            batch = requests[start_idx : start_idx + self.batch_size]
            prompts = [r.prompt for r in batch]
            start = time.perf_counter()
            inputs = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=float(kwargs.get("top_p", self.top_p)),
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            decoded = self._tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            latency = (time.perf_counter() - start) / max(1, len(batch))
            for request, text in zip(batch, decoded):
                results.append(
                    GenerationResponse(
                        request_id=request.request_id,
                        prompt=request.prompt,
                        raw_text=text.strip(),
                        backend=self.backend_name,
                        model=self.model_name,
                        latency_seconds=latency,
                        usage={},
                    )
                )
        return results
