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
        repetition_penalty: float = 1.0,
        # Keep 0 for ASIN-like alphanumeric IDs: HF n-gram repetition rules can corrupt short repeated substrings.
        no_repeat_ngram_size: int = 0,
        batch_size: int = 1,
        use_chat_template: bool = False,
        enable_thinking: bool | None = None,
        stop_at_json_end: bool = False,
        stop_strings: list[str] | None = None,
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
        self.repetition_penalty = float(repetition_penalty)
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)
        self.batch_size = max(1, int(batch_size))
        self.use_chat_template = bool(use_chat_template)
        self.enable_thinking = enable_thinking
        self.stop_at_json_end = bool(stop_at_json_end)
        self.stop_strings = [str(x) for x in (stop_strings or []) if str(x)]
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

    def _format_prompt(self, prompt: str) -> str:
        tokenizer = self._tokenizer
        if (
            self.use_chat_template
            and tokenizer is not None
            and getattr(tokenizer, "chat_template", None)
        ):
            template_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if self.enable_thinking is not None:
                template_kwargs["enable_thinking"] = bool(self.enable_thinking)
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    **template_kwargs,
                )
            except TypeError:
                template_kwargs.pop("enable_thinking", None)
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    **template_kwargs,
                )
        return prompt

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
        repetition_penalty = float(kwargs.get("repetition_penalty", self.repetition_penalty))
        no_repeat_ngram_size = int(kwargs.get("no_repeat_ngram_size", self.no_repeat_ngram_size))
        stop_at_json_end = bool(kwargs.get("stop_at_json_end", self.stop_at_json_end))
        stop_strings = [str(x) for x in kwargs.get("stop_strings", self.stop_strings) if str(x)]
        results: list[GenerationResponse] = []
        for start_idx in range(0, len(requests), self.batch_size):
            batch = requests[start_idx : start_idx + self.batch_size]
            prompts = [self._format_prompt(r.prompt) for r in batch]
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
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            decoded = self._tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            if stop_at_json_end:
                decoded = [_truncate_to_first_json_object_text(t) for t in decoded]
            if stop_strings:
                decoded = [_truncate_by_stop_strings(t, stop_strings) for t in decoded]
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


def _truncate_by_stop_strings(text: str, stop_strings: list[str]) -> str:
    raw = str(text or "")
    best = len(raw)
    for stop in stop_strings:
        i = raw.find(stop)
        if i >= 0:
            best = min(best, i)
    return raw[:best].strip()


def _truncate_to_first_json_object_text(text: str) -> str:
    raw = str(text or "")
    start = raw.find("{")
    if start < 0:
        return raw.strip()
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1].strip()
    return raw.strip()
