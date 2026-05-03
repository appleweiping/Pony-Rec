# Blocker Resolution

## DeepSeek Tiny Live Test

- Scope: tiny live connectivity test only.
- Workdir: `/home/ajifang/projects/fresh/uncertainty-llm4rec`
- Environment: existing `.venv_lora` conda prefix environment; no rebuild, install, or uninstall was performed.
- Python path: `/home/ajifang/projects/fresh/uncertainty-llm4rec/.venv_lora/bin/python3.11`
- DeepSeek key: present after sourcing `.env`; not printed; placeholder check returned `False`.
- CUDA preflight: `torch 2.4.1+cu124`, CUDA available, GPU `NVIDIA GeForce RTX 4090`.

Command run:

```bash
PYTHON=.venv_lora/bin/python3.11 bash scripts/test_deepseek_live.sh
```

Result:

- Success: `True`
- DeepSeek model used: `deepseek-v4-flash`
- Raw response output path: `outputs/deepseek_live_test/raw_responses.jsonl`
- Parsed response output path: `outputs/deepseek_live_test/parsed_responses.jsonl`
- Combined output path: `outputs/deepseek_live_test/deepseek_live_results.json`
- Manifest path: `outputs/deepseek_live_test/manifest.json`
- Latency logging: enabled and recorded per request.
- Token usage logging: enabled and recorded per request.
- Output count: `3` raw response rows and `3` parsed response rows.

Script fix:

- `scripts/test_deepseek_live.sh` already used `${PYTHON:-python}` and did not need a Python-entrypoint fix.
- The first live attempt failed before any API call because importing the backend path required the `openai` SDK, which then required missing `pydantic`.
- Because package installation was explicitly disallowed, the script was patched to use a direct `httpx` call to DeepSeek `/chat/completions`.
- The patched script still reads `configs/backends/deepseek_v4_flash.yaml`, uses `deepseek-v4-flash`, parses responses with `parse_pointwise_output`, records config hash, writes raw and parsed artifacts, and writes a manifest.

Not run:

- No DeepSeek multi-domain inference.
- No RecBole.
- No LoRA.
- No full experiments.
