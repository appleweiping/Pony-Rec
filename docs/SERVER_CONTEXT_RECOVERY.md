# Server Context Recovery

Recovery run from:

```text
/home/ajifang/projects/fresh/uncertainty-llm4rec
```

## Repository

- Remote: `origin https://github.com/appleweiping/Pony-Rec.git`
- Branch: `main`
- Commit: `0bf878a`
- Git cleanliness before writing this report: clean (`git status --short` returned no paths)
- Top-level project directories verified: `configs`, `docs`, `scripts`, `src`, `tests`

## Environment

- Python: `Python 3.13.5`
- pip: `pip 26.1 from /home/ajifang/miniconda3/lib/python3.13/site-packages/pip (python 3.13)`
- GPU status after host-level verification: available.
- GPU: `NVIDIA GeForce RTX 4090`
- Driver: `570.211.01`
- CUDA shown by `nvidia-smi`: `12.8`

- DeepSeek key presence after sourcing `.env`: `True`
- DeepSeek key placeholder check: `False`
- DeepSeek key length: `35`
- Qwen3-8B local model: configured in `configs/model/qwen3_8b_local.yaml` as `/home/ajifang/models/Qwen/Qwen3-8B`; that directory exists and contains local safetensor shards.

## Stable LoRA Environment

- Existing `.venv` preserved for debugging; it was not modified.
- New environment path: `/home/ajifang/projects/fresh/uncertainty-llm4rec/.venv_lora`
- Environment type: conda prefix environment, not a standard `python -m venv` layout.
- `conda-meta`: present.
- `.venv_lora/bin/python3.11`: present.
- `.venv_lora/bin/activate`: absent; this is expected for this conda-prefix verification path.
- Python executable: `/home/ajifang/projects/fresh/uncertainty-llm4rec/.venv_lora/bin/python`
- Direct interpreter path verified: `/home/ajifang/projects/fresh/uncertainty-llm4rec/.venv_lora/bin/python3.11`
- Python version: `3.11.15`
- Environment creation note: `python3.11` and `python3.12` were not available on `PATH`, so `.venv_lora` was created as an isolated conda-prefix environment with Python 3.11 at the requested path.
- Torch version: `2.4.1+cu124`
- Torch CUDA version: `12.4`
- CUDA availability: `True` in host-level verification.
- GPU name from torch: `NVIDIA GeForce RTX 4090`
- CUDA matmul: passed (`256 x 256` CUDA matmul returned a CUDA tensor).
- Exact torch install command that produced the final working torch stack:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

- Note: the requested `cu121` command was attempted first, but the sandboxed verification could not access CUDA/NVML and triggered the scripted `cu124` retry. Host-level verification with unrestricted GPU access passed on the final `cu124` stack.
- Transformers stack:
  - `transformers==4.44.2`
  - `accelerate==0.34.2`
  - `peft==0.12.0`
- bitsandbytes: `0.43.3`
- bitsandbytes import: passed.
- Direct-path imports verified: `transformers`, `accelerate`, `peft`, and `bitsandbytes` all imported successfully through `.venv_lora/bin/python3.11`.
- `conda run -p /home/ajifang/projects/fresh/uncertainty-llm4rec/.venv_lora ...`: available and verified.
- `conda run` Python executable: `/home/ajifang/projects/fresh/uncertainty-llm4rec/.venv_lora/bin/python`
- `conda run` CUDA availability: `True`

## Pytest

Command:

```bash
python3 -m pytest
```

Result: failed before test collection because the active Python environment does not have `pytest` installed.

```text
/home/ajifang/miniconda3/bin/python3: No module named pytest
```

## Raw Data Paths Found

Fresh clone:

- No files found under `data/raw` in `/home/ajifang/projects/fresh/uncertainty-llm4rec`.

Other paths under `/home/ajifang/projects` matching raw-data locations:

- `/home/ajifang/projects/uncertainty-llm4rec-codex-apr12-preserve-local/data/raw/movielens_1m/ml-1m/users.dat`
- `/home/ajifang/projects/uncertainty-llm4rec-codex-apr12-preserve-local/data/raw/movielens_1m/ml-1m/ratings.dat`
- `/home/ajifang/projects/uncertainty-llm4rec-codex-apr12-preserve-local/data/raw/movielens_1m/ml-1m/movies.dat`
- `/home/ajifang/projects/uncertainty-llm4rec-codex-apr12-preserve-local/data/raw/movielens_1m/ml-1m/README`

The broader requested filename scan also found many processed Amazon-domain artifacts under older project directories, but no Amazon raw review/meta files appeared in the first 100 matches.

## Blockers Before Live Tests

- Use the new `.venv_lora` environment for LoRA/local-model work; the default `python3` is still Python 3.13.
- `.env` contains a non-placeholder-looking `DEEPSEEK_API_KEY`.
- Confirm/copy required raw datasets into the fresh clone, because `data/raw` is empty or absent there.
- Do not treat older processed artifacts under `/home/ajifang/projects/*` as source-of-truth raw data without verification.

## DeepSeek Tiny Live Test

- Preflight command:

```bash
set -a
source .env
set +a
.venv_lora/bin/python3.11 - <<'PY'
import os, sys, torch
k = os.environ.get("DEEPSEEK_API_KEY", "")
print("python:", sys.executable)
print("DEEPSEEK_API_KEY present:", bool(k))
print("looks_like_placeholder:", any(s in k for s in ["这里", "你的", "真实key", "DeepSeekKey"]))
print("key_length:", len(k))
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

- Live test command:

```bash
PYTHON=.venv_lora/bin/python3.11 bash scripts/test_deepseek_live.sh
```

- Python path used: `/home/ajifang/projects/fresh/uncertainty-llm4rec/.venv_lora/bin/python3.11`
- DeepSeek model used: `deepseek-v4-flash`
- Raw response output path: `outputs/deepseek_live_test/raw_responses.jsonl`
- Parsed response output path: `outputs/deepseek_live_test/parsed_responses.jsonl`
- Combined output path: `outputs/deepseek_live_test/deepseek_live_results.json`
- Manifest path: `outputs/deepseek_live_test/manifest.json`
- Latency logging status: enabled (`latency_seconds` recorded per request).
- Token logging status: enabled (`token_usage` recorded per request).
- Result: success (`3` raw responses and `3` parsed responses written).
- Backend/config/script fix: `scripts/test_deepseek_live.sh` already respected `${PYTHON:-python}`, but the original backend import path failed because the current environment does not include `pydantic`, which is required by the installed `openai` SDK. Per the no-install constraint, the script was patched to use a direct `httpx` DeepSeek chat-completions tiny call while preserving config-driven model selection, parsing, config hash, latency logging, token logging, and manifest output.
