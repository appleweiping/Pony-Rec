# GPU Queue-to-8: Two Remaining Reviewer Experiments — READY-TO-FIRE PLAN

**Status:** scripts written + syntax-checked, panels + Llama model verified present, wall-time/disk estimated. **NOTHING has been run on the GPU.** These fire the instant the GPU frees.

**HARD CONSTRAINT:** The RTX 4090 is busy with the TGL LoRA training run (`scripts/train_cc_pace_lora.py`, **PID 4090183**, 48.3 GB VRAM). Do **NOT** launch anything below until that PID is gone and `nvidia-smi` shows the GPU free. Do not touch the LoRA.

Server: `ssh pony-rec-gpu`, project `~/projects/pony-rec-rescue-shadow-v6`, env `qwen_vllm` (vLLM 0.10.2), python `/home/ajifang/miniconda3/envs/qwen_vllm/bin/python`.

---

## What these experiments are (reviewer mapping)

| Exp | Gap-to-8 item | What it does | Backbone | Runs |
|-----|---------------|--------------|----------|------|
| **A. Second backbone** | Opus #1 (top remaining blocker) | Re-run the SAME C-CRP v3 pointwise scoring with **Llama-3.1-8B-Instruct** on sports/toys/home/tools, to show the win pattern + the η=0 negative result replicate on a **non-Qwen** backbone | Llama-3.1-8B | 4 |
| **B. Multi-seed CIs** | Opus #3 | Re-run **Qwen3-8B** scoring with seeds **2026/2027/2028** on the same 4 domains → **mean±std NDCG@10** (genuine generation/run variance at temp 0.1, vs the current paired-event bootstrap = sampling variance) | Qwen3-8B | 12 |

Both reuse the identical pipeline (prompt, parse, metrics, output layout). The only deltas are `--model` (A) and `--seed` (B).

---

## Pipeline facts (verified this session)

- **Scoring script (original):** `experiments/rsc/run_ccrp_v3_domain.py` → uses `src/llm/vllm_backend.py::VLLMBackend`.
- **Model:** passed via `--model` CLI arg (default `/home/ajifang/models/Qwen/Qwen3-8B`). Fully parameterized.
- **Temperature:** fixed `0.1` in the backend (`SamplingParams.temperature`); overrides each model's `generation_config.json` (Llama's 0.6 is ignored — confirmed). `max_new_tokens=100`, `top_p=0.95`, `max_model_len=1024`, `gpu_memory_utilization=0.85`, `enable_prefix_caching=True`, `dtype=float16`.
- **Prompt:** plain instruction ending `Return ONLY JSON: {"relevance_probability": 0.0, "reason": "one sentence"}` (history truncated to last 5, candidate_text to 200 chars).
- **Probability extraction:** **parsed numeric from generated TEXT** — `parse_score()` regex `"relevance_probability"\s*:\s*([\d.]+)` (fallback `(0\.\d+|1\.0)`), else `0.0`. **NOT** logprobs, **NOT** a yes/no token, **NOT** constrained decoding (JSON guided-decoding exists in the backend but is OFF by default here). → **tokenizer-agnostic**; no hard-coded token id to re-derive for Llama.
- **Chat template:** `VLLMBackend._format_prompt()` calls `tokenizer.apply_chat_template(...)` **generically** (no hard-coded Qwen format). The Qwen-specific `enable_thinking=False` kwarg is wrapped in `try/except TypeError`. Verified: Llama-3.1 tokenizer applies its own template correctly (`<|begin_of_text|><|start_header_id|>user...<|eot_id|><|start_header_id|>assistant`); it silently ignores `enable_thinking`, so the path is Llama-safe.
- **Output layout per run:** `scores.csv` (~87 MB), `report.json` (HR/NDCG@5/10/20 + MRR + n_users/n_prompts/inference_time_s), `user_ranks.jsonl` (~1 MB, per-user positive_rank for paired tests). No `predictions/` dir.
- **Panels (input):** `outputs/baselines/external_tasks/<domain>_large10000_100neg_test_same_candidate/ranking_test.jsonl` — confirmed **10000 users × 101 candidates** each, schema matches.

### THE ONE BLOCKER FOUND (handled — not a Qwen hard-coding issue)

The stock `VLLMBackend` **never sets a seed** in `SamplingParams` or `LLM()`, so the multi-seed experiment (B) cannot use it as-is. **Fix (already written, does NOT patch the shared backend):** a thin variant `experiments/rsc/run_ccrp_v3_domain_seeded.py` builds vLLM `LLM` + `SamplingParams` directly, mirroring every `VLLMBackend` default, and adds `--seed` (threaded into `SamplingParams.seed` + `LLM(seed=)`) and `--backbone` (provenance). Verified vLLM 0.10.2 accepts both seed routes. This same script serves **both** experiments (A = pass `--model`; B = pass `--seed`). The shared backend is left untouched (used by many other pipelines).

---

## Wall-time & disk

Measured original Qwen runs (from each `report.json` `inference_time_s`): sports 29,974 s · toys 31,148 s · home 32,012 s · tools 32,492 s → **~8.3–9.0 h/domain** (1.01 M prompts/domain). Llama-3.1-8B is the same 8B class with slightly **shorter** max prompt tokens (460–555 vs Qwen 481–612, all ≪ 1024 → no truncation) → budget **~8.5–9 h/domain**.

| | runs | serial wall-time | disk (scores.csv ~88 MB/run) |
|---|---|---|---|
| **A. Llama 2nd backbone** | 4 | **~35 h (~1.5 days)** | ~0.35 GB |
| **B. Qwen multi-seed** | 12 | **~104 h (~4.3 days)** | ~1.05 GB |
| **Total** | 16 | **~139 h (~5.8 days)** | **~1.4 GB** (25 GB free → fine) |

Score files are small; **no large predictions dir**. Disk is not a constraint. If GPU time is scarce, A is the higher-priority / cheaper experiment — do it first.

---

## EXACT COMMANDS, in priority order

### STEP 0 — confirm GPU is free (do NOT skip)
```bash
ssh pony-rec-gpu "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; \
  ps aux | grep train_cc_pace_lora | grep -v grep"
```
Proceed only when PID 4090183 is gone and VRAM is free.

### STEP 1 — SMOKE TEST FIRST (Llama, 20 sports users, ~1–2 min)
Validates the Llama template + that it actually emits parseable JSON probabilities (not all 0.0) BEFORE committing ~35 h.
```bash
ssh pony-rec-gpu 'cd ~/projects/pony-rec-rescue-shadow-v6 && \
  bash experiments/rsc/smoke_test_llama_ccrp_v3.sh 2>&1 | tee logs/smoke_llama_sports.log'
```
**Expected PASS:** `nonzero >= 30%`, `distinct_values >= 5`, a real NDCG@10. **If FAIL** (all 0.0 / degenerate): inspect raw_text — would mean Llama isn't following the JSON instruction; consider enabling the backend's JSON guided-decoding or a Llama-friendly prompt tweak. Do NOT launch the full run on FAIL.

### STEP 2 — A. Second-backbone Llama full run (4 domains, ~35 h) [highest priority]
```bash
ssh pony-rec-gpu 'cd ~/projects/pony-rec-rescue-shadow-v6 && \
  nohup bash experiments/rsc/run_ccrp_v3_second_backbone_llama.sh \
    > logs/ccrp_v3_llama_4domains.log 2>&1 &'
# monitor:
ssh pony-rec-gpu 'tail -f ~/projects/pony-rec-rescue-shadow-v6/logs/ccrp_v3_llama_4domains.log'
```
Outputs: `outputs/<domain>_large10000_100neg_ccrp_v3_llama/{report.json,scores.csv,user_ranks.jsonl}`.

### STEP 3 — B. Multi-seed Qwen full run (12 runs, ~104 h) [after A, or interleave]
```bash
ssh pony-rec-gpu 'cd ~/projects/pony-rec-rescue-shadow-v6 && \
  nohup bash experiments/rsc/run_ccrp_v3_multiseed_qwen.sh \
    > logs/ccrp_v3_multiseed_qwen.log 2>&1 &'
ssh pony-rec-gpu 'tail -f ~/projects/pony-rec-rescue-shadow-v6/logs/ccrp_v3_multiseed_qwen.log'
```
Outputs: `outputs/<domain>_large10000_100neg_ccrp_v3_seed{2026,2027,2028}/report.json` (+ scores.csv, user_ranks.jsonl).

> Run A and B serially (one GPU). If you want B to start immediately after A without manual intervention, chain them: `nohup bash -c 'bash .../run_ccrp_v3_second_backbone_llama.sh && bash .../run_ccrp_v3_multiseed_qwen.sh' > logs/queue_to8.log 2>&1 &`.

### STEP 4 — aggregate multi-seed → mean±std (CPU, seconds)
```bash
ssh pony-rec-gpu 'cd ~/projects/pony-rec-rescue-shadow-v6 && \
  PYTHONPATH=. /home/ajifang/miniconda3/envs/qwen_vllm/bin/python \
  experiments/rsc/aggregate_multiseed_ndcg.py'
```
Writes `outputs/summary/paper_critical/gpu_queue_to8/multiseed_ndcg_{summary.json,table.csv}` (mean±std for every metric; headline NDCG@10 mean±std; original unseeded NDCG@10 for cross-check).

---

## How each result plugs into the paper

- **A. Second-backbone replication table:** build a small table — for each of sports/toys/home/tools, C-CRP v3 NDCG@10 under **Qwen3-8B** (existing `outputs/<d>..._ccrp_v3/report.json`) vs **Llama-3.1-8B** (new `..._ccrp_v3_llama/report.json`), plus whether Llama still beats the strongest baseline in each domain (reuse the existing 8-baseline numbers). Headline claim: *the pointwise-posterior win pattern + the η=0 inert-machinery negative result replicate on a non-Qwen backbone* → kills the single-backbone external-validity objection. (η=0 replication: re-run the existing uncertainty/risk ablation with the Llama scores; the negative result is that adding the calibrated-uncertainty terms does not improve same-candidate ranking.)
- **B. Multi-seed mean±std:** replace / augment the current paired-event bootstrap CI in the main table with **NDCG@10 = mean ± std over 3 seeds** for the 4 gated domains → answers "are the gains stable across generations at temp 0.1, not just across the test sample?". The aggregator's `multiseed_ndcg_table.csv` is the drop-in source; confirm the 3 seeds bracket the published unseeded number.

---

## Files prepared (committed on branch `paper/reframe-major-revision`)

- `experiments/rsc/run_ccrp_v3_domain_seeded.py` — seed+model-parameterized scoring variant (serves both experiments)
- `experiments/rsc/smoke_test_llama_ccrp_v3.sh` — 20-user Llama smoke test + JSON-extraction validator
- `experiments/rsc/run_ccrp_v3_second_backbone_llama.sh` — A launcher (4 domains)
- `experiments/rsc/run_ccrp_v3_multiseed_qwen.sh` — B launcher (3 seeds × 4 domains)
- `experiments/rsc/aggregate_multiseed_ndcg.py` — mean±std aggregator
- `outputs/summary/paper_critical/gpu_queue_to8/PLAN.md` — this file
