# LoRA debug pilot: Qwen3-8B on reprocessed `amazon_beauty` (20 users, 19 negatives)

End-to-end smoke for the **PEFT LoRA** path on **local** weights and **`outputs/reprocessed_processed_source`** candidates. This is a **debug pilot** (`run_type=pilot`, `backend_type=lora`, `is_paper_result=false`), not a performance run.

## Exact command

From repo root (`/home/ajifang/projects/fresh/uncertainty-llm4rec`), using the LoRA venv:

```bash
.venv_lora/bin/python3.11 -m src.cli.run_lora_debug_reprocessed \
  --max_train_steps 12 \
  --max_seq_length 512
```

Defaults already match the requested pilot: `amazon_beauty`, seed `42`, `topk=19` (20 candidates including target), reprocess dir `outputs/reprocessed_processed_source`, output root `outputs/pilots/lora_qwen3_8b_processed_20u_c19_seed42_debug`, base model path below.

## Environment

| Item | Value |
|------|--------|
| Python | `.venv_lora/bin/python3.11` |
| **transformers** | **>= 4.51.0** required for `model_type: qwen3` (4.44.x raises `KeyError: 'qwen3'`). Verified run used **5.7.0** after `pip install -U 'transformers>=4.51.0'`. |
| Tokenizer | `AutoTokenizer.from_pretrained(..., use_fast=False)` avoids fast-tokenizer JSON parse errors on this checkpoint. |

## Base model and LoRA

| Item | Value |
|------|--------|
| Base model path | `/home/ajifang/models/Qwen/Qwen3-8B` |
| PEFT | `LoraConfig`: `r=8`, `lora_alpha=16`, `target_modules=["q_proj","k_proj","v_proj","o_proj"]` |
| Trainable parameters | **7,667,712** (see `meta/train_debug.json`) |
| Train steps | **12** optimizer steps (`max_steps` in HF `Trainer`) |
| Batch | `per_device_train_batch_size=1`, `gradient_accumulation_steps=4` |

## Data paths

| Role | Path |
|------|------|
| Reprocessed candidates (train/valid/test) | `outputs/reprocessed_processed_source/amazon_beauty/*_candidates.jsonl` |
| Processed item titles (prompt text) | `data/processed/amazon_beauty/` (via `--processed_dir`) |

Startup validates each split: target in `candidate_item_ids`, target not in history, no history IDs in negatives.

## Training loss and GPU memory

- **Loss log:** `outputs/pilots/lora_qwen3_8b_processed_20u_c19_seed42_debug/meta/train_loss.jsonl`
- **Trainer summary:** `train_loss` ≈ **3.044** over 12 steps (see console / `trainer_state.json` under adapter dir if present).
- **Snapshots** (`meta/train_debug.json`, bytes on a 50 GiB-class GPU):

  - Before load: ~50.4 GiB free.
  - After PEFT + base on GPU: ~16.4 GiB allocated, ~34.0 GiB free.
  - After train: ~16.5 GiB allocated, ~32.5 GiB free.

- **Adapter:** saved under `outputs/pilots/lora_qwen3_8b_processed_20u_c19_seed42_debug/adapter/`.

## Inference (adapter reload)

Inference uses **`HFLocalBackend`** with `adapter_path` pointing at the saved adapter (fresh model load per eval split in the driver). Post–test GPU snapshot is recorded in `debug_run_summary.json` (`gpu_mem_before_infer_reload`, `gpu_mem_after_infer`).

## Output layout (API pilot compatibility)

Per split (`valid`, `test`) under `.../amazon_beauty/<split>/`:

- `predictions/raw_responses.jsonl`
- `predictions/parsed_responses.jsonl`
- `predictions/rank_predictions.jsonl`
- `manifest.json`
- `eval/metrics.json`
- `eval/manifest.json`

`evaluate` is run with **`--candidates_source_path`** pointing at the matching `*_candidates.jsonl` in `outputs/reprocessed_processed_source/amazon_beauty/`.

## Confidence

LoRA generations do **not** supply calibrated verbalized confidence in JSON: **`parsed_responses.jsonl`** has `confidence: null` and `missing_confidence: true` for all rows in this run (**confidence availability rate = 0%**). Downstream metrics still use a default probability path where needed (see `eval/metrics.json` calibration block).

## Invalid output rate (parsed schema)

From `parsed_responses.jsonl`, **`invalid_output: true` for 20/20 rows** on both valid and test in this debug run (100% invalid flag), mostly incomplete JSON / listwise compliance despite partial `ranked_item_ids` in some lines. **HR / Recall / NDCG / MRR** in `eval/metrics.json` are still computed from the ranking pipeline where possible.

## Metrics snapshot (this machine, 12-step LoRA)

**Valid** — `amazon_beauty/valid/eval/metrics.json`:

- HR@5 / Recall@5: **0.15**; NDCG@5: **~0.103**; MRR@5: **0.0875**
- HR@10 / Recall@10: **0.15**; NDCG@10: **~0.103**; MRR@10: **0.0875**

**Test** — `amazon_beauty/test/eval/metrics.json`:

- HR@5 / Recall@5: **0.1**; NDCG@5: **~0.043**; MRR@5: **0.025**

(Metrics are **not** meaningful for model quality after 12 steps; they only confirm the eval wiring.)

## Blockers before a real LoRA run

1. **Transformers version:** pin **`transformers>=4.51.0`** (or the project-tested 5.x line) anywhere LoRA/Qwen3 runs; document in setup README.
2. **Tokenizer:** keep **`use_fast=False`** for this checkpoint unless the fast tokenizer file is fixed upstream.
3. **Output quality:** increase steps, tune `max_new_tokens`, decoding, and SFT format; current pilot uses minimal steps so JSON/listwise validity is poor (`invalid_output` high).
4. **Confidence:** either extend the prompt + parse path for verbalized confidence or treat LoRA pilots as **ranking-only** in analysis.
5. **Deprecation:** Transformers may warn that `torch_dtype` is deprecated in favor of `dtype` when loading the causal LM; harmless for now, can be cleaned in `run_lora_debug_reprocessed` / backend loaders.

## Uncertainty probe (Section 13 step 2)

After `rank_predictions.jsonl` exists:

```bash
.venv_lora/bin/python3.11 -m src.cli.run_uncertainty_probe \
  --predictions_path outputs/pilots/lora_qwen3_8b_processed_20u_c19_seed42_debug/amazon_beauty/valid/predictions/rank_predictions.jsonl
```

Writes `uncertainty_features.jsonl` and `uncertainty_probe_summary.json` next to the predictions file.

## Tests

```bash
.venv_lora/bin/python3.11 -m pytest
```

Relevant tests: `tests/test_lora_debug_reprocessed_validation.py`, `tests/test_phase1_backend_manifest_baselines.py`, `tests/test_uncertainty_features_probe.py`.
