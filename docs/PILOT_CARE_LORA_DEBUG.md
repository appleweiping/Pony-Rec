# CARE-LoRA debug pilot (Qwen3-8B, amazon_beauty, n=20)

**Purpose:** exercise **CARE as a training-data policy** (weights / pruning on supervised listwise JSON), plus **tiny PEFT LoRA** training and **HF reload inference**—**not** a performance study.  
**Flags:** `run_type=pilot`, `backend_type=lora`, `is_paper_result=false` on built samples and manifests.

**Why this is not rerank-only:** post-hoc CARE rerank reorders a fixed slate at inference time; CARE-LoRA **reshapes the empirical loss** (and optionally drops rows) so the adapter sees **reweighted teacher targets** tied to **reliability, echo-risk, and tail-safe** structure learned from DeepSeek + probe features.

**Why this is not A+B+C:** weights are a **single coupled policy** (`CARE_full_training`) derived from the same components as CARE rerank (via `compute_care_components` / stress), not three independent toggles.

---

## Source artifacts (read-only)

| Role | Path |
|------|------|
| Reprocessed candidates | `outputs/reprocessed_processed_source/amazon_beauty/{train,valid,test}_candidates.jsonl` |
| DeepSeek pilot (20 users / split) | `outputs/pilots/deepseek_v4_flash_processed_20u_c19_seed42/amazon_beauty/{valid,test}/predictions/rank_predictions.jsonl` |
| Uncertainty probe | `.../uncertainty_features.jsonl` |
| CARE rerank weights reference | `configs/methods/care_rerank_pilot.yaml` (shared `global` + `care_full` variant hyperparameters) |
| Optional diagnostics | `outputs/pilots/deepseek_v4_flash_processed_20u_c19_seed42/calibration_diagnostics/` |

**Training alignment note:** the builder defaults to **`--split valid`** so each of the **20 pilot users** has **DeepSeek predictions + uncertainty_features** merged with **`valid_candidates.jsonl`**. The output files are still named `*_train.jsonl` because they feed the **LoRA trainer** (SFT-style), not because they use the dataset `train` split.

---

## Exact commands

**A — Build CARE training JSONL + summaries**

```bash
cd /home/ajifang/projects/fresh/uncertainty-llm4rec
.venv_lora/bin/python3.11 -m src.cli.build_care_lora_data \
  --domain amazon_beauty \
  --split valid \
  --reprocess_dir outputs/reprocessed_processed_source \
  --deepseek_root outputs/pilots/deepseek_v4_flash_processed_20u_c19_seed42 \
  --processed_dir data/processed/amazon_beauty \
  --care_rerank_config configs/methods/care_rerank_pilot.yaml \
  --output_root outputs/pilots/care_lora_qwen3_8b_beauty_20u_c19_seed42_debug \
  --topk 19 \
  --seed 42
```

Writes under `.../data/`:  
`vanilla_lora_baseline_train.jsonl`, `care_full_train.jsonl`, `policy_summary.json`, `bucket_distribution_before_after.csv`, `pruned_samples.jsonl`, `weighted_samples.jsonl`, `data_manifest.json`.

**B+C — Train (5–20 steps) + infer valid/test**

```bash
.venv_lora/bin/python3.11 -m src.cli.run_care_lora_debug \
  --domain amazon_beauty \
  --reprocess_dir outputs/reprocessed_processed_source \
  --processed_dir data/processed/amazon_beauty \
  --base_model /home/ajifang/models/Qwen/Qwen3-8B \
  --output_root outputs/pilots/care_lora_qwen3_8b_beauty_20u_c19_seed42_debug \
  --data_split valid \
  --max_train_steps 12 \
  --topk 19 \
  --seed 42
```

Flags: `--skip_build_data` (reuse `data/`), `--skip_train` (only infer if adapters already exist).

**LoRA YAML (reference bundle):** `configs/lora/care_qwen3_8b_debug.yaml`

---

## Data policies (Part A)

| Policy | Behavior |
|--------|-----------|
| `vanilla_lora_baseline` | Keep all rows; `sample_weight=1.0`; teacher listwise JSON (target first). |
| `prune_high_uncertainty` | Drop invalid predictions, missing/tiny confidence, or very high uncertainty risk. |
| `downweight_high_confidence_wrong_risk` | Keep all; downweight when `high_confidence_wrong` or head-biased stress patterns. |
| `tail_safe_recovery` | Mild upweight for mid/tail targets only in a safe global-confidence band. |
| `CARE_full_training` | Combined scalar weight (reliability × echo downweight × tail factor) with prune gate; **no blind head/tail/confidence** rules (uses shared CARE component logic). |

Per-row JSON includes: `prompt`, `response`, `target_item_id`, `candidate_item_ids`, `original_rank`, `confidence`, `popularity_bucket`, `care_risk_features`, `sample_weight`, `policy`, `keep_drop_reason`, `source_paths`, `run_type`, `is_paper_result`.

---

## Training / hardware (Part B)

- **PEFT LoRA** on attention projections (`q_proj,k_proj,v_proj,o_proj`), `lora_r=8`, `lora_alpha=16`.
- **bf16** when CUDA + bf16 supported; else float32 load path.
- **`max_train_steps`** default **12** (stay within 5–20).
- **Trainable parameter count** and **GPU memory snapshots** (before load, after PEFT wrap, after train) are recorded in `train_manifest.json` under `train.*`.
- **Loss history:** `train_logs/vanilla_lora_baseline_loss.jsonl`, `train_logs/care_full_training_loss.jsonl`.
- **Adapters:** `adapters/vanilla_lora_baseline/`, `adapters/care_full_training/` (adapter weights + tokenizer save).

**CARE adapter loss:** `WeightedTrainer` scales the HF LM loss by `sample_weight` (batch size 1 in this debug).

---

## Inference layout (Part C)

For each adapter name:

`eval_runs/<adapter_name>/amazon_beauty/{valid,test}/predictions/{raw_responses,parsed_responses,rank_predictions}.jsonl`  
same fields as `run_lora_debug_reprocessed`, plus `confidence_available` on rank rows.  
`manifest.json` + `eval/metrics.json` via `python -m src.cli.evaluate`.

**References (no new training):** compare logits/metrics qualitatively to DeepSeek pilot and CARE rerank CSVs under `outputs/pilots/care_rerank_*`—**do not** claim gains at n=20.

---

## What changed vanilla vs CARE (typical run)

After `build_care_lora_data` on the frozen pilot, **`care_full_train.jsonl` often has fewer than 20 rows** (rows failing the CARE prune gate are listed in `pruned_samples.jsonl`). **`vanilla_lora_baseline_train.jsonl` always has 20 rows.**  
`policy_summary.json` + `bucket_distribution_before_after.csv` summarize per-policy keep counts and target popularity bucket mass before vs after pruning.

---

## Blockers before 100-user or all-domain LoRA

1. **Train vs inference split hygiene:** debug uses **valid** users for SFT; production must use true **train** users and hold out **valid** for model selection only.  
2. **Weight calibration:** `sample_weight` mapping needs tuning on larger n, not the fixed pilot heuristics.  
3. **GPU / memory:** Qwen3-8B + LoRA still needs a capable GPU; CPU-only is not supported for this path.  
4. **Teacher quality:** listwise teacher is **target-first** JSON—richer teachers or DPO pairs would change the objective.  
5. **Statistical power:** n=20 cannot support claims; rerun `evaluate` summaries only as **sanity**, not benchmarks.

---

## Tests

```bash
.venv_lora/bin/python3.11 -m pytest tests/test_care_lora_data.py -q
```

---

## Code map

- `src/methods/care_lora_data.py` — policy evaluation + training row schema  
- `src/cli/build_care_lora_data.py` — artifact writer  
- `src/cli/run_care_lora_debug.py` — train + infer driver  
- `configs/lora/care_qwen3_8b_debug.yaml` — documented hyperparameters / paths  
