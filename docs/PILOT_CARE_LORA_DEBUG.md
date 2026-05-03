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

**End-to-end (recommended):** build + train + infer from YAML.

```bash
cd /home/ajifang/projects/fresh/uncertainty-llm4rec
.venv_lora/bin/python3.11 -m src.cli.run_care_lora_debug \
  --config configs/lora/care_qwen3_8b_debug.yaml
```

**A — Build CARE training JSONL + summaries** (standalone)

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

**B+C — Train (5–20 steps) + infer valid/test** (CLI overrides YAML when both passed)

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

## Real server run (2026-05-03)

This section records one **completed** GPU run on the project server (`train_manifest.json` → `git_commit` **745af41** at data build time; orchestration includes local patches for `--config` and flat `eval_runs/` paths). **This is n=20 diagnostic data only; not a paper result.**

**Command**

```bash
cd /home/ajifang/projects/fresh/uncertainty-llm4rec
rm -rf outputs/pilots/care_lora_qwen3_8b_beauty_20u_c19_seed42_debug   # optional clean of prior pilot output only
.venv_lora/bin/python3.11 -m src.cli.run_care_lora_debug \
  --config configs/lora/care_qwen3_8b_debug.yaml
```

**Qwen3-8B load:** **Yes** — `AutoModelForCausalLM.from_pretrained` + PEFT wrap succeeded for both training phases (see `train_manifest.json` → `base_model`).

**Adapters trained and saved:** **Yes** — both `vanilla_lora_baseline` (20 rows, 12 steps) and `care_full_training` (15 rows after CARE prune, 12 steps). Adapter dirs contain `adapter_config.json` and weights. **Reload:** inference re-loaded the base model + adapter per split (four full reloads: vanilla valid/test, CARE valid/test); no load errors in the captured log.

**Data build (valid-aligned 20 users)**

| Artifact | Result |
|----------|--------|
| `vanilla_lora_baseline_train.jsonl` | **20** rows |
| `care_full_train.jsonl` | **15** rows (5 pruned by `CARE_full_training` gate) |
| `pruned_samples.jsonl` | **5** rows (same users dropped as in `policy_summary.json` → `counts.CARE_full_training.dropped`) |
| `sample_weight` on `care_full_train` | **min ≈ 0.663**, **max ≈ 1.255** |
| Target bucket mass (before → after CARE kept) | head **0%** → **0%**; mid **5%** → **6.67%**; tail **95%** → **93.33%** (`policy_summary.json`) |

**Training (`max_train_steps=12`)**

| Adapter | Trainable params | Final train loss (last step) | Peak GPU reserved (bytes) |
|---------|------------------|-------------------------------|---------------------------|
| `vanilla_lora_baseline` | **7,667,712** | **≈ 2.83** (step 12) | **17,857,249,280** |
| `care_full_training` | **7,667,712** | **≈ 9.00** (step 12; weighted LM loss) | **17,865,637,888** |

Full step losses are in `train_logs/*_loss.jsonl` and duplicated under `train_manifest.json` → `train.*.loss_history`.

**Inference + `evaluate`**

| Adapter | Split | HR@1 | HR@5 | NDCG@5 | MRR@5 |
|---------|-------|------|------|--------|-------|
| vanilla | valid | 0.05 | 0.05 | 0.05 | 0.05 |
| vanilla | test | 0.00 | 0.00 | 0.00 | 0.00 |
| CARE_full | valid | 0.05 | 0.05 | 0.05 | 0.05 |
| CARE_full | test | 0.00 | 0.00 | 0.00 | 0.00 |

(Values from each `eval_runs/<adapter>/<split>/eval/metrics.json` → `ranking`.)

**Invalid output / confidence**

- **`rank_predictions.jsonl`:** in this run, **all 20 rows per split** had `is_valid: false`, empty `predicted_ranking`, and `missing_confidence: true` — the 12-step adapter still **rambles in plain text** instead of emitting strict JSON (see `raw_response` excerpts). **Invalid rate ≈ 100%** on parsed listwise outputs for this debug budget.
- **`confidence_available`:** **false** on every emitted row (no usable verbalized confidence after parse failure).

**GPU memory:** peak **reserved** ≈ **16.6 GiB** during training (see table above). VRAM total on host **≈ 47.4 GiB** free before first load (`gpu_mem_before_load.mem_free_bytes` in manifest).

**Blockers before a 100-user LoRA pilot**

1. **Listwise JSON compliance:** need more SFT steps, lower temperature at decode, JSON/logit constraints, or rejection sampling — current **12-step** debug does not produce parseable rankings.  
2. **Train split hygiene:** still using **valid** users for SFT alignment to DeepSeek; real training must use **train** users with matched signals.  
3. **Weighted-loss scaling:** CARE-weighted CE runs **higher** than vanilla; validate stability and gradient norms when n↑.  
4. **Throughput:** four full base-model reloads for eval are expensive; consider a single long-lived HF backend for batched eval.

**Not statistically meaningful:** with **n=20** and broken parses on most rows, HR/NDCG numbers are **sanity checks only**, not evidence of ranking quality.

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

For each adapter name (`vanilla_lora_baseline`, `care_full_training`) and split (`valid`, `test`), files live **directly** under:

`eval_runs/<adapter_name>/<split>/`

— `raw_responses.jsonl`, `parsed_responses.jsonl`, `rank_predictions.jsonl`, `manifest.json`, and `eval/metrics.json` (same rank row shape as the DeepSeek pilot, plus `confidence_available` on each `rank_predictions` row).

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
