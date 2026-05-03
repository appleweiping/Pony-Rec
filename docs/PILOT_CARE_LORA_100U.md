# CARE-LoRA 100-User Limited Pilot (beauty, c19, seed42)

Scope: `amazon_beauty` only, pilot-only, `is_paper_result=false`.

## Executive status (strictgen attempt C accepted as negative)

- **Mainline strict JSON validity (original 100u pilot)** remains **~0.35–0.47** across adapters/splits (`repair_summary.json` under `outputs/pilots/care_lora_qwen3_8b_beauty_100u_c19_seed42_pilot/`). **CARE-LoRA strict output is not fixed**; do not describe it as resolved.
- **Strictgen** (inference-only decode/prompt iteration under `outputs/pilots/care_lora_qwen3_8b_beauty_100u_c19_seed42_strictgen/`, including **v3** re-inference) remains an **accepted negative** for **clearing the blocker**: latest `repair_summary.json` strict rates **match the registered pilot** (vanilla valid/test **0.47 / 0.43**, CARE valid/test **0.47 / 0.35**) — **no material lift above pilot**, not a new baseline. Earlier snapshots with worse CARE strictgen cells were **superseded** by v3 on-disk files.
- **`usable_ranking_rate_after_safe_repair=1.00`** still holds, but repaired rankings are **engineering usability** metrics (deterministic completion from allowed candidates), **not** pure model-generation quality. Report strict vs repaired separately.
- **Paper-result / broad scaling:** CARE-LoRA remains **blocked** until **strict generation** improves materially (e.g. **true constrained decoding**) or the task moves to a **non-generative scoring** formulation that does not depend on free-form JSON from the base LM.
- **Recommended next mainline (pilot):** **candidate_size=99** DeepSeek run **plus** CARE rerank on those outputs (API track, not LoRA).
- **Recommended LoRA-side research:** **true constrained decoding** (grammar / token masks over the candidate set) or **non-generative** reranking/scoring—**not** further n-gram heuristics on ASIN strings.

## Command

```bash
cd /home/ajifang/projects/fresh/uncertainty-llm4rec
.venv_lora/bin/python3.11 -m src.cli.run_care_lora_debug \
  --config configs/lora/care_qwen3_8b_beauty_100u_c19_pilot.yaml
```

Config highlights:

- `reprocess_dir: outputs/reprocessed_processed_source_100u_c19_seed42`
- `deepseek_pilot_root: outputs/pilots/deepseek_v4_flash_processed_100u_c19_seed42`
- `prompt_id: listwise_ranking_json_lora`
- `max_train_steps: 60` (conservative pilot budget)
- output: `outputs/pilots/care_lora_qwen3_8b_beauty_100u_c19_seed42_pilot/`

## Output structure

Per adapter/split (`vanilla_lora_baseline`, `care_full_training` x `valid`, `test`):

- `raw_responses.jsonl`
- `parsed_responses.jsonl`
- `rank_predictions.jsonl`
- `format_failure_taxonomy.csv`
- `repair_summary.json`
- `eval/metrics.json`
- `manifest.json`

## Two metric families (must be separated)

### 1) Strict-generation metrics

From `repair_summary.json`:

- `strict_json_valid_rate`
- `invalid_output_rate_strict`
- `confidence_available_rate_strict`

Observed:

- vanilla valid: `strict_json_valid_rate=0.47`
- vanilla test: `0.43`
- CARE valid: `0.47`
- CARE test: `0.35`

Optional strict-only ranking slice (computed on strict-valid rows only):

- vanilla valid strict rows `n=47`: `HR@5=0.298`, `NDCG@5=0.254`, `MRR@5=0.240`
- vanilla test strict rows `n=43`: `0.186 / 0.114 / 0.091`
- CARE valid strict rows `n=47`: `0.277 / 0.218 / 0.199`
- CARE test strict rows `n=35`: `0.200 / 0.143 / 0.125`

### 2) Safe-repaired metrics

From `repair_summary.json` + `eval/metrics.json`:

- `usable_ranking_rate_after_safe_repair`
- `invalid_output_rate_after_repair`
- `confidence_available_rate_after_repair`
- `repair_reason_counts`
- repaired ranking metrics (`HR/NDCG/MRR`)

Observed:

- all splits/adapters: `usable_ranking_rate_after_safe_repair=1.00`
- all splits/adapters: `invalid_output_rate_after_repair=0.00`
- confidence-after-repair equals strict confidence availability (repair does not invent confidence)

Repaired metrics snapshot:

- vanilla valid: `HR@5=0.32`, `NDCG@5=0.247`, `MRR@5=0.223`
- vanilla test: `0.20 / 0.144 / 0.126`
- CARE valid: `0.32 / 0.246 / 0.223`
- CARE test: `0.18 / 0.119 / 0.099`

Repair reasons:

- vanilla valid: `none=47`, `strict_json_invalid_then_candidate_completion=53`
- vanilla test: `none=43`, `...=57`
- CARE valid: `none=47`, `...=53`
- CARE test: `none=35`, `...=65`

## Interpretation constraints

- Repaired-ranking metrics are **not** pure model-generation quality.
- Strict-generation quality remains limited (roughly 35–47% strict JSON validity).
- This run is pilot engineering evidence only, not a paper claim.

## Before full-scale LoRA

At least one of:

1. materially improve strict generation validity, or
2. keep repaired-ranking reporting explicitly separated from strict-generation reporting.

---

## Strict JSON failure taxonomy (original 100u pilot)

Source: `outputs/pilots/care_lora_qwen3_8b_beauty_100u_c19_seed42_pilot/eval_runs/<adapter>/<split>/format_failure_taxonomy.csv` (`primary_failure` for rows with `strict_json_valid=False`).

Pattern across adapters/splits:

- **`incomplete_ranking` dominates** (typical strict-invalid primary bucket): model JSON parses but the `ranking` array is not a full permutation of the 19 allowed IDs (often 18 IDs, duplicates, or early EOS).
- **Duplicates / OOD** show up in `parsed_responses.jsonl` flags (`duplicate_item`, `output_not_in_candidate_set`) on some rows; taxonomy still often labels the primary failure as `incomplete_ranking` when the slate is not a valid full set.
- **Truncation / max_new_tokens**: in this pilot, strict-invalid rows did **not** systematically flag `truncation_or_max_new_tokens_issue=1` (so incompleteness is mostly **early EOS / formatting drift**, not obvious decode-length clipping).

Buckets you asked to track map roughly as:

| Bucket | Pilot evidence |
| --- | --- |
| no JSON | rare vs `incomplete_ranking` |
| incomplete ranking | primary driver |
| duplicate IDs | present on subset (see flags) |
| OOD IDs | present on subset (see flags) |
| malformed JSON | parser flags when JSON repair fails |
| thinking / prose leakage | suppressed via chat template `enable_thinking=false` + `/no_think`; residual risk if model ignores |
| truncation | not the dominant taxonomy signal in this pilot |
| confidence missing | tracked separately via `missing_confidence` / `confidence_available_rate_strict` |

---

## Strict JSON generation iteration (inference-only, same adapters)

**Goal:** improve `strict_json_valid_rate` without retraining and without folding repaired metrics into “generation quality”.

**Artifacts:** copy existing adapters from the 100u pilot into a separate output root, then rerun inference only:

```bash
cd /home/ajifang/projects/fresh/uncertainty-llm4rec
mkdir -p outputs/pilots/care_lora_qwen3_8b_beauty_100u_c19_seed42_strictgen
cp -r outputs/pilots/care_lora_qwen3_8b_beauty_100u_c19_seed42_pilot/adapters \
  outputs/pilots/care_lora_qwen3_8b_beauty_100u_c19_seed42_strictgen/

.venv_lora/bin/python3.11 -m src.cli.run_care_lora_debug \
  --config configs/lora/care_qwen3_8b_beauty_100u_c19_strictgen.yaml \
  --skip_build_data \
  --skip_train
```

Config: `configs/lora/care_qwen3_8b_beauty_100u_c19_strictgen.yaml` (pilot-only, `is_paper_result=false`).

### Code / decode controls added (HF local backend)

Wired through `build_backend()` and `run_care_lora_debug` YAML `inference:`:

- `repetition_penalty` (default `1.0` when unset)
- `no_repeat_ngram_size` (default `0` when unset)
- `stop_at_json_end` (optional): after `generate`, keep only the first brace-balanced `{...}` substring (drops trailing prose after a complete JSON object)
- `stop_strings` (optional): truncate decoded text at the earliest occurrence of any stop substring (use with care: `\n\n` can appear inside pretty-printed JSON)

Chat template path unchanged: `runtime.use_chat_template=true`, `runtime.enable_thinking=false` for Qwen3 listwise JSON runs.

### Constrained / schema-aware decoding

**Not implemented** in-repo yet: token-level / logits-masked JSON or permutations over the 19 ASINs would need extra stack support (e.g. grammar-guided decoding) beyond plain `model.generate`. The items below are **generation hygiene** only.

### Ablations (beauty 100u, same frozen adapters)

| Attempt | Decode / prompt idea | `strict_json_valid_rate` (approx.) | Notes |
| --- | --- | --- | --- |
| **Baseline** | original pilot run (`max_new_tokens=384`, no extra decode hooks) | vanilla valid/test **0.47 / 0.43**, CARE valid/test **0.47 / 0.35** | reference |
| **A** | `no_repeat_ngram_size=8` + mild `repetition_penalty` + `stop_at_json_end` + stop strings | **~0.00** all splits | **Invalid for ASIN JSON**: n-gram repetition rules corrupt 10-char IDs (spaces, wrong tail chars, glued strings). **Do not use** for this task. |
| **B** | remove n-gram repeat; keep `stop_at_json_end`; stricter prompt contract | vanilla **~0.31** valid/test; CARE **~0.23 / 0.32** | **Worse than baseline**; primary strict-invalid bucket remained `incomplete_ranking`. |
| **C + v3 (accepted negative)** | restore standard `listwise_ranking_json_lora` text; `max_new_tokens=512`; `stop_at_json_end=false`; ASIN-safe decode; v3 re-inference overwrites `.../strictgen/` | latest on-disk `repair_summary.json` (post–strictgen v3): vanilla valid/test **0.47 / 0.43**, CARE valid/test **0.47 / 0.35** — **same strict rates as the registered pilot**; `usable_ranking_rate_after_safe_repair=1.00` all four | **Accepted negative:** no **strict** validity improvement vs pilot; decode-only path does **not** clear the ~0.35–0.47 ceiling or substitute for constrained decoding / non-generative scoring. |

`usable_ranking_rate_after_safe_repair` stayed **1.00** across attempts (repair remains a separate bridge; still no target leakage in repair code).

### Remaining blocker

Pilot-scale **strict JSON validity** stays in the **~0.35–0.47** band on the **original 100u pilot** outputs. Until that moves materially **without** relying on repair, treat ranking metrics from repaired slates as **engineering-only**. **Strictgen v3** re-aligned strictgen on-disk strict rates with the pilot but **did not clear** the blocker (no lift above pilot). Next LoRA-side technical work is **true constrained decoding** or **non-generative scoring** over the candidate set—not more n-gram tricks on ASIN strings.
