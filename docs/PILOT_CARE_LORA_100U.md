# CARE-LoRA 100-User Limited Pilot (beauty, c19, seed42)

Scope: `amazon_beauty` only, pilot-only, `is_paper_result=false`.

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
