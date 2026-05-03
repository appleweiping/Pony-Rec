# DeepSeek 100-User Pilot (c19, seed42)

Run scope: pilot-only, `is_paper_result=false`, four domains, valid/test only.

## Command

```bash
cd /home/ajifang/projects/fresh/uncertainty-llm4rec
.venv_lora/bin/python3.11 -m src.cli.run_pilot_reprocessed_deepseek \
  --reprocess_dir outputs/reprocessed_processed_source_100u_c19_seed42 \
  --output_root outputs/pilots/deepseek_v4_flash_processed_100u_c19_seed42 \
  --backend_config configs/backends/deepseek_v4_flash.yaml \
  --domains amazon_beauty amazon_books amazon_electronics amazon_movies \
  --splits valid test \
  --seed 42 \
  --run_type pilot \
  --method llm_listwise
```

Uncertainty + calibration follow-ups:

```bash
# uncertainty probe on all rank_predictions.jsonl
.venv_lora/bin/python3.11 -m src.cli.run_uncertainty_probe --predictions_path <.../rank_predictions.jsonl>

# calibration diagnostics batch
.venv_lora/bin/python3.11 -m src.cli.run_calibration_diagnostics \
  --batch_glob "outputs/pilots/deepseek_v4_flash_processed_100u_c19_seed42/**/predictions/rank_predictions.jsonl" \
  --output_dir outputs/pilots/deepseek_v4_flash_processed_100u_c19_seed42/calibration_diagnostics \
  --confidence_field auto \
  --num_bins 10
```

## Output root

- `outputs/pilots/deepseek_v4_flash_processed_100u_c19_seed42/`

Per domain/split includes:

- `predictions/raw_responses.jsonl` (latency + token usage logs included per row)
- `predictions/parsed_responses.jsonl`
- `predictions/rank_predictions.jsonl`
- `manifest.json`
- `eval/metrics.json`

Top-level:

- `pilot_metrics_aggregate.csv`
- `pilot_run_summary.json`
- `calibration_diagnostics/` (+ `calibration_aggregate.csv`)

## Invalid output + confidence availability

From `pilot_run_summary.json`:

- Beauty: valid `0.12` invalid / `0.88` confidence; test `0.09` invalid / `0.91` confidence
- Books: valid `0.61` invalid / `0.42` confidence; test `0.67` invalid / `0.35` confidence
- Electronics: valid `0.13` invalid / `0.87` confidence; test `0.16` invalid / `0.84` confidence
- Movies: valid `0.39` invalid / `0.63` confidence; test `0.48` invalid / `0.54` confidence

## Ranking metrics snapshot (HR/NDCG/MRR @5)

- Beauty: valid `HR@5=0.39 NDCG@5=0.288 MRR@5=0.254`; test `0.36 / 0.233 / 0.191`
- Books: valid `0.70 / 0.549 / 0.499`; test `0.69 / 0.585 / 0.551`
- Electronics: valid `0.58 / 0.392 / 0.330`; test `0.53 / 0.335 / 0.271`
- Movies: valid `0.42 / 0.310 / 0.274`; test `0.38 / 0.298 / 0.271`

## Calibration diagnostics (ECE/Brier/reliability)

`calibration_diagnostics/calibration_aggregate.csv` contains per domain/split:

- `ECE`, `adaptive_ECE`, `Brier`
- `high_confidence_wrong_rate`
- `head/mid/tail` confidence means and risk slices
- confidence-correctness AUC proxy

Examples:

- Beauty valid: `ECE=0.6384`, `Brier=0.5590`, `high_confidence_wrong_rate=0.8409`
- Books test: `ECE=0.4320`, `Brier=0.4371`, `high_confidence_wrong_rate=0.5429`

Reliability bins per split are in:

- `calibration_diagnostics/<domain>_<split>/reliability_bins.csv`

## Blockers before c99 or full experiment

1. Domain inconsistency in parse validity (books/movies still high invalid-output rates).
2. High-confidence-wrong remains substantial in several splits.
3. Calibration quality (ECE/Brier) is weak enough that confidence-based decisions need stronger controls.
4. This is still pilot-scale (`100u`, `c19`) and not sufficient for paper-scale conclusions.
