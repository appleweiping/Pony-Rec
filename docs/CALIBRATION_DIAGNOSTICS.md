# Calibration diagnostics (CARE-Rec, pilot-scale)

This pipeline is **diagnostic engineering**, not a paper result. It measures how well a declared confidence (or score-derived proxy) matches **hit@1 correctness** under the same listwise setting as ranking evaluation.

## Command

Single run (DeepSeek / LoRA `rank_predictions.jsonl`):

```bash
.venv_lora/bin/python3.11 -m src.cli.run_calibration_diagnostics \
  --predictions_path outputs/pilots/deepseek_v4_flash_processed_20u_c19_seed42/amazon_beauty/valid/predictions/rank_predictions.jsonl \
  --output_dir tmp_outputs/calib_diag_deepseek_beauty_valid \
  --confidence_field auto \
  --num_bins 10
```

Optional merge with `uncertainty_features.jsonl` (same `user_id`):

```bash
.venv_lora/bin/python3.11 -m src.cli.run_calibration_diagnostics \
  --predictions_path <.../rank_predictions.jsonl> \
  --features_path <.../uncertainty_features.jsonl> \
  --output_dir <out>
```

Batch (all pilots under a tree):

```bash
.venv_lora/bin/python3.11 -m src.cli.run_calibration_diagnostics \
  --batch_glob "outputs/pilots/deepseek_v4_flash_processed_20u_c19_seed42/**/rank_predictions.jsonl" \
  --output_dir tmp_outputs/calib_diag_batch_deepseek
```

If `--features_path` is omitted, the CLI looks for `uncertainty_features.jsonl` beside `rank_predictions.jsonl` when present.

## Outputs

Per run directory:

| File | Role |
|------|------|
| `calibration_rows.jsonl` | Per-user: `is_correct_at_1`, `target_rank`, `confidence`, `confidence_source`, popularity flags, risk flags, optional channel values. |
| `calibration_summary.json` | Global ECE/MCE, Brier, adaptive ECE, NLL, AUROC, high/low-confidence rates, `compare_confidence_sources`. |
| `reliability_bins.csv` | Equal-width reliability table for plots. |
| `group_calibration_by_popularity.csv` | ECE/Brier/AUROC per `popularity_bucket`. |
| `high_confidence_error_by_bucket.csv` | Share of **wrong@1** when confidence ≥ 0.7, stratified by bucket. |

## Metrics (interpretation)

- **ECE / MCE (equal-width):** average (respectively max) gap between mean confidence and accuracy per bin. Lower is better for a **probability** meant to represent P(correct@1).
- **Adaptive ECE:** equal-mass bins on sorted confidence (common variant); useful when mass piles at extremes.
- **Brier:** mean squared error between confidence and binary correctness; strictly proper for probabilistic forecasts.
- **NLL:** binary negative log-likelihood when values are treated as probabilities in (0,1).
- **AUROC:** discrimination—whether higher confidence ranks correct instances above incorrect ones (manual implementation, no sklearn).

## Verbalized vs score-derived confidence

- **LLM verbalized** (`raw_confidence` from JSON) is **not** a calibrated probability unless post-hoc calibration is fitted and validated.
- **Score / margin / entropy** channels require **per-candidate scores or probabilities** in `rank_predictions.jsonl` (e.g. `candidate_probabilities`, `top1_score`). Current API listwise pilots often only expose verbalized confidence; other columns appear as `null` / NaN in summaries.
- `compare_confidence_sources` reports **which channels have enough finite data** (`n_valid`) before trusting ECE/Brier.

## Why popularity-stratified calibration

Echo-chamber and **popularity bias** couple with **miscalibrated certainty**: models can be overconfident on **head** items and underconfident on **tail** items. Group ECE and `high_confidence_error_by_bucket.csv` surface **confidence–popularity–error** structure needed for CARE exposure framing—not only a global scalar.

## RecBole / classical baselines

RecBole smoke outputs in this repo are **aggregate metrics**, not per-user `rank_predictions.jsonl` with scores. **Per-user top-1 scores or calibrated probabilities** are required to run the same diagnostics on classical baselines. Until exports include those fields, document **TODO**: extend RecBole export to write `rank_predictions.jsonl`-compatible rows with `top1_score` or `candidate_probabilities` (and correctness@1 under the same candidate protocol).

## Blockers before 100-user / multi-domain scale

1. **Sparse channels:** entropy/margin need vector scores; missing data dominates summaries.
2. **Hit@1 only:** diagnostics align with current pilot `correctness`; HR@K / Recall@K remain in `evaluate`, not duplicated here.
3. **Calibration vs ranking:** good NDCG does not imply calibrated confidence—report both separately.
4. **Split hygiene:** diagnostics on `valid`/`test` should not leak into training calibration fits (this CLI **fits nothing**; it only measures).

## Code map

- `src/uncertainty/calibration.py` — metrics + `build_calibration_diagnostic_rows`, `summarize_calibration_diagnostics` (isotonic/Platt fitters remain unchanged above in the same file).
- `src/cli/run_calibration_diagnostics.py` — CLI and CSV writers.
- Master plan: `docs/CODEX_TOPCONF_UNCERTAINTY_LLM4REC_PLAN.md`.
