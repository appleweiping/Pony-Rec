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

Batch (all pilots under a tree). Sub-runs are written to `{output_dir}/{domain}_{split}/`. After a batch, **`calibration_aggregate.csv`** is written under `{output_dir}/` unless `--no_aggregate` is set.

```bash
.venv_lora/bin/python3.11 -m src.cli.run_calibration_diagnostics \
  --batch_glob "outputs/pilots/deepseek_v4_flash_processed_20u_c19_seed42/**/rank_predictions.jsonl" \
  --output_dir outputs/pilots/deepseek_v4_flash_processed_20u_c19_seed42/calibration_diagnostics \
  --confidence_field auto \
  --num_bins 10
```

If `--features_path` is omitted, the CLI looks for `uncertainty_features.jsonl` beside `rank_predictions.jsonl` when present. Generate it with `python -m src.cli.run_uncertainty_probe --predictions_path <rank_predictions.jsonl>` if missing.

## Outputs

Per run directory:

| File | Role |
|------|------|
| `calibration_rows.jsonl` | Per-user: `is_correct_at_1`, `target_rank`, `confidence`, `confidence_source`, popularity flags, risk flags, optional channel values. |
| `calibration_summary.json` | Global ECE/MCE, Brier, adaptive ECE, NLL, AUROC, high/low-confidence rates, `compare_confidence_sources`. |
| `reliability_bins.csv` | Equal-width reliability table for plots. |
| `group_calibration_by_popularity.csv` | ECE/Brier/AUROC per `popularity_bucket`. |
| `high_confidence_error_by_bucket.csv` | Share of **wrong@1** when confidence ≥ 0.7, stratified by bucket. |
| `calibration_aggregate.csv` | **Batch only:** one row per domain×split with global ECE/Brier/AUROC and popularity-conditioned means (see below). |

## DeepSeek 20-user pilot findings (processed reprocess, 4 domains × valid/test)

Artifacts (local, not committed): `outputs/pilots/deepseek_v4_flash_processed_20u_c19_seed42/calibration_diagnostics/calibration_aggregate.csv` plus per-split folders. All **20/20** rows had finite verbalized confidence (`confidence_available_rate = 1.0`). **Entropy / margin / rank_score** channels stayed empty in `compare_confidence_sources` for this pilot—only **verbalized** confidence is usable today.

**Aggregate snapshot** (rounded; full precision in CSV):

| domain | split | ECE | adaptive_ECE | Brier | high_conf_wrong | low_conf_correct | AUROC |
|--------|-------|-----|----------------|-------|----------------|------------------|-------|
| amazon_beauty | valid | 0.57 | 0.60 | 0.51 | 0.87 | 0.20 | 0.38 |
| amazon_beauty | test | 0.67 | 0.64 | 0.58 | 0.83 | 0.50 | 0.21 |
| amazon_books | valid | 0.42 | 0.45 | 0.43 | 0.50 | 0.50 | 0.54 |
| amazon_books | test | 0.50 | 0.50 | 0.45 | 0.89 | 0.36 | 0.32 |
| amazon_electronics | valid | 0.66 | 0.66 | 0.54 | 0.89 | 0.00 | 0.81 |
| amazon_electronics | test | 0.57 | 0.57 | 0.49 | 0.83 | 0.00 | 0.38 |
| amazon_movies | valid | 0.32 | 0.25 | 0.28 | 0.70 | 0.11 | 0.70 |
| amazon_movies | test | 0.31 | 0.24 | 0.29 | 0.73 | 0.11 | 0.57 |

### Questions this slice answers

1. **Does confidence correlate with correctness?** **Weakly or inversely** in several slices: AUROC is often **below 0.5** (beauty valid/test, books test, electronics test), meaning verbalized confidence is **not** a reliable ranker of hit@1 correctness at n=20. Electronics **valid** is an exception (AUROC ≈ 0.81) but still has high ECE—discrimination can coexist with poor calibration on tiny n.

2. **Are wrong answers usually low confidence?** **No.** `high_confidence_wrong_rate` is **0.5–0.89** across splits when conditioning on confidence ≥ 0.7: many errors stay **high-confidence**.

3. **Do high-confidence wrong predictions exist?** **Yes**, consistently—this is the main qualitative signal for CARE (reliability-aware reranking).

4. **Are high-confidence wrong predictions concentrated on head/popular items?** **Cannot conclude from this pilot:** `head_confidence_mean` and `head_high_confidence_wrong_rate` are **NaN** because **no row used `head` popularity bucket** in these 20-user samples (reprocess buckets are almost entirely **tail/mid**). Use larger n or explicit head-heavy evaluation before claiming head concentration.

5. **Are tail targets lower-confidence?** **Tail mean confidence** is **0.32–0.76** by domain/split—tails are **not** systematically low-confidence; the model often assigns **high verbalized confidence** even for tail targets.

6. **Which confidence fields are usable?** **Verbalized (`raw_confidence`) only** for DeepSeek listwise pilots here. Parsed probability / logits / self-consistency were not populated in `rank_predictions.jsonl`.

7. **Ready for CARE rerank pilot?** **Engineering yes, science cautious:** the pipeline and aggregates are in place, but **raw verbalized scores should not be treated as probabilities** without a transform (temperature scaling, isotonic fit on held-out split, or rank-only CARE utility). CARE rerank can start with **rank-based** or **monotone-transformed** confidence.

8. **Before 100-user / domain scaling:** (a) increase n to populate **head/mid/tail** buckets; (b) log **logit / margin** channels where available; (c) stabilize AUROC/ECE variance with more users; (d) align **baseline** exports so score-based confidence can be compared under the same protocol.

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

- `src/uncertainty/calibration.py` — metrics + `build_calibration_diagnostic_rows`, `summarize_calibration_diagnostics`, **`aggregate_calibration_batch`** (isotonic/Platt fitters remain unchanged above in the same file).
- `src/cli/run_calibration_diagnostics.py` — CLI and CSV writers; batch mode writes **`calibration_aggregate.csv`**.
- Master plan: `docs/CODEX_TOPCONF_UNCERTAINTY_LLM4REC_PLAN.md`.
