# 100-User DeepSeek + CARE Rerank Pilot (candidate_size=99, seed42)

Scope: four domains, `max_users_per_domain=100`, **candidate_size=99**, `seed=42`, **pilot only**.  
`run_type=pilot`, `is_paper_result=false` on manifests. **No full experiment**, no statistical significance claims.

Compare baseline: **c19** (`outputs/reprocessed_processed_source_100u_c19_seed42` + `outputs/pilots/deepseek_v4_flash_processed_100u_c19_seed42` + rerank root `..._c19_...`).  
This run: **c99** reprocess + DeepSeek + uncertainty/calibration + CARE rerank as below.

### Gate status (read before any new experiment)

- **c99 is pilot only:** `run_type=pilot`, **`is_paper_result=false`** on DeepSeek manifests; this doc is not a paper claim.
- **Invalid rates did not blow up** vs the pre-run gate check (books/movies remain the weak domains; see §3).
- **CARE rerank c99 outputs are ready for human review** (aggregate CSVs + per-variant `eval/metrics.json` under the rerank root).
- **Do not proceed to a full experiment** until c99 findings (invalid/calibration/CARE deltas) are **manually accepted** and any domain-specific follow-ups are decided.

---

## 1) Exact commands

### Phase A — Reprocess (c99)

```bash
cd /home/ajifang/projects/fresh/uncertainty-llm4rec
.venv_lora/bin/python3.11 -m src.cli.reprocess_processed_source \
  --source_root data/processed \
  --output_dir outputs/reprocessed_processed_source_100u_c99_seed42 \
  --max_users_per_domain 100 \
  --candidate_size 99 \
  --seed 42
```

### Phase B — DeepSeek API pilot (c99)

```bash
.venv_lora/bin/python3.11 -m src.cli.run_pilot_reprocessed_deepseek \
  --reprocess_dir outputs/reprocessed_processed_source_100u_c99_seed42 \
  --output_root outputs/pilots/deepseek_v4_flash_processed_100u_c99_seed42 \
  --backend_config configs/backends/deepseek_v4_flash.yaml \
  --domains amazon_beauty amazon_books amazon_electronics amazon_movies \
  --splits valid test \
  --seed 42 \
  --run_type pilot \
  --method llm_listwise
```

`listwise_ranking_v1` slate length is **inferred from `candidate_item_ids`** per JSONL (CLI `--topk` optional override).

### Phase C — Uncertainty probe (per `rank_predictions.jsonl`)

```bash
while IFS= read -r -d '' f; do
  .venv_lora/bin/python3.11 -m src.cli.run_uncertainty_probe --predictions_path "$f"
done < <(find outputs/pilots/deepseek_v4_flash_processed_100u_c99_seed42 -name rank_predictions.jsonl -print0)
```

### Phase C — Calibration batch

```bash
.venv_lora/bin/python3.11 -m src.cli.run_calibration_diagnostics \
  --batch_glob "outputs/pilots/deepseek_v4_flash_processed_100u_c99_seed42/**/predictions/rank_predictions.jsonl" \
  --output_dir outputs/pilots/deepseek_v4_flash_processed_100u_c99_seed42/calibration_diagnostics \
  --confidence_field auto \
  --num_bins 10
```

Per-split outputs include: `reliability_bins.csv`, `group_calibration_by_popularity.csv`, `high_confidence_error_by_bucket.csv`, `calibration_summary.json`, `calibration_rows.jsonl`, plus top-level `calibration_aggregate.csv`.

### Phase D — CARE rerank (c99)

```bash
.venv_lora/bin/python3.11 -m src.cli.run_care_rerank_pilot \
  --pilot_root outputs/pilots/deepseek_v4_flash_processed_100u_c99_seed42 \
  --output_root outputs/pilots/care_rerank_deepseek_v4_flash_processed_100u_c99_seed42 \
  --config configs/methods/care_rerank_pilot.yaml \
  --reprocess_dir outputs/reprocessed_processed_source_100u_c99_seed42
```

Variants: `original_deepseek`, `confidence_only`, `popularity_penalty_only`, `uncertainty_only`, `care_full`.

---

## 2) Reprocess verification (Phase A)

Root: `outputs/reprocessed_processed_source_100u_c99_seed42/`

| Check | Result |
| --- | --- |
| Domains | `amazon_beauty`, `amazon_books`, `amazon_electronics`, `amazon_movies` |
| Rows per split | **100 / 100 / 100** train / valid / test per domain |
| `candidate_size` | **99** on all splits (reprocess log: single bucket `[99]` per split) |
| Leakage | `leakage_passed=True` per domain in CLI log |
| Popularity | `popularity_source: train_only` in each domain `manifest.json` |
| Manifests | `manifest.json` + `leakage_report.json` per domain |
| Path hygiene | No `srpd` / old split / old prediction path strings in sampled `manifest.json` bodies (spot-check); outputs reference `data/processed/...` and this reprocess root only |

---

## 3) DeepSeek invalid rate: c99 vs c19

Source: `pilot_run_summary.json` in each pilot root (`invalid_output_rate`, `confidence_availability_rate`).

| Domain | Split | c19 invalid | c19 conf. avail. | c99 invalid | c99 conf. avail. |
| --- | --- | --- | --- | --- | --- |
| beauty | valid | 0.12 | 0.88 | **0.20** | **0.96** |
| beauty | test | 0.09 | 0.91 | **0.15** | **0.95** |
| books | valid | 0.61 | 0.42 | **0.50** | **0.68** |
| books | test | 0.67 | 0.35 | **0.59** | **0.59** |
| electronics | valid | 0.13 | 0.87 | **0.35** | **0.95** |
| electronics | test | 0.16 | 0.84 | **0.29** | **0.97** |
| movies | valid | 0.39 | 0.63 | **0.32** | **0.80** |
| movies | test | 0.48 | 0.54 | **0.41** | **0.72** |

**Gate (books/movies):** invalid rates did **not** explode (no runaway toward ~1.0). **Beauty** and **electronics** invalid rates **rose vs c19** under the larger slate; **books** invalid improved modestly vs c19 but remains **~0.5–0.59**; **movies** mixed (valid improved, test still high). Rerank phase was **not** aborted.

---

## 4) Confidence availability (pilot summary)

See table in §3 (`confidence_availability_rate`). Interpreting together with §3: c99 shows **higher** numeric confidence availability on beauty/electronics than c19 in this summary, while **books/movies** remain the weakest domains.

---

## 5) Calibration comparison (c99 vs c19)

Source: `calibration_diagnostics/calibration_aggregate.csv` in each DeepSeek pilot root.  
Rows below: `ECE`, `adaptive_ECE`, `Brier`, `high_confidence_wrong_rate`, and **head/mid/tail** `*_confidence_mean` where present (CSV uses `nan` when a bucket is empty).

### amazon_beauty

| Split | Metric | c19 | c99 |
| --- | --- | --- | --- |
| valid | ECE | 0.6384 | **0.7890** |
| valid | adaptive ECE | 0.6384 | **0.7890** |
| valid | Brier | 0.5590 | **0.7050** |
| valid | high-conf wrong | 0.8409 | **0.9375** |
| valid | head/mid/tail conf mean | 0.838 / 0.724 / 0.721 | **0.903 / 0.825 / 0.850** |
| test | ECE | 0.6824 | **0.7923** |
| test | adaptive ECE | 0.6668 | **0.7923** |
| test | Brier | 0.5833 | **0.7081** |
| test | high-conf wrong | 0.9011 | **0.9474** |
| test | head/mid/tail conf mean | 0.674 / 0.740 / 0.758 | **0.906 / 0.796 / 0.844** |

### amazon_books

| Split | Metric | c19 | c99 |
| --- | --- | --- | --- |
| valid | ECE | 0.4040 | **0.5233** |
| valid | adaptive ECE | 0.3752 | **0.5033** |
| valid | Brier | 0.3968 | **0.4681** |
| valid | high-conf wrong | 0.6190 | **0.8088** |
| valid | head/mid/tail | nan | nan |
| test | ECE | 0.4320 | **0.5705** |
| test | adaptive ECE | 0.3964 | **0.5705** |
| test | Brier | 0.4371 | **0.5174** |
| test | high-conf wrong | 0.5429 | **0.8814** |
| test | head/mid/tail | nan | nan |

### amazon_electronics

| Split | Metric | c19 | c99 |
| --- | --- | --- | --- |
| valid | ECE | 0.5285 | **0.7491** |
| valid | adaptive ECE | 0.5085 | **0.7491** |
| valid | Brier | 0.4681 | **0.6472** |
| valid | high-conf wrong | 0.7816 | **0.9368** |
| valid | head/mid/tail | nan / 0.0 / 0.706 | nan / **0.85** / **0.809** |
| test | ECE | 0.5836 | **0.7652** |
| test | adaptive ECE | 0.5836 | **0.7652** |
| test | Brier | 0.4944 | **0.6603** |
| test | high-conf wrong | 0.8690 | **0.9381** |
| test | head/mid/tail | nan / 0.82 / 0.672 | nan / **0.82** / **0.825** |

### amazon_movies

| Split | Metric | c19 | c99 |
| --- | --- | --- | --- |
| valid | ECE | 0.3818 | **0.5663** |
| valid | adaptive ECE | 0.3818 | **0.5663** |
| valid | Brier | 0.3328 | **0.4738** |
| valid | high-conf wrong | 0.7414 | **0.8861** |
| valid | head/mid/tail | 0.75 / 0.0 / 0.476 | **0.39 / 0.39 / 0.657** |
| test | ECE | 0.4494 | **0.5594** |
| test | adaptive ECE | 0.4294 | **0.5584** |
| test | Brier | 0.3877 | **0.4626** |
| test | high-conf wrong | 0.8519 | **0.9437** |
| test | head/mid/tail | 0.5 / 0.39 / 0.407 | **0.273 / 0.815 / 0.594** |

**Reading:** moving from **19 to 99 candidates** makes the listwise task harder. In this pilot, **ECE / Brier / high-confidence-wrong** generally look **worse in c99 than c19**, especially beauty + electronics + movies test — i.e. miscalibration and overconfident-wrong risk are **more visible**, not cleaner, at c99.

---

## 6) CARE rerank c99 (`CARE_full` vs baselines)

Source: `outputs/pilots/care_rerank_deepseek_v4_flash_processed_100u_c99_seed42/care_rerank_aggregate.csv` (HR@5 / NDCG@5 / MRR@5; exposure from `exposure_shift.csv`).

### 6.0 Snapshot — `original_deepseek` vs `care_full` (HR@5 / NDCG@5 / MRR@5)

| domain | split | original HR@5 | care_full HR@5 | original NDCG@5 | care_full NDCG@5 |
| --- | --- | --- | --- | --- | --- |
| beauty | valid | 0.14 | 0.14 | 0.1027 | 0.1003 |
| beauty | test | 0.17 | 0.17 | 0.1130 | 0.1123 |
| books | valid | 0.42 | 0.42 | 0.3130 | 0.3130 |
| books | test | 0.44 | 0.44 | 0.3299 | 0.3299 |
| electronics | valid | 0.19 | 0.19 | 0.1168 | 0.1168 |
| electronics | test | 0.18 | 0.18 | 0.1245 | 0.1245 |
| movies | valid | 0.16 | 0.16 | 0.1089 | 0.1089 |
| movies | test | 0.19 | 0.19 | 0.1280 | 0.1280 |

### 6.1 `CARE_full` vs `original_deepseek`

Ranking deltas are **small** on most cells at c99 (same HR@5 on many domains/splits). **Beauty** shows the largest relative movement vs c19-era pattern but absolute HR@5 is **much lower** than c19 because the slate is harder (e.g. beauty test `HR@5`: c19 original ~`0.36` vs c99 original ~`0.17`).

**Exposure (`exposure_shift.csv`, beauty):** `care_full` slightly shifts top-1 mass toward tail vs `original_deepseek` (e.g. test head top-1 share `0.04 → 0.03`; valid `0.03 → 0.02`). **Books/electronics:** rerank leaves top-1 bucket shares **unchanged** in the recorded CSV (all tail top-1 `1.0` before/after for books; electronics similarly `0` head).

### 6.2 `CARE_full` vs `confidence_only`

**Identical** metrics on all domain/split rows in `care_rerank_aggregate.csv` for this run.

### 6.3 `CARE_full` vs `popularity_penalty_only`

**Beauty** shows small NDCG@5 / MRR@5 differences vs `popularity_penalty_only` (popularity variant moves head share more aggressively; see exposure rows for `popularity_penalty_only` on beauty). Other domains: **no** material HR@5 delta in the aggregate CSV.

### 6.4 `CARE_full` vs `uncertainty_only`

**Identical** to `original_deepseek` / `confidence_only` on all listed HR@5 cells (aggregate CSV).

### 6.5 High-confidence wrong (`high_confidence_wrong_changes.csv`)

Rerank **does not** collapse high-confidence-wrong rates in aggregate: `high_confidence_wrong_rate_before` ≈ `after` for most variants (beauty still ~`0.9` HC-wrong after). Use the CSV for row-level fixed/hurt counts if needed.

---

## 7) c99 vs c19 interpretation

1. **Stronger CARE signal?** **Not clearly.** At c99, ranking deltas vs `original_deepseek` remain **small**; the dominant effect is still **harder listwise ranking** (HR@5 drops vs c19), not a larger separation between CARE variants.
2. **Invalid rate worse?** **Mixed:** beauty + electronics **worse** than c19; books **slightly better** but still poor; movies **mixed**. None exploded past an emergency stop threshold.
3. **High-confidence wrong more visible?** **Yes in calibration aggregates:** beauty/electronics/movies show **higher** high-confidence-wrong and **higher** ECE/Brier in c99 vs c19 on several splits.
4. **Books/movies unstable?** **Yes.** Books invalid stays **~0.5–0.59**; calibration head/mid means are often **`nan`** for books (sparse buckets). Movies: improved invalid on valid vs c19, but calibration **worsens** on several metrics vs c19.

---

## 8) Decision gate (pilot-level)

| Question | Stance |
| --- | --- |
| Continue toward “full” c99 paper experiment? | **No.** Stay **pilot-only** until parse/invalid rates and calibration are under control on **books/movies** (and beauty/electronics regressions are understood). |
| Tune CARE coefficients? | **Optional small ablation**, but **not** the primary lever while listwise invalid + calibration remain this poor. |
| Improve prompt/parser for unstable domains? | **Yes, high priority** for books/movies (and monitoring beauty/electronics invalid upticks at large K). |
| Keep c99 as pilot only? | **Yes.** Treat this document as **engineering evidence**, not a paper result. |

---

## 9) Evidence class statement

- `run_type=pilot`, `is_paper_result=false` on DeepSeek per-split `manifest.json` (example: beauty valid manifest records `candidate_size: 99`).
- **No statistical significance** is claimed; `n=100` per split per domain is exploratory.

---

## 10) Output roots (artifacts)

| Phase | Path |
| --- | --- |
| A | `outputs/reprocessed_processed_source_100u_c99_seed42/` |
| B | `outputs/pilots/deepseek_v4_flash_processed_100u_c99_seed42/` |
| C | `.../calibration_diagnostics/` under DeepSeek root |
| D | `outputs/pilots/care_rerank_deepseek_v4_flash_processed_100u_c99_seed42/` |

DeepSeek per domain/split: `predictions/raw_responses.jsonl`, `parsed_responses.jsonl`, `rank_predictions.jsonl`, `manifest.json`, `eval/metrics.json`, plus top-level `pilot_run_summary.json`, `pilot_metrics_aggregate.csv`.
