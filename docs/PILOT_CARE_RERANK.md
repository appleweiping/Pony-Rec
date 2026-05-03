# CARE rerank pilot (DeepSeek 20-user, processed reprocess)

Pilot-only reranking over existing **listwise** `rank_predictions.jsonl` from  
`outputs/pilots/deepseek_v4_flash_processed_20u_c19_seed42/`.  
**`run_type=pilot`**, **`backend_type=rerank`**, **`is_paper_result=false`** on all emitted rows and manifests.

Outputs (local, not committed) live under:

`outputs/pilots/care_rerank_deepseek_v4_flash_processed_20u_c19_seed42/`

## Exact command

```bash
cd /home/ajifang/projects/fresh/uncertainty-llm4rec
.venv_lora/bin/python3.11 -m src.cli.run_care_rerank_pilot \
  --pilot_root outputs/pilots/deepseek_v4_flash_processed_20u_c19_seed42 \
  --output_root outputs/pilots/care_rerank_deepseek_v4_flash_processed_20u_c19_seed42 \
  --config configs/methods/care_rerank_pilot.yaml \
  --reprocess_dir outputs/reprocessed_processed_source
```

Prerequisites: per split, `predictions/uncertainty_features.jsonl` (from `run_uncertainty_probe`) and reprocess `*_candidates.jsonl`.

## Why this is not “A + B + C”

**CARE** here is a **single scoring policy** that treats listwise utility as a **base expected payoff**, then adjusts it with **reliability** (confidence minus **uncertainty risk**), **exposure harm** (**echo risk** only when stress signals suggest unreliable certainty—especially on **head** items), and a **tail recovery** term that rewards plausible mid/tail promotions only when global confidence is in a safe band.  
Isolated **variants** (below) turn off or blind parts of the objective so we can see which pressure moves rankings on n=20—**not** to claim optimal weights.

## Formula (default `care_full`)

Per candidate `i` (same slate `C` as DeepSeek):

- **`base_score(i)`** = `1 / log2(rank_model(i) + 1)` where `rank_model` is the position in the API’s list (after **sanitization**; see below).
- **`confidence_reliability(i)`**: use **`calibrated_confidence`** if finite; else **verbalized** `raw_confidence` **decayed** by rank:  
  `conf_global / (1 + decay * (rank−1))` — **one scalar per user** from the JSON; we do **not** have per-item verbalized probabilities (documented limitation).
- **`uncertainty_risk(i)`**: from `uncertainty_features.care_features` (verbalized uncertainty, missing-confidence boost, mild rank stress).
- **`echo_risk(i)`**: popularity bucket weight × **stress** (high when `high_confidence_wrong` or ≥0.7 confidence with **wrong@1** on the sanitized baseline). **Head** gets the largest weight; **tail** almost none—**no blind popularity penalty** in `care_full`.
- **`tail_recovery_bonus(i)`**: only for **tail/mid**, rank ≤6, and global confidence in **[0.32, 0.9]**.

**Total**

`score(i) = base(i) + α·(conf(i)−c₀) − β·unc(i) − γ·echo(i) + δ·tail(i)`  

with `c₀=0.5` (see `configs/methods/care_rerank_pilot.yaml` for α,β,γ,δ per variant).

## Slate sanitization

If the API list contains **IDs outside the slate** or **duplicates**, we **repair** to a permutation of `candidate_item_ids` (keep API order, drop bad IDs, append missing IDs in slate order).  
The **`original_deepseek`** variant is the **sanitized** baseline (may differ slightly from raw JSONL if the raw list was malformed). All variants use the same repaired slate for fair comparison.

## Variants (what each isolates)

| Variant | Intent |
|---------|--------|
| `original_deepseek` | **Identity** rerank: order = sanitized DeepSeek list (baseline). |
| `confidence_only` | Base + **α·(conf−0.5)** only. |
| `popularity_penalty_only` | Base − **γ·blind_popularity** (head>mid>tail) — **ablation** to show naive popularity penalty behavior. |
| `uncertainty_only` | Base − **β·uncertainty_risk**. |
| `care_full` | Full CARE terms as above. |

## Per-domain / split metrics (representative snapshot)

Values from `care_rerank_aggregate.csv` after the pilot run (HR@1 / HR@5; **n=20** — **noisy**).

| Domain | Split | Variant | HR@1 | HR@5 |
|--------|-------|---------|------|------|
| amazon_beauty | test | original_deepseek | 0.20 | 0.25 |
| amazon_beauty | test | popularity_penalty_only | 0.20 | **0.30** |
| amazon_beauty | test | care_full | 0.20 | 0.25 |
| amazon_beauty | valid | original_deepseek | 0.15 | 0.20 |
| amazon_beauty | valid | popularity_penalty_only | 0.15 | 0.20 |
| amazon_books | valid | all variants | 0.50 | 0.70 |
| amazon_electronics | test | all variants | 0.15 | 0.40 |

Several slices are **identical across variants** at HR@1: reranking rarely flips top-1 on this tiny pilot with current weights.

## Exposure shift (top-1 bucket shares)

See `exposure_shift.csv`. Example (**amazon_beauty / test**):

- **`popularity_penalty_only`**: `head_top1_share` **0.10 → 0.00** (drops head top-1 exposure entirely in this slice).
- **`care_full`**: **0.10 → 0.05** (one fewer head top-1 in 20 users).

## Examples: where CARE / variants move top-1

- **`care_full` + amazon_beauty + test**: user **`AE3335XF4PMHSXKTW5B7N7EALG3Q`** has **`top1_changed=True`** in `high_confidence_wrong_changes.csv` (still **HC wrong before and after**—rerank moved the top item but did not clear the high-confidence–wrong pattern because verbalized confidence is unchanged row-level).
- **`popularity_penalty_only`**: **2** top-1 changes on the full pilot CSV (mostly beauty test) — aligns with blind head penalty.

## Examples: where CARE can hurt ranking

- **`popularity_penalty_only` + amazon_beauty + valid`**: **NDCG@5** drops vs baseline (**0.333 → 0.317**) while HR@1 stays 0.15 — **listwise quality can degrade** when head items are down-weighted even if they were relevant.
- Blind popularity penalty can **reduce HR@5** without improving HR@1 (beauty test: 0.25→0.30 actually **increases** HR@5 here; check **electronics / movies** rows in the CSV for mixed behavior).

## Answering the diagnostic questions (this pilot only)

1. **Confidence vs correctness?** Still weak globally (see `confidence_correctness_auc_diag` column mirroring calibration CSV); rerank **does not fix** verbalized calibration.
2. **Wrong answers low confidence?** Not required for CARE; we still see many **HC-wrong** rows unchanged after rerank.
3. **HC-wrong exists after rerank?** **Yes** — same `raw_confidence` is carried; HC metric is recomputed on **new** top-1.
4. **HC-wrong concentrated on head?** **Inconclusive** — most slates are **tail-heavy**; `head_prediction_rate_after` is often **0**.
5. **Tail targets lower confidence?** Not modeled per-target; decay only by **rank position**.
6. **Usable confidence fields?** **Verbalized + decay** only; no logits / self-consistency in this pilot file.
7. **Ready for larger pilots?** **Engineering yes** (pipeline + artifacts); **science no** until weights tuned and n↑.
8. **Before 100-user / domain:** tune α–δ on **held-out** split; add **per-item** or **logit** channels; validate **target-in-slate** more; separate **calibration fit** from rerank policy.

## How this differs from other systems

| System | Difference |
|--------|----------------|
| **RecBole / classical** | Score-based ranking + optional calibrated **p(item)** from logits; CARE pilot here has **no per-item LLM probabilities**. |
| **DeepSeek baseline** | Single listwise JSON + one **verbalized** scalar; CARE consumes **probe features** + **popularity buckets** + **risk stress**. |
| **Naive popularity rerank** | `popularity_penalty_only` **deliberately** blinds stress—shows why CARE uses **conditioned** echo risk instead. |

## Artifacts per `variant/domain/split`

- `predictions/reranked_rank_predictions.jsonl`
- `predictions/care_scores.jsonl`
- `care_manifest.json`
- `eval/metrics.json` (includes `care_rerank_pilot` block: exposure shift, auxiliary rates, …)
- Repo root under output: **`care_rerank_aggregate.csv`**, **`exposure_shift.csv`**, **`high_confidence_wrong_changes.csv`**, **`pilot_run_meta.json`**

## Code map

- `src/methods/care_rerank.py` — scoring + `sanitize_listwise_ranking`
- `src/cli/run_care_rerank_pilot.py` — batch driver + CSV aggregates
- `configs/methods/care_rerank_pilot.yaml` — weights / variant switches
- `src/utils/manifest.py` — `backend_type_from_name("rerank") == "rerank"`

## Tests

```bash
.venv_lora/bin/python3.11 -m pytest tests/test_care_rerank.py tests/test_phase1_backend_manifest_baselines.py::test_backend_type_rerank_maps -q
```
