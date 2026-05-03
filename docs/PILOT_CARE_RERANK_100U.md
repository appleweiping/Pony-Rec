# CARE Rerank 100-User Pilot (c19, seed42)

Run scope: pilot-only, no paper claims, four domains, valid/test.

## Command

```bash
cd /home/ajifang/projects/fresh/uncertainty-llm4rec
.venv_lora/bin/python3.11 -m src.cli.run_care_rerank_pilot \
  --pilot_root outputs/pilots/deepseek_v4_flash_processed_100u_c19_seed42 \
  --output_root outputs/pilots/care_rerank_deepseek_v4_flash_processed_100u_c19_seed42 \
  --config configs/methods/care_rerank_pilot.yaml \
  --reprocess_dir outputs/reprocessed_processed_source_100u_c19_seed42
```

## Variants

- `original_deepseek`
- `confidence_only`
- `popularity_penalty_only`
- `uncertainty_only`
- `care_full`

## Required outputs

Under `outputs/pilots/care_rerank_deepseek_v4_flash_processed_100u_c19_seed42/`:

- per variant/domain/split:
  - `predictions/reranked_rank_predictions.jsonl`
  - `predictions/care_scores.jsonl`
  - `care_manifest.json`
  - `eval/metrics.json`
- top-level:
  - `care_rerank_aggregate.csv`
  - `exposure_shift.csv`
  - `high_confidence_wrong_changes.csv`

## CARE_full comparisons (requested pairwise)

From `care_rerank_aggregate.csv`:

- **CARE_full vs original_deepseek:** mostly identical outside beauty; small movement on beauty.
- **CARE_full vs confidence_only:** effectively identical in this pilot.
- **CARE_full vs popularity_penalty_only:** closest non-identical comparator on beauty (head-share reduction + slight metric shifts).
- **CARE_full vs uncertainty_only:** effectively identical in this pilot.

## Ranking / exposure / high-confidence-wrong summary

Observations from aggregate CSVs:

1. **Cross-domain consistency:** effect is **not uniformly strong**; most changes concentrate in `amazon_beauty`.
2. **Exposure shifts:** `care_full` reduces beauty head top-1 exposure (`0.08 -> 0.03` test; `0.04 -> 0.02` valid) and increases tail share correspondingly.
3. **Metrics impact:** beauty sees small trade-offs/improvements depending on split (`HR@5/NDCG@5/MRR@5` move modestly); other domains are nearly unchanged.
4. **High-confidence wrong:** little net reduction at 100u-c19 in this run; changed-top1 rate remains low.

## Fixed / hurt counts

`high_confidence_wrong_changes.csv` provides row-level before/after labels (`hc_wrong_before`, `hc_wrong_after`, `top1_changed`) for explicit fixed/hurt accounting by domain/split/variant.

## Interpretation guardrails

- No statistical significance claim is made here.
- This is pilot evidence for directionality and engineering viability only.
