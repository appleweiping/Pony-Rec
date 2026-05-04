# LightGCN Same-Candidate Baseline - 2026-05-04

This note records the fourth completed classical external baseline imported
through the Week8 same-candidate adapter.

## Status

- Baseline: `lightgcn`
- Status label: `same_schema_external_baseline`
- Artifact class: `completed_result`
- Protocol: train on exported train interactions, score the exact candidate
  rows, import through `main_import_same_candidate_baseline_scores.py`
- Coverage: `score_coverage_rate=1.0` on all four checked domains
- Score audit: `invalid_scores=0` and exact event-key coverage `1.0`
- Unified matrix: `outputs/summary/unified_method_matrix_week77_shadow_external.csv`
  rebuilt with `rows=46`

## Four-Domain Results

| domain | sample_count | NDCG@10 | MRR | note |
| --- | ---: | ---: | ---: | --- |
| beauty | 973 | 0.6277046260108826 | 0.5072285029119561 | Full exact-candidate coverage. |
| books | 500 | 0.5157244751873231 | 0.35986666666666667 | Full exact-candidate coverage. |
| electronics | 500 | 0.516161581801384 | 0.3609666666666667 | Full exact-candidate coverage. |
| movies | 500 | 0.5275590389741664 | 0.3751333333333333 | Full exact-candidate coverage. |

## Interpretation

LightGCN is a valid same-schema external baseline row. It is the strongest of
the four classical external baselines on Beauty in the current server run, but
it remains below the best Week7.7/Week7.9 method rows in the unified matrix.

With SASRec, GRU4Rec, BERT4Rec, and LightGCN complete, the classical external
baseline suite is now reproducible under the exact same candidate, split, and
metric protocol.

## Next Step

Use `unified_method_matrix_week77_shadow_external.csv` as the method-audit
source table, then run paired significance tests before making winner wording
in the paper.
