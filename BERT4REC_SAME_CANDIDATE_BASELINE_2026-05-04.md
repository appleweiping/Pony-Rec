# BERT4Rec Same-Candidate Baseline - 2026-05-04

This note records the third completed classical external baseline imported
through the Week8 same-candidate adapter.

## Status

- Baseline: `bert4rec`
- Status label: `same_schema_external_baseline`
- Artifact class: `completed_result`
- Protocol: train on exported train interactions, score the exact candidate
  rows, import through `main_import_same_candidate_baseline_scores.py`
- Coverage: `score_coverage_rate=1.0` on all four checked domains
- Score audit: `invalid_scores=0` and exact event-key coverage `1.0`

## Four-Domain Results

| domain | sample_count | NDCG@10 | MRR | note |
| --- | ---: | ---: | ---: | --- |
| beauty | 973 | 0.5381954605720659 | 0.3929085303186023 | Full exact-candidate coverage. |
| books | 500 | 0.4585595147515817 | 0.2901 | Full exact-candidate coverage. |
| electronics | 500 | 0.475589826877732 | 0.3115333333333334 | Full exact-candidate coverage. |
| movies | 500 | 0.47877585684061374 | 0.31626666666666664 | Full exact-candidate coverage. |

## Interpretation

BERT4Rec is a valid same-schema external baseline row. It is below GRU4Rec on
the four checked domains and remains below the best Week7.7/Week7.9 method
rows in the unified matrix.

Together with SASRec and GRU4Rec, it gives the paper table a three-model
classical sequential baseline block under the exact same candidate, split, and
metric protocol.

## Suite Status

LightGCN has now also been completed through the same adapter. The classical
same-candidate suite is complete for SASRec/GRU4Rec/BERT4Rec/LightGCN.
