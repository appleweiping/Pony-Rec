# GRU4Rec Same-Candidate Baseline - 2026-05-04

This note records the second completed classical external baseline imported
through the Week8 same-candidate adapter.

## Status

- Baseline: `gru4rec`
- Status label: `same_schema_external_baseline`
- Artifact class: `completed_result`
- Protocol: train on exported train interactions, score the exact candidate
  rows, import through `main_import_same_candidate_baseline_scores.py`
- Coverage: `score_coverage_rate=1.0` on all four checked domains
- Score audit: `invalid_scores=0` and exact event-key coverage `1.0`

## Four-Domain Results

| domain | sample_count | NDCG@10 | note |
| --- | ---: | ---: | --- |
| beauty | 973 | 0.5389659521392343 | Full exact-candidate coverage. |
| books | 500 | 0.5405199911085883 | Full exact-candidate coverage. |
| electronics | 500 | 0.5460856977197672 | Full exact-candidate coverage. |
| movies | 500 | 0.5364788038890503 | Full exact-candidate coverage. |

## Interpretation

GRU4Rec is a valid same-schema external baseline row. It is stronger than the
first SASRec run in all four checked domains, but it still remains below the
best Week7.7/Week7.9 method rows in the unified matrix.

This strengthens the baseline story: the main SRPD/shadow evidence is now
compared against two classical sequential recommender baselines under the exact
same candidate, split, and metric protocol.

## Next Baseline

The next classical candidate is BERT4Rec:

- Bidirectional masked-sequence recommender.
- Same export/import protocol.
- Same `source_event_id,user_id,item_id,score` score file contract.
