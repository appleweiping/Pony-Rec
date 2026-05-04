# SASRec Same-Candidate Baseline - 2026-05-04

This note records the first completed classical external baseline imported
through the Week8 same-candidate adapter.

## Status

- Baseline: `sasrec`
- Status label: `same_schema_external_baseline`
- Artifact class: `completed_result`
- Protocol: train on exported train interactions, score the exact candidate
  rows, import through `main_import_same_candidate_baseline_scores.py`
- Coverage: `score_coverage_rate=1.0` on all four checked domains

## Four-Domain Results

| domain | sample_count | NDCG@10 | MRR |
| --- | ---: | ---: | ---: |
| beauty | 973 | 0.5196854966540216 | 0.36894484412470024 |
| books | 500 | 0.4319889458564126 | 0.2572 |
| electronics | 500 | 0.45339197017757793 | 0.2846 |
| movies | 500 | 0.4604505457188672 | 0.2931666666666667 |

## Interpretation

SASRec is now a valid same-schema external baseline row, not a proxy and not a
reported-number comparison.

It is weaker than Week7.7 direct/structured-risk/SRPD and weaker than the
current shadow_v6 diagnostic bridge on the checked four-domain matrix. This is
useful evidence: the main method line is no longer only compared against simple
toy baselines, and the table now has a classical sequential recommender
baseline under the exact same candidate and metric protocol.

## Next Baseline

GRU4Rec has now been completed as the second lightweight external baseline:

- `main_train_gru4rec_same_candidate.py`

BERT4Rec and LightGCN remain the next candidates.
