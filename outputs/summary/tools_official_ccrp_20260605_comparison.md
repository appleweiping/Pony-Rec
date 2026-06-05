# tools official baseline vs C-CRP comparison

Read-only same-candidate comparison. This table does not start or rerun experiments.

| Rank | Method | Kind | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `ccrp_v3_qwen3base_pointwise` | internal_method | 0.193700 | 0.269600 | 0.393100 | 0.141864 | 0.166116 | 0.197040 | 0.155859 |
| 2 | `llmemb` | official_baseline | 0.136500 | 0.225700 | 0.363700 | 0.087458 | 0.115944 | 0.150506 | 0.106494 |
| 3 | `irllrec_intent` | official_baseline | 0.102000 | 0.165100 | 0.309500 | 0.065047 | 0.085257 | 0.121008 | 0.086702 |
| 4 | `llm2rec_sasrec` | official_baseline | 0.095700 | 0.162500 | 0.295400 | 0.060227 | 0.081479 | 0.114812 | 0.081014 |
| 5 | `rlmrec_graphcl` | official_baseline | 0.078400 | 0.135400 | 0.246500 | 0.050175 | 0.068389 | 0.095993 | 0.072201 |
| 6 | `llmesr_sasrec` | official_baseline | 0.071100 | 0.127000 | 0.221900 | 0.042729 | 0.060603 | 0.084332 | 0.063342 |
| 7 | `proex_profile` | official_baseline | 0.060200 | 0.117700 | 0.232900 | 0.037282 | 0.055676 | 0.084375 | 0.060718 |
| 8 | `promax_profile` | official_baseline | 0.056000 | 0.104600 | 0.201800 | 0.034683 | 0.050297 | 0.074582 | 0.056527 |
| 9 | `elmrec_graph` | official_baseline | 0.050100 | 0.101000 | 0.210100 | 0.029656 | 0.045871 | 0.073166 | 0.052376 |

## Gate Summary

- Observed C-CRP best on all seven metrics: `True`
- Holm-significant positive C-CRP deltas for all C-CRP-vs-official tests: `True`
- Number of paired tests: `56`
- Minimum delta across tests: `0.029400000000`
- Maximum Holm-adjusted p-value across tests: `9.71723076346e-07`

## Closest Official Baseline By Metric

| Metric | Best official baseline | Baseline | C-CRP | Delta | Holm p | 95% CI |
|---|---|---:|---:|---:|---:|---|
| HR@5 | `llmemb` | 0.136500 | 0.193700 | 0.057200 | 7.42937e-37 | [0.048500, 0.065400] |
| HR@10 | `llmemb` | 0.225700 | 0.269600 | 0.043900 | 1.52899e-16 | [0.033600, 0.053602] |
| HR@20 | `llmemb` | 0.363700 | 0.393100 | 0.029400 | 9.71723e-07 | [0.017300, 0.041100] |
| NDCG@5 | `llmemb` | 0.087458 | 0.141864 | 0.054406 | 3.38757e-55 | [0.048128, 0.060003] |
| NDCG@10 | `llmemb` | 0.115944 | 0.166116 | 0.050172 | 4.32805e-42 | [0.043960, 0.056064] |
| NDCG@20 | `llmemb` | 0.150506 | 0.197040 | 0.046533 | 4.08702e-37 | [0.040199, 0.052264] |
| MRR | `llmemb` | 0.106494 | 0.155859 | 0.049366 | 8.04336e-32 | [0.043726, 0.054651] |

Claim note: this is a tools-domain statistical gate. Multi-domain paper-level SOTA wording still requires the declared domain set, aligned baselines, and ARIS review.
