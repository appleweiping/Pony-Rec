# sports official baseline vs C-CRP comparison

Read-only same-candidate comparison. This table does not start or rerun experiments.

| Rank | Method | Kind | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `ccrp_v3_qwen3base_pointwise` | internal_method | 0.274500 | 0.381900 | 0.517200 | 0.198451 | 0.232862 | 0.267006 | 0.207566 |
| 2 | `llmemb` | official_baseline | 0.212400 | 0.338400 | 0.490000 | 0.138853 | 0.179500 | 0.217687 | 0.153883 |
| 3 | `irllrec_intent` | official_baseline | 0.157300 | 0.221500 | 0.401600 | 0.106422 | 0.126917 | 0.171285 | 0.124442 |
| 4 | `rlmrec_graphcl` | official_baseline | 0.121200 | 0.187900 | 0.300900 | 0.078580 | 0.100018 | 0.128182 | 0.097205 |
| 5 | `llm2rec_sasrec` | official_baseline | 0.110500 | 0.206000 | 0.365700 | 0.065148 | 0.095668 | 0.135617 | 0.088289 |
| 6 | `llmesr_sasrec` | official_baseline | 0.091600 | 0.156400 | 0.265000 | 0.054920 | 0.075812 | 0.103105 | 0.075115 |
| 7 | `proex_profile` | official_baseline | 0.082100 | 0.152700 | 0.277700 | 0.051683 | 0.074172 | 0.105406 | 0.074269 |
| 8 | `promax_profile` | official_baseline | 0.082500 | 0.138700 | 0.237000 | 0.054185 | 0.072153 | 0.096759 | 0.074105 |
| 9 | `elmrec_graph` | official_baseline | 0.053200 | 0.105400 | 0.201300 | 0.031705 | 0.048372 | 0.072350 | 0.053701 |

## Gate Summary

- Observed C-CRP best on all seven metrics: `True`
- Holm-significant positive C-CRP deltas for all C-CRP-vs-official tests: `True`
- Number of paired tests: `56`
- Minimum delta across tests: `0.027200000000`
- Maximum Holm-adjusted p-value across tests: `1.2191293148e-06`

## Closest Official Baseline By Metric

| Metric | Best official baseline | Baseline | C-CRP | Delta | Holm p | 95% CI |
|---|---|---:|---:|---:|---:|---|
| HR@5 | `llmemb` | 0.212400 | 0.274500 | 0.062100 | 1.06549e-35 | [0.052797, 0.071802] |
| HR@10 | `llmemb` | 0.338400 | 0.381900 | 0.043500 | 1.87286e-15 | [0.032500, 0.054600] |
| HR@20 | `llmemb` | 0.490000 | 0.517200 | 0.027200 | 1.21913e-06 | [0.016400, 0.038600] |
| NDCG@5 | `llmemb` | 0.138853 | 0.198451 | 0.059599 | 4.49235e-51 | [0.052483, 0.066504] |
| NDCG@10 | `llmemb` | 0.179500 | 0.232862 | 0.053362 | 1.32211e-42 | [0.046463, 0.060143] |
| NDCG@20 | `llmemb` | 0.217687 | 0.267006 | 0.049319 | 2.69063e-41 | [0.042774, 0.055551] |
| MRR | `llmemb` | 0.153883 | 0.207566 | 0.053683 | 5.89836e-36 | [0.047481, 0.059799] |

Claim note: this is a sports-domain statistical gate. Multi-domain paper-level SOTA wording still requires the declared domain set, aligned baselines, and ARIS review.
