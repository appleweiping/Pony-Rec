# home official baseline vs C-CRP comparison

Read-only same-candidate comparison. This table does not start or rerun experiments.

| Rank | Method | Kind | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `ccrp_v3_qwen3base_pointwise` | internal_method | 0.156100 | 0.226400 | 0.350500 | 0.109780 | 0.132394 | 0.163511 | 0.125948 |
| 2 | `llmemb` | official_baseline | 0.107900 | 0.185600 | 0.316900 | 0.068996 | 0.093906 | 0.126743 | 0.090123 |
| 3 | `irllrec_intent` | official_baseline | 0.082100 | 0.144300 | 0.287800 | 0.051090 | 0.070898 | 0.106620 | 0.074243 |
| 4 | `rlmrec_graphcl` | official_baseline | 0.068500 | 0.126800 | 0.245100 | 0.041268 | 0.059870 | 0.089322 | 0.063974 |
| 5 | `llmesr_sasrec` | official_baseline | 0.062100 | 0.116300 | 0.213900 | 0.037993 | 0.055376 | 0.079750 | 0.059737 |
| 6 | `proex_profile` | official_baseline | 0.060600 | 0.117700 | 0.229600 | 0.036629 | 0.054867 | 0.082901 | 0.059333 |
| 7 | `llm2rec_sasrec` | official_baseline | 0.057700 | 0.110100 | 0.215300 | 0.034208 | 0.050949 | 0.077196 | 0.056387 |
| 8 | `promax_profile` | official_baseline | 0.051400 | 0.101900 | 0.207600 | 0.030788 | 0.046918 | 0.073261 | 0.053475 |
| 9 | `elmrec_graph` | official_baseline | 0.050900 | 0.102100 | 0.201800 | 0.029717 | 0.046044 | 0.070886 | 0.051959 |

## Gate Summary

- Observed C-CRP best on all seven metrics: `True`
- Holm-significant positive C-CRP deltas for all C-CRP-vs-official tests: `True`
- Number of paired tests: `56`
- Minimum delta across tests: `0.033600000000`
- Maximum Holm-adjusted p-value across tests: `1.02169272554e-08`

## Closest Official Baseline By Metric

| Metric | Best official baseline | Baseline | C-CRP | Delta | Holm p | 95% CI |
|---|---|---:|---:|---:|---:|---|
| HR@5 | `llmemb` | 0.107900 | 0.156100 | 0.048200 | 9.30406e-31 | [0.040300, 0.056000] |
| HR@10 | `llmemb` | 0.185600 | 0.226400 | 0.040800 | 3.00426e-16 | [0.031000, 0.050000] |
| HR@20 | `llmemb` | 0.316900 | 0.350500 | 0.033600 | 1.02169e-08 | [0.021600, 0.045102] |
| NDCG@5 | `llmemb` | 0.068996 | 0.109780 | 0.040784 | 1.37596e-39 | [0.035212, 0.046294] |
| NDCG@10 | `llmemb` | 0.093906 | 0.132394 | 0.038488 | 2.37274e-32 | [0.032647, 0.043929] |
| NDCG@20 | `llmemb` | 0.126743 | 0.163511 | 0.036769 | 1.64307e-28 | [0.031196, 0.042277] |
| MRR | `llmemb` | 0.090123 | 0.125948 | 0.035825 | 1.43465e-21 | [0.030917, 0.040668] |

Claim note: this is a home-domain statistical gate. Multi-domain paper-level SOTA wording still requires the declared domain set, aligned baselines, and ARIS review.
