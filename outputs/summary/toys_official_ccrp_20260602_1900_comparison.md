# toys official baseline vs C-CRP comparison

Read-only same-candidate comparison. This table does not start or rerun experiments.

| Rank | Method | Kind | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `ccrp_v3_qwen3base_pointwise` | internal_method | 0.317200 | 0.396400 | 0.505900 | 0.245190 | 0.270799 | 0.298341 | 0.250305 |
| 2 | `llmemb` | official_baseline | 0.249900 | 0.350500 | 0.486600 | 0.172521 | 0.204850 | 0.239055 | 0.181380 |
| 3 | `llm2rec_sasrec` | official_baseline | 0.220200 | 0.317200 | 0.465200 | 0.147569 | 0.178873 | 0.216093 | 0.159216 |
| 4 | `irllrec_intent` | official_baseline | 0.156500 | 0.229300 | 0.409800 | 0.110492 | 0.133801 | 0.178585 | 0.131199 |
| 5 | `rlmrec_graphcl` | official_baseline | 0.128100 | 0.188500 | 0.305000 | 0.087160 | 0.106509 | 0.135400 | 0.105845 |
| 6 | `proex_profile` | official_baseline | 0.089500 | 0.161500 | 0.301700 | 0.058141 | 0.081017 | 0.116077 | 0.081217 |
| 7 | `promax_profile` | official_baseline | 0.092000 | 0.143500 | 0.241600 | 0.062896 | 0.079376 | 0.103876 | 0.081846 |
| 8 | `llmesr_sasrec` | official_baseline | 0.063700 | 0.117200 | 0.220300 | 0.037505 | 0.054568 | 0.080369 | 0.058450 |
| 9 | `elmrec_graph` | official_baseline | 0.054500 | 0.104300 | 0.201300 | 0.032593 | 0.048560 | 0.072780 | 0.054311 |

## Gate Summary

- Observed C-CRP best on all seven metrics: `True`
- Holm-significant positive C-CRP deltas for all C-CRP-vs-official tests: `True`
- Number of paired tests: `56`
- Minimum delta across tests: `0.019300000000`
- Maximum Holm-adjusted p-value across tests: `0.00040611199807`

## Closest Official Baseline By Metric

| Metric | Best official baseline | Baseline | C-CRP | Delta | Holm p | 95% CI |
|---|---|---:|---:|---:|---:|---|
| HR@5 | `llmemb` | 0.249900 | 0.317200 | 0.067300 | 1.20614e-46 | [0.058097, 0.076000] |
| HR@10 | `llmemb` | 0.350500 | 0.396400 | 0.045900 | 8.12069e-20 | [0.036500, 0.055202] |
| HR@20 | `llmemb` | 0.486600 | 0.505900 | 0.019300 | 0.000406112 | [0.008298, 0.029900] |
| NDCG@5 | `llmemb` | 0.172521 | 0.245190 | 0.072669 | 8.30969e-91 | [0.065975, 0.079289] |
| NDCG@10 | `llmemb` | 0.204850 | 0.270799 | 0.065948 | 1.56122e-78 | [0.059586, 0.071878] |
| NDCG@20 | `llmemb` | 0.239055 | 0.298341 | 0.059286 | 1.85724e-62 | [0.053155, 0.064947] |
| MRR | `llmemb` | 0.181380 | 0.250305 | 0.068925 | 1.5532e-59 | [0.062796, 0.074968] |

Claim note: this is a toys-domain statistical gate. Multi-domain paper-level SOTA wording still requires the declared domain set, aligned baselines, and ARIS review.
