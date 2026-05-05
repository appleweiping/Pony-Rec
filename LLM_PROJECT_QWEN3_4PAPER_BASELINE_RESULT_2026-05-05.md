# LLM Project Qwen3 4-Paper Baseline Result - 2026-05-05

This note records the completed four-paper-project baseline block after adding
LLMEmb-style and RLMRec-style to the earlier LLM2Rec-style and LLM-ESR-style
rows.

Completed same-schema paper-project baselines:

```text
LLM2Rec-style Qwen3-8B Emb. + SASRec
LLM-ESR-style Qwen3-8B Emb. + LLMESR-SASRec
LLMEmb-style Qwen3-8B Emb. + SASRec
RLMRec-style Qwen3-8B GraphCL
```

All four are completed same-candidate external baselines under the same split,
candidate set, metric implementation, and local Qwen3-8B embedding backbone.
They are style/adapted baselines, not official upstream reproductions.

## Rebuilt Artifacts

- Unified matrix:
  `outputs/summary/unified_method_matrix_week77_shadow_external_qwen_4paper.csv`
- Paper-ready table:
  `outputs/summary/paper_ready_baseline_comparison_week77_qwen_4paper.md`
- Paired statistical tests:
  `outputs/summary/week8_llm_project_qwen3_4paper_stat_tests/all_domains_significance_tests.csv`

The paper-ready table has `rows=48`, i.e. four domains with direct,
structured-risk, SRPD-best, four classical baselines, and four paper-project
same-backbone baselines.

## Four-Paper Results

| domain | LLM2Rec-style | LLM-ESR-style | LLMEmb-style | RLMRec-style | strongest observed paper-project row |
| --- | ---: | ---: | ---: | ---: | --- |
| beauty | 0.551090 | 0.573620 | 0.630114 | 0.634871 | RLMRec-style |
| books | 0.555974 | 0.475203 | 0.682281 | 0.694433 | RLMRec-style |
| electronics | 0.558289 | 0.482337 | 0.601886 | 0.628100 | RLMRec-style |
| movies | 0.572491 | 0.514004 | 0.665062 | 0.660390 | LLMEmb-style |

MRR:

| domain | LLM2Rec-style | LLM-ESR-style | LLMEmb-style | RLMRec-style |
| --- | ---: | ---: | ---: | ---: |
| beauty | 0.408565 | 0.437564 | 0.511802 | 0.518123 |
| books | 0.415167 | 0.310000 | 0.579033 | 0.594533 |
| electronics | 0.417733 | 0.318467 | 0.474533 | 0.507867 |
| movies | 0.435633 | 0.359733 | 0.555333 | 0.549967 |

## Statistical Readout

The expanded paired-test family is very conservative. In the pasted
four-paper stat output, comparisons involving LLMEmb-style or RLMRec-style have
`significant=False` after Holm correction, even when raw p-values and
confidence intervals are strong.

Against structured-risk:

| domain | LLMEmb-style delta | LLMEmb Holm result | RLMRec-style delta | RLMRec Holm result |
| --- | ---: | --- | ---: | --- |
| beauty | +0.016036 | not significant | +0.020793 | not significant |
| books | +0.042766 | not significant | +0.054919 | not significant |
| electronics | -0.056415 | not significant | -0.030200 | not significant |
| movies | +0.091880 | not significant | +0.087208 | not significant |

Against SRPD-best:

| domain | LLMEmb-style vs SRPD-best | RLMRec-style vs SRPD-best |
| --- | --- | --- |
| beauty | SRPD-best observed higher by 0.005252, not significant | SRPD-best observed higher by 0.000495, not significant |
| books | SRPD-best observed higher by 0.023427, not significant | SRPD-best observed higher by 0.011274, not significant |
| electronics | SRPD-best observed higher by 0.060215, not significant | SRPD-best observed higher by 0.034000, not significant |
| movies | LLMEmb-style observed higher by 0.118674, not Holm-significant | RLMRec-style observed higher by 0.114002, not Holm-significant |

Between the two new paper-project rows:

| domain | observed comparison | Holm result |
| --- | --- | --- |
| beauty | RLMRec-style higher by 0.004756 | not significant |
| books | RLMRec-style higher by 0.012153 | not significant |
| electronics | RLMRec-style higher by 0.026215 | not significant |
| movies | LLMEmb-style higher by 0.004672 | not significant |

## Interpretation

- The baseline block is now substantially stronger: four classical baselines
  plus four senior-recommended LLM-rec paper-project baselines.
- LLMEmb-style and RLMRec-style are the strongest observed paper-project rows.
- RLMRec-style is the strongest observed paper-project row on Beauty, Books,
  and Electronics; LLMEmb-style is strongest on Movies.
- The new rows are competitive with SRPD/structured-risk and observed higher in
  several domains, but the expanded Holm correction does not support
  significant winner wording.
- The safest paper claim is about baseline breadth and competitive observed
  performance, not statistically significant dominance.

## Paper-Safe Claim

Safe wording:

> We include four classical recommenders and four senior-recommended LLM-rec
> paper-project baselines under the same split, candidate set, metric
> implementation, and Qwen3-8B embedding backbone. The added LLMEmb-style and
> RLMRec-style rows are the strongest observed paper-project baselines, with
> RLMRec-style leading on Beauty, Books, and Electronics and LLMEmb-style
> leading on Movies. Under the expanded paired-test family, these observed gains
> should be reported as competitive/observed improvements rather than
> Holm-significant wins.

Also safe:

> The four-paper baseline block materially addresses baseline breadth: the
> comparison now includes sequential embedding, long-tail sequential,
> LLM-embedding adaptation, and graph/representation-style LLM-rec baselines.

Unsafe wording:

```text
We fully reproduce official LLM2Rec, LLM-ESR, LLMEmb, and RLMRec.
```

Why unsafe:

- All four rows are adapted same-schema baselines.
- They use local Qwen3-8B embeddings and same-candidate scoring wrappers.
- Official upstream preprocessing, training, and native evaluation assumptions
  were not fully reproduced.

## Decision

The current baseline adequacy claim can be updated to:

```text
4 classical same-candidate baselines
4 senior-recommended LLM-rec paper-project same-schema baselines
same split, same candidate set, same metric implementation
paired statistical tests completed
```

Use `observed_best`, `competitive`, and `not Holm-significant after expanded
correction` wording for the new LLMEmb-style/RLMRec-style readout.
