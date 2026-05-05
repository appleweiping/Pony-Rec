# LLM Project Qwen3 Baseline Combined Result - 2026-05-05

This note records the final same-schema readout after rebuilding the unified
matrix, the paper-ready table, and paired statistical tests with both
senior-recommended LLM-rec paper-project baselines.

Completed paper-project rows:

```text
LLM2Rec-style Qwen3-8B Emb. + SASRec
LLM-ESR-style Qwen3-8B Emb. + LLMESR-SASRec
```

Both rows are completed same-candidate external baselines under the same split,
candidate set, metric implementation, and local Qwen3-8B embedding backbone.
Neither row should be described as an official upstream reproduction.

## Rebuilt Artifacts

- Unified matrix:
  `outputs/summary/unified_method_matrix_week77_shadow_external_qwen_llmesr_llm2rec.csv`
- Paper-ready table:
  `outputs/summary/paper_ready_baseline_comparison_week77_qwen_llmesr_llm2rec.md`
- Paired statistical tests:
  `outputs/summary/week8_llm_project_qwen3_stat_tests/all_domains_significance_tests.csv`

The paper-ready table now contains two `paper_project_same_backbone_baseline`
rows per domain: one LLM2Rec-style row and one LLM-ESR-style row.

## Paper-Project Baseline Results

| domain | LLM2Rec-style NDCG@10 | LLM2Rec-style MRR | LLM-ESR-style NDCG@10 | LLM-ESR-style MRR | stronger LLM-project row |
| --- | ---: | ---: | ---: | ---: | --- |
| beauty | 0.551090 | 0.408565 | 0.573620 | 0.437564 | LLM-ESR-style, nominal only after Holm |
| books | 0.555974 | 0.415167 | 0.475203 | 0.310000 | LLM2Rec-style, significant |
| electronics | 0.558289 | 0.417733 | 0.482337 | 0.318467 | LLM2Rec-style, significant |
| movies | 0.572491 | 0.435633 | 0.514004 | 0.359733 | LLM2Rec-style, significant |

## Paired Statistical Readout

Against structured-risk:

| domain | LLM2Rec-style vs structured-risk | LLM-ESR-style vs structured-risk |
| --- | --- | --- |
| beauty | lower by 0.062988, significant | lower by 0.040458, significant |
| books | lower by 0.083540, significant | lower by 0.164311, significant |
| electronics | lower by 0.100012, significant | lower by 0.175964, significant |
| movies | lower by 0.000692, not significant | lower by 0.059179, significant |

Between the two LLM-project rows:

| domain | comparison | paired result |
| --- | --- | --- |
| beauty | LLM-ESR-style higher by 0.022530 | not Holm-significant |
| books | LLM2Rec-style higher by 0.080770 | significant |
| electronics | LLM2Rec-style higher by 0.075951 | significant |
| movies | LLM2Rec-style higher by 0.058487 | significant |

Main interpretation:

- LLM2Rec-style is the stronger paper-project baseline overall.
- LLM-ESR-style adds the second senior-recommended LLM-rec baseline and is
  nominally better on Beauty, but that Beauty advantage over LLM2Rec-style is
  not Holm-significant.
- Both LLM-project baselines remain below structured-risk on Beauty, Books, and
  Electronics.
- On Movies, LLM2Rec-style is statistically indistinguishable from
  structured-risk, while LLM-ESR-style is significantly lower.

## Context With Classical, SRPD, and Shadow

Classical baseline block:

- Completed same-candidate baselines: `SASRec`, `GRU4Rec`, `BERT4Rec`,
  `LightGCN`.
- Strongest classical row by domain is LightGCN on Beauty and GRU4Rec on
  Books, Electronics, and Movies.
- LLM2Rec-style is above the classical rows on Books, Electronics, and Movies,
  and below LightGCN on Beauty.
- LLM-ESR-style is mixed: useful as an additional LLM-rec paper-project row, but
  not the strongest non-uncertainty baseline overall.

SRPD and structured-risk block:

- SRPD remains the strongest self-trained framework line on Books and a
  leading line on Beauty/Electronics.
- Structured-risk remains a strong uncertainty-aware reference across all four
  domains.
- The new LLM-project rows do not overturn the main uncertainty-aware/SRPD
  conclusion.

Shadow block:

- Shadow-v6 remains a diagnostic decision bridge, not the final promoted
  method row.
- Its high Beauty/Electronics/Movies diagnostic scores should be discussed as
  evidence for the bridge direction, with the usual aligned-protocol caveat.

## Paper-Safe Claim

Safe wording:

> We evaluate four classical recommenders and two senior-recommended LLM-rec
> paper-project baselines under the same split, candidate set, metric
> implementation, and Qwen3-8B embedding backbone. The LLM2Rec-style baseline is
> the stronger paper-project row overall, while the LLM-ESR-style row provides an
> additional recent LLM-enhanced sequential baseline. These baselines do not
> eliminate the advantage of the uncertainty-aware/SRPD lines on Beauty, Books,
> and Electronics; on Movies, LLM2Rec-style is a near-tie with structured-risk.

Unsafe wording:

```text
We fully reproduce official LLM2Rec and official LLM-ESR.
```

Why unsafe:

- LLM2Rec-style uses local Qwen3-8B mean-pooled item embeddings and does not run
  the full official CSFT/IEM pipeline.
- LLM-ESR-style uses a same-candidate wrapper around the upstream
  `LLMESR_SASRec` model class and local Qwen3 embeddings, not the full official
  notebook/API preprocessing and native evaluation setup.

## Decision

The baseline block is now paper-ready enough for the current claim:

```text
4 classical same-candidate baselines
2 senior-recommended LLM-rec paper-project same-schema baselines
SRPD/structured-risk/shadow conclusions preserved under paired tests
```

Additional LLMEmb-style and RLMRec-style entrypoints have been added as the
next breadth extensions. Do not update this completed-result count from `2` to
`4` until both new baselines are run on all four domains, imported with full
same-candidate coverage, and included in paired statistical tests.
