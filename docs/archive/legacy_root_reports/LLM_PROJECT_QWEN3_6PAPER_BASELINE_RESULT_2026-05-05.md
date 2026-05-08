# LLM Project Qwen3 6-Paper Baseline Result - 2026-05-05

This note records the completed six-paper-project same-schema baseline block.
It supersedes the earlier two-paper and four-paper readouts:

```text
LLM_PROJECT_QWEN3_BASELINE_COMBINED_RESULT_2026-05-05.md
LLM_PROJECT_QWEN3_4PAPER_BASELINE_RESULT_2026-05-05.md
```

Completed same-candidate paper-project baselines:

```text
LLM2Rec-style Qwen3-8B Emb. + SASRec
LLM-ESR-style Qwen3-8B Emb. + LLMESR-SASRec
LLMEmb-style Qwen3-8B Emb. + SASRec
RLMRec-style Qwen3-8B GraphCL
IRLLRec-style Qwen3-8B IntentRep
SETRec-style Qwen3-8B Identifier
```

All six are completed same-candidate external baselines under the same split,
candidate set, metric implementation, and local Qwen3-8B embedding backbone.
They are style/adapted baselines, not official upstream reproductions.

## Rebuilt Artifacts

- Unified matrix:
  `outputs/summary/unified_method_matrix_week77_shadow_external_qwen_6paper.csv`
- Paper-ready table:
  `outputs/summary/paper_ready_baseline_comparison_week77_qwen_6paper.md`
- Paired statistical tests:
  `outputs/summary/week8_llm_project_qwen3_6paper_stat_tests/all_domains_significance_tests.csv`

The paper-ready table has six
`paper_project_same_backbone_baseline` rows per domain. The completed baseline
block is now:

```text
4 classical same-candidate baselines
6 senior-recommended LLM-rec paper-project same-schema baselines
```

## Six-Paper NDCG@10 Results

| domain | LLM2Rec-style | LLM-ESR-style | LLMEmb-style | RLMRec-style | IRLLRec-style | SETRec-style | strongest observed paper-project row |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| beauty | 0.551090 | 0.573620 | 0.630114 | 0.634871 | 0.662061 | 0.625298 | IRLLRec-style |
| books | 0.555974 | 0.475203 | 0.682281 | 0.694433 | 0.716744 | 0.646253 | IRLLRec-style |
| electronics | 0.558289 | 0.482337 | 0.601886 | 0.628100 | 0.604741 | 0.529358 | RLMRec-style |
| movies | 0.572491 | 0.514004 | 0.665062 | 0.660390 | 0.707149 | 0.591424 | IRLLRec-style |

## Six-Paper MRR Results

| domain | LLM2Rec-style | LLM-ESR-style | LLMEmb-style | RLMRec-style | IRLLRec-style | SETRec-style |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| beauty | 0.408565 | 0.437564 | 0.511802 | 0.518123 | 0.553083 | 0.504436 |
| books | 0.415167 | 0.310000 | 0.579033 | 0.594533 | 0.625033 | 0.530767 |
| electronics | 0.417733 | 0.318467 | 0.474533 | 0.507867 | 0.478900 | 0.380067 |
| movies | 0.435633 | 0.359733 | 0.555333 | 0.549967 | 0.610933 | 0.459200 |

## IRLLRec-Style and SETRec-Style Readout

IRLLRec-style is the strongest observed paper-project row on Beauty, Books,
and Movies. On Electronics, RLMRec-style remains the strongest observed
paper-project row, with IRLLRec-style close to LLMEmb-style and below
RLMRec-style.

SETRec-style adds useful baseline breadth but is weaker than IRLLRec-style in
all four domains. It is competitive with several earlier LLM-rec rows on
Beauty, Books, and Movies, but below the stronger graph/intent/embedding rows
overall.

## Statistical Readout

The six-paper paired-test family is larger and therefore very conservative
under Holm correction. In the pasted six-paper statistical output, comparisons
involving IRLLRec-style and SETRec-style have `significant=False` after Holm
correction, even when raw p-values and confidence intervals are strong.

Against structured-risk:

| domain | IRLLRec-style delta | IRLLRec Holm result | SETRec-style delta | SETRec Holm result |
| --- | ---: | --- | ---: | --- |
| beauty | +0.047984 | not significant | +0.011221 | not significant |
| books | +0.077229 | not significant | +0.006739 | not significant |
| electronics | -0.053560 | not significant | -0.128943 | not significant |
| movies | +0.133967 | not significant | +0.018241 | not significant |

Between the strongest six-paper rows:

| domain | observed comparison | Holm result |
| --- | --- | --- |
| beauty | IRLLRec-style higher than RLMRec-style by 0.027191 | not significant |
| books | IRLLRec-style higher than RLMRec-style by 0.022310 | not significant |
| electronics | RLMRec-style higher than IRLLRec-style by 0.023359 | not significant |
| movies | IRLLRec-style higher than LLMEmb-style by 0.042087 | not significant |

Use `observed higher`, `strongest observed`, and `competitive` wording. Do not
write `Holm-significant winner` for these six-paper comparisons.

## Interpretation

- The paper-facing baseline block now covers four classical recommenders and
  six senior-recommended LLM-rec paper-project baselines.
- IRLLRec-style is the strongest observed paper-project row on Beauty, Books,
  and Movies.
- RLMRec-style remains the strongest observed paper-project row on
  Electronics.
- SETRec-style broadens the comparison with an identifier/set-oriented
  baseline, but it is not the strongest row.
- The uncertainty-aware/SRPD conclusions should be discussed alongside these
  stronger baselines, with statistical wording constrained by the expanded Holm
  family.

## Paper-Safe Claim

Safe wording:

> We include four classical recommenders and six senior-recommended LLM-rec
> paper-project baselines under the same split, candidate set, metric
> implementation, and Qwen3-8B embedding backbone. IRLLRec-style is the
> strongest observed paper-project row on Beauty, Books, and Movies, while
> RLMRec-style remains strongest on Electronics. Under the expanded paired-test
> family, these should be reported as observed/competitive gains rather than
> Holm-significant wins.

Also safe:

> The six-paper baseline block materially improves baseline breadth by covering
> sequential LLM embedding, long-tail sequential LLM embedding, LLM embedding
> adaptation, graph/representation learning, intent representation, and
> identifier/set-oriented LLM-rec styles.

Unsafe wording:

```text
We fully reproduce official LLM2Rec, LLM-ESR, LLMEmb, RLMRec, IRLLRec, and SETRec.
```

Why unsafe:

- All six rows are adapted same-schema baselines.
- They use local Qwen3-8B embeddings and exact same-candidate scoring wrappers.
- Official upstream preprocessing, training, generative identifier pipelines,
  and native evaluation assumptions were not fully reproduced.

## Decision

The current baseline adequacy claim can be updated to:

```text
4 classical same-candidate baselines
6 senior-recommended LLM-rec paper-project same-schema baselines
same split, same candidate set, same metric implementation
paired statistical tests completed
```

This is the current final paper-ready baseline block as of 2026-05-05.
