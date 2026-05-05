# LLM-ESR-Style Qwen3 Baseline Result - 2026-05-05

This note records the second completed same-schema senior-recommended
paper-project baseline:

```text
LLM-ESR-style Qwen3-8B Emb. + LLMESR-SASRec
```

It is a completed same-candidate baseline row, but it is **not** an official
LLM-ESR reproduction.

## Status

- Baseline name: `llmesr_style_qwen3_sasrec`
- Display name: `LLM-ESR-style Qwen3-8B Emb. + LLMESR-SASRec`
- Status label: `same_schema_external_baseline`
- Artifact class: `completed_result`
- Table group: `paper_project_same_backbone_baseline`
- Backbone: local `Qwen3-8B` item embeddings
- Upstream model class: LLM-ESR `LLMESR_SASRec`
- Upstream repo: `https://github.com/liuqidong07/LLM-ESR`
- Protocol: exact same-candidate scoring through
  `main_train_score_llmesr_upstream_adapter.py` and
  `main_import_same_candidate_baseline_scores.py`
- Coverage: `score_coverage_rate=1.0` on all four checked domains

Paper-safe naming:

```text
LLM-ESR-style same-backbone adapter baseline
```

Do not call this row:

```text
official LLM-ESR reproduction
```

unless the full official LLM-ESR preprocessing, embedding notebooks, training
scripts, and native evaluation assumptions are reproduced.

## Four-Domain Result

| domain | sample_count | NDCG@10 | MRR | score_coverage_rate | checkpoint |
| --- | ---: | ---: | ---: | ---: | --- |
| beauty | 973 | 0.573620 | 0.437564 | 1.0 | `outputs/baselines/paper_adapters/beauty_llmesr_same_candidate_adapter/llmesr_upstream_model.pt` |
| books | 500 | 0.475203 | 0.310000 | 1.0 | `outputs/baselines/paper_adapters/books_llmesr_same_candidate_adapter/llmesr_upstream_model.pt` |
| electronics | 500 | 0.482337 | 0.318467 | 1.0 | `outputs/baselines/paper_adapters/electronics_llmesr_same_candidate_adapter/llmesr_upstream_model.pt` |
| movies | 500 | 0.514004 | 0.359733 | 1.0 | `outputs/baselines/paper_adapters/movies_llmesr_same_candidate_adapter/llmesr_upstream_model.pt` |

Training summary:

| domain | users | items | train_users | epochs | final_train_loss | trainable_params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| beauty | 973 | 1183 | 616 | 80 | 0.050468 | 13630016 |
| books | 500 | 3071 | 324 | 80 | 0.043055 | 21604928 |
| electronics | 500 | 2630 | 307 | 80 | 0.048028 | 19742144 |
| movies | 500 | 2863 | 331 | 80 | 0.038710 | 20726336 |

Readout:

- The row is fully covered and importable as a completed same-schema
  paper-project baseline.
- It is the second senior-recommended paper-project baseline after
  LLM2Rec-style Qwen3, so the paper-project baseline block is no longer a
  single-row block.
- It is generally weaker than LLM2Rec-style Qwen3 in the current four-domain
  run, except it is slightly higher on Beauty.

## Context Against Other Baselines

| method | beauty NDCG@10 | books NDCG@10 | electronics NDCG@10 | movies NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| LLM-ESR-style Qwen3-8B Emb. + LLMESR-SASRec | 0.573620 | 0.475203 | 0.482337 | 0.514004 |
| LLM2Rec-style Qwen3-8B Emb. + SASRec | 0.551090 | 0.555974 | 0.558289 | 0.572491 |
| LightGCN | 0.627705 | 0.515724 | 0.516162 | 0.527559 |
| GRU4Rec | 0.538966 | 0.540520 | 0.546086 | 0.536479 |

Interpretation:

- LLM-ESR-style is useful for coverage of recent senior-recommended LLM-rec
  paper projects.
- Its current score does not challenge the method-side uncertainty/SRPD story.
- It should be reported as a same-schema external baseline, not as evidence
  that the official LLM-ESR paper result was reproduced.

## Paper-Safe Claim

Safe wording:

> We include two senior-recommended LLM-rec paper-project baselines,
> LLM2Rec-style and LLM-ESR-style, both adapted to the same split, candidate
> set, and metric implementation using the local Qwen3-8B embedding backbone.

Also safe:

> LLM-ESR-style provides an additional recent LLM-enhanced sequential
> recommendation baseline, but in the current same-candidate evaluation it does
> not close the gap to uncertainty-aware/SRPD methods.

Unsafe wording:

> We fully reproduce official LLM-ESR.

Why unsafe:

- The current row uses local Qwen3-8B mean-pooled item embeddings rather than
  the original paper's exact notebook/API preprocessing.
- It uses a same-candidate wrapper around the upstream `LLMESR_SASRec` class.
- It imports exact same-candidate logits, not native LLM-ESR sampled-negative
  metrics.

## Remaining Gates

- Rebuild the unified and paper-ready baseline tables including
  `llmesr_style_qwen3_sasrec`.
- Run paired statistical tests including LLM-ESR-style before final paper
  wording.
- Keep the row label as LLM-ESR-style unless the official LLM-ESR pipeline is
  fully reproduced.
