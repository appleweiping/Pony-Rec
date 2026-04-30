# Generative Title Bridge

This is an extension bridge, not the primary claim, unless all required
experiments are completed.

The main paper remains candidate ranking and reranking. Generative title
recommendation opens a different evaluation problem: catalog grounding,
semantic matching, hallucination, and fallback policy.

## Required before main-table use

Generative title results may enter a main table only when all are completed:

- direct generation baseline
- retrieve-after-generate baseline
- shadow-verified generation
- catalog-grounded title-to-item mapping
- out-of-catalog rate
- hallucination rate
- unsupported confident generation rate
- accept, revise, fallback, and abstain rates
- at least two domains
- semantic or human audit for a sampled subset

## Status table schema

Reserved output:

```text
outputs/summary/generative_title_bridge_status.csv
```

Required fields:

- `generation_method`
- `catalog_mapping_method`
- `catalog_hit_rate`
- `recall_at_10_after_mapping`
- `ndcg_at_10_after_mapping`
- `out_of_catalog_rate`
- `hallucination_rate`
- `unsupported_confident_generation_rate`
- `accept_rate`
- `revision_rate`
- `fallback_rate`
- `status_label`

Only rows with `status_label=completed_result` can be cited as completed
results. Other rows are discussion or appendix planning material.
