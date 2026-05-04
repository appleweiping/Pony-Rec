# Summary Tables Guide

This document maps the generated summary CSV files under `outputs/summary/` to their intended use in paper writing. The goal is to keep the evidence layer stable: each table should answer a specific experimental question and should not drift into overlapping responsibilities.

## Core Tables

All paper-facing summary tables should include `status_label`. Rows labeled
`design_only`, `proxy_only`, or `future_extension` are not main-result rows.

### `candidate_protocol_audit.csv`

Required for candidate-ranking claims. It documents candidate set size,
positive count, negative sampling, hard-negative ratio, title duplicates,
valid/test user overlap, train/test item overlap, one-positive status, and
whether HR@K and Recall@K are numerically equivalent.

### `baseline_reliability_proxy_audit.csv`

Required for baseline fairness. It records whether a baseline score is a true
relevance-calibratable signal, a score/rank/stability reliability proxy, an
exposure policy certainty signal, or non-isomorphic to confidence.

### `baseline_paper_audit_matrix.csv`

Use this as the Week8 audit table for senior-recommended paper baselines under
`Paper/BASELINE/NH` and `Paper/BASELINE/NR`.

The table is not a result table. It records:

- paper title and detected method name
- code URL signal when visible in the PDF
- paper family
- runnable priority
- same-candidate/same-split/same-metric protocol gaps
- recommended next action

Rows labeled `B_adapter_candidate` are candidates for future same-schema
adapters. Rows labeled `C_proxy_only` or `D_related_only` must stay out of the
main result table unless converted into the ranking prediction schema.

### `significance_tests.csv` and `main_table_with_ci.csv`

Use these for winner claims. A row whose paired confidence interval crosses zero
or whose corrected p-value is not significant should be called `observed_best`,
not `winner`.

### `final_results.csv`

Use this as the primary cross-domain clean-results table.

Each row corresponds to one `(exp_name, domain, model, lambda)` setting and includes:

- diagnosis metrics
- test calibration metrics before/after calibration
- baseline ranking and exposure metrics
- uncertainty-aware rerank ranking and exposure metrics

Recommended use:

- main clean-result table
- cross-domain calibration discussion
- cross-domain reranking discussion

### `model_results.csv`

Use this as the main cross-model comparison table.

Each row corresponds to one `(domain, model, lambda)` experiment row and keeps:

- diagnosis metrics
- calibration metrics
- baseline ranking metrics
- rerank metrics

Recommended use:

- cross-model comparison on the clean pipeline
- model-centric analysis section

### `domain_model_summary.csv`

Grouped mean summary over `(domain, model)`.

Recommended use:

- compact appendix table
- quick sanity-check view over model behavior by domain

## Estimator Tables

### `estimator_results.csv`

Use this as the full multi-estimator comparison table.

Each row corresponds to one `(domain, model, estimator, lambda)` setting and includes:

- calibration metrics for the estimator-derived confidence signal
- baseline ranking/exposure metrics
- rerank metrics under the estimator-driven uncertainty

Recommended use:

- full estimator comparison appendix
- source table for custom filtered views

### `beauty_estimator_results.csv`

Beauty-only filtered estimator table.

Recommended use:

- main estimator-comparison table in the paper
- the central `Beauty x model x estimator` comparison view
- contains only the five primary clean Beauty experiment lines

### `beauty_estimator_supporting_results.csv`

Beauty-only supporting estimator table for non-main experiment variants.

Recommended use:

- appendix-level estimator support
- sensitivity or derived experiment bookkeeping without polluting the main Beauty estimator table

## Robustness Tables

### `robustness_results.csv`

Full clean-vs-noisy robustness summary.

Each row corresponds to one `(clean_exp, noisy_exp)` comparison and includes:

- ranking degradation
- calibration degradation
- high-confidence mistake changes

Recommended use:

- full robustness appendix
- detailed degradation inspection

### `robustness_brief.csv`

Compact robustness table.

Recommended use:

- main robustness table in the paper
- first clean/noisy claim

### `beauty_main_results.csv`

Beauty-only main-results table derived from `final_results.csv`.

Recommended use:

- main Beauty clean-results table
- first paper-facing table for the main domain

### `beauty_estimator_brief.csv`

Beauty-only estimator comparison brief table.

Recommended use:

- compact main-text estimator table
- quicker model-by-estimator comparison view

### `beauty_robustness_curve_brief.csv`

Beauty-only robustness curve table with noise-level organization.

Recommended use:

- main robustness curve table
- source table for a noise-level trend figure
- cross-model robustness support on Beauty (`deepseek` and `glm`)

### `beauty_reproducibility_brief.csv`

Beauty-only reproducibility brief table.

Recommended use:

- appendix stability table
- concise support for the reproducibility paragraph

### `beauty_consistency_sensitivity_brief.csv`

Beauty-only sampling-sensitivity brief table for `self-consistency`.

Recommended use:

- appendix table for consistency sensitivity
- source table for the claim that increasing temperature only partially activates consistency-based uncertainty

### `beauty_fused_alpha_brief.csv`

Beauty-only fusion-weight ablation brief table.

Recommended use:

- appendix table for `fused_alpha` analysis
- compact support for the claim that fused uncertainty is controllable but still weaker than calibrated verbalized confidence

## Auxiliary Tables

### `shadow_v1_to_v6_status_matrix.csv`

Use this as the shadow-family status table.

It separates:

- `shadow_v1`-`shadow_v5` as signal candidates
- `shadow_v6` as a signal-to-decision bridge

Recommended use:

- internal status audit
- appendix method-stage table
- guardrail against overclaiming incomplete shadow variants

### `unified_method_matrix_week77_shadow.csv`

Use this as the bridge between Week7.7 SRPD evidence and Week7.9 shadow
diagnostics.

Rows should include:

- Week7.7 direct candidate ranking
- Week7.7 structured-risk rerank
- Week7.7 SRPD variants
- full-replay `shadow_v1`
- diagnostic `shadow_v6`

Recommended use:

- decide whether `shadow_v6` remains a bridge/ablation or becomes a method
  candidate
- compare the trainable SRPD framework line against shadow diagnostics
- preserve `comparison_scope` so diagnostic rows do not enter main paper tables
  accidentally

### `weekly_summary.csv`

Compact clean-result view with the most important diagnosis, calibration, and rerank metrics.

Recommended use:

- internal progress tracking
- quick reporting

### `rerank_ablation.csv`

Alias-style clean summary focused on domain/lambda organization.

Recommended use:

- lambda-focused reporting
- backward compatibility with earlier Week1/Week2 summaries

## Reproducibility Tables

### `reproducibility_check.csv`

Raw repeated-run metrics for the reproducibility check.

### `reproducibility_delta.csv`

Absolute metric differences between repeated runs.

Recommended use:

- reproducibility subsection
- appendix stability evidence

## Suggested Paper Mapping

- Table 1: `final_results.csv`
- Table 2: `model_results.csv`
- Main Beauty table: `beauty_main_results.csv`
- Main estimator table: `beauty_estimator_brief.csv`
- Main robustness table: `beauty_robustness_curve_brief.csv`
- Reproducibility appendix table: `beauty_reproducibility_brief.csv`
- Consistency-sensitivity appendix table: `beauty_consistency_sensitivity_brief.csv`
- Fused-alpha appendix table: `beauty_fused_alpha_brief.csv`

## Regeneration

To rebuild the current summary layer:

```powershell
py -3.12 main_aggregate_all.py --output_root outputs
```
