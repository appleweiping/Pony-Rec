# Experiments Guide

This document maps the current repository structure to the main experiment families used in the project. It is meant to serve as the practical companion to the paper outline: each experiment family below points to the relevant entry scripts, config layers, and summary files.

## Config Layers

The repository is organized around three config layers:

- `configs/data/`: data preprocessing and sample-building settings
- `configs/model/`: backend, model, connection, and generation settings
- `configs/exp/`: concrete experiment runs with `exp_name`, `input_path`, `model_config`, and output root

In practice:

- switch domain by changing `configs/data/*.yaml`
- switch model by changing `configs/model/*.yaml`
- switch a concrete run by changing `configs/exp/*.yaml`

## Frozen Protocol Additions

Main-paper experiments must now also run the protocol layer:

```text
main_audit_candidate_protocol.py
main_baseline_reliability_audit.py
main_stat_tests.py
```

Candidate audit and baseline reliability proxy audit outputs are required
before promoting a row to `completed_result` in a paper-facing main table. See
`docs/paper_claims_and_status.md` for status-label rules.

## Main Experiment Families

### 1. Clean Diagnosis / Calibration / Reranking

This is the main Week1 pipeline:

```text
main_preprocess.py
-> main_build_samples.py
-> main_infer.py
-> main_eval.py
-> main_calibrate.py
-> main_rerank.py
```

Representative experiment names:

- `beauty_deepseek`
- `beauty_qwen`
- `beauty_glm`
- `beauty_kimi`
- `beauty_doubao`

Cross-domain small-subset validation:

- `movies_small_*`
- `books_small_*`
- `electronics_small_*`

Primary outputs:

- `outputs/{exp_name}/predictions/`
- `outputs/{exp_name}/calibrated/`
- `outputs/{exp_name}/reranked/`
- `outputs/{exp_name}/tables/`

### 2. Multi-Estimator Comparison

This is the Week2 Day3-Day4 line:

```text
main_self_consistency.py
-> main_uncertainty_compare.py
-> src/analysis/aggregate_estimator_results.py
```

Current estimator types:

- `verbalized_raw`
- `verbalized_calibrated`
- `consistency`
- `fused`

Key summary outputs:

- `outputs/summary/estimator_results.csv`
- `outputs/summary/beauty_estimator_results.csv`

### 3. Robustness / Noisy

This is the Week2 Day5 line:

```text
main_generate_noisy.py
-> main_infer.py
-> main_eval.py
-> main_calibrate.py
-> main_rerank.py
-> main_robustness.py
```

Current Beauty robustness settings now include:

- `beauty_deepseek_noisy_nl10/nl20/nl30`
- `beauty_glm_noisy_nl10/nl20/nl30`
- `beauty_qwen_noisy_nl10/nl20/nl30`
- `beauty_kimi_noisy_nl10/nl20/nl30`
- `beauty_doubao_noisy_nl10/nl20/nl30`

Key robustness outputs:

- `outputs/robustness/{clean_exp}_vs_{noisy_exp}/tables/robustness_table.csv`
- `outputs/robustness/{clean_exp}_vs_{noisy_exp}/tables/robustness_calibration_table.csv`
- `outputs/robustness/{clean_exp}_vs_{noisy_exp}/tables/robustness_confidence_table.csv`
- `outputs/summary/robustness_results.csv`
- `outputs/summary/robustness_brief.csv`

### 4. Experiment Aggregation

This is the Week2 Day6 line:

```text
main_aggregate_all.py
```

It calls:

- `src/analysis/aggregate_domain_results.py`
- `src/analysis/aggregate_model_results.py`
- `src/analysis/aggregate_estimator_results.py`
- `src/analysis/robustness_summary.py`

Run:

```powershell
py -3.12 main_aggregate_all.py --output_root outputs
```

Generated summary files under `outputs/summary/`:

- `final_results.csv`
- `weekly_summary.csv`
- `rerank_ablation.csv`
- `model_results.csv`
- `domain_model_summary.csv`
- `estimator_results.csv`
- `beauty_estimator_results.csv`
- `robustness_results.csv`
- `robustness_brief.csv`
- `beauty_main_results.csv`
- `beauty_estimator_brief.csv`
- `beauty_robustness_curve_brief.csv`
- `beauty_reproducibility_brief.csv`
- `beauty_consistency_sensitivity_brief.csv`
- `beauty_fused_alpha_brief.csv`

For table-level responsibilities and paper mapping, see:

- [docs/tables.md](tables.md)
- [docs/paper_outline.md](paper_outline.md)
- [docs/beauty_freeze_checklist.md](beauty_freeze_checklist.md)

## Suggested Reading Order for Paper Writing

If you are mapping code/results to the paper, the cleanest order is:

1. `outputs/summary/final_results.csv`
   Use for cross-domain and cross-model diagnosis/calibration/rerank discussion.
2. `outputs/summary/beauty_estimator_results.csv`
   Use for the main multi-estimator comparison table.
3. `outputs/summary/robustness_brief.csv`
   Use for the Beauty-first clean/noisy robustness claim and multi-model robustness comparison.
4. `outputs/summary/reproducibility_delta.csv`
   Use for the reproducibility appendix or stability note.

For Beauty-first paper writing, the most direct files are:

- `outputs/summary/beauty_main_results.csv`
- `outputs/summary/beauty_estimator_brief.csv`
- `outputs/summary/beauty_robustness_curve_brief.csv`
- `outputs/summary/beauty_reproducibility_brief.csv`
- `outputs/summary/beauty_consistency_sensitivity_brief.csv`
- `outputs/summary/beauty_fused_alpha_brief.csv`

## Current Practical Baselines

At the current stage, the most stable reference settings are:

- `beauty_deepseek` for the main clean baseline
- `beauty_*` for the five-model clean comparison layer
- `beauty_*_noisy_nl10/nl20/nl30` for the five-model robustness curve layer
- `beauty_*` for the full five-model estimator comparison
- `movies_small_deepseek` for the minimal cross-domain estimator validation

## Rescue-Branch Shadow / SRPD Matrix

The pre-Cursor rescue branch adds two audit-oriented summary entry points:

```bash
python main_summarize_shadow_v1_to_v6.py
python main_build_unified_method_matrix.py
```

`main_summarize_shadow_v1_to_v6.py` separates `shadow_v1`-`shadow_v5` signal
candidates from the `shadow_v6` decision bridge. It should be run against the
server shadow summary/raw output roots, not against incomplete local summaries.

`main_build_unified_method_matrix.py` compares the server-exported Week7.7
direct ranking, structured-risk rerank, and SRPD rows against full-replay
`shadow_v1` and diagnostic `shadow_v6`. The matrix keeps `comparison_scope`
visible because Week7.7 paper-candidate exports and Week7.9 shadow diagnostics
must not be silently merged into one evidence tier.

Recommended server command:

```bash
python main_build_unified_method_matrix.py \
  --week77_root ~/projects/uncertainty-llm4rec/export/week7_7_four_domain_final \
  --shadow_matrix_path outputs/summary/shadow_v1_to_v6_status_matrix.csv \
  --output_root outputs/summary \
  --output_name unified_method_matrix_week77_shadow
```

Interpretation rule:

- SRPD is the current self-trained ranking framework result line.
- `shadow_v1` is the current full-replay winner signal source.
- `shadow_v6` is a promising signal-to-decision diagnostic bridge until rerun
  or verified under the same paper-result protocol as SRPD.
- External baselines from related papers remain a separate validation gate
  unless they are reproduced under the same split, candidate set, and metrics.

## Week8 External Baseline Audit

The rescue branch adds a senior-paper audit entry point:

```bash
python main_audit_baseline_papers.py \
  --baseline_root Paper/BASELINE \
  --collections NH,NR \
  --output_root outputs/summary \
  --output_name baseline_paper_audit_matrix \
  --include_archives
```

This scans the local `Paper/BASELINE/NH` and `Paper/BASELINE/NR` PDFs and emits
`outputs/summary/baseline_paper_audit_matrix.csv/md`.

Use this table to choose which external baselines are worth adapting. The first
same-schema result layer should be SASRec/BERT4Rec/GRU4Rec/LightGCN-style
baselines, followed by paper-specific adapters such as OpenP5, SLMRec,
LLM-ESR, LLMEmb, LLM2Rec, RLMRec, IRLLRec, or SETRec when their repositories can
score the exact candidate set.

The same-candidate adapter entry points are:

```bash
python main_export_same_candidate_baseline_task.py
python main_train_sasrec_same_candidate.py
python main_train_gru4rec_same_candidate.py
python main_train_bert4rec_same_candidate.py
python main_train_lightgcn_same_candidate.py
python main_import_same_candidate_baseline_scores.py
```

The export script creates train interactions, exact candidate rows, and a
RecBole-style `.inter` file. `main_train_sasrec_same_candidate.py` and
`main_train_gru4rec_same_candidate.py`, `main_train_bert4rec_same_candidate.py`,
and `main_train_lightgcn_same_candidate.py` are the lightweight PyTorch
classical baseline trainers. The import script turns external model scores into
the repository's `rank_predictions.jsonl` format and evaluates them with the
same ranking metric code used elsewhere.

Once real external baseline summaries exist, pass
`--external_summary_glob "outputs/*/tables/same_candidate_external_baseline_summary.csv"`
to `main_build_unified_method_matrix.py` to add them to the unified matrix.

## Notes

- Current `100`-sample runs are best treated as stable research baselines, not final large-scale paper numbers.
- The summary layer is designed so later larger runs can replace or extend existing rows without changing file formats.
- `Paper/` local reports are intentionally not part of the reproducibility surface; the reproducibility surface is the config + script + summary stack.
