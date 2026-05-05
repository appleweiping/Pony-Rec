# Week8 Ours + External Combo and External-Only Diagnostics Plan - 2026-05-06

This note records two follow-up experiments after the six-paper external
baseline block:

```text
4 classical same-candidate baselines
6 senior-recommended LLM-rec paper-project same-schema baselines
```

## Question 1: Ours + Their Method

Goal:

```text
Can a combination of our framework row and a strong external paper-project row
produce a stronger final observed result?
```

Script:

```text
main_run_week8_ours_external_rank_fusion.py
```

Protocol:

- Load our Week7.7 `structured_risk` and SRPD-best predictions.
- Load completed external same-candidate paper-project predictions from
  `same_candidate_external_baseline_summary.csv`.
- Fuse rankings with normalized Borda rank scores:
  `combined = ours_weight * ours_rank_score + (1 - ours_weight) * external_rank_score`.
- Evaluate each pair and weight under the same ranking metric code path.
- Save best-by-pair and best-by-domain diagnostic rows.

Paper safety:

- A weight sweep on the test set is a diagnostic upper-bound.
- For main-paper claims, use a fixed weight such as `0.5` or select the weight
  on validation data before touching test.
- Safe wording is `diagnostic rank-fusion / complementary signal`, not a new
  standalone external baseline.

Server command:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
git pull --ff-only

python main_run_week8_ours_external_rank_fusion.py \
  --week77_root ~/projects/uncertainty-llm4rec/export/week7_7_four_domain_final \
  --external_summary_glob "outputs/*/tables/same_candidate_external_baseline_summary.csv" \
  --domains beauty,books,electronics,movies \
  --ours_methods structured_risk,srpd_best \
  --external_methods llm2rec_style_qwen3_sasrec,llmesr_style_qwen3_sasrec,llmemb_style_qwen3_sasrec,rlmrec_style_qwen3_graphcl,irllrec_style_qwen3_intent,setrec_style_qwen3_identifier \
  --weights 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --output_dir outputs/summary/week8_ours_external_rank_fusion

cat outputs/summary/week8_ours_external_rank_fusion/fusion_best_by_domain.csv
cat outputs/summary/week8_ours_external_rank_fusion/fusion_pair_best_metrics.csv
```

What to look for:

- `delta_vs_best_constituent_NDCG@10 > 0` means the fusion is observed higher
  than both the selected ours row and selected external row.
- If the best row uses a test-selected weight, report it only as a diagnostic
  upper-bound unless rerun with a pre-fixed or validation-selected weight.

## Question 2: External-Only Phenomenon

Goal:

```text
If we remove our structured-risk/SRPD method from the candidate method set, can
the external paper-project baselines still show the earlier base phenomenon?
```

Script:

```text
main_run_week8_external_only_phenomenon_diagnostics.py
```

Protocol:

- Exclude `structured_risk` and SRPD from the candidate method set.
- By default, use the best single external method as the base reference for
  event bins, keeping this diagnostic external-only.
- Optionally use `--base_reference direct` if the desired question is relative
  to the Week7.7 direct base ranking.
- Analyze the six external paper-project baselines by:
  - best single external method,
  - external-only per-event oracle,
  - base-rank bins,
  - positive-item popularity bins,
  - external-method disagreement bins.

Server command:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
git pull --ff-only

python main_run_week8_external_only_phenomenon_diagnostics.py \
  --week77_root ~/projects/uncertainty-llm4rec/export/week7_7_four_domain_final \
  --external_summary_glob "outputs/*/tables/same_candidate_external_baseline_summary.csv" \
  --domains beauty,books,electronics,movies \
  --external_methods llm2rec_style_qwen3_sasrec,llmesr_style_qwen3_sasrec,llmemb_style_qwen3_sasrec,rlmrec_style_qwen3_graphcl,irllrec_style_qwen3_intent,setrec_style_qwen3_identifier \
  --base_reference best_single_external \
  --output_dir outputs/summary/week8_external_only_phenomenon

cat outputs/summary/week8_external_only_phenomenon/external_only_oracle_summary.csv
cat outputs/summary/week8_external_only_phenomenon/external_only_base_rank_bins.csv
cat outputs/summary/week8_external_only_phenomenon/external_only_disagreement_bins.csv
cat outputs/summary/week8_external_only_phenomenon/external_only_popularity_bins.csv
```

What to look for:

- `oracle_gain_vs_best_single_NDCG@10 > 0` means external methods contain
  complementary event-level signals even without our method.
- Larger oracle gain in weaker best-single-external base-rank bins means the
  earlier base phenomenon is visible in the external-only setting.
- Larger oracle gain in high-disagreement bins means external method
  disagreement can serve as a diagnostic proxy for hard/uncertain events.

Safe wording:

> Without using our structured-risk/SRPD method as a candidate, the external
> paper-project baselines still show event-level complementarity: the
> external-only oracle is observed higher than the best single external method,
> especially in harder base/disagreement bins.

Unsafe wording:

```text
Their methods implement our uncertainty mechanism.
```

Why unsafe:

- The diagnostic only observes external-method complementarity and disagreement.
- It does not show that those methods contain the same uncertainty/risk module
  as our framework.
