# Project Lineage And File Index - 2026-05-06

This index is a lightweight organization layer. It intentionally avoids moving
active code files because many scripts, notes, and server commands refer to
their current paths.

## Active Mainline

Large-scale external baseline protocol:

```text
scripts/run_week8_large_scale_10k_100neg.sh
main_build_large_scale_same_candidate_runtime.py
main_export_llmesr_same_candidate_task.py
main_export_llm2rec_same_candidate_task.py
main_train_sasrec_same_candidate.py
main_train_gru4rec_same_candidate.py
main_train_bert4rec_same_candidate.py
main_train_lightgcn_same_candidate.py
main_train_llmemb_style_same_candidate.py
main_train_rlmrec_style_same_candidate.py
main_train_irllrec_style_same_candidate.py
main_train_setrec_style_same_candidate.py
main_build_external_only_baseline_comparison.py
main_run_week8_external_only_phenomenon_diagnostics.py
main_run_week8_external_paired_stat_tests.py
```

Current handoff and plans:

```text
CODEX_HANDOFF_WEEK8_2026-05-06.md
WEEK8_LARGE_SCALE_10K_100NEG_PLAN_2026-05-06.md
WEEK8_FUSION_EXTERNAL_ONLY_CONTRIBUTION_UPDATE_2026-05-06.md
WEEK8_FUTURE_FRAMEWORK_ROADMAP_2026-05-06.md
```

## Future Framework Scaffolds

Config:

```text
configs/week8_large_scale_future_framework.yaml
```

Scripts:

```text
main_build_week8_same_candidate_pointwise_inputs.py
main_run_week8_shadow_v6_gate_sweep.py
main_build_week8_generated_title_verification_scaffold.py
main_make_week8_future_framework_commands.py
main_build_week8_lora_framework_scaffold.py
scripts/run_week8_shadow_large_scale_diagnostic.sh
scripts/run_week8_light_large_scale_ablation.sh
scripts/run_week8_generated_title_verification_scaffold.sh
configs/week8_future_lora/
```

Purpose:

```text
Reuse large10000_100neg task packages for:
shadow_v1/v6 large-scale
old light/verbalized-confidence large-scale ablation
Signal LoRA planning
Decision / Generative LoRA planning
generated-title verification scaffolding
```

## Shadow Line

Core scripts:

```text
main_shadow_make_commands.py
main_eval_shadow.py
main_calibrate_shadow.py
main_build_shadow_v6_bridge.py
main_summarize_shadow_v1_to_v6.py
src/shadow/
prompts/shadow_v1_relevance_probability.txt
prompts/shadow_v2_topk_inclusion_probability.txt
prompts/shadow_v3_preference_strength.txt
prompts/shadow_v4_rank_position_distribution.txt
prompts/shadow_v5_intent_prototype_match.txt
prompts/shadow_v6_signal_to_decision.txt
```

Status docs:

```text
SHADOW_V1_TO_V6_STATUS_2026-05-04.md
SHADOW_V6_SERVER_DIAGNOSTIC_2026-05-04.md
docs/week7_9_shadow_observation_report.md
docs/shadow_method.md
```

Current interpretation:

```text
shadow_v1/v6 has positive diagnostic evidence, but is not yet a completed
trained large-scale method.
```

## Light / Verbalized Confidence Line

Core scripts:

```text
main_audit_light_pointwise_signal.py
main_eval.py
main_calibrate.py
src/uncertainty/verbalized_confidence.py
prompts/pointwise_yesno.txt
```

Status:

```text
Useful precursor observation and negative-control ablation. It should not
replace task-grounded shadow as the main future method.
```

## SRPD / Structured-Risk Line

Core scripts:

```text
main_build_srpd_rank_data.py
main_lora_train_rank.py
main_compare_srpd_frameworks.py
main_compare_framework.py
main_rank_rerank.py
src/training/
src/methods/uncertainty_ranker.py
configs/srpd/
configs/lora/
```

Status:

```text
Small/medium same-candidate paper-ready block is strong. Large-scale SRPD is
optional after external large-scale and shadow diagnostic priorities.
```

## External LLM-Rec Baselines

Result docs:

```text
LLM_PROJECT_QWEN3_6PAPER_BASELINE_RESULT_2026-05-05.md
LLM_PROJECT_QWEN3_4PAPER_BASELINE_RESULT_2026-05-05.md
LLM_PROJECT_QWEN3_BASELINE_COMBINED_RESULT_2026-05-05.md
TWO_MORE_SENIOR_BASELINES_ROUND2_PLAN_2026-05-05.md
PAPER_PROJECT_BASELINE_EXPANSION_PLAN_2026-05-05.md
```

Status:

```text
Completed six-paper same-candidate block. They are strong enough that the paper
story should emphasize complementarity and protocol rigor rather than a simple
standalone win.
```

## Historical/Audit Docs In Root

These root markdown files are useful but can later move to `docs/archive/` after
references are updated:

```text
BASELINE_PAPER_AUDIT_2026-05-04.md
BERT4REC_SAME_CANDIDATE_BASELINE_2026-05-04.md
GRU4REC_SAME_CANDIDATE_BASELINE_2026-05-04.md
LIGHTGCN_SAME_CANDIDATE_BASELINE_2026-05-04.md
SASREC_SAME_CANDIDATE_BASELINE_2026-05-04.md
LOCAL_EXPERIMENT_ASSET_AUDIT_2026-05-04.md
MAINLINE_ALIGNMENT_AUDIT_2026-05-04.md
UNIFIED_METHOD_MATRIX_INTERPRETATION_2026-05-04.md
WEEK8_UNIFIED_METHOD_AND_BASELINE_PLAN_2026-05-04.md
```

Do not move them during an active server run unless necessary.

## Next-Codex Reading Order

Use this order:

```text
1. CODEX_HANDOFF_WEEK8_2026-05-06.md
2. WEEK8_LARGE_SCALE_10K_100NEG_PLAN_2026-05-06.md
3. WEEK8_FUTURE_FRAMEWORK_ROADMAP_2026-05-06.md
4. PROJECT_LINEAGE_AND_FILE_INDEX_2026-05-06.md
5. SHADOW_V1_TO_V6_STATUS_2026-05-04.md
```

Then check the active server log before launching any new heavy run.
