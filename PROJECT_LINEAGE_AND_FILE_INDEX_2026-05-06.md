# Project Lineage And File Index - 2026-05-06

This index is a lightweight organization layer. It intentionally avoids moving
active code files because many scripts, notes, and server commands refer to
their current paths.

## Canonical Milestone Layer

Start future project navigation from:

```text
docs/milestones/README.md
docs/top_conference_review_gate.md
docs/server_runbook.md
```

Milestone map:

| milestone | role | canonical file |
| --- | --- | --- |
| M0 Week1-4 / pony12 | observation and confidence diagnosis | `docs/milestones/M0_week1_4_pony12_observation.md` |
| M1 Pony framework | observation to framework | `docs/milestones/M1_pony_framework_week5_6.md` |
| M2 Light series | verbalized-confidence boundary / negative control | `docs/milestones/M2_light_series.md` |
| M3 Shadow series | task-grounded uncertainty signals | `docs/milestones/M3_shadow_series.md` |
| M4 Baseline system | fairness, classical/paper-style/official baselines | `docs/milestones/M4_baseline_system.md` |
| M5 Four-domain validation | small-domain to 100neg four-domain protocol | `docs/milestones/M5_four_domain_same_candidate.md` |
| M6 Complete recommender roadmap | official baselines, Shadow, LoRA, generated-title system | `docs/milestones/M6_complete_recommender_system.md` |

The old root markdown files are retained as evidence and handoff logs. The
milestone layer decides which evidence is main-claim eligible, supplementary,
or roadmap-only.

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

Official external-baseline upgrade helpers:

```text
configs/official_external_baselines.yaml
OFFICIAL_EXTERNAL_BASELINE_UPGRADE_PLAN_2026-05-07.md
main_audit_official_external_repos.py
main_make_official_external_adapter_plan.py
main_import_same_candidate_baseline_scores.py
```

Official upgrade standard:

```text
pinned official or official-code-level implementation
+ official algorithm / loss / scoring head preserved
+ unified Qwen3-8B base model for text or LLM representations
+ LoRA, adapter, semantic identifier, intent representation, graph contrastive
  module, or embedding artifact trained and retained according to that
  baseline's official algorithm
+ unchanged same-candidate task rows
+ unchanged score schema: source_event_id,user_id,item_id,score
+ shared importer, score coverage audit, metrics, and paired tests
```

Do not edit `configs/official_external_baselines.yaml` unless the contract
itself changes.

Current handoff and plans:

```text
docs/milestones/README.md
docs/server_runbook.md
docs/top_conference_review_gate.md
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

Current row family:

```text
llm2rec_style_qwen3_sasrec
llmesr_style_qwen3_sasrec
llmemb_style_qwen3_sasrec
rlmrec_style_qwen3_graphcl
irllrec_style_qwen3_intent
setrec_style_qwen3_identifier
```

These are paper-style adapted baselines, not official reproductions. The target
official family is:

```text
llm2rec_official_qwen3base_sasrec
llmesr_official_qwen3base_sasrec
llmemb_official_qwen3base
rlmrec_official_qwen3base_graphcl
irllrec_official_qwen3base_intent
setrec_official_qwen3base_identifier
```

Per-method file map:

| method | current local scripts | official-upgrade focus |
| --- | --- | --- |
| LLM2Rec | `main_export_llm2rec_same_candidate_task.py`, `main_prepare_llm2rec_upstream_adapter.py`, `main_generate_llm2rec_sentence_embeddings.py`, `main_score_llm2rec_same_candidate_adapter.py` | pin upstream checkout, preserve official SASRec path, verify score coverage |
| LLM-ESR | `main_export_llmesr_same_candidate_task.py`, `main_generate_llmesr_sentence_embeddings.py`, `main_train_score_llmesr_upstream_adapter.py` | audit official class usage, preserve handled-data/scoring architecture |
| LLMEmb | `main_train_llmemb_style_same_candidate.py` | replace style adapter with official repo data/training/scoring bridge |
| RLMRec | `main_train_rlmrec_style_same_candidate.py` | build official graph data, preserve GraphCL/objective, export candidate scores |
| IRLLRec | `main_train_irllrec_style_same_candidate.py` | preserve official intent representation construction and scoring head |
| SETRec | `main_train_setrec_style_same_candidate.py` | preserve semantic identifier/set representation algorithm and scoring head |

Shared downstream files:

```text
main_import_same_candidate_baseline_scores.py
main_build_external_only_baseline_comparison.py
main_run_week8_external_only_phenomenon_diagnostics.py
main_run_week8_external_paired_stat_tests.py
outputs/baselines/external_tasks/
outputs/summary/
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
1. docs/milestones/README.md
2. docs/server_runbook.md
3. docs/top_conference_review_gate.md
4. CODEX_HANDOFF_WEEK8_2026-05-06.md
5. OFFICIAL_EXTERNAL_BASELINE_UPGRADE_PLAN_2026-05-07.md
6. WEEK8_LARGE_SCALE_10K_100NEG_PLAN_2026-05-06.md
7. WEEK8_FUTURE_FRAMEWORK_ROADMAP_2026-05-06.md
8. PROJECT_LINEAGE_AND_FILE_INDEX_2026-05-06.md
9. SHADOW_V1_TO_V6_STATUS_2026-05-04.md
```

Then check the active server log before launching any new heavy run.
