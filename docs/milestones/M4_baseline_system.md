# M4: Baseline System

## Role

This milestone is the fairness and reviewer-defense layer. It prevents the
project from comparing incompatible reported numbers or mixing protocols.

## Baseline Families

| family | examples | status |
| --- | --- | --- |
| classical same-candidate | SASRec, GRU4Rec, BERT4Rec, LightGCN | local trainers available |
| paper-style LLM-rec | LLM2Rec-style, LLM-ESR-style, LLMEmb-style, RLMRec-style, IRLLRec-style, SETRec-style | supplementary sanity block |
| official-code-level LLM-rec | pinned LLM2Rec, LLM-ESR, LLMEmb, RLMRec, IRLLRec, SETRec | contract written, adapters unfinished |

## Non-Negotiable Contract

```text
same train/valid/test event rows
same candidate items per event
same metric importer
same score schema: source_event_id,user_id,item_id,score
same paired-test unit
```

For official external baselines:

```text
unified Qwen3-8B base model
primary comparison variant = official code + Qwen3-8B base + declared
adaptation mode + baseline official default or recommended hyperparameters
frozen Qwen3-8B base except method-declared adapter or representation artifact
trained and retained according to the baseline's official algorithm
official repo URL and pinned commit recorded
checkpoint / adapter / score provenance recorded
exact finite score coverage audited before table import
```

Full fine-tuning and retuned-baseline comparisons are explicitly labeled
supplementary/sensitivity variants. They must not be silently mixed with the
primary standardized-backbone default-hyperparameter table. Official rows enter
the main table only with `implementation_status=official_completed`, exact
finite score coverage, and fairness provenance.

## Paper Role

Use paper-style baselines as supplementary controlled adaptations. Use
official-code-level rows for final official external-baseline claims only after
the provenance checklist passes.

## Related Files

- `configs/official_external_baselines.yaml`
- `configs/baseline/week8_external_same_candidate_manifest.yaml`
- `OFFICIAL_EXTERNAL_BASELINE_UPGRADE_PLAN_2026-05-07.md`
- `main_audit_official_external_repos.py`
- `main_audit_official_fairness_policy.py`
- `main_make_official_external_adapter_plan.py`
- `main_import_same_candidate_baseline_scores.py`
- `main_build_external_only_baseline_comparison.py`
- `BASELINE_PAPER_AUDIT_2026-05-04.md`
