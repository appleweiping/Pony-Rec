# M4: Baseline System

## Role

This milestone is the fairness and reviewer-defense layer. It prevents the
project from comparing incompatible reported numbers or mixing protocols.

## Baseline Families

| family | examples | status |
| --- | --- | --- |
| classical same-candidate | SASRec, GRU4Rec, BERT4Rec, LightGCN | local trainers available |
| paper-style LLM-rec | LLM2Rec-style, LLM-ESR-style, LLMEmb-style, RLMRec-style, IRLLRec-style, SETRec-style | supplementary sanity block |
| official-code-level LLM-rec | pinned LLM2Rec, LLM-ESR, LLMEmb, RLMRec, IRLLRec, SETRec | LLM2Rec, LLM-ESR, and LLMEmb completed/imported across declared four-domain protocol; RLMRec run-stage support implemented pending server validation; remaining two need official run-stage adapters |
| internal formal methods | C-CRP, SRPD | must use same score schema/importer before table claims |

## Non-Negotiable Contract

```text
same train/valid/test event rows
same candidate items per event
same metric importer
same score schema: source_event_id,user_id,item_id,score
same paired-test unit
```

Internal method rows are held to the same score gate:

```text
C-CRP -> status_label=same_schema_internal_method
SRPD  -> status_label=same_schema_internal_ablation unless leakage-clean
         train/eval, weighted loss, and native candidate scores are complete
```

C-CRP is the main method. SRPD is our trainable framework/ablation line; it
does not substitute for external baselines.

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

## Current Official Baseline Status

As of 2026-05-09, LLM2Rec is the first completed official-code-level external
LLM-rec baseline under the declared Qwen3-8B same-candidate protocol:

```text
LLM2Rec official qwen3base SASRec
-> beauty supplementary smaller-N completed/imported
-> books large10000 100neg completed/imported
-> electronics large10000 100neg completed/imported
-> movies large10000 100neg completed/imported
-> summary:
   outputs/summary/week8_llm2rec_official_qwen3base_fourdomain_summary.csv
   outputs/summary/week8_llm2rec_official_qwen3base_fourdomain_summary.md
```

The production pattern is now fixed for the remaining official baselines:
single-domain run, provenance/coverage audit, minimal evidence package copied
off server, documented intermediate cleanup, same-candidate import, then
method-level summary. A baseline is not complete until all declared domains are
imported and summarized. LLM-ESR and LLMEmb have now followed that path across
the declared domains. RLMRec is the next runner-enabled method, pending server
rows with `official_completed`, `blockers=[]`, and exact score coverage. The
remaining methods without run-stage support are IRLLRec and SETRec.

For comparison reporting, prefer `NDCG@5`, `NDCG@10`, `HR@5`, and `HR@10` in
the main baseline view. Keep `@20` columns in the extended tables when they are
available, because they help show whether gains hold beyond the top-10 slice.
The current official baseline block is a floor, not the final width target: six
official-code-level LLM-rec baselines are now the minimum, and the working goal
is to reach eight by adding two more current-year recommendation baselines from
DBLP/GitHub.

## Related Files

- `configs/official_external_baselines.yaml`
- `configs/baseline/week8_external_same_candidate_manifest.yaml`
- `OFFICIAL_EXTERNAL_BASELINE_UPGRADE_PLAN_2026-05-07.md`
- `main_audit_official_external_repos.py`
- `main_audit_official_fairness_policy.py`
- `main_make_official_external_adapter_plan.py`
- `main_import_same_candidate_baseline_scores.py`
- `main_select_ccrp_variant_on_valid.py`
- `main_export_srpd_scores_from_predictions.py`
- `main_build_external_only_baseline_comparison.py`
- `docs/archive/legacy_root_reports/BASELINE_PAPER_AUDIT_2026-05-04.md`
