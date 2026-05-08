# Official External Baseline Upgrade Plan - 2026-05-07

This note upgrades the baseline target from paper-style adapters to
official-code-level baselines under the same-candidate protocol. The important
change is the standard of evidence, not the candidate protocol or result schema:
all final rows use the same train/valid/test candidate packages and the same
`source_event_id,user_id,item_id,score` score import path.

For future agents, root-level `AGENTS.md` is binding. In particular, an
`official` runner name, wrapper, or adapter plan row is not evidence by itself;
only unblocked provenance plus exact score/import/stat-test gates can upgrade a
row to `official_completed`.

## Target Standard

Final paper-facing external baselines should satisfy:

```text
official algorithm / official-code-level implementation
+ unified same-candidate train/valid/test protocol
+ unified Qwen3-8B base model for LLM/text representation
+ frozen Qwen3-8B base except method-declared LoRA, adapter, identifier,
  representation learner, graph/intent module, or downstream checkpoint that is
  part of the baseline's official algorithm
+ baseline official default or recommended hyperparameters unless a protocol
  bridge requires an explicit documented override
+ our framework hyperparameters selected on validation only or fixed before test
+ exact candidate-score export
+ unified importer, metrics, coverage audit, and paired tests
```

The required score schema remains:

```text
source_event_id,user_id,item_id,score
```

For main-table official rows, those four columns are an exact key contract:
`source_event_id,user_id,item_id` must match the candidate rows one-for-one,
keys must be unique, scores must be finite numeric values, and extra or missing
candidate scores are disallowed. Scores may be method-native and uncalibrated;
the importer only requires that they rank candidates within the same event.

Full-catalog metrics from an external repository must not be mixed into the
main same-candidate table.

The candidate protocol is intentionally unchanged:

```text
same event rows
same candidate items per event
same positives and sampled negatives
same split discipline
same importer
same metric definitions
same paired-test unit
```

The only acceptable adaptation is the bridge needed to feed those exact rows
into the official method and recover exact candidate scores.

## Fairness Policy

The default paper-facing policy follows the common academic comparison pattern:

```text
official source code
+ our dataset and same-candidate interface
+ unified Qwen3-8B base model
+ method-declared adapter or representation artifact when the official method
  uses one
+ baseline official default/recommended hyperparameters
+ our method tuned on validation only
```

This is the primary comparison variant:

```text
official_code_qwen3base_default_hparams_declared_adaptation
```

Full fine-tuning is allowed only as a separate variant:

```text
official_code_qwen3_8b_full_finetune_default_hparams_supplementary
```

Do not mix full fine-tuning rows into the primary comparison table.
If the project later retunes every baseline on validation, report that as a
stronger sensitivity or robustness variant, not as an unmarked replacement for
the default-hyperparameter table.

The primary table also requires `implementation_status=official_completed`.
Rows marked `style_adapter_only` or `partial_official_adapter_exists` are
supplementary/pilot evidence until the official provenance and exact score audit
pass.

## Current State

The current Week8 full-external script is a strong protocol sanity check, but
not yet a six-method official reproduction suite.

| baseline | current state | official upgrade status |
| --- | --- | --- |
| LLM2Rec | upstream adapter/scorer exists | closest to official-code adaptation |
| LLM-ESR | upstream handled-data adapter/scorer exists | close, needs commit-pinned audit |
| LLMEmb | local paper-style implementation | needs official repo adapter |
| RLMRec | local paper-style implementation | needs official repo adapter |
| IRLLRec | local paper-style implementation | needs official repo adapter |
| SETRec | local paper-style implementation | needs official repo adapter |

Do not label the current `*_style_*` rows as official reproductions.

Safe label:

```text
paper-style same-candidate adapted baselines
```

Future label after this plan is completed:

```text
official-code-level same-candidate adapted baselines
```

## Pinned Official Sources

The implementation contract is machine-readable in:

```text
configs/official_external_baselines.yaml
```

Pinned sources:

| method | official repo | pinned HEAD |
| --- | --- | --- |
| LLM2Rec | `https://github.com/HappyPointer/LLM2Rec` | `73b481f710f67166ab958f4985d27b27fb410871` |
| LLM-ESR | `https://github.com/Applied-Machine-Learning-Lab/LLM-ESR` | `e5dc388c12509c88c65536ecd8d231325993d4ef` |
| LLMEmb | `https://github.com/Applied-Machine-Learning-Lab/LLMEmb` | `3458a5e225062e94b4f1a01e41f3ec82089f0407` |
| RLMRec | `https://github.com/HKUDS/RLMRec` | `22413752246de3dee8ab0d509f7f7a8889080f95` |
| IRLLRec | `https://github.com/wangyu0627/IRLLRec` | `ee8330b456e568f87869ec6c0c553c55d43fce6e` |
| SETRec | `https://github.com/Linxyhaha/SETRec` | `2ed9a75ad1ad3784c61bba3c68cbedbe3cfce2d7` |

## Adapter Contract

Each official adapter must:

1. Install or materialize the unified task package in the official repository's
   native data format.
2. Preserve the official algorithm, recommender architecture, loss/objective,
   adapter or representation-learning step, and scoring head.
3. Use the unified Qwen3-8B base model whenever the method consumes an LLM/text
   representation.
4. Train and retain the method-specific LoRA, adapter, semantic identifier,
   intent representation, graph contrastive module, or item embedding artifact
   according to the official method's own algorithm.
5. Replace only the representation source or data bridge required by the
   same-candidate protocol.
6. Train/select checkpoints without using test metrics.
7. Emit exact same-candidate score CSVs using the shared schema.
8. Import results through `main_import_same_candidate_baseline_scores.py`.
9. Write provenance:
   - official repo URL
   - pinned commit
   - local checkout path
   - preserved modules
   - protocol changes
   - Qwen3-8B base-model source
   - LoRA/adapter or method-specific representation artifact path
   - checkpoint path
   - score coverage

Minimum provenance row fields:

```text
baseline_name
domain
comparison_variant
official_repo
pinned_commit
local_repo_path
official_entrypoints_used
preserved_algorithm_components
same_candidate_task_path
qwen3_base_model_path
llm_adaptation_mode
baseline_hyperparameter_source
baseline_hyperparameter_overrides
baseline_extra_tuning_allowed
default_hparam_source_file_or_url
override_reason
validation_tuned
our_hyperparameter_selection_rule
validation_selection_metric
adapter_or_representation_artifact_path
checkpoint_path
score_csv_path
score_coverage
imported_result_path
implementation_status
audit_status
```

## Work Order

Recommended sequence:

1. Audit local official checkouts and commit pins with
   `main_audit_official_external_repos.py`.
2. Generate or refresh the method-by-method adapter plan with
   `main_make_official_external_adapter_plan.py`.
3. Confirm that the shared same-candidate packages exist for every domain/split
   needed by the target comparison.
4. Confirm that the Qwen3-8B base path and method-declared adapter or
   representation source are available and recorded for all methods that
   consume text/LLM representations.
5. Finish LLM2Rec official-code adaptation because the current adapter path is
   already close.
6. Finish LLM-ESR official-code adaptation because the handled-data adapter is
   already close.
7. Implement LLMEmb official adapter.
8. Implement RLMRec official adapter.
9. Implement IRLLRec official adapter.
10. Implement SETRec official adapter.
11. For each method/domain, export exact score CSVs and import them through the
    shared importer.
12. Verify score coverage equals 1.0 or explicitly document exclusions before
    any row enters a main table.
13. Rebuild the final comparison table using only `*_official_qwen3base_*`
    rows for the official baseline claim.
14. Run paired statistical tests and keep non-significant differences labeled
    as observed differences, not wins.

The current paper-style rows may remain as a supplementary sanity check, but
they should not be the final answer if the claim is "official algorithm-level
baselines."

## Per-Method Upgrade Checklist

Use this checklist for each of LLM2Rec, LLM-ESR, LLMEmb, RLMRec, IRLLRec, and
SETRec:

```text
[ ] official repo cloned locally
[ ] checkout matches pinned commit
[ ] official train/eval/scoring entry points identified
[ ] official algorithm components listed and preserved
[ ] unified same-candidate train/valid/test package installed
[ ] Qwen3-8B base model wired as shared text/LLM source
[ ] official adapter/representation artifact trained or loaded as required
[ ] default/recommended hyperparameter source recorded
[ ] any protocol-bridge override justified
[ ] checkpoint selected on validation only
[ ] exact candidate scores exported with source_event_id,user_id,item_id,score
[ ] shared importer run successfully
[ ] score coverage audit passes
[ ] provenance record written
[ ] paired-test inputs generated
```

Do not mark a row as official until every item that applies to the method is
complete.

## Paper-Safe Wording

Before this upgrade is complete:

```text
We include controlled paper-style adaptations of representative LLM-enhanced
recommendation frameworks under a unified same-candidate protocol.
```

After this upgrade is complete:

```text
We adapt official or official-code-level implementations of six representative
LLM-enhanced recommendation frameworks to our unified same-candidate protocol,
standardizing split, candidates, metrics, and the Qwen3-8B base model while
using the method-declared adapter or representation-learning regime required by
each official algorithm. Full fine-tuning and retuned-baseline variants are
reported separately when run.
```
