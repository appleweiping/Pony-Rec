# Official External Baseline Upgrade Plan - 2026-05-07

This note upgrades the baseline target from paper-style adapters to
official-code-level baselines under the same-candidate protocol.

## Target Standard

Final paper-facing external baselines should satisfy:

```text
official algorithm / official-code-level implementation
+ unified same-candidate train/valid/test protocol
+ unified Qwen3-8B LoRA backbone for LLM/text representation
+ exact candidate-score export
+ unified importer, metrics, coverage audit, and paired tests
```

The required score schema remains:

```text
source_event_id,user_id,item_id,score
```

Full-catalog metrics from an external repository must not be mixed into the
main same-candidate table.

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
2. Preserve the official recommender architecture, loss, and scoring head.
3. Replace only the LLM/text representation source with the shared Qwen3-8B
   LoRA encoder where the method uses LLM/text representations.
4. Train/select checkpoints without using test metrics.
5. Emit exact same-candidate score CSVs using the shared schema.
6. Import results through `main_import_same_candidate_baseline_scores.py`.
7. Write provenance:
   - official repo URL
   - pinned commit
   - local checkout path
   - preserved modules
   - protocol changes
   - backbone replacement
   - checkpoint path
   - score coverage

## Work Order

Recommended sequence:

1. Audit local official checkouts and commit pins.
2. Finish LLM2Rec official-code adaptation because the current adapter path is
   already close.
3. Finish LLM-ESR official-code adaptation because the handled-data adapter is
   already close.
4. Implement LLMEmb official adapter.
5. Implement RLMRec official adapter.
6. Implement IRLLRec official adapter.
7. Implement SETRec official adapter.
8. Rebuild the final comparison table using only `*_official_qwen3_lora_*`
   rows for the official baseline claim.

The current paper-style rows may remain as a supplementary sanity check, but
they should not be the final answer if the claim is "official algorithm-level
baselines."

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
standardizing split, candidates, metrics, and the Qwen3-8B LoRA text backbone.
```
