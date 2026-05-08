# Week8 Future Framework Roadmap - 2026-05-06

Legacy roadmap note: keep this as historical planning context. The current
agent contract and source of truth are `AGENTS.md`, `README.md`,
`docs/milestones/README.md`, and `docs/paper_claims_and_status.md`.

This note keeps the post-large-scale plan explicit so later Codex chats do not
forget the shadow/light/LoRA/generated-title direction.

Canonical milestone view:

```text
docs/milestones/M6_complete_recommender_system.md
docs/server_runbook.md
docs/top_conference_review_gate.md
```

This is an M6 roadmap document. It should not be read as evidence that the full
recommendation system is already complete.

## Current Priority

Finish the running large-scale external baseline protocol first:

```text
Books/Electronics/Movies
10k users per domain
100 sampled negatives + 1 positive
same candidate rows
classical baselines + external paper-project style baselines
external-only comparison, phenomenon diagnostics, paired tests
```

In parallel with documentation/planning only, keep the official-baseline target
clear:

```text
The final official external baseline tier reuses this same-candidate protocol,
but replaces paper-style local approximations with pinned official or
official-code-level adapters. Those adapters use the unified Qwen3-8B base
model, retain each baseline's official LoRA/adapter or representation-learning
algorithm, and emit the unchanged source_event_id,user_id,item_id,score schema.
```

The command currently being run on the server is:

```bash
bash scripts/run_week8_large_scale_10k_100neg.sh
```

This run does not include shadow, old light, LoRA, or generated-title
verification. It creates the protocol foundation they should later use.

It also does not yet constitute the official external-baseline tier; the
official upgrade is tracked separately in
`OFFICIAL_EXTERNAL_BASELINE_UPGRADE_PLAN_2026-05-07.md`.

## Why This Order

Do not start by training a large LoRA stack before the 101-candidate protocol is
validated. The safer top-conference path is:

```text
1. Validate the large-scale same-candidate protocol.
2. Upgrade external LLM-rec baselines from paper-style rows to official or
   official-code-level same-candidate rows when making official baseline claims.
3. Add large-scale shadow/light diagnostics under the exact same protocol.
4. Select gates and teachers on validation, not test.
5. Only then train Signal LoRA and Decision/Generative LoRA.
6. Move from candidate ranking to catalog-grounded generated-title verification.
```

This keeps the work original and avoids stitching unrelated baselines together.

## Lines And Roles

| line | role | status |
| --- | --- | --- |
| external large-scale | robustness gate and strong baseline map | running |
| official external upgrade | final official baseline standard | planned/contract written |
| old light/verbalized confidence | negative-control / precursor observation | future optional large-scale ablation |
| shadow_v1 | task-grounded uncertainty signal | future large-scale diagnostic |
| shadow_v6 | validation-selected signal-to-decision gate | future large-scale diagnostic |
| Signal LoRA | train local model to emit shadow signal | after shadow diagnostic |
| Decision LoRA | train accept/revise/fallback or pairwise correction behavior | after v6 gate selection |
| Generative LoRA | generated-title recommendation | after verification scaffold |
| generated-title verification | catalog grounding + shadow verification | scaffold first, full method later |

## Implemented Scaffolds In This Commit

Config:

```text
configs/week8_large_scale_future_framework.yaml
```

Data bridge:

```text
main_build_week8_same_candidate_pointwise_inputs.py
```

Converts same-candidate ranking JSONL into pointwise rows for shadow or old
light inference. This is the key guardrail: future methods reuse the same
large10000/100neg candidate rows.

Shadow gate selection:

```text
main_run_week8_shadow_v6_gate_sweep.py
```

Sweeps `shadow_v6` gate thresholds on validation and evaluates the selected
gate once on test.

Generated-title verification scaffold:

```text
main_build_week8_generated_title_verification_scaffold.py
```

Builds a catalog-grounded verification dataset using catalog titles as a proxy.
This is not the final generative result; replace `generated_title` with actual
model-generated titles before claiming a generative recommender.

Command generator:

```text
main_make_week8_future_framework_commands.py
```

Generates scripts for future shadow, old-light, and generated-title scaffold
runs after the external large-scale run is healthy.

LoRA config scaffold:

```text
main_build_week8_lora_framework_scaffold.py
```

Writes template configs for Signal LoRA, Decision LoRA, and Generative LoRA.
These configs are deliberately marked scaffold-only until their referenced
teacher datasets and verification rows exist.

## How To Generate Future Commands

Direct tracked wrappers:

```bash
# Shadow v1/v6 large-scale diagnostic after the external baseline run finishes.
bash scripts/run_week8_shadow_large_scale_diagnostic.sh

# Old light/verbalized-confidence large-scale negative-control ablation.
bash scripts/run_week8_light_large_scale_ablation.sh

# Catalog-title proxy verification scaffold, no model inference.
bash scripts/run_week8_generated_title_verification_scaffold.sh
```

Smoke versions:

```bash
MAX_EVENTS=20 bash scripts/run_week8_shadow_large_scale_diagnostic.sh
MAX_EVENTS=20 bash scripts/run_week8_light_large_scale_ablation.sh
MAX_EVENTS=20 bash scripts/run_week8_generated_title_verification_scaffold.sh
```

The wrappers generate command scripts under `outputs/summary/` and logs under
`outputs/summary/logs/`.

Manual command generation:

Shadow only:

```bash
python main_make_week8_future_framework_commands.py \
  --config configs/week8_large_scale_future_framework.yaml \
  --stage shadow \
  --output_path outputs/summary/week8_large10000_100neg_shadow_commands.sh
```

Old light / verbalized confidence negative-control:

```bash
python main_make_week8_future_framework_commands.py \
  --config configs/week8_large_scale_future_framework.yaml \
  --stage light \
  --output_path outputs/summary/week8_large10000_100neg_light_commands.sh
```

Generated-title verification scaffold:

```bash
python main_make_week8_future_framework_commands.py \
  --config configs/week8_large_scale_future_framework.yaml \
  --stage generated_title \
  --output_path outputs/summary/week8_large10000_100neg_generated_title_scaffold_commands.sh
```

Smoke command generation:

```bash
python main_make_week8_future_framework_commands.py \
  --config configs/week8_large_scale_future_framework.yaml \
  --stage all \
  --max_events 20 \
  --output_path outputs/summary/week8_large10000_100neg_future_framework_smoke_commands.sh
```

LoRA config scaffolds:

```bash
python main_build_week8_lora_framework_scaffold.py \
  --config configs/week8_large_scale_future_framework.yaml \
  --output_dir configs/week8_future_lora
```

Do not run full shadow/light inference until the external baseline run has
finished and the task packages are confirmed healthy.

## Shadow Large-Scale Target

First target:

```text
same large10000_100neg valid/test task packages
-> pointwise shadow_v1 rows
-> Qwen3 shadow_v1 inference
-> valid calibration
-> direct anchor ranking on valid/test
-> validation-selected shadow_v6 gate
-> test metrics and paired tests
```

Required reporting:

```text
NDCG@10, MRR, HR@10
calibration metrics
intervention rate / changed ranking fraction
matched signal rate
fallback rate and fallback reasons
gate thresholds selected on valid
paired tests within the 101-candidate protocol
```

Paper-safe wording:

```text
Shadow large-scale is a validation-selected diagnostic extension under the same
101-candidate protocol, not a test-selected post-hoc reranker.
```

## Old Light Large-Scale Target

Old light should not become the main trained method. Its role is:

```text
Direct verbalized confidence contains useful but unstable signal.
This motivated task-grounded shadow signals.
```

Large-scale old-light is useful only as a negative-control or precursor
ablation:

```text
same large10000_100neg pointwise rows
-> pointwise yes/no + verbalized confidence
-> valid calibration
-> calibration/collapse diagnostics
```

Do not spend heavy LoRA budget here unless a later result clearly reverses the
collapse story.

## LoRA Framework Plan

Signal LoRA:

```text
Input: large-scale pointwise rows plus shadow_v1 teacher outputs.
Target: task-grounded relevance/uncertainty JSON or compact score fields.
Gate: must match teacher calibration and ranking utility on validation/test.
```

Decision LoRA:

```text
Input: candidate rows plus shadow_v1 signal, anchor score/rank, and v6 selected gate.
Target: accept/revise/fallback or pairwise correction.
Gate: must beat anchor and preserve calibrated fallback behavior.
```

Generative LoRA:

```text
Input: user history.
Target: generated title or grounded item-title candidate.
Verifier: catalog grounding + shadow verification.
Gate: report generated-title success only after verification and catalog match.
```

Do not present LoRA as complete until there are trained adapters, startup checks,
validation logs, test metrics, and artifact manifests.

## Generated-Title Verification Plan

The intended final method form is:

```text
user history
-> generated title
-> catalog grounding
-> shadow verification
-> accept / revise / fallback
```

The current scaffold only creates catalog-title proxy verification rows. It is
useful for validating data plumbing and verifier metrics, but it is not a final
generative recommendation result.

## File Organization

To avoid breaking imports and scripts, do not move `main_*.py` files casually.
Use documentation indices first.

Suggested mental organization:

```text
PROJECT_LINEAGE_AND_FILE_INDEX_2026-05-06.md
  human map of active, future, and legacy files

configs/week8_large_scale_future_framework.yaml
  future shadow/light/LoRA/generated-title protocol config

outputs/summary/week8_*_commands.sh
  generated server command scripts
```

Historical root markdown files can later be moved into `docs/archive/` only if
all references are updated and no active server instructions point to old paths.

## Official Baseline Next Steps

Before spending heavy training time on future framework LoRAs, make the
external baseline status unambiguous:

```text
[ ] keep configs/official_external_baselines.yaml unchanged unless the contract changes
[ ] audit pinned official checkouts
[ ] complete LLM2Rec official adapter/scorer
[ ] complete LLM-ESR official adapter/scorer
[ ] implement LLMEmb official adapter/scorer
[ ] implement RLMRec official adapter/scorer
[ ] implement IRLLRec official adapter/scorer
[ ] implement SETRec official adapter/scorer
[ ] export exact source_event_id,user_id,item_id,score files
[ ] import with main_import_same_candidate_baseline_scores.py
[ ] verify coverage, provenance, metrics, and paired-test inputs
[ ] rebuild official comparison using only *_official_qwen3base_* rows
```

The official baseline upgrade does not change the shadow/light/LoRA roadmap's
candidate rows or score schema. It only changes how external baseline scores
are produced: from local paper-style approximations to pinned official
algorithms with the shared Qwen3-8B base model and method-appropriate
LoRA/adapter artifacts.

## Setting References

The current 101-candidate design follows common sampled-ranking practice:

- RecBole supports `uni100` and `pop100` sampled evaluation settings:
  `https://recbole.io/docs/user_guide/config/evaluation_settings.html`
- RecBole also documents `uniN` / `popN` as sample-based ranking modes where
  each positive item is paired with N sampled negative items:
  `https://recbole.io/evaluation.html`
- BERT4Rec uses leave-one-out evaluation with 100 randomly sampled negative
  items, sampled according to item popularity:
  `https://arxiv.org/pdf/1904.06690`
- Sampling studies warn that conclusions can depend on the negative sampler, so
  popularity sampling is the main protocol and uniform is a sensitivity option:
  `https://arxiv.org/abs/2106.10621`

Keep paired tests within the same protocol. Do not compare a Week7.7
six-candidate row directly against a Week8 101-candidate row without stating
the protocol difference.
