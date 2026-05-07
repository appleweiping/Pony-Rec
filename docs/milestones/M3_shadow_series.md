# M3: Shadow Series

## Role

The Shadow series is the task-grounded uncertainty signal family. It shifts the
project from self-reported model confidence to recommendation-specific
uncertainty evidence.

## Signal Family

```text
shadow_v1 relevance probability
shadow_v2 top-k inclusion probability
shadow_v3 preference strength
shadow_v4 rank-position distribution
shadow_v5 intent prototype match
shadow_v6 signal-to-decision bridge
```

## Current Interpretation

Shadow v1/v6 have positive diagnostic evidence, but the complete trained
large-scale Shadow system is not done yet.

Main missing pieces before a full Shadow-system claim:

- large-scale 101-candidate Shadow v1 inference;
- validation-selected shadow_v6 gate;
- accept/revise/fallback controller;
- Signal LoRA and Decision LoRA training;
- exact same-candidate score export and paired tests.

## Paper Role

Use Shadow as task-grounded uncertainty evidence and future method extension
unless the missing large-scale protocol items are completed.

Safe wording:

```text
Shadow is a task-grounded uncertainty signal family with positive diagnostic
evidence. The full trained Shadow recommendation system remains a staged
extension until evaluated under the shared same-candidate protocol.
```

## Related Files

- `SHADOW_V1_TO_V6_STATUS_2026-05-04.md`
- `SHADOW_V6_SERVER_DIAGNOSTIC_2026-05-04.md`
- `docs/week7_9_shadow_observation_report.md`
- `docs/shadow_method.md`
- `src/shadow/`
- `prompts/shadow_v1_relevance_probability.txt`
- `prompts/shadow_v6_signal_to_decision.txt`
- `scripts/run_week8_shadow_large_scale_diagnostic.sh`

