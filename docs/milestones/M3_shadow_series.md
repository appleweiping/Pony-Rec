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

Shadow v1/v6 have positive diagnostic evidence. The formal internal method
track now treats C-CRP as the main task-grounded uncertainty method and SRPD as
the trainable framework/ablation line.

The formal C-CRP track is:

```text
Qwen3-8B pointwise shadow observation
-> validation-only calibration
-> validation-only C-CRP mode/weight/eta/ablation selection
-> exact same-candidate score export
-> unified importer and paired-test gate
```

The three paper-facing C-CRP score families are:

```text
confidence_only              # calibrated confidence / SRPD-facing confidence baseline
evidence_only                # evidence support minus counterevidence
confidence_plus_evidence     # calibrated confidence combined with evidence, with risk penalty
```

Main missing pieces before a full Shadow-system claim:

- large-scale 101-candidate Shadow v1 inference for all target domains;
- validation-selected C-CRP configuration and imported exact score rows;
- confidence-collapse diagnostics for confidence-only variants;
- validation-selected shadow_v6 gate if the bridge is claimed;
- SRPD leakage-clean teacher generation and weighted-loss training if SRPD is claimed;
- exact same-candidate score export and paired tests.

## Paper Role

Use Shadow as task-grounded uncertainty evidence and future method extension
unless the missing large-scale protocol items are completed.

Safe wording:

```text
C-CRP is our formal task-grounded uncertainty method under the controlled
same-candidate protocol. SRPD is our trainable framework/ablation line and
becomes paper-facing only after leakage, weighted-loss, exact-score export, and
paired-test gates pass.
```

## Related Files

- `docs/archive/legacy_root_reports/SHADOW_V1_TO_V6_STATUS_2026-05-04.md`
- `docs/archive/legacy_root_reports/SHADOW_V6_SERVER_DIAGNOSTIC_2026-05-04.md`
- `docs/week7_9_shadow_observation_report.md`
- `docs/shadow_method.md`
- `main_select_ccrp_variant_on_valid.py`
- `main_export_srpd_scores_from_predictions.py`
- `src/shadow/`
- `prompts/shadow_v1_relevance_probability.txt`
- `prompts/shadow_v6_signal_to_decision.txt`
- `scripts/run_week8_shadow_large_scale_diagnostic.sh`

