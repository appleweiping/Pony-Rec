# M1: Pony Framework / Week5-6

## Role

This milestone turns the early observation into a framework. The key shift is
from "can the model say it is confident?" to "can uncertainty be made useful
for candidate-ranking decisions?"

## Framework Layers

```text
pointwise diagnosis
-> pairwise preference and correction
-> candidate ranking / reranking
-> validation-selected calibration
-> decision-time risk adjustment
```

## What It Established

- Same-schema candidate ranking is the primary experimental surface.
- Calibration must be fit on validation and applied once to test.
- Structured risk and SRPD-style training provide a stronger decision layer
  than raw confidence alone.
- Exposure, coverage, robustness, and paired tests are part of the core
  evaluation, not optional reporting.

## Paper Role

Use this milestone as the method bridge:

```text
Framework: task-grounded calibrated uncertainty converts an unreliable
confidence observation into a controlled candidate-ranking decision layer.
```

## Related Files

- `docs/paper_claims_and_status.md`
- `docs/candidate_protocol.md`
- `docs/calibration_protocol.md`
- `docs/from_teacher_line_to_srpd_bridge.md`
- `docs/archive/legacy_root_reports/UNIFIED_METHOD_MATRIX_INTERPRETATION_2026-05-04.md`
- `docs/archive/legacy_root_reports/WEEK8_UNIFIED_METHOD_AND_BASELINE_PLAN_2026-05-04.md`

