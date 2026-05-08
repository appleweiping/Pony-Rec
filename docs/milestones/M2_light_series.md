# M2: Light Series

## Role

The Light series is the old verbalized-confidence / lightweight-signal line.
It is not LightGCN. Keep this distinction explicit.

## What It Tested

- Local 8B and early local execution paths.
- Direct yes/no relevance confidence and lightweight risk signals.
- Confidence collapse and calibration boundary cases.
- Whether simple verbalized confidence could scale into the main method.

## Current Interpretation

The Light line is useful as a precursor and negative-control ablation:

```text
Direct verbalized confidence contains signal, but it is not stable enough to be
the final task-grounded recommendation uncertainty mechanism.
```

## Paper Role

Use this milestone to explain why the project moved toward Shadow and C-CRP:

- Light shows the value and limits of raw/verbalized confidence.
- Shadow/C-CRP are the response to those limits.
- Light should not replace task-grounded uncertainty as the main method.

## Related Files

- `main_audit_light_pointwise_signal.py`
- `main_eval.py`
- `main_calibrate.py`
- `src/uncertainty/verbalized_confidence.py`
- `prompts/pointwise_yesno.txt`
- `docs/archive/legacy_root_reports/WEEK8_FUTURE_FRAMEWORK_ROADMAP_2026-05-06.md`

