# M0: Week1-4 / pony12 Observation

## Role

This milestone is the project origin. It should be preserved as the source of
the core observation, not promoted as the final system.

## Core Question

```text
Is LLM recommendation confidence informative, and if so why is it still
unreliable as a direct decision signal?
```

## What It Established

- Verbalized LLM confidence can correlate with recommendation correctness.
- Raw confidence is unstable across domains and failure modes.
- Miscalibration, confidence collapse, and domain-dependent behavior make
  direct confidence unsafe as a final ranking signal.
- Calibration and risk-aware reranking are plausible, but need strict
  validation-to-test discipline.

## Paper Role

Use this milestone as diagnosis:

```text
Observation: confidence contains signal, but self-reported confidence is not
reliable enough for recommendation decisions without task-grounding,
calibration, and protocol control.
```

Do not use this milestone to claim a full recommender system, full-catalog
SOTA, or official external-baseline superiority.

## Related Files

- `docs/April 24th evidence posterior confidence repair plan.md`
- `docs/April 24th evidence posterior confidence repair report.md`
- `docs/April 24th evidence posterior full Beauty Day2 report.md`
- `docs/April 24th evidence posterior full Beauty expansion plan.md`
- `docs/April 25th evidence posterior Day6 to Day7 handoff report.md`
- `part5_artifact_map.md`

