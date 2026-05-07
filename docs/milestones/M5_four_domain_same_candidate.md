# M5: Small-Domain to Four-Domain Same-Candidate Validation

## Role

This milestone upgrades the project from small-domain observation to
cross-domain validation under a harder sampled-ranking protocol.

## Domain Policy

```text
Books / Electronics / Movies:
  main large-domain 10k-user 100neg protocol when enough eligible users exist

Beauty:
  supplementary smaller-N same-candidate 100neg run
```

Beauty must not be described as a 10k main-domain result when eligible users are
fewer than 10k.

## Protocol Boundary

Do not mix these as direct row-level comparisons:

| protocol | candidates | role |
| --- | ---: | --- |
| Week7.7 compact replay | 6 candidates | small/medium controlled evidence |
| Week8 100neg | 101 candidates | large-scale robustness gate |

The same-candidate protocol remains the bridge, but candidate count and domain
N must stay visible in tables.

## Paper Role

Use this milestone to show robustness and cross-domain structure:

- tighter estimates on larger domains;
- harder candidate sets;
- external-method complementarity under 100 negatives;
- Beauty as aligned supplementary evidence.

## Related Files

- `CODEX_HANDOFF_WEEK8_2026-05-06.md`
- `WEEK8_LARGE_SCALE_10K_100NEG_PLAN_2026-05-06.md`
- `WEEK8_FUSION_EXTERNAL_ONLY_CONTRIBUTION_UPDATE_2026-05-06.md`
- `WEEK8_OURS_EXTERNAL_COMBO_AND_EXTERNAL_ONLY_PLAN_2026-05-06.md`
- `scripts/run_week8_large_scale_10k_100neg.sh`
- `main_build_large_scale_same_candidate_runtime.py`
- `main_run_week8_external_paired_stat_tests.py`

