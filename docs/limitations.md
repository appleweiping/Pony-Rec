# Limitations

The current main claim is intentionally bounded.

## Candidate ranking

Unless `candidate_protocol_audit.csv` reports full-catalog availability, results
are controlled sampled-candidate ranking results. They should not be described
as full-catalog recommender SOTA.

## Baselines

Internal SRPD and shadow variants are ablations. External recommender baselines
and same-schema LLM direct ranking remain necessary for strong paper claims.
Proxy related-work rows are for positioning only.

## Calibration

Calibration can improve reliability metrics while leaving ranking utility
unchanged. Calibrator selection must use validation only. Fallback internal
splits are not main-table evidence.

## Statistical power

Small differences in NDCG or MRR require paired confidence intervals and
multiple-comparison correction. Without this, a method can only be called
observed best.

## Generative recommendation

Generated titles are not primary evidence unless catalog grounding,
hallucination, unsupported confident generation, and semantic audit are
completed.

## Compute and reproducibility

Model backend versions, seeds, data hashes, and output manifests must be
recorded for any result promoted from `runnable_not_complete` to
`completed_result`.
