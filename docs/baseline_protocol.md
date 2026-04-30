# Baseline Protocol

The baseline layer separates same-schema evidence from proxy positioning.

## Baseline groups

Main paper comparisons should include these groups when available under the
same schema:

- Non-LLM recommenders: SASRec, BERT4Rec, GRU4Rec, or LightGCN.
- Simple recommendation priors: popularity, recency, history overlap, BM25 or
  title embedding.
- LLM direct ranking: same prompt and candidate set, no uncertainty signal.
- Uncertainty baselines: raw confidence, Platt or isotonic calibrated
  confidence, self-consistency, entropy or logprob when available.

Internal SRPD/shadow variants are ablations, not substitutes for external
baselines.

## Reliability proxy audit

The old "baseline confidence formulation audit" is renamed:

```text
baseline reliability proxy audit
```

Run:

```bash
python main_baseline_reliability_audit.py \
  --config configs/baseline_reliability/week7_9_manifest.yaml \
  --output_path outputs/summary/baseline_reliability_proxy_audit.csv
```

The schema includes:

- `baseline_name`
- `baseline_family`
- `confidence_semantics`
- `calibration_target`
- `is_relevance_calibratable`
- `can_compute_ece`
- `can_run_selective_analysis`
- `risk_of_unfair_comparison`
- `protocol_gap`
- `status_label`

## ECE boundary

ECE and Brier are valid only for relevance-calibratable signals:

- `self_reported_confidence`
- `calibrated_relevance_probability`

Signals such as `exposure_policy_certainty`, candidate order, or pure
popularity are not relevance probabilities. They may appear in exposure or
policy audit tables, but not in the relevance calibration table.

## Related-work reported numbers

Reported numbers from different protocols must not enter the same main ranking
table. They can appear only in a proxy table with `protocol_gap`, for example:

- `different_candidate_space`
- `full_catalog_vs_sampled`
- `different_backbone`
- `no_confidence_output`
- `not_relevance_calibratable`
