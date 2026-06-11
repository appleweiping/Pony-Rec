# Four-Domain C-CRP Component Ablation Diagnostic

- OK: `True`
- Table eligibility: `supplementary_diagnostic_only`
- Delta convention: `removal_minus_full`
- Tie epsilon: `1e-12`

Positive deltas mean the component removal was nonworse or better than full C-CRP.

| ablation | metric | mean delta | nonworse domains | classification |
| --- | --- | ---: | ---: | --- |
| without_boundary_uncertainty | NDCG@10 | 0 | 4 | removal_nonworse_in_3plus_domains_harmful_or_redundant |
| without_calibration_gap | NDCG@10 | -0.000784954060099 | 2 | mixed_diagnostic |
| without_evidence_support | NDCG@10 | -0.000128057425853 | 1 | directionally_supportive_not_significant |
| without_counterevidence | NDCG@10 | 0.00168456132483 | 4 | removal_nonworse_in_3plus_domains_harmful_or_redundant |
| without_risk_penalty | NDCG@10 | 0.000824890949846 | 4 | removal_nonworse_in_3plus_domains_harmful_or_redundant |
