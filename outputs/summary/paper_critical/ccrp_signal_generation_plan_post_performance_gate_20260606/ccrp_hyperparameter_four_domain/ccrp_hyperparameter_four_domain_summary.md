# Four-Domain C-CRP Hyperparameter Sensitivity

- OK: `True`
- Paper claim ready: `True`
- Table eligibility: `supplementary_hyperparameter_stability_only`
- Metric: `NDCG@10`
- Controls: `eta, weight_grid_label`

| control | stable domains | max relative drop | worst test rank | classification |
| --- | ---: | ---: | ---: | --- |
| eta | 4/4 | 1.24629498263e-05 | 2 | stable_with_domain_specific_best_values |
| weight_grid_label | 4/4 | 0.000297434403908 | 3 | stable_with_domain_specific_best_values |
