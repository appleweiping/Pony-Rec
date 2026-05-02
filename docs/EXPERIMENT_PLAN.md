# Experiment Plan

Primary domains: Beauty, Electronics, Books, Movie.

Main experiments:

- DeepSeek zero-shot listwise and pointwise recommendation.
- Verbalized, self-consistency, perturbation, and hybrid uncertainty.
- Validation-only calibration with isotonic and Platt scaling.
- Uncertainty-aware reranking, abstention, list truncation, and exploration-aware reranking.
- Popularity-confidence and echo-chamber diagnostics.
- Standard sanity and RecBole baselines.
- Server-side LoRA ablations: base, uncertainty-pruned, uncertainty-weighted, and curriculum.

Robustness:

- Candidate size: 19, 99, 199, 499.
- K-core: 3, 5, 10.
- Prompt sensitivity and temperature sensitivity.
- History-length and domain transfer analyses.
