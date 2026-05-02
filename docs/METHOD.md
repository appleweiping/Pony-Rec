# Method

The codebase supports uncertainty-aware LLM4Rec as a decision pipeline:

1. Generate recommendation predictions with a pointwise/listwise/generative prompt.
2. Parse outputs without silently repairing invalid predictions into valid ones.
3. Estimate uncertainty from verbalized confidence, self-consistency, perturbation consistency, hybrid confidence, logprob/entropy when available, and semantic dispersion when embeddings are available.
4. Fit calibration only on validation predictions.
5. Apply calibrated confidence to test predictions.
6. Use uncertainty for reranking, selective abstention, list truncation, exploration, exposure diagnostics, and LoRA data filtering/weighting.

Implemented command path:

```bash
python -m src.cli.infer --config configs/experiments/smoke_mock.yaml --split test
python -m src.cli.calibrate --valid_path ... --test_path ... --output_path ...
python -m src.cli.rerank --input_path ... --output_path ... --lambda_penalty 0.5
python -m src.cli.evaluate --predictions_path ... --output_dir ...
```
