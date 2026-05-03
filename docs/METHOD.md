# Method

## CARE rerank (pilot)

**CARE** (this repository’s rerank pilot) is framed as **calibrated expected utility under uncertainty and echo-risk constraints**: listwise **base utility** from the model ordering, plus a **reliability** term, minus **uncertainty risk** and **conditioned echo risk**, plus a **guarded tail-recovery** bonus. It is **not** “confidence plus popularity plus uncertainty” as independent knobs stacked without interaction.

Design principles (see `docs/PILOT_CARE_RERANK.md` and `configs/methods/care_rerank_pilot.yaml`):

- **Popularity** is **not** blindly penalized. Echo risk scales with **head / mid / tail** buckets only when **stress** signals (e.g. high-confidence–wrong pattern, miscalibration hints from diagnostics) justify treating a **head** top pick as risky—not because the item is popular.
- **Tail** items are **not** blindly promoted. The **tail recovery** term applies only under a **safe** global confidence band and rank proximity; otherwise tail mass is unchanged.
- **Confidence** is **not** blindly trusted. Verbalized confidence is **decayed by rank** when per-item probabilities are missing; **calibrated** confidence is preferred when available; **uncertainty features** inflate risk when confidence is missing or inconsistent.
- **`popularity_penalty_only`** is a **deliberate ablation** (blind head-heavy penalty), **not** the CARE method—it exists to contrast with **conditioned** echo risk in `care_full`.

Implemented entrypoint: `python -m src.cli.run_care_rerank_pilot`. Audit tables and manifest checks: `python -m src.cli.summarize_care_rerank_pilot --output_root outputs/pilots/care_rerank_deepseek_v4_flash_processed_20u_c19_seed42`.

---

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
