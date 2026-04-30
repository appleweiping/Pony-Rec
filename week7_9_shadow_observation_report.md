# Week7.9 Shadow Observation Report

Date: 2026-04-30

## Current Position

The current shadow line is still in the observation, signal-screening, calibration, and robustness stage. It is not LoRA training yet.

Current pipeline:

```text
fixed Qwen3-8B
-> prompt-only shadow signal
-> pointwise AUROC / ECE / Brier
-> valid-to-test calibration
-> uncertainty-aware rerank
-> noisy_nl10 robustness
-> winner signal selection
```

Later training pipeline:

```text
winner shadow signal
-> real shadow_v6 bridge
-> Signal LoRA
-> Decision / Generative LoRA
```

## Actions Recorded

1. Rebuilt the week7.9 small-prior summary over `shadow_v1` to `shadow_v5` after confirming that the earlier visible summary only covered `shadow_v3` to `shadow_v5`.

2. Confirmed that `beauty/books x shadow_v1-v5` clean core was complete:

```text
ready_core_rows=10/10
```

3. Selected `shadow_v1`, `shadow_v2`, and `shadow_v5` as the top3 candidates for noisy robustness:

```text
shadow_v1: cross-domain stable candidate
shadow_v2: Beauty-oriented candidate
shadow_v5: rerank-oriented / Books candidate
```

4. Diagnosed and fixed two workflow failures:

- generated scripts were cd-ing into the old server repository path.
- `*_noisy_nl10` input files were missing before noisy inference.

5. Re-ran top3 noisy after fixing the script path and materializing missing noisy inputs. The run completed:

```text
ready_core_rows=6/6
```

6. Launched `shadow_v1 full_replay` across Beauty, Books, Electronics, and Movies with clean and noisy branches. This run was still in progress when this report was written.

7. Updated `main_shadow_make_commands.py` so future scripts:

- cd into the current project directory by default.
- support explicit `--project_dir`.
- prepend missing noisy input generation commands when `--include_noisy` is enabled.
- emit `python3.12` commands by default through the manifest-level `python_cmd`.

## Clean + Noisy Top3 Results

| Domain | Variant | Clean AUROC | Noisy AUROC | Clean NDCG@10 | Noisy NDCG@10 | Noisy Drop |
|---|---|---:|---:|---:|---:|---:|
| Beauty | shadow_v1 | 0.583760 | 0.558568 | 0.639430 | 0.627760 | 0.011670 |
| Beauty | shadow_v2 | 0.589210 | 0.562276 | 0.639430 | 0.627760 | 0.011670 |
| Beauty | shadow_v5 | 0.581185 | 0.562645 | 0.639430 | 0.627760 | 0.011670 |
| Books | shadow_v1 | 0.724627 | 0.676185 | 0.646938 | 0.650194 | -0.003256 |
| Books | shadow_v2 | 0.691828 | 0.646643 | 0.646938 | 0.650194 | -0.003256 |
| Books | shadow_v5 | 0.700265 | 0.658214 | 0.651628 | 0.656782 | -0.005154 |

## Interpretation

`shadow_v1` is the current best winner candidate.

Reasons:

- strongest Books clean AUROC and noisy AUROC.
- competitive Beauty behavior.
- best cross-domain balance among the top3.
- cleanest semantic bridge to generated-title verification: calibrated relevance posterior plus evidence support and counterevidence.

`shadow_v2` remains a Beauty-oriented ablation. It is slightly stronger on Beauty AUROC but weaker on Books.

`shadow_v5` remains a rerank-oriented ablation. It gives the best Books rerank metrics but has weaker signal stability than v1.

`shadow_v4` should not be promoted now because Beauty calibration was unstable in the v1-v5 clean summary.

## Scientific Meaning

The current evidence answers:

```text
Which prompt-only task-grounded uncertainty signal is worth training later?
```

It does not yet answer:

```text
Can a fine-tuned local model learn this signal?
Can the signal improve generative title recommendation?
Can v6 convert the signal into a robust accept / revise / fallback controller?
```

Current answer:

```text
shadow_v1 is the first candidate to carry into full replay and later v6 / LoRA work.
```

## Relationship To Light, SRPD, And Generative Title

Project positioning:

```text
light
  old verbalized-confidence observation and failure boundary

SRPD
  evidence that an uncertainty teacher can be distilled into decision behavior

shadow
  next-generation task-grounded uncertainty signal

generative title
  final recommendation task form
```

Candidate ranking is a controlled microscope, not the final deployment form.

Final target direction:

```text
user history
-> generated item title
-> catalog grounding
-> shadow verification
-> accept / revise / fallback
```

The top-conference contribution should be framed as:

```text
uncertainty-calibrated generated-title verification
```

The system should not merely generate a title. It should verify whether the generated title is supported by the user's history, calibrate the posterior support, estimate uncertainty, and then accept, revise, or fallback.

## v6 Status

`shadow_v6` now has the first real bridge implementation for the ranking-side decision layer.

Already present:

- prompt
- schema
- parser fields
- first-pass scoring formula
- command-generation support
- winner-signal bridge builder
- anchor-rank + calibrated shadow signal fusion
- correction gate
- fallback flag
- pair type and pair weight fields

Still missing:

- generated-title verification path.
- accept / revise / fallback controller for generated title outputs.
- chosen/rejected or pair-weight training export for Decision LoRA.
- Decision LoRA loop.

Therefore, v6 should now be treated as:

```text
implemented ranking bridge, not yet generated-title controller
```

The current v6 bridge consumes:

```text
direct candidate ranking anchor
+ shadow_v1 calibrated score
+ shadow_v1 uncertainty
-> correction gate
-> decision_score
-> fallback_flag
-> pair_type / pair_weight
```

The bridge writes reusable rows and reranked predictions through:

```text
python3.12 main_build_shadow_v6_bridge.py
```

## Next Steps

While `shadow_v1 full_replay` is running:

1. Do not start another heavy GPU inference run.
2. Keep the full-replay interpretation criteria fixed:

```text
pointwise AUROC
calibrated ECE / Brier
noisy AUROC drop
rerank NDCG / MRR
coverage / head exposure / long-tail coverage
```

3. If `shadow_v1` remains stable across all four domains, promote it to the official shadow winner.

4. If `shadow_v1` is stable on Beauty/Books but weak on Electronics/Movies, keep it as the positive-domain winner and run `shadow_v5` or v6 bridge as a repair ablation.

5. After full replay, run the v6 bridge on the validated winner signal:

```text
shadow_v1 calibrated score + uncertainty + anchor ranking
-> correction gate
-> accept / fallback / pair weight
```

6. Only after the winner signal and v6 bridge are validated, start Signal LoRA.
