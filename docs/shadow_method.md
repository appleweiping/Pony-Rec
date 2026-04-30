# Shadow Method: C-CRP

The main shadow method is C-CRP: Calibrated Candidate Relevance Posterior.

Shadow v2-v6 are design variants and appendix-only unless they are rerun under
the same split, seed, candidate set, and statistical protocol.

## Main method

C-CRP estimates a calibrated candidate relevance posterior and decomposes
uncertainty into three components:

```text
U = alpha * U_boundary
  + beta  * U_calibration_gap
  + gamma * U_evidence
```

Where:

- `U_boundary = 4 * p_cal * (1 - p_cal)`
- `U_calibration_gap = abs(p_raw - p_cal)`
- `U_evidence = 1 - clamp(evidence_support - counterevidence_strength, 0, 1)`

The risk-adjusted ranking score is:

```text
score = p_cal * (1 - U)^eta
```

## Weight rule

`alpha`, `beta`, and `gamma` must be fixed before test or selected on
validation only. The default fixed weights are:

```text
alpha = 0.5
beta  = 0.3
gamma = 0.2
```

## Required ablations

The main method table may include these ablations:

- C-CRP without calibration gap
- C-CRP without evidence support
- C-CRP without counterevidence
- C-CRP without risk penalty

## Entry point

```bash
python main_shadow_ccrp_eval.py \
  --input_path outputs/beauty_shadow/predictions/test_raw.jsonl \
  --output_dir outputs/summary/shadow_ccrp \
  --ablation full \
  --status_label completed_result
```

Prompt-only shadow diagnostics are not ranking main-table evidence. Use
`--prompt_only true` or a non-completed status label when only the signal
parser has been exercised.

## Outputs

- C-CRP scored records
- C-CRP diagnostic summary
- risk-coverage curve
- optional ranked records when pointwise candidate rows are available
