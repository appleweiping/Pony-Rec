# Calibration Protocol

Calibration must be validation-to-test only. Test data cannot select the
calibrator method, weights, threshold, or hyperparameters.

## Entry point

```bash
python main_calibrate.py \
  --exp_name beauty_deepseek \
  --valid_path outputs/beauty_deepseek/predictions/valid_raw.jsonl \
  --test_path outputs/beauty_deepseek/predictions/test_raw.jsonl \
  --method isotonic \
  --strict_split_check true \
  --allow_user_overlap false
```

## Split checks

The default behavior is strict:

- `--strict_split_check true`
- `--allow_user_overlap false`
- `--allow_item_overlap true`

User overlap between validation and test raises an error by default. Item
overlap is allowed by default because warm-item recommendation evaluation is
often intended; set `--allow_item_overlap false` for cold-item audits.

## Fallback split

The single-file fallback path is labeled
`fallback_internal_validation_only`. It is useful for debugging but is not main
table eligible.

## Metadata

`calibration_split_metadata.csv` records:

- validation/test sample and user counts
- user overlap count and rate
- item counts and item overlap
- split hash
- source file SHA256 hashes
- requested and effective calibrator method
- non-empty calibration bin counts
- `main_table_eligible`

## Isotonic guard

Isotonic calibration is high variance on small validation sets. If validation
rows are below `--min_isotonic_valid_samples`, the script fits Platt scaling
instead and records both requested and effective methods. This decision uses
validation size only.

## Uncertainty definition

For the old verbalized-confidence line, uncertainty remains
`1 - calibrated_confidence`. For the C-CRP line, uncertainty is not defined as
`1 - p`; see `docs/shadow_method.md`.

## Confidence intervals

The calibration entry point writes bootstrap intervals for ECE and Brier:

```text
outputs/{exp_name}/tables/calibration_metric_ci.csv
```
