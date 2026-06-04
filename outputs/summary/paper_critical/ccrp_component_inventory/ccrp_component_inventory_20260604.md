# C-CRP Component Inventory

- Status: `paper_critical_ccrp_component_inventory`
- Paper-claim ready: `False`
- Component count: `12`
- Blocked by: `missing_full_scale_uncertainty_or_recomputable_signal_rows`
- Code risk formula: `base_score * ((1 - uncertainty) ** eta)`
- Figure formula aligned: `True`

## Components

| Component | Kind | Selector handle | Execution status |
| --- | --- | --- | --- |
| `calibrated_posterior_base` | base_signal | `score_mode=confidence_only or score_mode=full` | requires_full_scale_signal_rows |
| `score_mode_family` | mode_comparison | `score_modes=confidence_only,evidence_only,confidence_plus_evidence,full` | requires_full_scale_signal_rows |
| `boundary_uncertainty` | leave_one_component_out | `ablation=without_boundary_uncertainty` | requires_full_scale_signal_rows |
| `calibration_gap` | leave_one_component_out | `ablation=without_calibration_gap` | requires_full_scale_signal_rows |
| `evidence_support_insufficiency` | leave_one_component_out | `ablation=without_evidence_support` | requires_full_scale_signal_rows |
| `counterevidence` | leave_one_component_out | `ablation=without_counterevidence` | requires_full_scale_signal_rows |
| `risk_penalty` | leave_one_component_out | `ablation=without_risk_penalty` | requires_full_scale_signal_rows |
| `eta_risk_exponent` | hyperparameter | `--etas` | requires_full_scale_signal_rows |
| `confidence_weight` | hyperparameter | `--confidence_weights` | requires_full_scale_signal_rows |
| `uncertainty_weight_triple` | hyperparameter | `--weight_grid` | requires_full_scale_signal_rows |
| `raw_vs_calibrated_posterior` | conceptual_not_currently_executable | `no_current_cli_handle` | requires_full_scale_signal_rows |
| `temperature_prompt_variants` | conceptual_or_separate_runner | `not_part_of_main_select_ccrp_variant_on_valid` | requires_full_scale_signal_rows |

## Overclaim Risks

- missing_full_scale_uncertainty_or_recomputable_signal_rows
- Do not report component ablations from score-only formal C-CRP scores.
- Do not claim counterevidence contribution unless source rows contain a non-empty counterevidence column.
- Do not claim raw-vs-calibrated posterior or temperature/prompt variants as completed C-CRP LOO ablations without new audited handles.
