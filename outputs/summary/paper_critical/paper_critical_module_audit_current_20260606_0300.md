# Paper-Critical Module Audit

- Generated UTC: `2026-06-05T18:59:47+00:00`
- Paper ready: `False`
- Signal rows available: `False`
- Framework overview scaffold ready: `True`
- Component inventory ready: `True`
- Guarded plan ready: `True`
- Four-domain evidence consistent: `True`
- Phase 2.5 storage launch allowed: `False`
- Storage free bytes: `12406411264`
- Storage deficit bytes: `3699716096`

## Module Status

- `observation_motivation`: `blocked_missing_signal_rows`; paper claim ready = `False`
  blockers: missing_full_scale_uncertainty_or_recomputable_signal_rows, server_disk_below_phase2_5_floor, phase2_5_experiment_launch_not_allowed
- `component_ablation`: `blocked_missing_signal_rows`; paper claim ready = `False`
  blockers: missing_full_scale_uncertainty_or_recomputable_signal_rows, server_disk_below_phase2_5_floor, phase2_5_experiment_launch_not_allowed
- `hyperparameter_analysis`: `blocked_missing_signal_rows`; paper claim ready = `False`
  blockers: missing_full_scale_uncertainty_or_recomputable_signal_rows, server_disk_below_phase2_5_floor, phase2_5_experiment_launch_not_allowed
- `framework_overview`: `review_ready`; paper claim ready = `True`

## Next Action

With the official-baseline evidence package consistent, do not start paper-critical C-CRP modules until full-scale valid/test uncertainty signal rows are located or regenerated under the same-candidate protocol and the Phase 2.5 storage gate allows launch. Then run observation, ablation, and hyperparameter gates with validation-only selection and exact score-coverage audits.
