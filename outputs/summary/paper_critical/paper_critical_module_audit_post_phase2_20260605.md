# Paper-Critical Module Audit

- Generated UTC: `2026-06-05T14:18:52+00:00`
- Paper ready: `False`
- Signal rows available: `False`
- Framework overview scaffold ready: `True`
- Component inventory ready: `True`
- Guarded plan ready: `True`

## Module Status

- `observation_motivation`: `blocked_missing_signal_rows`; paper claim ready = `False`
  blockers: missing_full_scale_uncertainty_or_recomputable_signal_rows
- `component_ablation`: `blocked_missing_signal_rows`; paper claim ready = `False`
  blockers: missing_full_scale_uncertainty_or_recomputable_signal_rows
- `hyperparameter_analysis`: `blocked_missing_signal_rows`; paper claim ready = `False`
  blockers: missing_full_scale_uncertainty_or_recomputable_signal_rows
- `framework_overview`: `draft_scaffold_ready`; paper claim ready = `False`
  blockers: final_paper_layout_and_reviewer_polish

## Next Action

With the official-baseline gate complete, do not start paper-critical C-CRP modules until full-scale valid/test uncertainty signal rows are located or regenerated under the same-candidate protocol. Then run observation, ablation, and hyperparameter gates with validation-only selection and exact score-coverage audits.
