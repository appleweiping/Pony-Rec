# Paper-Critical Module Audit

- Generated UTC: `2026-06-11T17:00:34+00:00`
- Paper ready: `False`
- Signal rows available: `False`
- Framework overview scaffold ready: `True`
- Component inventory ready: `True`
- Observation execution support ready: `True`
- Component-ablation execution support ready: `True`
- Hyperparameter execution support ready: `True`
- Guarded plan ready: `True`
- Four-domain evidence consistent: `True`
- Phase 2.5 storage launch allowed: `True`
- Storage free bytes: `25656160256`
- Storage deficit bytes: `0`
- Storage safe-now recoverable bytes: `0`
- Storage approval decision required: `True`
- Storage cleanup decision required: `False`
- Storage recommended candidate: `/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`
- Candidate would clear minimum gate: `True`

## Module Status

- `observation_motivation`: `blocked_missing_signal_rows`; paper claim ready = `False`
  blockers: missing_full_scale_uncertainty_or_recomputable_signal_rows
- `component_ablation`: `blocked_missing_signal_rows`; paper claim ready = `False`
  blockers: missing_full_scale_uncertainty_or_recomputable_signal_rows
- `hyperparameter_analysis`: `blocked_missing_signal_rows`; paper claim ready = `False`
  blockers: missing_full_scale_uncertainty_or_recomputable_signal_rows
- `framework_overview`: `review_ready`; paper claim ready = `True`

## Next Action

With the official-baseline evidence package consistent, do not start paper-critical C-CRP modules until full-scale valid/test uncertainty signal rows are located or regenerated under the same-candidate protocol and the Phase 2.5 storage gate allows launch. Then run observation, ablation, and hyperparameter gates with validation-only selection and exact score-coverage audits.
