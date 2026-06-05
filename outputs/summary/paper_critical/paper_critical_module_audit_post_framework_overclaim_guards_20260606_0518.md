# Paper-Critical Module Audit

- Generated UTC: `2026-06-05T21:17:55+00:00`
- Paper ready: `False`
- Signal rows available: `False`
- Framework overview scaffold ready: `True`
- Component inventory ready: `True`
- Observation execution support ready: `True`
- Component-ablation execution support ready: `True`
- Hyperparameter execution support ready: `True`
- Guarded plan ready: `True`
- Four-domain evidence consistent: `True`
- Phase 2.5 storage launch allowed: `False`
- Storage free bytes: `12397707264`
- Storage deficit bytes: `3708420096`
- Storage safe-now recoverable bytes: `0`
- Storage approval decision required: `True`
- Storage cleanup decision required: `True`
- Storage recommended candidate: `/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`
- Candidate would clear minimum gate: `True`

## Module Status

- `observation_motivation`: `blocked_missing_signal_rows`; paper claim ready = `False`
  blockers: missing_full_scale_uncertainty_or_recomputable_signal_rows, server_disk_below_phase2_5_floor, phase2_5_experiment_launch_not_allowed
- `component_ablation`: `blocked_missing_signal_rows`; paper claim ready = `False`
  blockers: missing_full_scale_uncertainty_or_recomputable_signal_rows, server_disk_below_phase2_5_floor, phase2_5_experiment_launch_not_allowed
- `hyperparameter_analysis`: `blocked_missing_signal_rows`; paper claim ready = `False`
  blockers: missing_full_scale_uncertainty_or_recomputable_signal_rows, server_disk_below_phase2_5_floor, phase2_5_experiment_launch_not_allowed
- `framework_overview`: `review_ready`; paper claim ready = `True`

## Next Action

Full-scale valid/test uncertainty signal rows remain missing and the Phase 2.5 storage gate is closed. The current storage audit found no safe-now recoverable bytes; the approval-required candidate /home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy would clear the minimum disk gate. Record an explicit archive/retention decision before any delete command, then rerun the storage gate and only then launch guarded signal-row generation.
