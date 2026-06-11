# Phase 2.5 Hyperparameter Analysis Design Review Packet

Generated: 2026-06-11

Status: design review required before execution. GPT-5.5 xhigh gave a
proceeding design score of 8.24/10, while the engineering reviewer blocked at
7.5/10 until code/audit support existed. The code/audit support has now been
implemented and tested locally; do not launch the hyperparameter module until a
post-hardening design/code re-review clears the gate.

## Current State

- Official baseline and C-CRP comparison evidence is complete for the current
  same-candidate claim.
- Phase 2.5 component ablation is closed as supplementary diagnostic evidence.
- Phase 2.5 observation/motivation is closed as
  `motivation_only_not_main_table_sota` evidence.
- Remaining paper-critical experiment module before final claim/overclaim
  review: real C-CRP hyperparameter curves.
- Server preflight on 2026-06-11 found no matching Pony/C-CRP/baseline Python
  process, GPU `0%`, `15 MiB / 49140 MiB`, and about `18G` free disk.

## Existing Evidence and Gaps

For each of `sports`, `toys`, `home`, and `tools`, the server has:

- valid signal rows:
  `outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/ccrp_signal_rows_<domain>/valid/valid_ccrp_signal_rows.csv`
  with `1,010,001` lines including header.
- test signal rows:
  `outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/ccrp_signal_rows_<domain>/test/test_ccrp_signal_rows.csv`
  with `1,010,001` lines including header.
- valid/test candidate rows:
  `outputs/baselines/external_tasks/<domain>_large10000_100neg_<split>_same_candidate/candidate_items.csv`
  with `1,010,001` lines including header.
- valid/test ranking tasks:
  `outputs/baselines/external_tasks/<domain>_large10000_100neg_<split>_same_candidate/ranking_<split>.jsonl`
  with `10,000` lines.
- a current `valid_ccrp_sweep.csv` in `ccrp_ablation_<domain>/`, but it has
  only three data rows: `eta in {0.5,1.0,2.0}` under
  `score_mode=full`, `ablation=full`, `confidence_weight=0.7`, and
  `weight_grid_label=0.5,0.3,0.2`.

Current gaps:

- no `test_ccrp_sweep.csv` exists;
- no full weight-grid curve exists;
- no valid+test stability package exists;
- `confidence_weight` is not a main-method hyperparameter under
  `score_mode=full`; in `src/shadow/ccrp.py` it only affects
  `score_mode=confidence_plus_evidence`;
- the old guarded-plan template used to mention `--score_modes` and
  `--ablations` for `scripts/misc/main_select_ccrp_variant_on_valid.py`, but
  the current selector CLI does not support those flags. This has been fixed in
  `scripts/audit/main_plan_ccrp_signal_generation.py`; the hyperparameter
  module now uses the saved-signal sweep builder below as its execution entry
  point.

## Post-Review Hardening Implemented

- Added `scripts/analysis/main_build_ccrp_hyperparameter_sweep.py` to build
  valid/test sweep inputs from saved signal rows only. It writes metrics and
  provenance but does not retain per-grid scores, prediction JSONL,
  checkpoints, or temp scored rows.
- Updated `scripts/analysis/main_plot_ccrp_hyperparameter_sweep.py` to consume
  valid/test sweeps, embed source provenance, and filter by the explicit
  `control` column so weight-grid rows cannot contaminate eta curves.
- Hardened `scripts/audit/main_audit_phase2_5_module_package.py` so the
  hyperparameter module requires `test_not_used_for_selection=true`, source
  row-count provenance, cleanup status, coverage `1.0`, audit and degeneracy
  flags, and `eta`/`weight_grid_label` main controls. It rejects
  `confidence_weight` as a full-mode main curve.
- Updated the guarded plan so it calls
  `scripts/analysis/main_build_ccrp_hyperparameter_sweep.py` before plotting
  and removed stale selector flags.
- Focused local checks passed on 2026-06-11:
  `python -m pytest tests\test_audit_paper_critical_modules.py tests\test_build_ccrp_hyperparameter_sweep.py tests\test_ccrp_hyperparameter_sweep_plot.py tests\test_audit_phase2_5_module_package.py tests\test_plan_ccrp_signal_generation.py -q`
  -> `58 passed`.

## Proposed Evidence Design

Build a server-side sweep generator, or safely extend an existing one, that
evaluates a pre-registered grid from saved signal rows only. No LLM re-query is
allowed. The generator must write, per domain:

- `valid_ccrp_hyperparameter_sweep.csv`
- `test_ccrp_hyperparameter_sweep.csv`
- `ccrp_hyperparameter_sweep_provenance.json`
- command/log/config snippets suitable for the Phase 2.5 module package audit

Rows must include:

- `domain`, `split`, `control`, `control_value`, `score_mode`, `ablation`,
  `eta`, `confidence_weight`, `weight_boundary`, `weight_calibration_gap`,
  `weight_evidence`, `weight_grid_label`;
- full metrics: `MRR`, `HR@5`, `HR@10`, `HR@20`, `NDCG@5`, `NDCG@10`,
  `NDCG@20`;
- coverage/audit fields: `candidate_key_count`, `score_key_count`,
  `score_coverage_rate`, `missing_score_keys`, `extra_score_keys`,
  `duplicate_score_keys`, `invalid_scores`, `blank_score_keys`, `audit_ok`,
  `degeneracy_audit_ok`, `score_degeneracy_event_count`,
  `constant_score_event_count`, `tie_pair_rate`;
- `expected_events=10000`, `expected_candidates_per_event=101`,
  `tie_break_seed=20260607`.

The test sweep is reporting-only. It must not be used to choose the main method
or change the already reported C-CRP configuration.

## Controls

### Main Controls

These controls are paper-facing stability curves for the actual main C-CRP
method.

1. `eta` risk exponent:
   - fixed: `score_mode=full`, `ablation=full`,
     `weight_grid_label=0.5,0.3,0.2`, `confidence_weight=0.7`;
   - grid: `{0, 0.25, 0.5, 1, 2, 4}`;
   - `eta=0` is the no-risk-penalty diagnostic endpoint;
   - `eta=1` is the preregistered main setting.

2. `weight_grid_label` for uncertainty components:
   - fixed: `score_mode=full`, `ablation=full`, `eta=1.0`,
     `confidence_weight=0.7`;
   - grid:
     - `0.5,0.3,0.2` current setting;
     - `0.7,0.2,0.1` boundary-heavy;
     - `0.4,0.4,0.2` calibration-heavy;
     - `0.4,0.2,0.4` evidence-heavy;
     - `0.33,0.33,0.34` balanced.

### Diagnostic Control

`confidence_weight` is diagnostic only, because it is a no-op for the main
`score_mode=full`. If included, evaluate it under
`score_mode=confidence_plus_evidence`, `ablation=full`, `eta=1.0`,
`weight_grid_label=0.5,0.3,0.2`, with grid `{0.1,0.3,0.5,0.7,0.9}`.
It must not be used to claim stability of the main C-CRP method.

## Selection Discipline

- Main C-CRP remains the preregistered configuration already used in the
  comparison and component modules.
- Valid curves may describe selection sensitivity.
- Test curves may describe stability only.
- Provenance must include `test_not_used_for_selection=true`.
- If the valid-best value is not within 5% relative drop of the test-best value
  for a main control, the module must downgrade to diagnostic-only wording for
  that control.
- If a control is flat or nearly inert, report it as weak/inert rather than
  pretending it is a necessary tuning knob.

## Execution Policy

- Run one domain at a time on `pony-rec-gpu`.
- Use `/home/ajifang/miniconda3/envs/qwen_vllm/bin/python`.
- Do not keep bulky per-grid `scores.csv`, prediction JSONL, checkpoints, or
  temporary scored rows after each row's metrics/audit/provenance are captured.
- Keep light local evidence only: sweep CSVs, provenance, plots, configs,
  commands, logs, package audits, local/server manifest comparison, and any
  small summary tables.
- Do not delete source signal rows, candidate/ranking files, canonical configs,
  final C-CRP selected scores, package audits, or comparison tables.

## Required Gates

Per domain:

- `valid_ccrp_hyperparameter_sweep.csv` and `test_ccrp_hyperparameter_sweep.csv`
  exist and have identical schema.
- Main controls have valid and test curves with at least 5 plotted values for
  `weight_grid_label` and 6 plotted values for `eta`.
- Diagnostic `confidence_weight`, if present, is labeled diagnostic.
- Every row has `candidate_key_count=1,010,000`,
  `score_coverage_rate=1.0`, zero missing/extra/duplicate/invalid score keys,
  finite full metrics, and `audit_ok=true`.
- Package audit passes with `ok=true` and `paper_claim_ready=true` for main
  controls only; diagnostic controls must not be required for paper readiness.

Cross-domain:

- Aggregate only after all four per-domain packages pass.
- Summarize valid-best/test-best agreement and relative drop for each main
  control across the four domains.
- Paper wording is stability/sensitivity only, not a new SOTA result and not a
  test-selected hyperparameter claim.

## Reviewer Questions

1. Are the main controls (`eta`, `weight_grid_label`) sufficient and correctly
   separated from diagnostic `confidence_weight`?
2. Is the proposed grid broad enough for a top-conference stability figure
   without wasting GPU/disk?
3. Is a server-side saved-signal sweep generator the right execution route, or
   should the selector be extended?
4. Package audit has been changed so `confidence_weight` is no longer a
   required main-control gate; it is diagnostic-only.
5. Does the plan reach the project's required >=8/10 design gate?
