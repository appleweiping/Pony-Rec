# Paper-Critical Experiment Plan (2026-06-03)

This plan is a paper-readiness gate, not optional polish. It adds the method
story evidence that must exist before final writing or GPT-5.5/Codex xhigh
review.

## Current Constraint

Do not start these runs while the active Tools `promax_profile` official row is
running. Tools `proex_profile` has completed and passed server-final,
large-artifact, local-light sync, and local-light audit gates; the protected
server process is now runner PID `3279573` and adapter PID `3279582`, with PID
files `baselines_new_domains_tools_promax_20260604_164630.runner.pid` and
`baselines_new_domains_tools_promax_20260604_164630.adapter.pid`, and log
`baselines_new_domains_tools_promax_20260604_164630.log`. At the 2026-06-04
17:19 CST monitor checkpoint, exactly one matching ProMax adapter process was
active, Qwen3 `hf_mean_pool` progress was `134256/269711`, GPU use was normal
for the run, `/` had `10,764,537,856` bytes free (`95%` used), and no failure
markers were present. Queue paper-critical server work only after this row
completes and passes evidence gates, or after an audited failure/recovery
decision frees the GPU.

All experiment execution runs on `pony-rec-gpu`. Local work is limited to
planning, code review, documentation, plotting design, and version control.

## Module A: Observation / Motivation Study

Goal: show why task-grounded uncertainty is useful in this framework before
introducing C-CRP as a decision module.

Representative scope:

- Start with Sports and Toys because both have the full 8-official baseline
  block, imported C-CRP evidence, domain gates, comparison tables, and paired
  tests.
- Use 3-4 representative methods rather than every model/domain: C-CRP, the
  best official row in the domain, one strong LLM-rec official baseline, and
  one weaker-but-completed official baseline if it clarifies the phenomenon.
- Add Home/Tools only after their official blocks pass; do not block the first
  motivation figure on unfinished domains.

Phenomena to test:

- Uncertainty bins vs. positive rank / HR@10 / NDCG@10.
- High-uncertainty events where the best single baseline fails but risk-aware
  C-CRP improves or avoids overconfident errors.
- Disagreement/instability between representative official methods as a proxy
  for hard recommendation events.
- Long-tail/popularity split behavior when candidate metadata is available.

Required outputs:

- `observation_event_bins.csv`: event-level uncertainty/disagreement bins with
  full row counts.
- `observation_summary.csv/json`: HR@5/@10/@20, NDCG@5/@10/@20, MRR by bin and
  method.
- One paper-ready figure, preferably `fig_uncertainty_motivation.pdf/png`.
- Provenance JSON with domains, methods, input paths, git commit, commands,
  row counts, and status labels.

Numeric gates:

- Each included domain must have exact same-candidate score/import gates for
  the selected methods.
- Event-level joins must retain at least 99.9% of the C-CRP event rows; lower
  retention is a blocker unless explained.
- Ranking eval inputs must have unique event IDs, no eval events outside the
  C-CRP uncertainty input, finite positive integer ranks, and `num_candidates`
  equal to the expected candidate count whenever that column is present.
- The figure must be backed by tabular data and not by manual screenshot-only
  inspection.

Implementation anchor:

- `scripts/analysis/main_build_uncertainty_observation_study.py`
- `scripts/audit/main_discover_ccrp_uncertainty_sources.py` for scanning
  candidate artifact headers before choosing paths for the stricter audit.
- `scripts/audit/main_plan_ccrp_signal_generation.py` for producing a
  guarded, non-executing Sports/Toys command plan after the active official
  baseline row gates pass.
  Generated shell plans start with `exit 2` and retain
  `TODO_*_CCRP_SIGNAL_JSONL_OR_CSV` placeholders until real full-scale signal
  artifacts have passed audit.
- `scripts/analysis/main_export_ccrp_scored_rows_from_signal.py` for rebuilding
  scored rows with `ccrp_uncertainty` columns from saved signal inputs and a
  fixed validation-selected C-CRP config, without any LLM re-query.
- `scripts/audit/main_audit_ccrp_uncertainty_sources.py` for the required
  preflight audit of candidate signal/scored-row files.

Command template:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
python scripts/analysis/main_build_uncertainty_observation_study.py \
  --domain sports \
  --uncertainty_scores_path <CCRP_SIGNAL_OR_SCORE_ROWS_WITH_UNCERTAINTY> \
  --ccrp_eval_path outputs/sports_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate/tables/ranking_eval_records.csv \
  --method_eval llmemb=outputs/sports_large10000_100neg_llmemb_official_qwen3base_same_candidate/tables/ranking_eval_records.csv \
  --method_eval rlmrec=outputs/sports_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate/tables/ranking_eval_records.csv \
  --method_eval proex=outputs/sports_large10000_100neg_proex_profile_official_qwen3base_same_candidate/tables/ranking_eval_records.csv \
  --output_dir outputs/summary/paper_critical/observation_sports \
  --expected_events 10000 \
  --expected_candidates_per_event 101 \
  --min_join_rate 0.999
```

Do not pass final C-CRP `scores.csv` if it only contains
`source_event_id,user_id,item_id,score`; the script intentionally rejects
score-only files without an uncertainty column. If necessary, regenerate the
C-CRP signal rows from saved non-test-selected inputs without re-querying the
LLM, then record that command and sha256 in the provenance.
As of 2026-06-04, the script also writes
`artifact_class=paper_critical_observation_motivation`,
`paper_claim_scope=motivation_only_not_main_table_sota`, full
MRR/HR@5/@10/@20/NDCG@5/@10/@20 metric requirements, and explicit claim limits
into provenance.

Discovery status (2026-06-03): a read-only server inventory found only one
complete C-CRP selector provenance/scored-row package:
`outputs/summary/week8_large10000_100neg_ccrp_formal/beauty/`, where
`ccrp_internal_provenance.json` points to
`outputs/beauty_supplementary_smallerN_100neg_qwen3_shadow_v1/calibrated/{valid,test}_calibrated.jsonl`
and `ccrp_selected_test_scored_rows.csv` includes `ccrp_uncertainty` and its
component columns. This Beauty package is useful for script/interface smoke
checks but is supplementary smaller-N evidence, not the Sports/Toys full-scale
motivation result. The visible Sports/Toys/Home/Tools formal C-CRP directories
currently contain score-only `scores.csv`, `report.json`, and
`user_ranks.jsonl`; their imported same-candidate tables also lack uncertainty
columns. Therefore the next observation/ablation launch must first resolve
full-scale `VALID_CCRP_SIGNAL_JSONL_OR_CSV` and `TEST_CCRP_SIGNAL_JSONL_OR_CSV`
paths, or regenerate scored rows from existing saved signal inputs without any
LLM re-query.

Preflight audit template:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
python scripts/audit/main_discover_ccrp_uncertainty_sources.py \
  --root outputs \
  --domain sports \
  --domain toys \
  --output_json outputs/summary/paper_critical/ccrp_uncertainty_source_discovery.json \
  --output_csv outputs/summary/paper_critical/ccrp_uncertainty_source_discovery.csv

python scripts/audit/main_audit_ccrp_uncertainty_sources.py \
  --candidate_items_path outputs/baselines/external_tasks/sports_large10000_100neg_test_same_candidate/candidate_items.csv \
  --source sports_scores=outputs/sports_large10000_100neg_ccrp_v3/scores.csv \
  --source <LABEL>=<CANDIDATE_SIGNAL_OR_SCORED_ROWS_PATH> \
  --expected_events 10000 \
  --expected_candidates_per_event 101 \
  --output_json outputs/summary/paper_critical/ccrp_uncertainty_source_audit_sports.json \
  --output_csv outputs/summary/paper_critical/ccrp_uncertainty_source_audit_sports.csv
```

Proceed only when the target full-scale artifact is classified as
`paper_ready_uncertainty_rows` or, for regeneration, `recomputable_signal_rows`
with exact or near-exact candidate-key coverage. `score_only_not_uncertainty`
is a blocker for observation/motivation evidence.

If a full-scale `recomputable_signal_rows` artifact is found, rebuild the
paper-critical scored rows without querying the LLM:

```bash
python scripts/analysis/main_export_ccrp_scored_rows_from_signal.py \
  --domain sports \
  --signal_path <TEST_CCRP_SIGNAL_JSONL_OR_CSV> \
  --candidate_items_path outputs/baselines/external_tasks/sports_large10000_100neg_test_same_candidate/candidate_items.csv \
  --selected_config_json <SELECTED_VALID_CONFIG_OR_PROVENANCE_JSON> \
  --output_dir outputs/summary/paper_critical/ccrp_scored_rows_sports \
  --expected_rows 1010000
```

The resulting `ccrp_scored_rows.csv` is the preferred
`--uncertainty_scores_path` input for the observation script, and
`ccrp_scores.csv` must pass exact same-candidate coverage before any imported
ranking/ablation claim is made.
The source signal artifact may use either `candidate_item_id` or `item_id` for
candidate identity; the audit, rebuild helper, and validation selector now
share that alias policy. A coverage failure after a passing
`recomputable_signal_rows` audit should therefore be treated as a real key
mismatch, not a known schema-alias limitation.

Validated preflight result (updated 2026-06-04): the remote discovery helper
is now safe for absolute server roots because it matches domain/name tokens
against paths relative to the scan root, not against `/home/...`. Fixed-filter
header discovery plus a broader token sweep over Sports, Toys, Home, and Tools
found only the four full-scale C-CRP formal `scores.csv` files and no
paper-ready or recomputable uncertainty/signal rows. A follow-up project-root
scan with `--root .` and broad C-CRP/shadow/signal/calibration/bridge/rows
tokens produced the same four score-only candidates and no additional signal
rows outside `outputs/`. Full audits against each domain's test
`candidate_items.csv` reported exact candidate-key coverage
(`1,010,000/1,010,000`, 10,000 events) for all four files, but classified every
file as `score_only_not_uncertainty` with failure
`missing_uncertainty_column`. This confirms that the formal score files are
usable for ranking import/audit but not for uncertainty-bin motivation evidence.
Local evidence copies are under `outputs/summary/paper_critical/` with names
`ccrp_uncertainty_source_discovery_fullscale*_fixed_filter_20260604_*` and
`ccrp_uncertainty_source_discovery_projectroot_broad_fixed_filter_20260604_0520.*`,
plus
`ccrp_uncertainty_source_audit_{sports,toys,home,tools}_fixed_filter_20260604_0502.*`.

Static trace result (2026-06-04): `scripts/audit/main_trace_ccrp_formal_signal_path.py`
confirms that `experiments/rsc/run_ccrp_v3_domain.py` only requests
`relevance_probability` and writes `scores.csv`, `report.json`, and
`user_ranks.jsonl`; it does not preserve evidence/counterevidence fields,
`ccrp_uncertainty`, selected scored rows, or internal provenance. The selector
route in `scripts/misc/main_select_ccrp_variant_on_valid.py` can write
`ccrp_selected_test_scored_rows.csv` and `ccrp_internal_provenance.json`, but
only when given real valid/test signal paths. Do not attempt to infer
paper-ready uncertainty rows from formal `scores.csv` alone. Evidence:
`outputs/summary/paper_critical/ccrp_formal_signal_path_trace_20260604_0535.json`.

Guarded next-step plan (2026-06-04): local plan artifact
`outputs/summary/paper_critical/ccrp_signal_generation_plan/ccrp_signal_generation_plan_20260604.*`
records the Sports/Toys discovery, per-domain source audit, C-CRP validation
selection, component-ablation summary, observation-study, hyperparameter-plot,
and module-package-audit command templates. It is intentionally not executable
as generated; the shell exits before any command and requires replacing
`TODO_VALID_*`/`TODO_TEST_*` signal paths with artifacts classified as
`recomputable_signal_rows` or `paper_ready_uncertainty_rows`.

## Module B: C-CRP Component Ablation

Goal: show which C-CRP components matter and honestly identify weak components.

Implementation anchors:

- `src/shadow/ccrp.py`
- `scripts/misc/main_select_ccrp_variant_on_valid.py`
- `scripts/analysis/main_build_ccrp_component_ablation_summary.py`
- `configs/week8_large_scale_future_framework.yaml`

Components to ablate or compare:

- score mode: `confidence_only`, `evidence_only`,
  `confidence_plus_evidence`, `full`;
- boundary uncertainty: `without_boundary_uncertainty`;
- calibration gap: `without_calibration_gap`;
- evidence support/insufficiency: `without_evidence_support`;
- counterevidence: `without_counterevidence`;
- risk penalty: `without_risk_penalty`;
- eta: at least `0.5,1.0,2.0`;
- confidence weight for `confidence_plus_evidence`: at least `0.5,0.7,0.9`;
- C-CRP weight triples: default `0.5,0.3,0.2` plus a small validation-only
  sensitivity grid.

Inventory status (2026-06-04): local helper
`scripts/audit/main_build_ccrp_component_inventory.py` writes
`outputs/summary/paper_critical/ccrp_component_inventory/ccrp_component_inventory_20260604.{json,md}`.
It covers the current executable selector handles plus two conceptual risks
that are not currently executable LOO handles: raw-vs-calibrated posterior and
temperature/prompt variants. Do not claim those conceptual items as completed
ablations unless new audited handles are implemented. The inventory also
records that the current C-CRP risk formula is multiplicative:
`base_score * ((1 - uncertainty) ** eta)`.

Protocol:

- Use validation-only selection for any selected C-CRP configuration.
- Export exact test scores with `source_event_id,user_id,item_id,score`.
- Import through the shared same-candidate score gate.
- Report full metrics @5/@10/@20 + MRR for every ablation row.
- If an ablation matches or beats full C-CRP, do not hide it. Label the
  removed component as weak, redundant, or needing redesign, and revise the
  method wording accordingly.

Command template:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
python scripts/misc/main_select_ccrp_variant_on_valid.py \
  --domain sports \
  --valid_ranking_path outputs/baselines/external_tasks/sports_large10000_100neg_valid_same_candidate/ranking_task.jsonl \
  --test_ranking_path outputs/baselines/external_tasks/sports_large10000_100neg_test_same_candidate/ranking_task.jsonl \
  --valid_candidate_items_path outputs/baselines/external_tasks/sports_large10000_100neg_valid_same_candidate/candidate_items.csv \
  --test_candidate_items_path outputs/baselines/external_tasks/sports_large10000_100neg_test_same_candidate/candidate_items.csv \
  --valid_signal_path <VALID_CCRP_SIGNAL_JSONL_OR_CSV> \
  --test_signal_path <TEST_CCRP_SIGNAL_JSONL_OR_CSV> \
  --output_dir outputs/summary/paper_critical/ccrp_ablation_sports \
  --score_modes confidence_only,evidence_only,confidence_plus_evidence,full \
  --ablations full,without_boundary_uncertainty,without_calibration_gap,without_evidence_support,without_counterevidence,without_risk_penalty \
  --etas 0.5,1.0,2.0 \
  --confidence_weights 0.5,0.7,0.9 \
  --weight_grid "0.5,0.3,0.2;0.7,0.2,0.1;0.4,0.4,0.2;0.4,0.2,0.4" \
  --selection_metric NDCG@10 \
  --import_scores

python scripts/analysis/main_build_ccrp_component_ablation_summary.py \
  --selector_dir outputs/summary/paper_critical/ccrp_ablation_sports \
  --output_dir outputs/summary/paper_critical/ccrp_ablation_sports \
  --domain sports \
  --expected_events 10000 \
  --expected_candidates_per_event 101 \
  --metric NDCG@10 \
  --ablations full,without_boundary_uncertainty,without_calibration_gap,without_evidence_support,without_counterevidence,without_risk_penalty
```

The `<VALID_CCRP_SIGNAL_JSONL_OR_CSV>` and `<TEST_CCRP_SIGNAL_JSONL_OR_CSV>`
paths must be resolved before launch. If C-CRP v3 kept only final `scores.csv`
and not the signal rows with uncertainty fields, regenerate the scored rows
from saved signal inputs without re-querying the LLM, and record the command.
As of the 2026-06-03 inventory, this path-resolution step remains open for
Sports/Toys/Home/Tools; only the Beauty supplementary smaller-N selector
package has a saved signal path and scored rows with C-CRP uncertainty columns.

Required outputs:

- `valid_ccrp_sweep.csv`
- `selected_valid_config.json`
- `selected_test_metrics.csv`
- `ccrp_internal_provenance.json`
- imported same-candidate `tables/`
- `component_ablation_summary.csv`
- `component_ablation_summary.json`
- `component_ablation_provenance.json`
- compact component-ablation PNG/PDF plots

Numeric gates:

- Exact score coverage must be `1.0`.
- No finite-score or degeneracy audit failure.
- Test reporting must use configs selected on validation or pre-fixed before
  test.
- The dedicated component builder must see `selected_on=valid`, a valid-split
  `selected_valid_config.json`, `score_mode=full` by default, and every
  expected ablation in `valid_ccrp_sweep.csv`.
- At least Sports and Toys must pass before the ablation story is paper-ready;
  broader-domain ablations can follow after Home/Tools finish.

## Module C: Hyperparameter Analysis

Goal: show C-CRP is not a single-point overfit.

Primary sweeps:

- eta: `0,0.25,0.5,1.0,1.5,2.0,3.0`;
- confidence weight: `0.1,0.3,0.5,0.7,0.9`;
- weight triples: a small simplex-style grid over boundary, calibration gap,
  evidence weights, always normalized;
- uncertainty temperature or calibration-temperature controls when the active
  C-CRP runner/config exposes them, including
  `experiments/rsc/run_ccrp_v3_temperature.py` as an implementation audit
  target before launch;
- optional Shadow gate controls from
  `scripts/experiments/main_run_week8_shadow_v6_gate_sweep.py`:
  gate thresholds, uncertainty thresholds, and anchor conflict penalties.

SRPD-only sweeps:

- Use learning-rate/lambda sweeps only for SRPD rows that are already
  leakage-clean and paper-relevant.
- A safe learning-rate grid is `1e-1,1e-2,1e-3,1e-4,1e-5`, but skip unsafe
  values if the optimizer/config would make them meaningless or unstable.

Required plots:

- `fig_hyper_eta_curve.pdf/png`
- `fig_hyper_confidence_weight_curve.pdf/png`
- `fig_hyper_weight_simplex_or_lines.pdf/png`
- `fig_hyper_temperature_curve.pdf/png` if temperature is an active control
- SRPD learning-rate/lambda curves only if SRPD is used in the paper story.
- `ccrp_hyperparameter_curve_summary.csv` and
  `ccrp_hyperparameter_curve_provenance.json` with sweep paths, hashes, fixed
  controls, min-values gate, status label, and paper-claim scope.

Implementation anchor:

- `scripts/analysis/main_plot_ccrp_hyperparameter_sweep.py`

Command template after a validation sweep exists:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
python scripts/analysis/main_plot_ccrp_hyperparameter_sweep.py \
  --domain sports \
  --sweep_csv outputs/summary/paper_critical/ccrp_ablation_sports/valid_ccrp_sweep.csv \
  --test_sweep_csv outputs/summary/paper_critical/ccrp_ablation_sports/test_ccrp_sweep.csv \
  --output_dir outputs/summary/paper_critical/hyperparameter_sports \
  --metric NDCG@10 \
  --score_mode full \
  --ablation full \
  --eta 1.0 \
  --confidence_weight 0.5 \
  --weight_grid_label "0.5,0.3,0.2" \
  --controls eta,confidence_weight,weight_grid_label \
  --min_values 3
```

The plotting script is an evidence-packaging step, not a model runner. When a
test sweep is available, pass it via `--test_sweep_csv` so validation and test
curves are reported separately. If no test sweep is supplied, the provenance is
marked `valid_only`, which is useful for planning but not sufficient for the
paper stability claim. The script fails by default when any requested curve has
fewer than three values. Use `--allow_incomplete` only for diagnostic output
that will not support a paper stability claim.
As of 2026-06-04, the plotter requires `audit_ok` and
`degeneracy_audit_ok` columns by default; use `--no-require_audit_ok` only for
diagnostic output. Provenance labels valid-only output as
`validation_only_hyperparameter_selection_curve`, labels incomplete or
audit-not-enforced output as diagnostic-only, and only labels audited valid+test
curves as `paper_critical_hyperparameter_curve_ready`.

Numeric gates:

- Curves must report validation and test separately.
- A single selected setting is not enough; at least three values per plotted
  control are required.
- Audit columns must exist and pass for paper-facing curves.
- If the method is only good at one isolated value, downgrade the stability
  claim and report sensitivity.

## Module D: Framework Overview Figure

Goal: provide a clean paper figure that explains the method without implying a
full-catalog or generative-title recommender claim.

Required content:

- same-candidate task construction;
- LLM task-grounded scoring / signal extraction;
- calibration and evidence fields;
- C-CRP uncertainty decomposition: boundary ambiguity, calibration gap,
  evidence insufficiency/counterevidence;
- risk-adjusted candidate ranking;
- shared metric importer and evidence gates.

The risk formula shown in the figure must match `src/shadow/ccrp.py`: current
C-CRP uses multiplicative risk adjustment,
`base_score * ((1 - uncertainty) ** eta)`, not a subtractive
`posterior - eta * uncertainty` formula.

Output:

- editable source (`.pptx`, `.drawio`, `.fig`, or equivalent);
- export as `.pdf` and `.png`;
- short figure caption draft.

Implementation anchor:

- `scripts/analysis/main_build_framework_overview_figure.py`

Command template:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
python scripts/analysis/main_build_framework_overview_figure.py \
  --output_dir outputs/summary/paper_critical/framework_overview
```

The generated `.svg` is the editable source for this figure; `.pdf` and `.png`
are paper/export artifacts. The caption and provenance must keep the claim
boundary explicit: controlled same-candidate ranking, not full-catalog or
generative-title recommendation.

## Evidence Package Standard

For each finished module, create a lightweight package under:

```text
outputs/summary/paper_critical/<module_name>/
```

Include:

- commands and log snippets;
- configs and selected hyperparameters;
- git commit;
- input path manifest with sha256 where feasible;
- row-count and join-count audits;
- metrics CSV/JSON and final tables;
- generated plots;
- provenance notes and status labels;
- local/server manifest comparison.

Before using a finished Phase 2.5 module in paper claims, run the local package
audit:

```powershell
python scripts\audit\main_audit_phase2_5_module_package.py `
  --module observation_motivation `
  --package_dir outputs\summary\paper_critical\<module_name> `
  --output_json outputs\summary\paper_critical\<module_name>\phase2_5_module_package_audit.json `
  --output_md outputs\summary\paper_critical\<module_name>\phase2_5_module_package_audit.md
```

Use `--module component_ablation` for the leave-one-component-out package and
`--module hyperparameter_analysis` for curve packages. The audit is read-only:
it does not launch experiments, copy files, or delete files. It fails closed if
the package lacks status labels, full metrics where required, row counts,
figures, provenance, input hashes, log snippets, or local/server manifest
comparison. It also requires the component-ablation package to include an
explicit `component_ablation_summary.csv` covering every expected ablation, so
a validation-only sweep cannot be relabeled as completed leave-one-component-out
evidence. The component audit also checks selected-valid/test artifacts,
imported same-candidate tables, exact score coverage, and audit/degeneracy
fields. The hyperparameter audit requires the expected controls
(`eta`, `confidence_weight`, and `weight_grid_label` by default) with valid and
test curves, producer audit-summary fields, and package-contained figures.

Exclude by default:

- huge raw `scores.csv`;
- full prediction JSONL unless needed for paired tests and not reconstructible;
- model checkpoints and embeddings unless needed for resume/verification.

## Safe Disk Policy

Before any paper-critical run, check:

```bash
df -h /
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
ps aux | grep python | grep -v grep
```

If disk is tight, audit large files first. Delete only duplicate, garbage, temp,
cache, repeated logs, obsolete archives, or completed intermediate artifacts
whose final server/local gates and manifests have already passed. Do not delete
data splits, source code, canonical configs, important evidence, current
outputs, active/resumable checkpoints, or files from other projects. If unsure,
report a proposed deletion list before acting.

## ARIS Plan Audit

Verdict: PROCEED after the current Tools ProEx row is resolved and the required
full-scale C-CRP signal rows are located or regenerated without LLM re-query
leakage.

Evidence quality: 8/10. The modules target reviewer-critical motivation,
component necessity, and hyperparameter stability.

Rigor: 8/10 if validation-only selection, exact same-candidate imports, and
full @5/@10/@20 + MRR reporting are preserved.

Gates: 8/10. Numeric gates are defined for coverage, joins, plotted values, and
readiness.

Consolidated audit update (2026-06-06 03:48 CST): the paper-critical module
audit now checks component-ablation execution support before returning
`ok=true`. This includes the selector full-metric path, dedicated
`main_build_ccrp_component_ablation_summary.py` builder, module-package audit
requirements, and guarded-plan command templates. Current checkpoint:
`outputs/summary/paper_critical/paper_critical_module_audit_post_component_execution_support_20260606_0348.{json,md,sha256}`.
It reports execution support ready but paper readiness false because full-scale
valid/test uncertainty signal rows remain missing and the storage launch gate
is still closed.

Feasibility: 7/10. Most work reuses existing C-CRP signal/score infrastructure,
but the signal-row paths must be audited before launch.

Paper potential: 8/10. These modules directly address likely top-conference
objections about motivation, non-stitching novelty, ablation completeness, and
overfitting.

Blocking issue: do not start these server runs until the active Tools ProEx row
is complete/gated or cleanly failed, and do not proceed if the required C-CRP
signal rows cannot be located or regenerated without LLM re-query leakage.
