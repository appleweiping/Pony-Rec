# Paper-Critical Experiment Plan (2026-06-03)

This plan is a paper-readiness gate, not optional polish. It adds the method
story evidence that must exist before final writing or GPT-5.5/Codex xhigh
review.

## Current Constraint

Do not start these runs while the active Home `rlmrec_graphcl` official row is
running. The current server row has runner PID `3178395`, adapter PID
`3178403`, and log `baselines_new_domains_home_rlmrec_20260603_2028.log`.
Queue the work only after that row completes and passes evidence gates, or
after an audited failure/recovery decision frees the GPU.

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
- The figure must be backed by tabular data and not by manual screenshot-only
  inspection.

Implementation anchor:

- `scripts/analysis/main_build_uncertainty_observation_study.py`

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
  --min_join_rate 0.999
```

Do not pass final C-CRP `scores.csv` if it only contains
`source_event_id,user_id,item_id,score`; the script intentionally rejects
score-only files without an uncertainty column. If necessary, regenerate the
C-CRP signal rows from saved non-test-selected inputs without re-querying the
LLM, then record that command and sha256 in the provenance.

## Module B: C-CRP Component Ablation

Goal: show which C-CRP components matter and honestly identify weak components.

Implementation anchors:

- `src/shadow/ccrp.py`
- `scripts/misc/main_select_ccrp_variant_on_valid.py`
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
```

The `<VALID_CCRP_SIGNAL_JSONL_OR_CSV>` and `<TEST_CCRP_SIGNAL_JSONL_OR_CSV>`
paths must be resolved before launch. If C-CRP v3 kept only final `scores.csv`
and not the signal rows with uncertainty fields, regenerate the scored rows
from saved signal inputs without re-querying the LLM, and record the command.

Required outputs:

- `valid_ccrp_sweep.csv`
- `selected_valid_config.json`
- `selected_test_metrics.csv`
- `ccrp_internal_provenance.json`
- imported same-candidate `tables/`
- ablation summary table and a compact plot if useful

Numeric gates:

- Exact score coverage must be `1.0`.
- No finite-score or degeneracy audit failure.
- Test reporting must use configs selected on validation or pre-fixed before
  test.
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

Numeric gates:

- Curves must report validation and test separately.
- A single selected setting is not enough; at least three values per plotted
  control are required.
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

Output:

- editable source (`.pptx`, `.drawio`, `.fig`, or equivalent);
- export as `.pdf` and `.png`;
- short figure caption draft.

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

Verdict: PROCEED after current Home RLMRec row is resolved.

Evidence quality: 8/10. The modules target reviewer-critical motivation,
component necessity, and hyperparameter stability.

Rigor: 8/10 if validation-only selection, exact same-candidate imports, and
full @5/@10/@20 + MRR reporting are preserved.

Gates: 8/10. Numeric gates are defined for coverage, joins, plotted values, and
readiness.

Feasibility: 7/10. Most work reuses existing C-CRP signal/score infrastructure,
but the signal-row paths must be audited before launch.

Paper potential: 8/10. These modules directly address likely top-conference
objections about motivation, non-stitching novelty, ablation completeness, and
overfitting.

Blocking issue: do not start these server runs until Home RLMRec is complete or
cleanly failed, and do not proceed if the required C-CRP signal rows cannot be
located or regenerated without LLM re-query leakage.
