# Paper Claims and Experiment Status

This file freezes the paper scope so the project does not drift into a mixed
system log.

## Working title

Actionable Uncertainty for LLM-based Recommendation.

This title replaces the more engineering-sounding "Task-Grounded Uncertainty
for LLM-based Recommendation" as the public-facing project/paper name. The
technical claim remains task-grounded calibrated uncertainty under controlled
same-candidate evaluation; the new title must not be used to broaden the paper
into full-catalog or generic recommender SOTA claims.

## Primary claim

Task-grounded calibrated uncertainty improves controlled candidate ranking and
reranking reliability under same-schema evaluation.

The main paper studies whether uncertainty signals can be made decision-useful
for LLM-based recommendation. The contribution is not a new model backbone and
not a generic recommender system.

## Primary contribution statements

1. Diagnosis: verbalized LLM confidence is informative in recommendation, but
   unreliable under miscalibration, confidence collapse, and domain-dependent
   failure.
2. Method: C-CRP, a task-grounded calibrated candidate relevance posterior with
   boundary uncertainty, calibration gap, and evidence insufficiency.
3. Decision: risk-adjusted candidate ranking/reranking evaluated with utility,
   calibration, coverage, exposure, robustness, and paired statistical tests.

C-CRP is the main internal method line. SRPD is the trainable
framework/ablation line and becomes paper-facing only after leakage-clean
teacher generation, weighted-loss training when claimed, exact same-candidate
score export, and paired-test gates pass.

## Paper-critical readiness modules

Official-baseline completion is necessary but no longer sufficient for paper
readiness. Before final writing or GPT-5.5/Codex xhigh review, the paper must
also include:

1. An observation/motivation study explaining why uncertainty should be used in
   this framework. It should use representative completed baselines and fair
   same-candidate settings, not paid/SOTA general-model exhaustiveness, and it
   must produce a paper-ready figure or table showing the uncertainty
   phenomenon. Script entry:
   `scripts/analysis/main_build_uncertainty_observation_study.py`; it requires
   real event-level uncertainty fields and intentionally rejects score-only
   C-CRP files. Current artifact audit status: the server has a complete
   `ccrp_selected_test_scored_rows.csv`/`ccrp_internal_provenance.json` pair
   only for the older Beauty supplementary smaller-N selector run; the
   Sports/Toys/Home/Tools formal C-CRP outputs currently visible are score-only
   and cannot support the motivation figure until real signal/scored rows are
   located or regenerated without LLM re-query leakage. Fixed-filter discovery
   and full audits on 2026-06-04 confirmed that each visible new-domain formal
   C-CRP `scores.csv` has 1,010,000 rows, 10,000 events, and candidate coverage
   `1.0`, but all four are `score_only_not_uncertainty` because they lack an
   uncertainty column. Preflight entries:
   `scripts/audit/main_discover_ccrp_uncertainty_sources.py`,
   `scripts/audit/main_remote_discover_ccrp_uncertainty_sources.py`, and
   `scripts/audit/main_audit_ccrp_uncertainty_sources.py`. A 2026-06-04
   project-root broad scan also found no additional full-scale signal rows
   outside `outputs/`. Static trace
   `scripts/audit/main_trace_ccrp_formal_signal_path.py` shows the formal v3
   runner only requested `relevance_probability` and did not preserve
   evidence/counterevidence, `ccrp_uncertainty`, selected scored rows, or
   internal provenance, so the paper-critical uncertainty rows cannot be
   reconstructed from formal `scores.csv` alone. Guarded planning entry:
   `scripts/audit/main_plan_ccrp_signal_generation.py` writes a planning-only
   JSON/shell command package for Sports/Toys signal discovery, source audit,
   validation selection, ablation, observation, and hyperparameter plots. The
    generated shell exits before running commands and remains
    `planning_only_not_executed` until real full-scale signal paths pass audit.
    On 2026-06-05 01:50 CST, the generator and plan artifact were hardened to
    require no active official baseline row or matching baseline Python process,
    replacing stale row-specific wording while preserving the non-executing
    shell guard.
    Consolidated module audit entry:
   `scripts/audit/main_audit_paper_critical_modules.py` writes
    `outputs/summary/paper_critical/paper_critical_module_audit_20260604.{json,md}`
    without running experiments. The current audit reports
    `paper_ready=false`, `framework_overview_scaffold_ready=true`,
    `guarded_plan_ready=true`, and `signal_rows_available=false`, so the
    observation, component-ablation, and hyperparameter modules remain blocked
    by missing full-scale uncertainty or recomputable signal rows.
    Post-Phase2 checkpoint:
    `outputs/summary/paper_critical/paper_critical_module_audit_post_phase2_20260605.{json,md}`
    and
    `outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/`
    update this state after the Sports/Toys/Home/Tools official+C-CRP gate
    certificate. The guarded plan now covers all four new domains and records
    `will_start_experiment=false`; the generated shell exits with `exit 2`
    before any command. The audit still reports `paper_ready=false` and
    `signal_rows_available=false`, so observation, component ablation, and
    hyperparameter claims remain blocked until full-scale valid/test
    uncertainty signal rows are located or regenerated and audited under the
    same-candidate protocol.
    Storage preflight/cleanup for this next gate is recorded in
    `outputs/summary/paper_critical/server_storage_preflight_20260605.*`,
    `outputs/summary/tools_llmesr_completed_adapter_embedding_cleanup_20260605.{json,sha256}`,
    and `outputs/summary/paper_critical/server_storage_postcleanup_20260605.*`.
    The cleanup removed only completed Tools LLM-ESR paper-adapter staging
    files after final evidence/local-light/domain gates passed, recovering
    `/` to `12,342,898,688` bytes free / `94%` used. Final scores,
    provenance, audits, imported tables, final model checkpoints, task splits,
    and other projects were preserved. This is enough to clear the 10GiB danger
    floor but not enough for the recommended Phase 2.5 regeneration target, so
    signal-row generation remains gated on additional disk or another
    archive-backed cleanup decision.
    A follow-up read-only storage-retention audit is recorded in
    `outputs/summary/paper_critical/server_storage_phase2_5_retention_audit_20260605.{json,md,sha256}`.
    It found no sufficient routine safe-now cleanup: small completed staging
    remnants are only about `57M`, while high-yield candidates are protected
    completed artifacts such as the Tools LLM2Rec upstream embedding
    (`5,662,687,360` bytes) or final LLMEmb/LLM-ESR checkpoints (`3.8G`-`6.8G`).
    Those require an explicit archive/retention decision before deletion. No
    deletion was performed and no Phase 2.5 experiment was launched.
    Guarded planner:
    `scripts/audit/main_plan_phase2_5_retention_cleanup.py` now creates a
    planning-only retention cleanup package. Current artifact:
    `outputs/summary/paper_critical/retention_cleanup_plan_20260605/tools_llm2rec_upstream_embedding_retention_cleanup_plan_20260605.{json,sh,sha256}`.
    It targets the completed Tools LLM2Rec upstream embedding only as an
    approval-required option, records `will_delete=false`,
    `will_delete_files=false`, and `will_execute_cleanup=false`, and its shell
    exits with `exit 2` before `sha256sum` or `rm --`. The planned expected
    embedding sha256 is
    `306618d974eb4133d9cda87bae3251e17d793aa6f5a8cb38d558b549ed31d56e`.
   Observation-builder guard hardening on 2026-06-04: the motivation script now
   rejects duplicate ranking-eval events, eval events absent from the C-CRP
   uncertainty input, invalid positive ranks, and `num_candidates` mismatches
   when ranking records include that column. Its provenance labels the output
   as `artifact_class=paper_critical_observation_motivation` with
   `paper_claim_scope=motivation_only_not_main_table_sota`, records the full
   required metric set (MRR, HR@5/@10/@20, NDCG@5/@10/@20), and includes
   explicit claim limits so representative motivation evidence cannot be
   mistaken for main-table SOTA evidence.
2. A component ablation study over every nontrivial C-CRP design component
   found in the implementation/docs. At minimum, audit score mode, boundary
   uncertainty (`without_boundary_uncertainty`), calibration gap
   (`without_calibration_gap`), evidence support/insufficiency
   (`without_evidence_support`), counterevidence, risk penalty, eta,
   confidence weight, and C-CRP weight triples where the code path supports
   them. A component whose removal is neutral or better must be reported as
   weak or misdesigned rather than hidden.
3. A hyperparameter analysis over real method controls, with curves and
   validation-only selection clearly separated from test reporting. Candidate
   controls include eta, confidence weight, weight triples, uncertainty
   thresholds/gates, anchor conflict penalties, and SRPD learning-rate/lambda
   controls when SRPD rows are used. C-CRP sweep plotting entry:
   `scripts/analysis/main_plot_ccrp_hyperparameter_sweep.py`, consuming
   validation `valid_ccrp_sweep.csv` outputs and requiring at least three values
   per paper-facing curve by default. Paper-facing stability wording requires
   validation and test curves to be reported separately, using `--test_sweep_csv`
   when the test sweep is available. Guard hardening on 2026-06-04 requires
   `audit_ok` and `degeneracy_audit_ok` columns by default, labels valid-only
   outputs as `validation_only_hyperparameter_selection_curve` with
   `paper_claim_scope=validation_only_not_stability_claim`, labels incomplete
   or audit-not-enforced curves as diagnostic-only, and reserves
   `paper_critical_hyperparameter_curve_ready` for audited valid+test curves.
4. A clean framework overview figure showing the full pipeline and where
   uncertainty and C-CRP components enter the ranking decision. Figure builder:
   `scripts/analysis/main_build_framework_overview_figure.py`; generated SVG
   is the editable source and PDF/PNG are export artifacts. Draft package:
   `outputs/summary/paper_critical/framework_overview/` with SVG, PDF, PNG,
   caption, provenance, and `framework_overview_manifest.sha256`. This package
   is paper-critical evidence scaffolding, not yet a final camera-ready figure.
   It was regenerated at 2026-06-04 17:31 CST against git commit `9badd19`;
   `python -m pytest tests\test_framework_overview_figure.py
   tests\test_audit_paper_critical_modules.py` passed (`4 passed`) and the PNG
   was visually checked as nonblank/readable.
   Review-ready checkpoint 2026-06-05 23:20 CST: the framework package was
   regenerated and the module audit now treats it as a figure-only
   review-ready artifact when the SVG/PDF/PNG/caption/provenance files match
   the sha256 manifest, the SVG contains the required pipeline labels, the PNG
   is valid and at least `1800x900`, provenance records
   `status_label=paper_critical_framework_overview_review_ready`,
   `paper_claim_ready=true`, the controlled same-candidate claim boundary, and
   the multiplicative risk formula
   `risk_score = base_score * (1 - uncertainty)^eta`. Evidence:
   `outputs/summary/paper_critical/framework_overview/` and
   `outputs/summary/paper_critical/paper_critical_module_audit_post_framework_review_20260605.{json,md,sha256}`.
   The audit reports framework overview `status=review_ready` and
   `paper_claim_ready=true`, but keeps overall `paper_ready=false` because
   observation, component-ablation, and hyperparameter claims still lack
   full-scale valid/test uncertainty signal rows.

Each module needs status labels, row counts, commands, configs, seeds when
applicable, provenance notes, plots/tables, and a lightweight local evidence
package before it can support paper claims.

The current execution specification is
`docs/paper_critical_experiment_plan_2026-06-03.md`.

## Not primary claims unless completed

- Generative title recommendation.
- Full-catalog SOTA.
- Universal cross-domain winner.
- LoRA distillation as main novelty.
- Shadow v2-v6 as independent main methods.
- SRPD as a main-method substitute for external baselines unless its formal
  train/eval gates are completed.
- Proxy comparisons against reported numbers from incompatible protocols.

## Status labels

Every paper-facing summary table must include `status_label`.

- `completed_result`: runnable and completed under the stated protocol.
- `runnable_not_complete`: code path exists, but the result is not complete
  enough for a main table.
- `design_only`: method or plan is documented but not fully run under the
  frozen protocol.
- `proxy_only`: useful for related-work positioning or fairness audit, but not
  same-schema evidence.
- `future_extension`: intentionally outside the main paper claim.

Rows with `design_only`, `proxy_only`, or `future_extension` must not enter the
main result table. Rows from fallback internal validation splits are not main
table eligible.

Agent drift guard: every new experiment, baseline, model module, or generated
table must receive a status label and milestone eligibility before it is
described as evidence. If a future agent cannot prove eligibility from
provenance and audits, the row is diagnostic or supplementary.

## Milestone claim eligibility

The canonical milestone map lives in `docs/milestones/README.md`.

| milestone | paper role | main-table eligible |
| --- | --- | --- |
| M0 Week1-4 / pony12 | diagnosis and motivation | no |
| M1 Pony framework | method and protocol bridge | yes, if completed under frozen protocol |
| M2 Light series | precursor / negative-control ablation | supplementary unless explicitly completed under main protocol |
| M3 Shadow series | task-grounded signal family | diagnostic unless large-scale protocol and validation gates are complete |
| M4 Baseline system | fairness and reviewer defense | yes for completed classical rows; official external rows only after provenance passes |
| M5 Four-domain validation | robustness evidence | yes when 100neg outputs, audits, and paired tests are complete |
| M6 Complete recommender system | future full-system roadmap | no until official baselines, Shadow, LoRA, and generated-title verification are completed |

The `*_style_*` LLM-rec rows are paper-style supplementary rows, not official
reproductions. Official external-baseline claims require the
`*_official_qwen3base_*` family plus provenance and score coverage.

Internal Shadow/Light/LoRA ablations cannot substitute for missing external
baselines. If official external rows are unfinished, the paper wording must
state the baseline gap and keep those rows out of main official claims.

## Main table eligibility

A row can enter the main ranking table only if all are true:

- `artifact_class == completed_result`
- `status_label` matches the row type, for example
  `same_schema_external_baseline`, `same_schema_internal_method`, or a
  documented completed ablation label.
- Same-schema data, prompt, candidate construction, and metric definitions are
  used.
- External baseline rows follow the declared comparison variant, usually
  `official_code_qwen3base_default_hparams_declared_adaptation`.
- Official external rows require `implementation_status=official_completed`;
  `style_adapter_only` and `partial_official_adapter_exists` rows stay
  supplementary.
- Internal formal rows require exact score export and import. C-CRP uses
  `status_label=same_schema_internal_method`; SRPD uses
  `status_label=same_schema_internal_ablation` unless all trainable-framework
  gates are satisfied.
- C-CRP formal rows must include validation-only selection provenance, input
  hashes, exact score coverage, and score-degeneracy audit output.
- SRPD rows that use rank-order fallback scores are internal ablations only;
  they must not be presented as native trainable scorer results or promoted to
  the main method table.
- Runner or plan filenames containing `official` are not enough. The row must
  have unblocked provenance from the pinned official repo and pass score/import
  gates.
- Baseline hyperparameters come from the official default/recommended setting,
  unless an override is explicitly recorded for protocol compatibility.
- Our method's gates, weights, and hyperparameters are selected on validation or
  fixed before test.
- Full-finetune and retuned-baseline variants are not mixed into the primary
  main comparison table.
- External score files have exact unique `source_event_id,user_id,item_id`
  coverage with finite numeric scores.
- Calibration method and C-CRP weights are selected on validation or fixed
  before test.
- `calibration_split_metadata.csv` has zero user overlap unless the table is
  explicitly labeled non-main.
- Candidate protocol audit exists for the domain/split.
- Statistical tests provide paired confidence intervals for the supported claim.

## Safe wording

Use "observed best" for results whose confidence interval crosses zero or whose
paired test is not significant after correction. Use "winner" only for results
that pass the configured statistical rule.

## C-CRP v3 Progress (2026-05-31)

### Multi-Domain Results

C-CRP v3 uses a profile-enhanced prompt that asks the LLM to infer user
preferences before scoring each candidate. All domains use 10k users,
101 candidates (1 positive + 100 negative), Qwen3-8B via vLLM.

| Domain | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR | vs Best Baseline |
|--------|------|-------|-------|--------|---------|---------|-----|------------------|
| beauty (973u) | 0.157 | 0.229 | 0.369 | 0.111 | 0.134 | 0.169 | 0.128 | #2 (ProEx=0.253) |
| books | 0.374 | **0.476** | 0.592 | 0.300 | **0.333** | 0.362 | 0.306 | **SOTA** (+0.8% vs LLMEmb) |
| electronics | 0.218 | **0.299** | 0.418 | 0.157 | **0.183** | 0.213 | 0.168 | **SOTA** (+22% vs LLMEmb) |
| movies | 0.145 | 0.208 | 0.331 | 0.108 | 0.128 | 0.159 | 0.127 | #5 (LLMEmb=0.334) |
| sports | 0.275 | 0.382 | 0.517 | 0.198 | 0.233 | 0.267 | 0.208 | domain gate PASS |
| toys | 0.317 | 0.396 | 0.506 | 0.245 | 0.271 | 0.298 | 0.250 | domain gate PASS |
| home | 0.156 | 0.226 | 0.351 | 0.110 | 0.132 | 0.164 | 0.126 | domain gate PASS |
| tools | 0.194 | 0.270 | 0.393 | 0.142 | 0.166 | 0.197 | 0.156 | domain gate PASS |

Status: `completed_result` for beauty/books/electronics/movies/sports/toys/home/tools.
The C-CRP v3 batch completed without FAILED/OOM/Traceback markers in
`ccrp_v3_all_domains.log`; new-domain official baselines entered Phase 2 on
2026-05-31 with a sports single-domain production run.
For sports/toys/home/tools, each report records `n_users=10000`,
`n_prompts=1010000`, the test same-candidate `data_path`, `scores.csv` with
1,010,000 scored candidate rows plus header, and `user_ranks.jsonl` with
10,000 rows.
Toys-specific audit checkpoint (2026-06-02 07:18 CST): the C-CRP core result
under `outputs/toys_large10000_100neg_ccrp_v3` is metric-complete
(HR@5/10/20 `0.3172 / 0.3964 / 0.5059`, NDCG@5/10/20
`0.2451904009717959 / 0.27079859856897753 / 0.298341205798594`, MRR
`0.2503049488607351`, `n_users=10000`, `scores.csv` `1,010,001` lines, and
`user_ranks.jsonl` `10,000` rows). At 2026-06-02 17:09 CST the raw scores were
imported into
`outputs/toys_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate`
through the existing same-candidate importer without `--allow_partial_scores`.
The import reported `score_coverage_rate=1.000000`. After the final toys
`llmesr_sasrec` official row completed, the follow-up toys domain gate recorded
`ccrp_ok=true`, `official_ok_count=8`, `official_all_ok=true`, and
`gate_ok=true`. The toys comparison/statistical gate then recorded C-CRP rank
1 on all seven metrics and all 56 C-CRP-vs-official paired tests positive and
Holm-significant. The local lightweight C-CRP import package keeps the imported
tables and gate summary; the large imported prediction JSONL remains
server-only.
Original-domain C-CRP v3 formal reports are present under
`outputs/ccrp_v3_formal/<domain>/report.json`, and the old four-domain
official-baseline comparison is present at
`outputs/ccrp_v3_formal/main_comparison_table.csv`.
Artifact audit note (2026-05-31): the old four-domain C-CRP reports were not
missing; earlier searches that only checked `outputs/*ccrp_v3/report.json`
missed the `outputs/ccrp_v3_formal/<domain>/report.json` layout. The old
four-domain 8-baseline comparison table is metric-complete, but some
method-specific old-domain baseline directories are table-only under the
current strict evidence gate because final provenance/audit files are not
co-located with the imported tables. Reconcile those evidence packs before
paper submission; do not rerun or relabel them without a provenance decision.

### Strategy for SOTA

C-CRP v3 achieves SOTA on books and electronics under the current comparison
table, and sports/toys/home/tools now pass domain-level official-baseline and
paired-test gates. Paper-wide SOTA wording is still not submission-ready until
the four-domain comparison is consolidated into the paper table, ARIS
claim/citation audits pass, the paper-critical observation/ablation/
hyperparameter/figure modules are complete, and GPT-5.5/Codex xhigh review
reaches at least 8/10.

### Remaining for paper submission

1. Consolidate the completed canonical 8-official-baseline gates for
   sports/toys/home/tools into the final paper comparison evidence. Do not
   launch more official baseline rows unless a later audit finds a concrete
   invalid row. `scripts/run_baselines_new_domains.sh` remains aligned to
   exclude SETRec while it is blocked/supplementary, supports single-domain
   production via `DOMAINS_OVERRIDE`, and audits/imports complete
   `@5/@10/@20 + MRR` same-candidate metrics after each completed score file.
   Sports, toys, home, and tools are now 8/8 complete and have passed their
   domain/comparison/paired-test gates. Tools `proex_profile` launched at
   2026-06-04 14:25 CST as the first Tools row and completed at
   2026-06-04 16:08 CST with `implementation_status=official_completed`,
   `blockers=[]`, exact `score_coverage_rate=1.0`, score audit/imported
   tables, server-final audit, server large-artifact sha256 manifest,
   lightweight local sync, and local-light audit all passing. Full metrics over
   10,000 users and 101 candidates are HR@5/10/20
   `0.0602 / 0.1177 / 0.2329`, NDCG@5/10/20
   `0.037281705859706714 / 0.055676376797898205 / 0.08437492971571317`, and
   MRR `0.06071849976691817`; `scores.csv` has `1,010,001` lines,
   predictions had `10,000` lines before post-gate deletion, and
   `tables/ranking_eval_records.csv` has `10,001` lines. The local
   lightweight package is
   `outputs/baselines/official_adapters/tools_large10000_100neg_proex_profile_official_qwen3base_same_candidate/`.
   Server-only `scores.csv`, deleted prediction metadata, and
   `proex_official_model.pt` are covered by
   `server_large_artifact_manifest.sha256` and
   `prediction_deletion_manifest.json`. The completed intermediate adapter was
   removed after final evidence and local backup passed, recovering `/` to
   about `12G` free / `95%` used while preserving final evidence. Tools
   `promax_profile` launched at 2026-06-04 16:46 CST as the second single-row
   Tools official baseline with runner PID `3279573`, adapter PID `3279582`,
   and log `baselines_new_domains_tools_promax_20260604_164630.log`, then
   completed at 2026-06-04 19:59 CST with `implementation_status=official_completed`,
   `blockers=[]`, exact `score_coverage_rate=1.0`, score audit/imported
   tables, server-final audit, server large-artifact sha256 manifest,
   lightweight local sync, and local-light audit all passing. Full metrics over
   10,000 users and 101 candidates are HR@5/10/20
   `0.056 / 0.1046 / 0.2018`, NDCG@5/10/20
   `0.03468275603534166 / 0.05029722685396016 / 0.07458228366305956`, and MRR
   `0.056527355267188224`; `scores.csv` has `1,010,001` lines, predictions
   had `10,000` lines before post-gate deletion, and
   `tables/ranking_eval_records.csv` has `10,001` lines. The local
   lightweight package is
   `outputs/baselines/official_adapters/tools_large10000_100neg_promax_profile_official_qwen3base_same_candidate/`.
   Server-only `scores.csv`, deleted prediction metadata, and
   `promax_official_model.pt` are covered by
   `server_large_artifact_manifest.sha256` and
   `prediction_deletion_manifest.json`; the manifest records sha256 values
   `2a9797b945fef73e76a1db18efe7ef037f2b47732c29ebaaf44ece01b33ac781`
   for scores, `b2123ea945285b9ee7ca940819382191fbae6af945cee09fb741b7c5ca95c717`
   for the deleted prediction JSONL, and
   `c0e17d003ba1e055e65d38dfac4dc96483f4a8744e201e3daf168a71f31890fc`
   for the final model. The completed ProMax intermediate adapter was removed
   after final evidence and local backup passed, with cleanup manifests
   `outputs/summary/tools_promax_completed_adapter_cleanup_manifest_20260604.sha256`
   and `outputs/summary/tools_promax_completed_adapter_cleanup_du_20260604.txt`;
   final scores, provenance, audits, imported tables, and model were preserved.
   Tools `elmrec_graph` launched at 2026-06-04 20:46 CST as the third
   single-row Tools official baseline with runner PID `3301337`, adapter PID
   `3301345`, and log `baselines_new_domains_tools_elmrec_20260604_204602.log`.
   It completed at 2026-06-04 22:39 CST with
   `implementation_status=official_completed`, `blockers=[]`, exact
   `score_coverage_rate=1.0`, score audit/imported tables, server-final audit,
   server large-artifact sha256 manifest, lightweight local sync, and
   local-light audit all passing. Full metrics over 10,000 users and 101
   candidates are HR@5/10/20 `0.0501 / 0.101 / 0.2101`, NDCG@5/10/20
   `0.029656030656687697 / 0.045870649973376774 / 0.07316592297455926`, and
   MRR `0.05237582779698271`; `scores.csv` has `1,010,001` lines,
   predictions had `10,000` lines before post-gate deletion, and
   `tables/ranking_eval_records.csv` has `10,001` lines. The local lightweight
   package is
   `outputs/baselines/official_adapters/tools_large10000_100neg_elmrec_graph_official_qwen3base_same_candidate/`.
   Server-only `scores.csv`, deleted prediction metadata, and
   `elmrec_official_model.pt` are covered by `server_large_artifact_manifest.*`
   and `prediction_deletion_manifest.json`; the manifest records sha256 values
   `13e8aa52ed5c69fa8c9b04006907b907043dd5fbc26b1829e3542d2ed58b050c` for
   scores, `d49457a2d7fc7a5877f565f51913f7110fac438bd6cadd871a8e2da68237c4fd`
   for the deleted prediction JSONL, and
   `637e253d09007a93dfa1fc3f78ba4209b2d052b76a22f629e6e1ad8bf375a22d` for the
   final model. The completed ElmRec intermediate adapter was removed after
   final evidence and local backup passed, with cleanup manifests
   `outputs/summary/tools_elmrec_completed_adapter_cleanup_manifest_20260604.sha256`
   and `outputs/summary/tools_elmrec_completed_adapter_cleanup_du_20260604.txt`;
   final scores, provenance, audits, imported tables, and model were preserved.
   Tools `llmemb` launched at 2026-06-04 23:10 CST as the fourth single-row
   Tools official baseline with runner PID `3317251`, adapter PID `3317260`,
   and log `baselines_new_domains_tools_llmemb_20260604_231030.log`; it
   completed at 2026-06-05 00:43 CST with
   `implementation_status=official_completed`, `blockers=[]`, exact
   `score_coverage_rate=1.0`, and passing server-final audit, server
   large-artifact manifest, lightweight local sync, and local-light audit.
   Full metrics over 10,000 users and 101 candidates are HR@5/10/20
   `0.1365 / 0.2257 / 0.3637`, NDCG@5/10/20
   `0.087457824217457 / 0.11594350972806679 / 0.15050644138929892`, and MRR
   `0.10649354669900822`; `scores.csv` has `1,010,001` lines, predictions had
   `10,000` lines before post-gate deletion, and
   `tables/ranking_eval_records.csv` has `10,001` lines. The local lightweight
   package is
   `outputs/baselines/official_adapters/tools_large10000_100neg_llmemb_official_qwen3base_same_candidate/`.
   The server-only scores, deleted prediction metadata, and
   `llmemb_official_model.pt` are covered by `server_large_artifact_manifest.*`
   and `prediction_deletion_manifest.json`; cleanup manifests are
   `outputs/summary/tools_llmemb_completed_adapter_cleanup_manifest_20260605.sha256`
   and `outputs/summary/tools_llmemb_completed_adapter_cleanup_du_20260605.txt`.
   During the run, a completed Home LLM2Rec checkpoint was deleted under
   emergency disk approval after sha256/size manifesting; local record:
   `outputs/summary/home_llm2rec_checkpoint_deletion_manifest_20260604.json`.
   Final scores, provenance, audits, imported tables, and model were preserved,
   and post-cleanup `/` returned to about `15G` free / `93%` used. After a
   fresh preflight, Tools
   `irllrec_intent` launched at 2026-06-05 01:04 CST as the fifth single-row
   Tools official baseline with runner PID `3326805`, adapter PID `3326813`,
   log `baselines_new_domains_tools_irllrec_20260605_0058.log`, and heartbeat
   `monitor-tools-irllrec`; its launch snapshot
   `outputs/summary/tools_irllrec_launch_monitor_20260605.json` shows Qwen3
   embedding progress `2056/269711`, GPU `95%`, disk `13.34G` free, and no
   failure markers. It completed normally at 2026-06-05 05:19 CST with
   `implementation_status=official_completed`, `blockers=[]`, exact
   `score_coverage_rate=1.0`, score audit/imported tables, server-final audit,
   server large-artifact manifest, lightweight local sync, and local-light
   audit all passing. Full metrics over 10,000 users and 101 candidates are
   HR@5/10/20 `0.102 / 0.1651 / 0.3095`, NDCG@5/10/20
   `0.06504709833452535 / 0.08525707170530923 / 0.12100829406900945`, and MRR
   `0.08670154590435823`; `scores.csv` has `1,010,001` lines, predictions have
   `10,000` lines, and `tables/ranking_eval_records.csv` has `10,001` lines.
   The local lightweight package is
   `outputs/baselines/official_adapters/tools_large10000_100neg_irllrec_intent_official_qwen3base_same_candidate/`.
   Server-only `scores.csv`, predictions, and `irllrec_official_model.pt` are
   covered by `server_large_artifact_manifest.*`; provenance records the
   IRLLRec scalability bridge `deterministic_node_cap`. The completed
   intermediate adapter was removed after gates and local backup passed, with
   cleanup manifest
   `outputs/summary/tools_irllrec_completed_adapter_cleanup_manifest_20260605.sha256`,
   recovering `/` to about `14G` free / `93%` used while preserving final
   evidence. Tools is now 5/8 complete and still needs `rlmrec_graphcl`,
   `llm2rec_sasrec`, and `llmesr_sasrec` plus the Tools
   domain/comparison/paired-test gates. After a fresh preflight, Tools
   `rlmrec_graphcl` launched at 2026-06-05 05:41 CST as the sixth single-row
   Tools official baseline with runner PID `3347729`, adapter PID `3347738`,
   log `baselines_new_domains_tools_rlmrec_20260605_054158.log`, heartbeat
   `monitor-tools-rlmrec`, guarded completion-gate plan
   `outputs/summary/official_completion_gate_plan/tools_rlmrec_graphcl_completion_gates_20260605.{json,ps1}`,
   and launch snapshot `outputs/summary/tools_rlmrec_launch_monitor_20260605.json`.
   The launch snapshot found one matching adapter process, no completion or
   failure markers, adapter dir about `1000M`, final output placeholder-only,
   and disk about `12.35G` free / `94%` used. A 2026-06-05 05:53 CST monitor
   check found runner PID `3347729` and adapter PID `3347738` still alive, one
   matching adapter process, Qwen3 embedding progress `35360/269711`, GPU
   active, disk about `12.35G` free / `94%` used, no failure markers, and no
   final scores/provenance/tables yet. A 2026-06-05 06:03 CST follow-up found
   the same PIDs alive and unique, Qwen3 embedding progress `79528/269711`,
   disk still about `12.35G` free / `94%` used, and no completion/failure
   markers or final scores/provenance/tables. A 2026-06-05 06:15 CST follow-up
   found Qwen3 embedding progress `128944/269711`, the same PIDs alive and
   unique, disk still about `12.35G` free / `94%` used, and no
   completion/failure markers or final scores/provenance/tables. A
   2026-06-05 06:25 CST follow-up found Qwen3 embedding progress
   `172352/269711`, the same PIDs alive and unique, disk still about `12.35G`
   free / `94%` used, and no completion/failure markers or final
   scores/provenance/tables. A 2026-06-05 06:33 CST follow-up found Qwen3
   embedding progress `209840/269711`, the same PIDs alive and unique, disk
   still about `12.35G` free / `94%` used, and no completion/failure markers or
   final scores/provenance/tables. This is active monitor-only evidence, not a
   completed row.
   A 2026-06-05 07:02 CST follow-up found Qwen3 embedding complete
   (`269711/269711`) and official RLMRec training started, latest observed at
   `epoch=110` with train loss `1.519527`. The same runner/adapter PIDs were
   alive and unique, GPU was active, final RLMRec evidence remained
   placeholder-only, and no completion/failure marker or final
   scores/provenance/tables existed. Disk fell below the warning line to about
   `8.6G` free / `96%` used as the active adapter grew to about `5.2G`; after
   verifying server-final and local-light evidence for completed Tools IRLLRec,
   Codex removed only its server-side `predictions/rank_predictions.jsonl`
   with sha256 manifest
   `outputs/summary/tools_irllrec_prediction_cleanup_manifest_20260605.sha256`.
   Scores, provenance, audits, imported tables, and model were preserved; disk
   recovered only to about `9.3G` free, still below the 10G warning threshold.
   This is active monitor-only evidence, not a completed RLMRec row.
   A 2026-06-05 08:01 CST follow-up confirmed the active Tools RLMRec run is
   on the expected official default `3000`-epoch path (`--rlmrec_epochs=3000`
   / trainer `--epochs=3000`) and had reached epoch `590/3000` with train loss
   `1.509551`. Runner PID `3347729` and adapter PID `3347738` were alive and
   unique, no completion/failure/final-artifact markers existed, final evidence
   remained placeholder-only, and disk remained warning-level at about `9.3G`
   free / `96%` used. A repeat large-file/cache/temp/archive audit found no
   safe deletion candidate. This remains active monitor-only evidence, not a
   completed RLMRec row.
   A 2026-06-05 08:46 CST follow-up found the same runner/adapter PIDs alive
   and unique, with official RLMRec training past the first material monitor
   checkpoint at epoch `1000/3000` and train loss `1.505931`. Final RLMRec
   evidence still remained placeholder-only, with no final scores/provenance,
   score audit, imported tables, predictions, completion marker, OOM/no-space,
   killed, traceback, or error marker. Disk remained warning-level at about
   `9.3G` free / `96%` used and no new safe cleanup candidate was identified.
   This remains active monitor-only evidence, not a completed RLMRec row.
   A 2026-06-05 09:45 CST follow-up found the same runner/adapter PIDs alive
   and unique, with official RLMRec training past the halfway monitor
   checkpoint on the default `3000`-epoch path: latest observed epoch
   `1510/3000` had train loss `1.506936`, after epoch `1500` loss
   `1.507642`. Final RLMRec evidence still remained placeholder-only (`4.0K`)
   with no final scores, provenance, score audit, imported tables,
   predictions, completion marker, OOM/no-space, killed, traceback, or error
   marker. Disk remained warning-level at about `9.3G` free / `96%` used; a
   repeat large-file/cache/temp/archive and prediction cleanup audit found no
   safe meaningful deletion candidate. This remains active monitor-only
   evidence, not a completed RLMRec row.
   A 2026-06-05 10:36 CST follow-up found the same runner/adapter PIDs alive
   and unique, with exactly one matching RLMRec adapter and zero matching
   IRLLRec adapters. Official RLMRec training crossed the two-thirds monitor
   checkpoint on the default `3000`-epoch path at epoch `2000/3000` with train
   loss `1.506144`. Final RLMRec evidence still remained placeholder-only
   (`4.0K`) with no final scores, provenance, score audit, imported tables,
   predictions, completion marker, OOM/no-space, killed, traceback, or error
   marker. Disk remained warning-level at about `9.3G` free / `96%` used; a
   fresh cleanup audit found no safe deletion target because caches/tmp were
   small, no temp/archive/part/core files existed, and the largest files were
   active RLMRec intermediates, protected task splits, retained completed
   checkpoints/evidence, or historical C-CRP/fusion summaries. This remains
   active monitor-only evidence, not a completed RLMRec row.
   A 2026-06-05 11:49 CST follow-up found the same runner/adapter PIDs alive
   and unique, with official RLMRec training past the five-sixths monitor
   checkpoint at epoch `2610/3000`; the epoch `2600` loss was `1.505511` and
   the latest epoch `2610` loss was `1.504630`. Final RLMRec evidence still
   remained placeholder-only (`4.0K`) with no final scores, provenance, score
   audit, imported tables, predictions, completion marker, OOM/no-space,
   killed, traceback, or error marker. Disk remained warning-level at about
   `9.3G` free / `96%` used; a repeat large-file/cache/temp/archive and
   prediction cleanup audit found no safe deletion target because visible
   reclaimable candidates were active RLMRec intermediates, protected task
   splits, retained completed checkpoints/evidence, or the legacy Electronics
   ELMRec prediction JSONL without server-final/local-light deletion proof.
   This remains active monitor-only evidence, not a completed RLMRec row.
   A 2026-06-05 12:49 CST completion/gate pass then recorded Tools
   `rlmrec_graphcl` as the sixth gated Tools official row. The wrapper log
   reached epoch `3000/3000` with final train loss `1.505858`, wrote
   `implementation_status=official_completed`, `blockers=0`, saved predictions
   and tables, reported `score_coverage_rate=1.000000`, and ended with
   `DONE rlmrec_graphcl on tools` / `All baseline runs complete`. Server-final
   audit, server large-artifact manifest, lightweight local sync, and
   local-light audit all passed. Full metrics over 10,000 users and 101
   candidates are HR@5/10/20 `0.0784 / 0.1354 / 0.2465`, NDCG@5/10/20
   `0.05017501611537314 / 0.06838865570840932 / 0.09599330874161652`, and MRR
   `0.07220064580885768`; `scores.csv` has `1,010,001` lines, predictions had
   `10,000` lines before post-gate deletion, and
   `tables/ranking_eval_records.csv` has `10,001` lines. The local lightweight
   package is
   `outputs/baselines/official_adapters/tools_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate/`.
   Post-gate cleanup removed only the server prediction JSONL and completed
   intermediate adapter directory with manifests
   `prediction_deletion_manifest.json`,
   `outputs/summary/tools_rlmrec_completed_adapter_cleanup_manifest_20260605.sha256`,
   and `outputs/summary/tools_rlmrec_completed_adapter_cleanup_du_20260605.txt`,
   preserving final scores, provenance, audits, imported tables, model,
   manifests, and local evidence. Disk recovered to about `15G` free / `93%`
   used. Tools is now 6/8 official rows gated and still needs `llm2rec_sasrec`,
   `llmesr_sasrec`, and the Tools domain/comparison/paired-test gates.
   A first Tools `llm2rec_sasrec` launch at 2026-06-05 13:02 CST then failed
   before a stable adapter process or score file existed. The traceback was a
   validation-export exact-one-positive failure for
   `source_event_id='AGEY75LYLXUAHG3KW5KF5ICMKA4A'`. Audit found the Tools
   valid `candidate_items.csv` was truncated: original sha256
   `4302712bb7dbe0a8cfde99b0a2727c8de0818d250b65e7cc3bc0b8ad01fa6f2b`,
   `587252` logical rows, partial final event, and a malformed user-id-only
   row. The CSV was rebuilt from `ranking_valid.jsonl` and independently
   validated as PASS with `1,010,000` rows, `10,000` events, `101`
   candidates/event, and exactly one positive/label per event. Manifests are
   `outputs/summary/tools_valid_candidate_items_repair_20260605T051943Z.json`
   and
   `outputs/summary/tools_valid_candidate_items_repair_validation_20260605T0520Z.json`.
   No Tools LLM2Rec row is active or table-eligible; disk is about `14G` free /
   `93%` used, so relaunch is pending a fresh preflight and storage-margin
   decision.
   A 2026-06-05 13:46 CST relaunch checkpoint then started exactly one Tools
   `llm2rec_sasrec` row after sidecar ARIS review of storage and duplicate-run
   risk. Runner PID `3413921`, adapter PID `3413930`, log
   `baselines_new_domains_tools_llm2rec_20260605_134355.log`, heartbeat
   `monitor-tools-llm2rec`, and launch snapshot
   `outputs/summary/tools_llm2rec_launch_monitor_20260605.json`. The snapshot
   found one adapter process, no failure markers, active Qwen3 embedding
   progress around `[hf_mean_pool] encoded 3888/345622`, GPU about `95%` /
   `16089 MiB`, adapter directory `1.1G`, final output placeholder-only, and
   disk about `12G` free / `94%` used. This is active monitor-only evidence,
   not a completed or table-eligible row.
   A 2026-06-05 13:57 CST follow-up found the same runner/adapter alive, cleared
   three stale diagnostic grep/bash processes that were causing false duplicate
   matches, and observed Qwen3 embedding progress around
   `[hf_mean_pool] encoded 53296/345622`; disk remained about `12G` free /
   `94%` used and no failure/completion marker was present. Snapshot:
   `outputs/summary/tools_llm2rec_monitor_checkpoint_20260605_1355.json`.
   A 2026-06-05 15:23 CST monitor confirmed the relaunch failed due disk
   exhaustion rather than completing. Runner PID `3413921` and adapter PID
   `3413930` had exited, GPU was idle, and
   `baselines_new_domains_tools_llm2rec_20260605_134355.log` ended with
   `OSError: [Errno 28] No space left on device` after Qwen3 embedding reached
   `345622/345622`. The LLM2Rec training log showed `torch.save` failed while
   saving the SASRec checkpoint, so the partial checkpoint was corrupt. There
   is no `fairness_provenance.json`, no `scores.csv`, and no score audit; the
   row is failed and not table-eligible. Emergency cleanup removed only failed
   or already-repaired artifacts: the corrupt validation CSV backup, failed
   adapter-side embedding copy, regenerable adapter CSVs, and corrupt partial
   checkpoint. It preserved the reusable upstream embedding at
   `/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`.
   Post-cleanup disk is about `8.1G` free / `96%` used, still below the `10G`
   monitor threshold. Do not import this row and do not start the next baseline
   until additional audited space is freed; the recovery relaunch should pass
   the preserved path through `--llm2rec_item_embedding_path`.
   A 2026-06-05 16:05 CST recovery/storage checkpoint removed the completed
   Sports and Toys LLM2Rec full checkpoints with sha256/size manifests after a
   GPT-5.5 xhigh sidecar ARIS storage audit and live evidence checks. The
   manifests are
   `outputs/summary/sports_llm2rec_checkpoint_deletion_manifest_20260605.json`
   and `outputs/summary/toys_llm2rec_checkpoint_deletion_manifest_20260605.json`.
   Sports/Toys LLM2Rec scores, provenance, score audits, run summaries,
   imported tables, server-final audits, and local-light packages were
   preserved; only the full `.pth` checkpoints were deleted. The wrapper now
   supports `LLM2REC_ITEM_EMBEDDING_PATH_OVERRIDE` and the server copy passed
   `bash -n`. Exactly one Tools LLM2Rec recovery launched at 2026-06-05
   15:59 CST with runner PID `3423029`, adapter PID `3423037`, official
   training PID `3423221`, log
   `baselines_new_domains_tools_llm2rec_recovery_20260605_155904.log`, and
   heartbeat `monitor-tools-llm2rec-recovery`. Its adapter command includes
   `--llm2rec_item_embedding_path /home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`,
   so Qwen3 embedding is reused. At the 16:05 CST snapshot it was active around
   epoch `30+`, had saved a Tools checkpoint, and disk was about `10G` free /
   `95%` used. This is active monitor-only evidence, not a completed row.
   A 2026-06-05 17:04 CST monitor/storage checkpoint confirmed the recovery
   row was still active with runner PID `3423029`, adapter PID `3423037`, and
   training PID `3423221`; duplicate counts were exactly one LLM2Rec adapter
   and one `ToolsSameCandidate100Neg` training child. No final provenance,
   `scores.csv`, or imported ranking table existed yet. Training-log
   validation metrics were still improving at epoch `330`, latest observed
   HR@5/10/20
   `0.26969999074935913 / 0.3407999873161316 / 0.414900004863739` and
   NDCG@5/10/20
   `0.20475752651691437 / 0.22773440182209015 / 0.2465011030435562`; these
   are validation-only progress numbers, not final table metrics. Disk briefly
   crossed the monitor threshold at about `9.9G` free / `95%` used. A
   large-file/cache/temp/archive audit found no safe project-output deletion:
   large visible artifacts were active, final evidence/model artifacts,
   same-candidate task data, or an older Electronics ELMRec prediction whose
   server-final audit is `ok=false`. After approval, only conda package cache
   cleanup was run; no project outputs, evidence, checkpoints, embeddings, task
   splits, or other projects were deleted. Post-cleanup `/` was about `11G`
   free (`10,773,983,232` bytes) / `95%` used.
   A 2026-06-05 18:07-18:11 CST follow-up found the same recovery row still
   active with runner PID `3423029`, adapter PID `3423037`, and training PID
   `3423221`; duplicate counts remained exactly one LLM2Rec adapter and one
   `ToolsSameCandidate100Neg` training child. No final provenance,
   `scores.csv`, score audit, or imported ranking table existed yet.
   Training-log validation progress reached epoch `715`, latest observed
   validation HR@5/10/20
   `0.8866999745368958 / 0.9272000193595886 / 0.9584000110626221` and
   NDCG@5/10/20
   `0.8021516799926758 / 0.8153268098831177 / 0.8232611417770386`; these are
   validation-only progress numbers, not final table metrics. Disk was only
   about 34 MiB above the 10 GiB guard (`10,773,143,552` bytes free), so only
   non-project cache cleanup was run: conda reported no unused
   tarballs/packages/tempfiles, pip cache purge removed `17.0 MB`, and npm
   cache clean was attempted. No project outputs, evidence, checkpoints,
   embeddings, task splits, or other projects were deleted. The post-cleanup
   safety check confirmed the same PIDs alive, duplicate counts `1/1`, no final
   evidence, and disk `10,820,177,920` bytes free / `95%` used.
   At 2026-06-05 18:38 CST, Tools `llm2rec_sasrec` completed normally under
   the recovery run. The wrapper reported `implementation_status=official_completed`,
   `[2026-06-05 18:38:04] DONE llm2rec_sasrec on tools`, and `=== All baseline
   runs complete ===`; tracked PIDs exited and duplicate counts were `0/0`.
   Server-final audit passed with `ok=true`, `failures=[]`, `warnings=[]`;
   server large-artifact manifest passed; local-light sync and local-light
   audit both passed. Local lightweight evidence is under
   `outputs/baselines/official_adapters/tools_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate/`.
   Full metrics over `10,000` users and `101` candidates are HR@5/10/20
   `0.0957 / 0.1625 / 0.2954`, NDCG@5/10/20
   `0.060227320850546905 / 0.08147852827156639 / 0.11481218118869342`, and MRR
   `0.08101396652891538`. Score audit/import coverage passed with
   `candidate_rows=1010000`, `score_rows=1010000`, `matched_keys=1010000`,
   no missing/extra/duplicate keys, finite scores `1010000`, and
   `score_coverage_rate=1.0`. Row counts: `scores.csv` `1,010,001` lines,
   `predictions/rank_predictions.jsonl` `10,000` lines before post-gate
   cleanup, and `tables/ranking_eval_records.csv` `10,001` lines. Provenance
   has `implementation_status=official_completed`, `blockers=[]`,
   `score_coverage_rate=1.0`, no test-set model selection, and no extra
   baseline tuning. Post-gate cleanup preserved final scores, provenance,
   audits, imported tables, server-final audit, large-artifact manifest, SASRec
   checkpoint, upstream embedding, and compact adapter metadata. Because disk
   remained tight, only the server-side prediction JSONL was deleted with
   manifest
   `outputs/summary/tools_llm2rec_prediction_deleted_post_gate_20260605.{json,sha256}`
   and then the completed-row adapter staging CSVs were deleted with manifest
   `outputs/summary/tools_llm2rec_completed_adapter_staging_cleanup_20260605.{json,sha256}`.
   The deleted prediction had `10,000` lines, size `804,958,794` bytes, and
   sha256 `211a037a71020955e3488fdcd53f8d6710505bdad23e13c60e7a869d83e99148`;
   adapter staging cleanup removed `1,100,523,516` bytes. Post-cleanup disk is
   `11,772,899,328` bytes free / `95%` used. Tools is now 7/8 official rows
   gated; the remaining Tools official row is `llmesr_sasrec`.
   After a fresh no-process/GPU/disk/duplicate-output preflight, the final
   Tools official row `llmesr_sasrec` launched at 2026-06-05 18:52 CST with
   runner PID `3440278`, adapter PID `3440287`, log
   `baselines_new_domains_tools_llmesr_20260605_185250.log`, pidfile
   `outputs/summary/tools_llmesr_launch_20260605_185250.pid`, launch snapshot
   `outputs/summary/tools_llmesr_launch_monitor_20260605_185250.json`, and
   heartbeat `monitor-tools-llm-esr`. The 18:57 CST stable follow-up showed
   exactly one LLM-ESR adapter process, no training child yet, GPU `99%` /
   `16091 MiB`, embedding progress about `11288/269711`, no failure markers,
   final output placeholder-only, and adapter dir about `1005M`. Because
   adapter staging pushed disk below the strict 10 GiB guard, only non-project
   caches were cleaned (`.cache`, `.codex/.tmp`, Chrome GPU cache, and inactive
   `.cursor-server/bin` after verifying no Cursor process). No project
   evidence, checkpoints, embeddings, task splits, configs, source code, or
   other project files were deleted. Cache-cleanup record:
   `outputs/summary/tools_llmesr_launch_cache_cleanup_20260605.txt`.
   Post-cleanup disk was `10,866,053,120` bytes free / `95%` used. This
   LLM-ESR row is active monitor-only evidence until final score/provenance
   gates pass.
   A 2026-06-05 19:04-19:06 CST monitor/cache checkpoint found the same row
   still active with runner PID `3440278` and adapter PID `3440287`, duplicate
   counts exactly one adapter and zero training child, GPU about `91-96%` /
   `16189 MiB`, embedding progress about `47824/269711`, no final evidence,
   and no failure/OOM/no-space markers. Disk dipped just below the strict 10
   GiB guard (`10,726,035,456` bytes free), so only non-project Chrome
   cache/GPU cache and VSCode-server logs were cleared. No project outputs,
   evidence, checkpoints, embeddings, task splits, configs, source code, or
   other project files were deleted. Cleanup record:
   `outputs/summary/tools_llmesr_monitor_cache_cleanup_20260605_1904.txt`.
   Post-cleanup disk was `10,757,963,776` bytes free / `95%` used.
   A 2026-06-05 19:09-19:13 CST disk-emergency checkpoint found the active
   Tools `llmesr_sasrec` row still alive with runner PID `3440278` and adapter
   PID `3440287`, duplicate counts exactly one adapter and zero training child,
   GPU about `96%` / `16213 MiB`, embedding progress from about `59448/269711`
   to `77056/269711`, and no final evidence or failure markers. Disk was below
   the strict 10 GiB guard (`10,633,687,040` bytes free). No safe completed
   prediction deletion existed, and conda/pip/npm cache cleanup freed `0` bytes
   (`outputs/summary/tools_llmesr_monitor_conda_cache_cleanup_20260605_1909.txt`).
   After explicit destructive-action approval, the completed Tools LLM2Rec
   SASRec checkpoint was deleted only after verifying gated evidence and
   writing manifests:
   `outputs/summary/tools_llm2rec_completed_checkpoint_cleanup_for_llmesr_disk_20260605.{json,sha256}`.
   The deleted checkpoint was `5,665,876,357` bytes, sha256
   `8ad7ce0316befeb8ee6b3482546ffe3e301e42e9a6b1e10ee608689ea5ece414`.
   Final LLM2Rec scores, provenance, score audit, run summary, imported
   tables, server-final audit, large-artifact manifest, and local-light package
   were preserved; disk recovered to `16,293,425,152` bytes free / `92%` used.
   A 2026-06-05 19:55-20:01 CST monitor then found the active Tools
   `llmesr_sasrec` row still alive and unique with runner PID `3440278` and
   adapter PID `3440287`. The Qwen3 embedding pass completed
   (`269711/269711`) and the row entered official LLM-ESR training at
   `[llmesr] epoch=1 train_loss=1.398057`. There was still no final
   `scores.csv`, provenance, score audit, run summary, imported ranking table,
   or prediction JSONL, so this remains active monitor-only evidence. Disk was
   tight at `11,804,352,512` bytes free / `95%` used; a read-only storage audit
   found the main new pressure is the active adapter embedding
   `outputs/baselines/paper_adapters/tools_large10000_100neg_llmesr_official_adapter/llm_esr/handled/itm_emb_np.pkl`
   (`4.12G`), which is not safe to delete while training is active. Other large
   visible files are protected completed models/checkpoints/task splits or
   non-deletable legacy prediction evidence. No cleanup was performed.
   At 2026-06-05 21:19 CST, Tools `llmesr_sasrec` completed as the eighth
   Tools official row with wrapper markers `implementation_status=official_completed`,
   `blockers=0`, `DONE llmesr_sasrec on tools`, and
   `All baseline runs complete`. Server-final audit, server large-artifact
   manifest, local-light sync, and local-light audit all passed. Full metrics
   over 10,000 users and 101 candidates are HR@5/10/20
   `0.0711 / 0.1270 / 0.2219`, NDCG@5/10/20
   `0.042728964614829223 / 0.060602849768892623 / 0.08433244535733923`, and
   MRR `0.06334161303438132`; row counts are `scores.csv` `1,010,001`,
   prediction JSONL `10,000` before cleanup, and
   `tables/ranking_eval_records.csv` `10,001`. The local lightweight package is
   `outputs/baselines/official_adapters/tools_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/`.
   Post-gate cleanup removed only the completed server prediction JSONL and
   two adapter staging CSVs with manifest
   `outputs/summary/tools_llmesr_post_gate_cleanup_20260605.sha256`, preserving
   final scores, provenance, audits, imported tables, server-final certificate,
   `llmesr_official_model.pt`, and upstream embedding artifacts. The valid
   pre-cleanup Tools IRLLRec server-final certificate was restored from the
   local-light package after a post-cleanup failed audit had overwritten the
   server copy; the failed audit was preserved as
   `server_final_evidence_audit_post_cleanup_failed_20260605.json`.
   Tools C-CRP raw scores were imported into
   `outputs/tools_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate`
   with exact `score_coverage_rate=1.0`. The post-cleanup Tools domain gate
   `outputs/summary/tools_official_ccrp_gate_post_cleanup_20260605.{json,csv}`
   records `official_ok_count=8`, `official_all_ok=true`, `ccrp_ok=true`, and
   `gate_ok=true`. Tools C-CRP metrics are HR@5/10/20
   `0.1937 / 0.2696 / 0.3931`, NDCG@5/10/20
   `0.14186375906171483 / 0.16611553052793934 / 0.19703986741872317`, and MRR
   `0.15585924577949772`. The comparison/statistical package
   `outputs/summary/tools_official_ccrp_20260605_*` records
   `claim_gate=tools_domain_pass`: C-CRP is observed-best on all seven metrics,
   all 56 C-CRP-vs-8-official paired tests have positive deltas and are
   Holm-significant, `min_delta=0.0294`, `min_ci_low=0.0173`, and
   `max_holm_p_value=9.7172307634557e-07`. After local table sync and the
   passed domain/comparison gates, the server-only imported C-CRP prediction
   JSONL was removed with
   `outputs/tools_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate/prediction_deletion_manifest.json`;
   the post-cleanup gate still passes by certificate. This is a Tools-domain
   pass, not final paper readiness.
   A compact four-new-domain ARIS gate certificate was then built locally as
   `outputs/summary/new_domains_official_ccrp_cross_domain_20260605_*`. It
   covers Sports, Toys, Home, and Tools and records passing blocking checks:
   8/8 official-code-level baseline rows per domain, exact same-candidate gate
   status, C-CRP rank 1 on all seven metrics, and 56/56 positive
   Holm-significant C-CRP-vs-official paired tests per domain. The safe claim
   is limited to the four new 10k-user/101-candidate same-candidate domains.
   The certificate explicitly does not support paper-ready SOTA,
   full-catalog recommender SOTA, large-effect wording, universal winner
   wording, or a proven uncertainty-mechanism claim. It also records
   non-blocking local artifact gaps for Sports/Toys C-CRP raw event restat, so
   it is a comparison-gate summary rather than a self-contained local raw
   reproduction package.
   Home
   LLM2Rec completed at 2026-06-04 09:49 CST after a disk-full partial-copy
   recovery and passed score audit/import, server-final audit, server
   large-artifact manifest, lightweight local sync, and local-light audit.
   Full metrics over 10,000 users and 101 candidates are HR@5/10/20
   `0.0577 / 0.1101 / 0.2153`, NDCG@5/10/20
   `0.034207889197971464 / 0.05094946457092549 / 0.07719596686909931`, and
   MRR `0.0563865859396318`; `scores.csv` has `1,010,001` lines,
   predictions had `10,000` lines before post-gate deletion, and
   `tables/ranking_eval_records.csv` has `10,001` lines. The local
   lightweight package is
   `outputs/baselines/official_adapters/home_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate/`.
   The post-gate server prediction deletion manifest is
   `outputs/summary/home_llm2rec_prediction_deleted_after_gates_20260604.sha256`.
   The completed LLM2Rec intermediate adapter and upstream symlink were then
   removed with sha256 cleanup manifest
   `outputs/summary/home_llm2rec_completed_adapter_cleanup_manifest_20260604.sha256`,
   recovering `/` to about `12G` free / `95%` used without touching final
   evidence. Home `llmesr_sasrec` launched at 2026-06-04 10:14 CST as the
   eighth Home row and completed at 2026-06-04 13:09 CST with
   `implementation_status=official_completed`, `blockers=[]`, exact
   `score_coverage_rate=1.0`, score audit/imported tables, server-final audit,
   server large-artifact sha256 manifest, lightweight local sync, and
   local-light audit all passing. Full metrics over 10,000 users and 101
   candidates are HR@5/10/20 `0.0621 / 0.1163 / 0.2139`, NDCG@5/10/20
   `0.037993209299003045 / 0.055376101596196485 / 0.0797502336556021`, and
   MRR `0.059737054548523474`; `scores.csv` has `1,010,001` lines,
   predictions had `10,000` lines before post-gate deletion, and
   `tables/ranking_eval_records.csv` has `10,001` lines. The local
   lightweight package is
   `outputs/baselines/official_adapters/home_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/`.
   The post-gate prediction/staging cleanup manifest is
   `outputs/summary/home_llmesr_post_gate_cleanup_20260604.sha256`, and final
   scores/provenance/audits/imported tables/`llmesr_official_model.pt` were
   preserved. The Home C-CRP raw scores were then imported into
   `outputs/home_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate`
   with exact coverage and NDCG@10 `0.13239420796539653`. The Home domain gate
   wrote `outputs/summary/home_official_ccrp_gate_20260604.{json,csv}` with
   `official_ok_count=8`, `official_all_ok=true`, `ccrp_ok=true`, and
   `gate_ok=true`. The comparison/statistical package
   `outputs/summary/home_official_ccrp_20260604_*` records C-CRP rank 1 on all
   seven metrics, all 56 C-CRP-vs-official paired tests positive and
   Holm-significant, `min_delta=0.0336`, `min_ci_low=0.0216`, and
   `max_holm_p_value=1.0216927255359559e-08`. This is a Home-domain pass only,
   not a paper-wide SOTA claim. Toys
   `llmesr_sasrec` completed at 2026-06-02 18:59 CST after a disk-full
   recovery as `implementation_status=official_completed`, `blockers=[]`, and
   `score_coverage_rate=1.0`. Full metrics over 10,000 users and 101
   candidates are HR@5/10/20 `0.0637 / 0.1172 / 0.2203`,
   NDCG@5/10/20
   `0.037504900117522603 / 0.05456849726033091 / 0.08036871527121744`, and MRR
   `0.05844977379835533`; row counts are `scores.csv` `1,010,001`,
   predictions `10,000`, and `tables/ranking_eval_records.csv` `10,001`.
   Server-final audit, lightweight sync, and local-light audit passed. The
   local lightweight package is
   `outputs/baselines/official_adapters/toys_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/`.
   The final toys domain gate and comparison/statistical gate are under
   `outputs/summary/toys_official_gate_final_20260602_1900.*` and
   `outputs/summary/toys_official_ccrp_20260602_1900_*`. C-CRP is rank 1 on
   all seven toys metrics; all 56 paired tests are positive and
   Holm-significant. Current server disk recovered from about `5.9G` to `17G`
   free after removing only verified completed upstream staging data for
   sports/toys LLMEmb and LLM-ESR, with cleanup manifest
   `outputs/summary/upstream_completed_sports_toys_llmemb_llmesr_cleanup_manifest_20260602.sha256`.
   Final official evidence and local lightweight packages were preserved.
   Home `proex_profile` launched at 2026-06-02 19:45 CST with runner PID
   `3004208`, adapter PID `3004218`, and log
   `baselines_new_domains_home_proex_20260602_1950.log`. It completed at
   2026-06-02 22:00 CST as `implementation_status=official_completed` with
   `blockers=[]`, exact `score_coverage_rate=1.0`, server-final audit PASS,
   lightweight sync PASS, and local-light audit PASS. Full metrics over 10,000
   users and 101 candidates are HR@5/10/20=`0.0606/0.1177/0.2296`,
   NDCG@5/10/20=`0.03662857786324662/0.054867449700296195/0.08290060869107069`,
   and MRR=`0.05933326491258513`. Row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local lightweight
   package is
   `outputs/baselines/official_adapters/home_large10000_100neg_proex_profile_official_qwen3base_same_candidate/`;
   server-only large artifacts remain covered by
   `server_large_artifact_manifest.sha256`. After final evidence and local
   backup passed, the completed intermediate adapter was removed with cleanup
   manifest
   `outputs/summary/home_proex_completed_adapter_cleanup_manifest_20260602.sha256`,
   recovering disk from about `8.2G` to `16G` free without touching final
   evidence. Home `promax_profile` launched at 2026-06-02 22:14 CST with
   runner PID `3026043`, adapter PID `3026052`, and log
   `baselines_new_domains_home_promax_20260602_2215.log`. It completed at
   2026-06-03 02:53 CST as `implementation_status=official_completed` with
   `blockers=[]`, exact `score_coverage_rate=1.0`, server-final audit PASS,
   lightweight sync PASS, and local-light audit PASS. Full metrics over 10,000
   users and 101 candidates are HR@5/10/20=`0.0514/0.1019/0.2076`,
   NDCG@5/10/20=`0.030788292596664168/0.04691808776215203/0.07326077825489297`,
   and MRR=`0.053474908740382465`. Row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local lightweight
   package is
   `outputs/baselines/official_adapters/home_large10000_100neg_promax_profile_official_qwen3base_same_candidate/`;
   server-only large artifacts remain covered by
   `server_large_artifact_manifest.sha256`. After final evidence and local
   backup passed, the completed intermediate adapter was removed with cleanup
   manifest
   `outputs/summary/home_promax_completed_adapter_cleanup_manifest_20260602.sha256`,
   recovering disk from about `7.5G` to `15G` free without touching final
   evidence. Home `elmrec_graph` launched at 2026-06-03 03:02 CST with runner
   PID `3061705`, adapter PID `3061714`, and log
   `baselines_new_domains_home_elmrec_20260603_0302.log`. It completed at
   2026-06-03 05:47 CST as `implementation_status=official_completed` with
   `blockers=[]`, exact `score_coverage_rate=1.0`, server-final audit PASS,
   lightweight sync PASS, and local-light audit PASS. Full metrics over 10,000
   users and 101 candidates are HR@5/10/20=`0.0509/0.1021/0.2018`,
   NDCG@5/10/20=`0.029717257242599254/0.0460440741915887/0.0708856096588022`,
   and MRR=`0.05195852255617441`. Row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local lightweight
   package is
   `outputs/baselines/official_adapters/home_large10000_100neg_elmrec_graph_official_qwen3base_same_candidate/`.
   At that 2026-06-03 05:47 CST checkpoint, home had 3/8 completed official
   rows. At the 2026-06-03 05:48 CST
   checkpoint no Pony/C-CRP/baseline Python process was active, GPU was idle,
   and disk was tight at about `6.5G` free. The completed ElmRec intermediate
   adapter was removed after exact realpath checks and a 16-file sha256 cleanup
   manifest
   `outputs/summary/home_elmrec_completed_adapter_cleanup_manifest_20260603.sha256`;
   a post-cleanup server-final audit remained `ok=true`, and final scores,
   provenance, audits, predictions, imported tables, model, and local
   lightweight evidence were preserved. Disk recovered to about `14G` free.
   Home `llmemb` launched at 2026-06-03 06:08 CST as the fourth home row. The
   first run reached exact score export but failed during checkpoint save at
   `100%` disk, so the orphan score was quarantined and not imported as
   official evidence. After removing only failed-run/generated staging storage
   and patching LLMEmb handled embedding staging to symlink the large
   `itm_emb_np.pkl`, the true symlink rerun launched at 2026-06-03 09:16 CST
   with log `baselines_new_domains_home_llmemb_symlink_rerun_20260603_0920.log`.
   It completed as `implementation_status=official_completed` with
   `blockers=[]`, exact `score_coverage_rate=1.0`, server-final audit PASS,
   lightweight sync PASS, and local-light audit PASS. Full metrics over 10,000
   users and 101 candidates are HR@5/10/20=`0.1079/0.1856/0.3169`,
   NDCG@5/10/20=`0.06899578967097944/0.09390612986107003/0.12674255822842873`,
   and MRR=`0.09012268660291177`. Row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local lightweight
   package is
   `outputs/baselines/official_adapters/home_large10000_100neg_llmemb_official_qwen3base_same_candidate/`;
   server-only large artifacts remain covered by
   `server_large_artifact_manifest.sha256`. At that checkpoint, Home had 4/8
   completed official rows before IRLLRec. The completed LLMEmb intermediate adapter was then removed
   with cleanup manifest
   `outputs/summary/home_llmemb_completed_adapter_cleanup_manifest_20260603.sha256`;
   a post-cleanup server-final audit remained `ok=true`, and disk recovered to
   about `7.1G` free. Home `irllrec_intent` then launched at 2026-06-03 13:55
   CST as the fifth home row after a clean preflight; runner PID `3147646`,
   adapter PID `3147655`, log
   `baselines_new_domains_home_irllrec_20260603_1355.log`. At the 13:57 CST
   checkpoint it was CPU-side active with about `6.1G` free and no final
   scores/provenance/imported tables, so no fifth home result row was eligible
   at that checkpoint. At the 2026-06-03 17:21 CST heartbeat, the active row had completed
   Qwen3 embedding and reached official training epoch `1220`, but disk fell to
   about `30M` free. Emergency cleanup removed only the completed Toys LLM2Rec
   intermediate adapter and two server-only prediction JSONLs from already
   gated toys rows (`rlmrec_graphcl` and `irllrec_intent`) after sha256
   manifests; scores, provenance, audits, imported tables, models, and
   local-light packages were preserved. Disk recovered to about `2.0G`, and the
   active home IRLLRec process continued to epoch `1280` by 17:27 CST. The
   cleanup manifests are
   `outputs/summary/toys_llm2rec_completed_adapter_cleanup_manifest_20260603_irllrec_disk.sha256`
   and
   `outputs/summary/toys_predictions_deleted_for_home_irllrec_disk_20260603.sha256`.
   At the 17:43 CST follow-up, disk had slipped to about `1.7G`, so the
   already gated Toys ProEx server-only prediction JSONL was removed after
   confirming server-final and local-light audits; scores, provenance, audits,
   imported tables, models, and local-light packages were preserved. The
   additional manifest is
   `outputs/summary/toys_proex_prediction_deleted_for_home_irllrec_disk_20260603.sha256`;
   disk recovered to about `2.5G`, and active training reached epoch `1480` by
   17:45 CST. At the 18:08 CST follow-up, home IRLLRec remained active through
   epoch `1740` with no fatal/OOM/no-space markers, but disk was still tight.
   After confirming sports ProMax server-final audit `ok=true` and local-light
   audit `ok=true`, only the already gated server-side Sports ProMax prediction
   JSONL was removed with sha256 manifest
   `outputs/summary/sports_promax_prediction_deleted_for_home_irllrec_disk_20260603.sha256`.
   Sports ProMax scores, provenance, audits, imported tables, model, and
   local-light package were preserved; disk recovered to about `3.1G`.
   At the 18:45 CST continuation check, Home IRLLRec was still active through
   epoch `2140` with no fatal/OOM/no-space markers, but disk remained tight.
   After confirming server-final audit `ok=true`, local-light audit `ok=true`,
   `implementation_status=official_completed`, `blockers=[]`, and
   `score_coverage_rate=1.0` for completed Home `proex_profile`,
   `promax_profile`, `elmrec_graph`, and `llmemb`, only those four rows'
   server-side `predictions/rank_predictions.jsonl` files were removed under
   the documented disk-pressure exception. The sha256 manifest is
   `outputs/summary/home_completed_predictions_deleted_for_irllrec_disk_20260603.sha256`.
   Their scores, provenance, audits, imported tables, models, and local-light
   packages were preserved; disk recovered to about `6.0G`, and the active Home
   IRLLRec runner/adapter remained active at that checkpoint. Home
   `irllrec_intent` completed at
   2026-06-03 20:05 CST as `implementation_status=official_completed` with
   `blockers=[]`, exact `score_coverage_rate=1.0`, server-final audit PASS,
   lightweight sync PASS, and local-light audit PASS. Full metrics over 10,000
   users and 101 candidates are HR@5/10/20=`0.0821/0.1443/0.2878`,
   NDCG@5/10/20=`0.05108975290090002/0.07089752436733526/0.10662011242627113`,
   and MRR=`0.07424325424843974`. Row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local lightweight
   package is
   `outputs/baselines/official_adapters/home_large10000_100neg_irllrec_intent_official_qwen3base_same_candidate/`;
   server-only large artifacts remain covered by
   `server_large_artifact_manifest.sha256`. After final evidence and local
   backup passed, the completed IRLLRec intermediate adapter was removed with
   cleanup manifest
   `outputs/summary/home_irllrec_completed_adapter_cleanup_manifest_20260603.sha256`;
   a post-cleanup server-final audit remained `ok=true`, final scores,
   provenance, audits, predictions, imported tables, model, and local-light
   package were preserved, and disk recovered from about `4.7G` to `12G` free.
   Home `rlmrec_graphcl` launched at 2026-06-03 20:28 CST as the sixth home row
   after fresh process/GPU/disk and duplicate-output preflight, then completed
   at 2026-06-04 06:49 CST as `implementation_status=official_completed`,
   `blockers=[]`, and `score_coverage_rate=1.0`. The launch log was
   `baselines_new_domains_home_rlmrec_20260603_2028.log`; the final training
   line reached `[rlmrec-official] epoch=3000 train_loss=1.546134`. RLMRec
   passed server-final evidence audit, server large-artifact sha256 manifest,
   lightweight local sync, and local-light audit. Full metrics over 10,000
   users and 101 candidates are HR@5/10/20=`0.0685/0.1268/0.2451`,
   NDCG@5/10/20=`0.04126795456098191/0.059869974785318684/0.08932211717259728`,
   and MRR=`0.06397404670906748`. Row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local lightweight
   package is
   `outputs/baselines/official_adapters/home_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate/`;
   server-only large artifacts remain covered by
   `server_large_artifact_manifest.sha256`. Home now has 6/8 completed official
   rows; remaining Home rows are `llm2rec_sasrec` and `llmesr_sasrec`. Home is
   still not domain-gate eligible until all eight official rows and C-CRP
   imported evidence pass the same checks. Before launching Home LLM2Rec, the
   completed non-final RLMRec intermediate adapter was removed after cleanup
   manifest
   `outputs/summary/home_rlmrec_completed_adapter_cleanup_manifest_20260604.sha256`,
   exact realpath check, and post-cleanup RLMRec server-final audit PASS; final
   RLMRec evidence was preserved. Disk recovered to about `19G` free / `91%`
   used. Home `llm2rec_sasrec` then launched at 2026-06-04 07:19 CST as the
   seventh home row with runner PID `3236678`, adapter PID `3236688`, and log
   `baselines_new_domains_home_llm2rec_20260604_071902.log`. At the first
   stable check it was in Qwen3 `hf_mean_pool` embedding at about `944/568891`,
   GPU was about `96%`, and `/` was about `18G` free / `91%` used. It is not
   table-eligible until final score/provenance/import/server-final,
   large-artifact manifest, local-light sync, local-light audit, and full
   metric/row-count gates pass.
   Historical sports run record: sports started from
   `baselines_new_domains_sports.log` with runner PID `2794722`; the active
   child at the 2026-05-31 22:32 CST checkpoint was sports `llmemb` PID
   `2794731`. It had completed the `hf_mean_pool` embedding pass
   (`233470/233470`) and entered `llmemb-sasrec` training, with log lines
   through epoch 35. By the 2026-05-31 22:52 CST checkpoint,
   `llmemb-sasrec` had completed 200 epochs and the `llmemb` training phase
   had reached epoch 175. GPU was about `83%`, `16301 MiB / 49140 MiB`; disk
   was about `36G` free. At 2026-05-31 22:56 CST, sports `llmemb` completed
   as `official_completed` with `blockers=[]`, `score_coverage_rate=1.0`, and
   exact audit `audit_ok=True`. Metrics: HR@5/10/20=`0.2124/0.3384/0.4900`,
   NDCG@5/10/20=`0.1388527216/0.1795004215/0.2176868359`,
   MRR=`0.1538831336`, `n_users=10000`, `score_rows=1010000`, and
   `candidate_rows=1010000`. Lightweight local evidence is under
   `outputs/baselines/official_adapters/sports_large10000_100neg_llmemb_official_qwen3base_same_candidate/`;
   server-only large artifacts remain on the server. The runner then advanced
   to sports `proex_profile`. At 2026-06-01 00:25 CST, sports `proex_profile`
   also completed as `official_completed` with `blockers=[]`,
   `score_coverage_rate=1.0`, and exact audit `audit_ok=True`. Metrics:
   HR@5/10/20=`0.0821/0.1527/0.2777`,
   NDCG@5/10/20=`0.0516826556/0.0741722663/0.1054064715`,
   MRR=`0.0742689715`, `sample_count=10000`, `score_rows=1010000`, and
   `candidate_rows=1010000`. Lightweight local evidence is under
   `outputs/baselines/official_adapters/sports_large10000_100neg_proex_profile_official_qwen3base_same_candidate/`;
   server-only large artifacts remain on the server. At 2026-06-01 03:04 CST,
   sports `promax_profile` completed as `official_completed` with
   `blockers=[]`, `score_coverage_rate=1.0`, and exact audit `audit_ok=True`.
   Metrics: HR@5/10/20=`0.0825/0.1387/0.2370`,
   NDCG@5/10/20=`0.0541847954/0.0721533411/0.0967593591`,
   MRR=`0.0741052747`, `sample_count=10000`, `score_rows=1010000`, and
   `candidate_rows=1010000`. Lightweight local evidence is under
   `outputs/baselines/official_adapters/sports_large10000_100neg_promax_profile_official_qwen3base_same_candidate/`;
   server-only large artifacts remain on the server. The runner advanced to
   sports `elmrec_graph`; at the 2026-06-01 03:05 CST checkpoint disk was
   about `20G` free (`90%` used), so storage is a close watch item but not yet
   a blocker. At 2026-06-01 04:37 CST, sports `elmrec_graph` completed as
   `official_completed` with `blockers=[]`, `score_coverage_rate=1.0`, and
   exact audit `audit_ok=True`. Metrics: HR@5/10/20=`0.0532/0.1054/0.2013`,
   NDCG@5/10/20=`0.0317045493/0.0483716358/0.0723504733`,
   MRR=`0.0537009851`, `sample_count=10000`, `score_rows=1010000`, and
   `candidate_rows=1010000`. Lightweight local evidence is under
   `outputs/baselines/official_adapters/sports_large10000_100neg_elmrec_graph_official_qwen3base_same_candidate/`;
   server-only large artifacts remain on the server. The runner advanced to
   sports `irllrec_intent`; at the 2026-06-01 04:38 CST checkpoint disk was
   about `15G` free (`93%` used), so storage is a close watch item but no
   space/OOM/CUDA failure has been observed.
   At 2026-06-01 05:31 CST, local lightweight evidence for the four completed
   sports official rows was expanded to include inspect provenance, JSON/TXT
   score audits, run summaries, imported summary/metric/coverage/exposure
   tables, and per-event `tables/ranking_eval_records.csv`; local/server
   size and line-count checks matched. The four completed methods' server
   `outputs/baselines/paper_adapters/` working directories were then removed
   after path checks, while final scores/provenance/audits/tables/predictions
   and checkpoints were preserved. Disk recovered to about `33G` free.
   At 2026-06-01 06:36 CST, the same runner remains active on sports
   `irllrec_intent` (child PID `2835275`), with log progress through epoch
   `1190/3000`, latest train loss `0.625393`, GPU about `98%`, `16295 MiB /
   49140 MiB`, disk about `29G` free (`85%` used), and no fatal/OOM/no-space
   markers. Four sports official rows are complete and locally backed up;
   `irllrec_intent`, `rlmrec_graphcl`, `llm2rec_sasrec`, and
   `llmesr_sasrec` are still incomplete. The local package gate
   `scripts/audit/main_audit_official_evidence_package.py` passed on all four
   completed lightweight evidence packages and is now the required check
   before recording future copied official evidence as backed up.
   At 2026-06-01 06:44 CST, the same gate was run in `server_final` mode on
   `pony-rec-gpu` for the four completed sports output directories; all four
   passed, including final server-side `scores.csv`, predictions, provenance,
   full metrics, coverage/exposure tables, and per-event evaluation records.
   The active `irllrec_intent` process was left untouched.
   At 2026-06-01 07:18 CST, sports `irllrec_intent` remained active under
   runner PID `2794722` and child PID `2835275`; the log had reached epoch
   `2040/3000`, latest train loss `0.624872`, GPU was about `75%` with
   `16295 MiB / 49140 MiB`, disk remained about `29G` free (`85%` used), and
   fatal/OOM/CUDA/no-space/runtime-error scans remained clean. No fifth sports
   official row is final yet, so no new paper-facing metric row has been added.
   To prevent local evidence omissions after future rows finish,
   `scripts/audit/main_sync_official_evidence_package.py` now performs
   allowlist-based lightweight sync and size/sha256 verification before the
   `local_light` evidence audit.
   At 2026-06-01 07:32 CST, sports `irllrec_intent` was still active at epoch
   `2320/3000` with latest train loss `0.625049`; GPU/memory were about
   `69%` and `16295 MiB / 49140 MiB`, disk remained about `29G` free, and
   fatal scans were clean. No final IRLLRec score/provenance/table package
   exists yet, so paper-facing sports official evidence remains four completed
   rows.
   At 2026-06-01 08:10 CST, sports `irllrec_intent` became the fifth completed
   sports official row: `implementation_status=official_completed`,
   `blockers=[]`, `score_coverage_rate=1.0`, and exact score coverage passed.
   Metrics are HR@5/10/20=`0.1573/0.2215/0.4016`,
   NDCG@5/10/20=`0.10642150916142634/0.12691703149297534/0.17128490034441315`,
   and MRR=`0.12444202662842994` over 10,000 users and 101 candidates.
   Server-final audit, local lightweight sync, and local-light audit all
   passed; the row is eligible as official sports evidence, pending the full
   eight-baseline sports comparison and paired tests. The runner then advanced
   to sports `rlmrec_graphcl`. At 2026-06-01 08:24 CST, RLMRec remained active
   under runner PID `2794722` and child PID `2851207`, generating Qwen
   embeddings at `57992/233470`; GPU was about `95%` with
   `16285 MiB / 49140 MiB`, disk was `32G` free, and fatal/OOM/CUDA/no-space
   scans remained clean. A read-only scan found one empty malformed output
   directory named
   `outputs/sports_large10000_100neg_TRAIN_METHODS_OVERRIDE=_official_qwen3base_same_candidate/`.
   It contains no files and is not evidence; the local runner now validates
   method tokens before creating output directories so a misquoted override
   cannot create another malformed method directory. At 2026-06-01 08:34 CST,
   RLMRec was still in Qwen embedding generation at `105376/233470` with no
   fatal/OOM/CUDA/no-space markers; its output directory still had only
   `inspect_fairness_provenance.json`, so no sixth sports official result is
   available yet. At 2026-06-01 09:05 CST, RLMRec reached Qwen embedding
   progress `233470/233470`, but child PID `2851207` was still active after
   embedding completion and the output directory still contained only
   `inspect_fairness_provenance.json`; no final `scores.csv`, provenance,
   audit, run summary, or imported metrics table exists yet, so the sports
   official evidence count remains five completed rows. At 2026-06-01 09:08
   CST, RLMRec entered official training and logged epoch 10 train loss
   `1.675038`; GPU was `100%` with `19943 MiB / 49140 MiB`, disk was `28G`
   free, and no final artifacts existed yet. The zero-degree graph
   normalization warning is recorded as non-fatal because the implementation
   maps infinite inverse degrees to `0.0`, but final eligibility still depends
   on successful score/provenance/audit/table outputs. At 2026-06-01 09:19
   CST, RLMRec was still training at epoch 140 with train loss `1.490221`,
   clean fatal/OOM/no-space scans, and only `inspect_fairness_provenance.json`
   in the output directory; the sports official evidence count remains five.
   At 2026-06-01 09:53 CST, RLMRec passed epoch 500 (`train_loss=1.480699`)
   and reached epoch 510 (`train_loss=1.482085`) with clean fatal/OOM/CUDA/
   no-space scans and stable `28G` free disk. No final `scores.csv`, final
   provenance, audit, run summary, or imported metrics table exists yet, so
   the sports official evidence count remains five. At 2026-06-01 10:41 CST,
   RLMRec passed epoch 1000 (`train_loss=1.477797`) and reached epoch 1030
   (`train_loss=1.478778`) with clean fatal/OOM/CUDA/no-space scans and stable
   `28G` free disk. The output directory still contained only
   `inspect_fairness_provenance.json`, so the sports official evidence count
   remains five. At 2026-06-01 12:10 CST, RLMRec passed epoch 2000
   (`train_loss=1.476514`) with clean fatal/OOM/CUDA/no-space scans and stable
   `28G` free disk. The output directory still contained only
   `inspect_fairness_provenance.json`; no final score/provenance/audit/table
   package existed yet, so the sports official evidence count remained five.
   At 2026-06-01 13:50 CST, sports `rlmrec_graphcl` completed and passed all
   row gates: `implementation_status=official_completed`, `blockers=[]`,
   `score_coverage_rate=1.0`, server-final audit PASS, lightweight sync PASS,
   and local-light audit PASS. Metrics over 10,000 users and 101 candidates:
   HR@5/10/20=`0.1212/0.1879/0.3009`,
   NDCG@5/10/20=`0.078580389191345/0.10001773336299705/0.12818232277286493`,
   MRR=`0.09720456858848743`, `sample_count=10000`,
   `avg_candidates=101.0`, `score_rows=1010000`, and
   `candidate_rows=1010000`. Server row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). Sports now has six
   completed official rows, but no sports SOTA claim is allowed until LLM2Rec,
   LLM-ESR, the complete eight-baseline comparison table, and paired tests
   finish. The runner then reached sports `llm2rec_sasrec` and stopped during
   adapter export because validation candidate histories were incorrectly
   looked up in the test-task train file. The local fix uses
   `valid_same_candidate/train_interactions.csv` for validation histories and
   passed a targeted unit test; it still needs server pull and resumed LLM2Rec
   execution before any LLM2Rec row is eligible. At 2026-06-01 14:03 CST, the
   fix had been deployed to the dirty server worktree without resetting
   unrelated experiment changes, and sports `llm2rec_sasrec` was resumed as a
   single-row job. The real sports run passed the previous adapter-export
   blocker and entered Qwen3 `hf_mean_pool` embedding generation at about
   `3432/283760` under PID `2870575`. This is progress only; the LLM2Rec row
   remains ineligible until final scores, provenance, audits, complete metrics,
   and row-count checks pass. At 2026-06-01 15:14 CST, sports
   `llm2rec_sasrec` had completed full Qwen3 item embedding
   (`283760/283760`) and produced matching 4,649,140,352-byte embedding files
   under the Pony adapter directory and the upstream LLM2Rec
   `item_info/SportsSameCandidate100Neg/` path. The run then stopped before
   official SASRec training with
   `FileNotFoundError: [Errno 2] No such file or directory: 'python'` from
   `_train_with_official_entrypoint`. This is a wrapper environment bug, not
   a completed or failed metric row. Local fix changes the subprocess command
   to start with `sys.executable` while preserving the official
   `evaluate_with_seqrec.py` entrypoint and SASRec arguments; targeted local
   tests passed (`tests/test_llm2rec_upstream_adapter.py`: 5 passed,
   `tests/test_llm2rec_same_candidate_export.py`: 3 passed). The LLM2Rec row
   remains not paper/table eligible until the server resume produces final
   `scores.csv`, provenance, score audit, imported full `@5/@10/@20 + MRR`
   metrics, row-count checks, server-final audit, and local-light evidence
   audit. At 2026-06-01 15:51 CST, the fixed runner had been verified on the
   server and sports `llm2rec_sasrec` was active again: adapter PID `2875446`
   and upstream official `evaluate_with_seqrec.py` PID `2875559`. The upstream
   official command uses the existing Qwen3 embedding file rather than
   regenerating it. The training log reached epoch 15 validation and saved
   checkpoints at epochs 5 and 10. This is a running checkpoint only; no final
   LLM2Rec row, metric table, provenance, or score audit is available yet.
   At 2026-06-01 15:56 CST, sports `llm2rec_sasrec` completed as an official
   row with `implementation_status=official_completed`, `blockers=[]`,
   `score_coverage_rate=1.0`, server-final audit PASS, lightweight sync PASS,
   and local-light audit PASS. Full same-candidate metrics over 10,000 users
   and 101 candidates are HR@5/10/20=`0.1105/0.206/0.3657`,
   NDCG@5/10/20=`0.06514778914391295/0.09566791850988236/0.13561659669926907`,
   and MRR=`0.08828933028385053`. Row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). Sports now has seven
   completed official rows; no sports SOTA claim is allowed until LLM-ESR,
   the full eight-baseline table, and paired tests finish. At 2026-06-01
   16:28 CST, the completed LLM2Rec intermediate adapter directory was removed
   after server-final/local-light audit checks, recovering about `5.3G`
   without touching final LLM2Rec evidence or the upstream embedding. Sports
   `llmesr_sasrec` was then launched as the eighth sports official row under
   runner PID `2877443` and adapter PID `2877452`; it is in Qwen3
   `hf_mean_pool` embedding at about `51472/233470`. This is a running
   checkpoint only: LLM-ESR has no final scores/provenance/imported metrics
   yet and is not table-eligible. At 2026-06-01 16:50 CST, LLM-ESR was still
   running under the same PIDs, with embedding progress about
   `141696/233470`, GPU sample `95%`, `16285 MiB / 49140 MiB`, and about
   `22G` free disk. No LLM-ESR `scores.csv`, final provenance, score audit,
   imported table, or predictions exist yet. The completed sports rows were
   also rechecked for the user's full-metric concern: the seven completed rows
   all have HR@5/@10/@20, NDCG@5/@10/@20, MRR, `sample_count=10000`,
   `avg_candidates=101.0`, exact 1,010,000/1,010,000 score coverage, final
   provenance, score audit, and imported `ranking_eval_records.csv`. The four
   earliest rows (`llmemb`, `proex_profile`, `promax_profile`, and
   `elmrec_graph`) lacked only the newer standardized
   `server_final_evidence_audit.json`; that audit was backfilled with
   `ok=true` on the server and local-light audits passed after copying the new
   JSONs into the local lightweight packages. No scores changed. At
   2026-06-01 17:15 CST, LLM-ESR completed Qwen3 embedding
   (`233470/233470`) and entered official training, with logged losses
   epoch 1 `1.374167` and epoch 5 `0.361412`. The row is still not
   table-eligible: no final LLM-ESR `scores.csv`, final provenance, score
   audit, predictions, imported tables, or local lightweight evidence package
   exists yet. Disk is a watch item at about `15G` free (`93%` used), but a
   read-only storage review found no meaningful safe cleanup while the active
   LLM-ESR adapter is running and completed-row final evidence is protected.
   At 2026-06-01 18:42 CST, sports `llmesr_sasrec` completed as the eighth
   sports official row with `implementation_status=official_completed`,
   `blockers=[]`, and `score_coverage_rate=1.0`. Server-final audit,
   lightweight sync, and local-light audit passed. Full metrics over 10,000
   users and 101 candidates are HR@5/10/20=`0.0916/0.1564/0.2650`,
   NDCG@5/10/20=`0.054919833257876506/0.0758115528438973/0.10310478593304104`,
   and MRR=`0.0751149958885503`. Row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). Sports now has all
   eight official rows complete, but sports SOTA is still not claimable until
   the full sports comparison table and paired tests are built. The completed
   LLM-ESR intermediate adapter directory was removed only after final evidence
   and local lightweight backup passed, recovering disk from `9.4G` to `14G`
   free without touching final outputs. At 2026-06-01 19:08 CST, a read-only
   domain gate with `scripts/audit/main_audit_domain_official_gate.py`
   generated `outputs/summary/sports_official_ccrp_gate_20260601.json` and
   `.csv` on the server and local. The gate returned
   `official_ok_count=8`, `ccrp_ok=true`, and `gate_ok=true`: all eight
   official rows plus `ccrp_v3_qwen3base_pointwise` have complete
   HR@5/@10/@20, NDCG@5/@10/@20, MRR, 10,000 users, 101 candidates,
   `score_coverage_rate=1.0`, expected row counts, and no gate failures.
   This confirms sports result completeness, not a SOTA claim; paired tests
   and the full comparison table remain required before any winner wording.
   At 2026-06-01 19:20 CST, `scripts/experiments/main_build_domain_official_comparison.py`
   built that sports comparison/statistical gate from existing server
   artifacts. The generated local/server outputs are
   `outputs/summary/sports_official_ccrp_20260601_comparison.csv`,
   `outputs/summary/sports_official_ccrp_20260601_comparison.md`,
   `outputs/summary/sports_official_ccrp_20260601_paired_tests.csv`, and
   `outputs/summary/sports_official_ccrp_20260601_paired_summary.json`.
   C-CRP is rank 1 and observed-best on all seven full metrics. The paired
   family has 56 tests (8 official baselines x HR@5/@10/@20, NDCG@5/@10/@20,
   and MRR), all with 10,000 paired events, positive deltas, paired-bootstrap
   95% CIs above zero, and Holm-significant p-values. Against the strongest
   official row (`llmemb`), the closest margin is HR@20 delta `0.0272` with
   CI `[0.0164, 0.0386]` and Holm p `1.219129314796352e-06`. This supports a
   sports-domain passed gate only; paper-wide SOTA wording still requires the
   declared domain set and ARIS review.
2. Import and audit each remaining new-domain baseline row with exact score coverage,
   full @5/@10/@20 metrics, provenance, and row-count checks. At
   2026-06-01 19:48 CST, the next-domain phase began with a storage preflight:
   disk was only `14G` free, so disposable user caches under
   `/home/ajifang/.cache` were removed after path verification, recovering
   about `5G` and leaving about `19G` free without deleting project outputs or
   protected evidence. Toys `proex_profile` was then launched as a
   single-domain/single-method official row with runner PID `2893793`, adapter
   PID `2893803`, PID file `baselines_new_domains_toys_proex.pid`, and log
   `baselines_new_domains_toys_proex_20260601_194414.log`. At the 19:48 check
   it was in Qwen3 `hf_mean_pool` embedding at about `7088/215034`, with GPU
   `95%` and disk about `18G` free. This row is running and not yet
   table-eligible; do not include it in claims until final score/provenance,
   audits, imported full metrics, row counts, local-light sync, and paired
   evidence gates pass. At 2026-06-01 21:13 CST, toys `proex_profile`
   completed as `official_completed` with `blockers=[]` and
   `score_coverage_rate=1.0`. Server-final audit, lightweight sync, and
   local-light audit passed. Full metrics over 10,000 users and 101 candidates
   are HR@5/10/20=`0.0895/0.1615/0.3017`,
   NDCG@5/10/20=`0.058141214365017416/0.0810170703641553/0.11607709818340411`,
   and MRR=`0.08121671352544663`; row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local package is
   `outputs/baselines/official_adapters/toys_large10000_100neg_proex_profile_official_qwen3base_same_candidate/`.
   A server-side sha256 manifest records the final large files while keeping
   `scores.csv`, predictions, and `proex_official_model.pt` server-only. The
   completed intermediate adapter directory was removed after audits, recovering
   disk from about `14G` to `18G` free without touching final evidence. Toys
   now has 1/8 completed official baselines. At 2026-06-01 21:31 CST, toys
   `promax_profile` was running as the next single-row official baseline under
   runner PID `2899989` and adapter PID `2899998`, with log
   `baselines_new_domains_toys_promax_20260601_212808.log`; it was in Qwen3
   embedding at about `1312/215034` and is not table-eligible until the same
   gates pass. At 2026-06-02 00:02 CST, toys `promax_profile` completed as
   `official_completed` with `blockers=[]` and `score_coverage_rate=1.0`.
   Server-final audit, lightweight sync, and local-light audit passed. Full
   metrics over 10,000 users and 101 candidates are
   HR@5/10/20=`0.0920/0.1435/0.2416`,
   NDCG@5/10/20=`0.06289618254810064/0.07937554863319267/0.10387644003990415`,
   and MRR=`0.08184625622431366`; row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local package is
   `outputs/baselines/official_adapters/toys_large10000_100neg_promax_profile_official_qwen3base_same_candidate/`.
   A server-side sha256 manifest records final large files while keeping
   `scores.csv`, predictions, and `promax_official_model.pt` server-only. The
   completed intermediate adapter directory was removed after audits, recovering
   disk from about `13G` to `17G` free without touching final evidence. Toys
   now has 2/8 completed official baselines. At 2026-06-02 00:10 CST, toys
   `elmrec_graph` was running as the next single-row official baseline under
   runner PID `2906447` and adapter PID `2906455`, with log
   `baselines_new_domains_toys_elmrec_20260602_000729.log`; at the 2026-06-02
   00:15 CST monitor check it was still active and had advanced to about
   `21872/215034` Qwen3 embeddings, with no fatal/OOM/no-space markers. It is
   not table-eligible until the same gates pass. At 2026-06-02 01:36 CST,
   toys `elmrec_graph` completed as `official_completed` with `blockers=[]` and
   `score_coverage_rate=1.0`. Server-final audit, lightweight sync, and
   local-light audit passed. Full metrics over 10,000 users and 101 candidates
   are HR@5/10/20=`0.0545/0.1043/0.2013`,
   NDCG@5/10/20=`0.03259298673054038/0.04856005753116525/0.07278039157879498`,
   and MRR=`0.05431081812612059`; row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local package is
   `outputs/baselines/official_adapters/toys_large10000_100neg_elmrec_graph_official_qwen3base_same_candidate/`.
   A server-side sha256 manifest records final large files while keeping
   `scores.csv`, predictions, and `elmrec_official_model.pt` server-only. The
   completed intermediate adapter directory was removed after audits, recovering
   disk from about `12G` to `16G` free without touching final evidence. At
   2026-06-02 03:04 CST, toys `llmemb` completed as the fourth official row
   with `implementation_status=official_completed`, `blockers=[]`, and
   `score_coverage_rate=1.0`. Server-final and local-light audits passed with
   full metrics over 10,000 users and 101 candidates: HR@5/10/20
   `0.2499 / 0.3505 / 0.4866`, NDCG@5/10/20
   `0.17252113274887534 / 0.20485045979333913 / 0.23905481091819092`, and MRR
   `0.1813804118284203`. Row counts passed for `scores.csv` (`1,010,001`
   lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local lightweight
   package is
   `outputs/baselines/official_adapters/toys_large10000_100neg_llmemb_official_qwen3base_same_candidate/`;
   final `scores.csv`, predictions, and `llmemb_official_model.pt` remain
   server-only and covered by `server_large_artifact_manifest.sha256`. After
   audits and local sync passed, the completed intermediate adapter directory
   was removed, recovering disk from about `4.0G` to `8.3G` free without
   touching final evidence. Toys now has 4/8 completed official baselines. At
   2026-06-02 03:16 CST, toys `irllrec_intent` was launched as the fifth
   single-row official baseline under runner PID `2923429` and adapter PID
   `2923437`, with log
   `baselines_new_domains_toys_irllrec_20260602_031623.log`; at the 03:19 CST
   check it was in Qwen3 `hf_mean_pool` embedding at about `1400/215034`, with
   GPU `96%` and disk about `7.3G` free. This row is running and not
   table-eligible until final score/provenance, audits, imported full metrics,
   row counts, local-light sync, and paired evidence gates pass. At 04:12 CST,
   after embeddings completed and training reached epoch 30, disk had fallen to
   about `4.0G` free; only disposable pip cache/temp paths were removed after
   scope checks, recovering disk to about `4.4G` free without touching project
   evidence or active adapters. At 04:35 CST, IRLLRec was still running at
   epoch `500/3000`; GPU was active and disk remained about `4.4G` free. The
   final evidence directory still lacked scores/provenance/audits/imported
   tables, so toys remains 4/8 complete. A read-only cleanup audit did not find
   another safe project artifact to remove: the old
   `books_large10000_100neg_llmesr_adapter` contains historical adapter mapping
   data while the corresponding final books directory is table-only and no
   local lightweight package was found, so it was left intact. At 04:47 CST,
   IRLLRec had reached `epoch=760/3000` and still had no final evidence
   package. Three user-level cache directories
   (`.vscode-server/data/CachedExtensionVSIXs`, Chrome `component_crx_cache`,
   and Code `CachedData`) were removed after realpath allowlist checks,
   recovering disk from about `4.4G` to `4.6G` free without touching project
   outputs, active adapters, models, Python site-packages, or other projects.
   At 04:56 CST, IRLLRec had reached `epoch=940/3000` with no final evidence
   files and a clean error scan. Five inactive VSCode remote server cache
   directories under `.vscode-server/cli/servers/Stable-*` were removed after
   a no-process check and realpath prefix verification, recovering disk to
   about `6.4G` free without touching project outputs, final evidence, active
   adapters, models, conda/Python environments, or other projects.
   At 2026-06-02 06:35 CST, toys `irllrec_intent` completed as the fifth
   official row with `implementation_status=official_completed`, `blockers=[]`,
   and `score_coverage_rate=1.0`. Server-final audit and local-light audit
   passed with full metrics over 10,000 users and 101 candidates: HR@5/10/20
   `0.1565 / 0.2293 / 0.4098`, NDCG@5/10/20
   `0.11049209461545026 / 0.13380144693674725 / 0.1785851471792316`, and MRR
   `0.1311986744710446`. Row counts passed for `scores.csv` (`1,010,001`
   lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local lightweight
   package is
   `outputs/baselines/official_adapters/toys_large10000_100neg_irllrec_intent_official_qwen3base_same_candidate/`;
   final `scores.csv`, predictions, and `irllrec_official_model.pt` remain
   server-only and covered by `server_large_artifact_manifest.sha256`. After
   audits and local sync passed, the completed IRLLRec intermediate adapter
   directory was removed, recovering disk from about `5.4G` to `9.7G` free
   without touching final evidence. Toys now has 5/8 completed official
   baselines. At 2026-06-02 06:44 CST, toys `rlmrec_graphcl` was launched as
   the next single-row official baseline with runner PID `2937284`, adapter PID
   `2937292`, and log `baselines_new_domains_toys_rlmrec_20260602_064443.log`;
   at the 06:47 CST check it was in Qwen3 `hf_mean_pool` embedding at about
   `1664/215034`, GPU `99%`, and disk about `8.8G` free. This row is running
   and not table-eligible until final score/provenance, audits, imported full
   metrics, row counts, local-light sync, and paired evidence gates pass.
3. Full @5/@10/@20 comparison table across all domains
4. Statistical significance tests (paired t-test, 20+ seeds or bootstrap)
5. Paper writing
6. GPT-5.5/Codex review cycle (target: 8/10)
