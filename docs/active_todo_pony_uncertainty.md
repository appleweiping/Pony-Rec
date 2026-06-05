# Uncertainty Active TODO

Last updated: 2026-06-06 02:25 CST

This is the cumulative execution TODO for the active Uncertainty goal. It is a
handoff artifact, not a claim of paper readiness. Update it after each completed
official row, blocker, cleanup decision, comparison-table build, or review
cycle.

## Hard Invariants

- Do not stop, restart, or duplicate active server experiments unless a verified
  blocker requires an explicit recovery decision.
- GPU/CPU experiments run on `pony-rec-gpu`, not on the local workstation.
- No toy rows enter paper-facing tables: full-scale means 10,000 users and 101
  candidates per user under the same-candidate protocol.
- Primary official-baseline rows require final provenance with
  `implementation_status=official_completed`, `blockers=[]`, exact score
  coverage `1.0`, complete `@5/@10/@20 + MRR` metrics, and row-count checks.
- Local evidence packages are lightweight: keep useful audit/provenance/tables
  locally, keep huge `scores.csv`, predictions, checkpoints, and embeddings
  server-side unless a recovery/archive decision says otherwise.
- Every important status change must update shared memory and canonical project
  docs, then commit and push from the local repository.
- Current multi-agent routing: Claude reviewer tooling is unavailable in this
  thread. When collaboration/review is required, use GPT-5.5 xhigh sub-agents.

## Paper-Critical Readiness Addendum (2026-06-03)

Working display title: **Actionable Uncertainty for LLM-based Recommendation**.
This replaces the more engineering-sounding "Task-Grounded Uncertainty for
LLM-based Recommendation" as the repo/paper-facing name, while preserving the
frozen technical claim: task-grounded calibrated uncertainty under controlled
same-candidate evaluation.

The paper is not ready after official-baseline completion alone. Before moving
to final writing or claiming readiness, add and gate these top-priority modules:

1. Observation/motivation study: show why uncertainty is needed in this
   framework using representative existing baselines under fair settings. This
   can use a few representative completed baselines/domains rather than every
   method/domain, but it must produce a paper-ready figure or table and record
   data paths, metrics, commands, row counts, provenance notes, and the git
   commit. Script entry:
   `scripts/analysis/main_build_uncertainty_observation_study.py`. Current
   blocker to resolve before running it for Sports/Toys: the already imported
   C-CRP final `scores.csv` files are score-only, so a signal/score row file
   with a real uncertainty column must be located or regenerated without LLM
   re-query leakage. Read-only server discovery on 2026-06-03 found that the
   only current `ccrp_selected_test_scored_rows.csv`/`ccrp_internal_provenance`
   pair is the older Beauty supplementary smaller-N run under
   `outputs/summary/week8_large10000_100neg_ccrp_formal/beauty/`; Sports,
   Toys, Home, and Tools formal C-CRP directories currently expose only
   score-only `scores.csv`, `report.json`, and `user_ranks.jsonl` plus imported
   ranking tables. Do not use those score-only files as uncertainty evidence.
   Audit helper:
   `scripts/audit/main_audit_ccrp_uncertainty_sources.py`, which classifies
   candidate files as `paper_ready_uncertainty_rows`,
   `recomputable_signal_rows`, or `score_only_not_uncertainty` and checks
   candidate-key coverage when `candidate_items.csv` is supplied.
   Remote stdin preflight on 2026-06-04 with the fixed-filter discovery wrapper
   confirmed the visible full-scale new-domain C-CRP artifacts are only
   `outputs/{sports,toys,home,tools}_large10000_100neg_ccrp_v3/scores.csv`.
   Header discovery and a broader token sweep found no paper-ready or
   recomputable signal rows for those four domains. Full audits against each
   domain's test `candidate_items.csv` confirmed every C-CRP score file has
   1,010,000 rows, 10,000 events, exact candidate-key coverage `1.0`, and
   status `score_only_not_uncertainty` with failure
   `missing_uncertainty_column`. Evidence files:
   `outputs/summary/paper_critical/ccrp_uncertainty_source_discovery_fullscale_fixed_filter_20260604_0458.*`,
   `outputs/summary/paper_critical/ccrp_uncertainty_source_discovery_fullscale_broad_fixed_filter_20260604_0459.*`,
   `outputs/summary/paper_critical/ccrp_uncertainty_source_discovery_projectroot_broad_fixed_filter_20260604_0520.*`,
   and
   `outputs/summary/paper_critical/ccrp_uncertainty_source_audit_{sports,toys,home,tools}_fixed_filter_20260604_0502.*`.
   The project-root broad scan used `--root .` and still found only the same
   four score-only files, so the next productive action is not another blind
   server sweep but locating the original full-scale signal-generation path or
   regenerating rows from saved non-test-selected signal inputs.
   Static trace helper:
   `scripts/audit/main_trace_ccrp_formal_signal_path.py`, with evidence
   `outputs/summary/paper_critical/ccrp_formal_signal_path_trace_20260604_0535.json`,
   confirms `experiments/rsc/run_ccrp_v3_domain.py` only requests
   `relevance_probability`, writes `scores.csv`, `report.json`, and
   `user_ranks.jsonl`, and does not write
   `ccrp_selected_test_scored_rows.csv`,
   `ccrp_internal_provenance.json`, evidence/counterevidence fields, or
   `ccrp_uncertainty`. Therefore paper-ready uncertainty rows cannot be
   rebuilt from formal `scores.csv` alone; the selector route still requires
   real valid/test signal paths.
   Observation-builder guard hardening on 2026-06-04 09:02 CST: the script now
   rejects duplicate ranking-eval events, eval events not present in the C-CRP
   uncertainty input, invalid positive ranks, and `num_candidates` mismatches
   when ranking records expose that column. Its provenance now records
   `artifact_class=paper_critical_observation_motivation`,
   `paper_claim_scope=motivation_only_not_main_table_sota`, the required full
   metric set (`MRR`, HR@5/@10/@20, NDCG@5/@10/@20), expected candidate count,
   and explicit claim limits. This strengthens the future motivation figure
   gate but does not remove the current missing-signal-row blocker.
   Discovery helper:
   `scripts/audit/main_discover_ccrp_uncertainty_sources.py`, which scans CSV
   and JSONL headers for domain-specific C-CRP/shadow/signal/scored artifacts
   before the stricter row/key audit. On 2026-06-04 the helper's path filter
   was fixed to match domain/name tokens against paths relative to the scan
   root, preventing `--domain home` from matching every absolute `/home/...`
   server path. Remote wrapper:
   `scripts/audit/main_remote_discover_ccrp_uncertainty_sources.py`.
   Rebuild helper:
   `scripts/analysis/main_export_ccrp_scored_rows_from_signal.py`, to convert a
   future located `recomputable_signal_rows` artifact plus a fixed selected
   C-CRP config into `ccrp_scored_rows.csv`/`ccrp_scores.csv`/provenance without
   LLM re-query. On 2026-06-03 this rebuild path and
   `scripts/misc/main_select_ccrp_variant_on_valid.py` were hardened to accept
   both `candidate_item_id` and `item_id` in signal rows, matching the
   uncertainty-source audit's allowed schema aliases; this prevents a
   preflight-passing `item_id` signal artifact from failing later score
   coverage only because of column naming.
   Guarded plan helper:
   `scripts/audit/main_plan_ccrp_signal_generation.py` generates a
   non-executing JSON/shell plan for Sports/Toys C-CRP signal evidence
   discovery, source audit, validation selection, ablation, observation, and
   hyperparameter plotting. The generated shell begins with `exit 2` and uses
   `TODO_*_CCRP_SIGNAL_JSONL_OR_CSV` placeholders, so it cannot accidentally
   launch work before real full-scale signal paths are filled. Current local
   plan artifact:
   `outputs/summary/paper_critical/ccrp_signal_generation_plan/ccrp_signal_generation_plan_20260604.*`.
   Guard wording hardening on 2026-06-05 01:50 CST: the plan generator,
   generated JSON, and focused tests now name the precondition generically as
   no active official baseline row or matching baseline Python process,
   removing stale row-specific wording from an older Home RLMRec handoff while
   preserving the non-executing `exit 2` shell guard.
   Readiness audit helper:
   `scripts/audit/main_audit_paper_critical_modules.py` now consolidates the
   paper-critical module state without running experiments. Current artifact:
   `outputs/summary/paper_critical/paper_critical_module_audit_20260604.{json,md}`.
   It reports `paper_ready=false`, `framework_overview_scaffold_ready=true`,
   `component_inventory_ready=true`, `guarded_plan_ready=true`, and
   `signal_rows_available=false`; therefore the observation, ablation, and
   hyperparameter modules remain blocked by
   `missing_full_scale_uncertainty_or_recomputable_signal_rows`.
   Post-Phase2 audit checkpoint 2026-06-05 22:19 CST: after the
   Sports/Toys/Home/Tools official+C-CRP gate certificate was pushed, Codex
   generated an all-four-domain guarded signal plan at
   `outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/`
   and a current module audit snapshot at
   `outputs/summary/paper_critical/paper_critical_module_audit_post_phase2_20260605.{json,md}`;
   hashes are recorded in
   `outputs/summary/paper_critical/post_phase2_paper_critical_audit_manifest_20260605.sha256`.
   The plan covers Sports/Toys/Home/Tools, records `will_start_experiment=false`,
   and the generated shell exits with `exit 2` before any command. The audit
   remains `paper_ready=false` and `signal_rows_available=false`, with
   observation, component ablation, and hyperparameter analysis all blocked by
   `missing_full_scale_uncertainty_or_recomputable_signal_rows`. Focused tests
   passed:
   `python -m pytest tests\test_audit_paper_critical_modules.py tests\test_plan_ccrp_signal_generation.py tests\test_build_ccrp_component_inventory.py tests\test_uncertainty_observation_study.py tests\test_ccrp_hyperparameter_sweep_plot.py`
   (`21 passed`). The next server action is not a blind experiment launch: first
   recover disk above the danger threshold with an audited safe-deletion list,
   then locate or regenerate full-scale valid/test uncertainty signal rows under
   the same-candidate protocol.
   Storage cleanup checkpoint 2026-06-05 22:34 CST: ARIS read-only storage
   audit plus a GPT-5.5 xhigh sidecar found no active experiment process, idle
   GPU, and `/` at `7.85GB` free / `97%` used. The only high-impact safe
   cleanup was completed Tools LLM-ESR paper-adapter staging under
   `outputs/baselines/paper_adapters/tools_large10000_100neg_llmesr_official_adapter/llm_esr/handled/`.
   After verifying Tools LLM-ESR `server_final_evidence_audit.ok=true`,
   `implementation_status=official_completed`, `blockers=[]`,
   `score_coverage_rate=1.0`, local-light package pass, and Tools domain
   gate pass, Codex recorded sha256/size manifests and deleted exactly five
   staging files: `itm_emb_np.pkl`, `pca64_itm_emb_np.pkl`,
   `sim_user_100.csv`, `sim_user_100.pkl`, and `inter.txt`. Cleanup manifest:
   `outputs/summary/tools_llmesr_completed_adapter_embedding_cleanup_20260605.{json,sha256}`.
   Local audit artifacts:
   `outputs/summary/paper_critical/server_storage_preflight_20260605.*` and
   `outputs/summary/paper_critical/server_storage_postcleanup_20260605.*`.
   Total deleted size was `4,493,778,837` bytes; final scores, provenance,
   score audits, run summary, imported tables, final `llmesr_official_model.pt`,
   C-CRP artifacts, task splits, and other projects were not touched.
   Post-cleanup disk is `12,342,898,688` bytes free / `94%` used. This clears
   the 10GiB danger floor but remains below the recommended 15GiB hard stop
   for signal-row regeneration; do not launch Phase 2.5 regeneration until
   disk is expanded or another archive-backed cleanup raises free space.
   Retention audit checkpoint 2026-06-05 22:49 CST: Codex ran an ARIS
   read-only storage-retention audit plus a GPT-5.5 xhigh sidecar request for
   the remaining Phase 2.5 disk gate. Server preflight again found no relevant
   Python experiment process, idle GPU, and `/` at `12,342,841,344` bytes free
   / `94%` used, leaving a `3,763,286,016` byte deficit to a strict `15GiB`
   launch floor. Routine safe-now cleanup is not enough: the only clear
   low-risk candidates were small completed staging/temp remnants such as the
   Tools LLM2Rec paper-adapter directory (~`57M`) and `tmp_llm2rec_sync`
   (~`40K`). The only high-yield candidates are protected artifacts requiring
   an explicit archive/retention decision before deletion, especially
   `/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`
   (`5,662,687,360` bytes) or completed final LLMEmb/LLM-ESR model checkpoints
   (`3.8G`-`6.8G` each). No deletion was performed, and no Phase 2.5 experiment
   was launched. Evidence:
   `outputs/summary/paper_critical/server_storage_phase2_5_retention_audit_20260605.{json,md,sha256}`.
   Minimum next action remains: expand disk or approve/archive exactly one
   high-yield completed artifact with sha256/size manifesting and post-delete
   gate checks before signal-row regeneration.
   Guarded retention-decision plan 2026-06-05 22:58 CST: Codex added
   `scripts/audit/main_plan_phase2_5_retention_cleanup.py` and generated a
   non-executing plan for the highest-yield candidate, the completed Tools
   LLM2Rec upstream embedding. Artifact:
   `outputs/summary/paper_critical/retention_cleanup_plan_20260605/tools_llm2rec_upstream_embedding_retention_cleanup_plan_20260605.{json,sh,sha256}`.
   The generated shell exits with `exit 2` before preflight, `sha256sum`, or
   `rm --`, records `will_delete=false`, `will_delete_files=false`, and
   `will_execute_cleanup=false`, and requires explicit token
   `APPROVE_DELETE_COMPLETED_TOOLS_LLM2REC_UPSTREAM_EMBEDDING_20260605` after
   a retention/archive decision. Expected embedding sha256 from local
   `fairness_provenance.json` is
   `306618d974eb4133d9cda87bae3251e17d793aa6f5a8cb38d558b549ed31d56e`. The
   plan includes exact-target `realpath` and size/hash guards, evidence
   prechecks, pre/post Tools domain-gate and comparison checks, sha256/size
   manifesting, and post-delete disk checks. Focused test:
   `python -m pytest tests\test_plan_phase2_5_retention_cleanup.py`
   (`3 passed`). No deletion was performed.
   Read-only retention helper checkpoint 2026-06-05 23:42 CST: Codex added
   `scripts/audit/main_audit_phase2_5_storage_retention.py`, a reusable
   SSH-based audit helper that runs only `ps`, `nvidia-smi`, `df`, `find`,
   `du`, and log/stat reads, then classifies large artifacts as protected,
   safe-now low-yield, approval-required, or requiring another audit before
   deletion. Current evidence:
   `outputs/summary/paper_critical/server_storage_phase2_5_retention_audit_current_20260605.{json,md,sha256}`.
   It confirms no active project Python process, GPU idle, `12,342,640,640`
   free bytes, a `3,763,486,720` byte deficit to the strict `15GiB` floor,
   safe-now recoverable bytes only `64,611,717`, and
   `experiment_launch_allowed=false`. Eight high-yield candidates would each
   require explicit archive/retention approval, led by the completed Tools
   LLM2Rec upstream embedding and completed LLMEmb/LLM-ESR checkpoints. No
   deletion or experiment launch was performed. Focused verification:
   `python -m pytest tests\test_audit_phase2_5_storage_retention.py
   tests\test_plan_phase2_5_retention_cleanup.py` (`5 passed`), and broader
   paper-critical tooling verification passed (`24 passed`).
   Safe-now cleanup checkpoint 2026-06-06 00:10-00:15 CST: Codex added
   `scripts/audit/main_cleanup_phase2_5_safe_now_remnants.py`, verified it
   locally, then streamed it to the server to remove only the exact fixed
   low-yield targets already classified as disposable:
   `outputs/baselines/paper_adapters/tools_large10000_100neg_llm2rec_official_adapter`,
   `outputs/baselines/paper_adapters/tools_large10000_100neg_llmesr_official_adapter`,
   and `tmp_llm2rec_sync`. The helper wrote a file-level sha256/size manifest
   before deletion and removed `64,574,853` bytes; post-delete presence is
   false for all three targets. Local evidence:
   `outputs/summary/paper_critical/phase2_5_safe_now_cleanup_manifest_20260605.json`,
   `phase2_5_safe_now_cleanup_post_domain_gate_20260605.{json,csv}`,
   `phase2_5_safe_now_cleanup_post_comparison_20260605/`,
   `server_storage_phase2_5_safe_now_postcleanup_20260605.{json,md}`, and
   `phase2_5_safe_now_cleanup_evidence_20260605.sha256`. Post-cleanup disk is
   `12,407,840,768` free bytes / `94%` used, still below the strict `15GiB`
   floor by `3,698,286,592` bytes; `experiment_launch_allowed=false`.
   Post-cleanup Tools gate remains `official_ok_count=8`,
   `official_all_ok=true`, `ccrp_ok=true`, and `gate_ok=true`; post-cleanup
   comparison keeps C-CRP observed-best on all seven metrics and all 56
   C-CRP-vs-official paired tests positive and Holm-significant. No protected
   final scores, provenance, audits, imported tables, checkpoints, task splits,
   upstream embeddings, or other projects were deleted. Remaining action is
   unchanged: disk expansion or explicit archive/retention approval for one
   high-yield completed artifact before Phase 2.5 signal-row regeneration.
   Ranked retention recommendation checkpoint 2026-06-06 00:20 CST: Codex
   upgraded `scripts/audit/main_audit_phase2_5_storage_retention.py` to assign
   deterministic retention-risk tiers/ranks and emit a
   `recommended_approval_candidate` without deleting anything. Current ranked
   evidence:
   `outputs/summary/paper_critical/server_storage_phase2_5_retention_audit_ranked_20260606.{json,md,sha256}`.
   It reports no active project Python process, GPU idle, `12,407,414,784`
   free bytes, `3,698,712,576` byte deficit to the strict `15GiB` floor, and
   no safe-now recoverable bytes remaining. The recommended approval-required
   target is the completed Tools LLM2Rec upstream embedding
   `/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`
   (`5,662,687,360` bytes), classified as
   `approval_required_external_embedding_cache`; deleting it after explicit
   archive/retention approval would raise expected free space to
   `18,070,102,144` bytes and clear the minimum gate. This is only a ranked
   decision surface; deletion still requires the existing approval token path,
   sha256/size manifesting, and post-delete domain/comparison gates.
   Ranked guarded plan refresh 2026-06-06 00:30 CST: after a fresh read-only
   preflight still found no matching project Python process, GPU idle, and `/`
   at `12,407,390,208` free bytes / `94%` used, Codex updated
   `scripts/audit/main_plan_phase2_5_retention_cleanup.py` so the guarded
   approval plan cites the ranked audit explicitly. New artifact:
   `outputs/summary/paper_critical/retention_cleanup_plan_20260606/tools_llm2rec_upstream_embedding_ranked_retention_cleanup_plan_20260606.{json,sh,sha256}`.
   The JSON records `ranked_retention_audit_source`,
   `recommended_by_ranked_audit=true`, tier
   `approval_required_external_embedding_cache`, rank `20`, ranked-audit
   `current_free_bytes=12,407,414,784`, and
   `expected_free_bytes_after_delete=18,070,102,144`. The generated shell still
   exits with `exit 2` before `sha256sum` or `rm --`, records
   `will_delete=false`, `will_delete_files=false`, and
   `requires_explicit_approval=true`. No deletion or experiment launch was
   performed. Verification:
   `python -m pytest tests\test_audit_paper_critical_modules.py
   tests\test_plan_ccrp_signal_generation.py
   tests\test_build_ccrp_component_inventory.py
   tests\test_uncertainty_observation_study.py
   tests\test_ccrp_hyperparameter_sweep_plot.py
   tests\test_framework_overview_figure.py
   tests\test_plan_phase2_5_retention_cleanup.py
   tests\test_audit_phase2_5_storage_retention.py
   tests\test_cleanup_phase2_5_safe_now_remnants.py` (`29 passed`).
   Local/server evidence consistency checkpoint 2026-06-06 00:40 CST: Codex
   added `scripts/audit/main_audit_local_server_evidence_consistency.py` to
   compare local lightweight official-baseline packages against their copied
   server large-artifact manifests without SSH, copying, deletion, or
   experiment launch. The helper verifies required local-light files, hashes
   every `light_evidence_sync_manifest.json` checked file, checks
   `server_large_artifact_manifest.json` for `scores.csv`,
   `predictions/rank_predictions.jsonl`, and model/checkpoint records, and
   fails if server-only bulk artifacts are present locally by default. Current
   Tools audit artifact:
   `outputs/summary/paper_critical/local_server_evidence_consistency_tools_20260606.{json,md,sha256}`.
   Result: `ok=true`, `row_count=8`, `ok_count=8`, `failure_count=0`;
   all eight Tools official local packages are consistent with their copied
   server manifests. Verification:
   `python -m pytest tests\test_local_server_evidence_consistency.py
   tests\test_server_large_artifact_manifest.py
   tests\test_remote_official_evidence_audit.py
   tests\test_remote_server_large_artifact_manifest.py
   tests\test_audit_phase2_5_storage_retention.py
   tests\test_plan_phase2_5_retention_cleanup.py
   tests\test_cleanup_phase2_5_safe_now_remnants.py` (`22 passed`).
   Four-new-domain consistency checkpoint 2026-06-06 00:50 CST: heartbeat
   preflight found no matching project Python process, GPU idle, and `/` at
   `12,407,324,672` free bytes / `94%` used, so no server experiment was
   launched. Codex ran the local-only consistency helper across
   Sports/Toys/Home/Tools:
   `outputs/summary/paper_critical/local_server_evidence_consistency_new_domains_20260606.{json,md,sha256}`.
   Result: `ok=false`, `row_count=32`, `ok_count=11`, `failure_count=51`.
   Tools remains fully consistent (`8/8` ok); Home has RLMRec, LLM2Rec, and
   LLM-ESR consistent; the remaining Sports/Toys/Home rows are blocked only on
   missing copied `server_large_artifact_manifest.json` and/or `.sha256` in the
   local lightweight packages. Representative inspection confirms those older
   rows still have local-light manifests and `server_final_evidence_audit.json`
   with `ok=true`, complete metrics, exact score coverage, and row counts, so
   this is an evidence-packaging backfill gap rather than a baseline failure.
   Next safe action is to backfill or regenerate small server-large manifests
   for the missing rows where server scores/predictions/model files or accepted
   deletion certificates exist, then rerun the same consistency audit. Do not
   launch Phase 2.5 signal generation until disk is above the launch floor.
   Verification checkpoint 2026-06-05 03:27 CST: while Tools IRLLRec remained
   active and untouched, Codex reran the local paper-critical tooling audit
   (`python scripts/audit/main_audit_paper_critical_modules.py --root .`) and
   focused tests:
   `python -m pytest tests\test_audit_paper_critical_modules.py tests\test_framework_overview_figure.py tests\test_uncertainty_observation_study.py tests\test_ccrp_hyperparameter_sweep_plot.py tests\test_build_ccrp_component_inventory.py`
   (`19 passed`). The audit summary remained `ok=true`, `paper_ready=false`,
   `framework_overview_scaffold_ready=true`, `component_inventory_ready=true`,
   `guarded_plan_ready=true`, and `signal_rows_available=false`; no local or
   server experiment was launched.
2. Component ablation: identify every nontrivial C-CRP component from the
   implementation and docs, then run leave-one-component-out variants under the
   same candidate protocol and validation/test discipline. Known component
   handles include score mode, boundary uncertainty
   (`without_boundary_uncertainty`), calibration gap
   (`without_calibration_gap`), evidence support/insufficiency
   (`without_evidence_support`), counterevidence, risk penalty, eta,
   confidence weight, and the C-CRP weight triple. If removing a component is
   neutral or better, report it honestly and mark the component as weak or
   needing redesign. Inventory helper:
   `scripts/audit/main_build_ccrp_component_inventory.py`; current artifact
   `outputs/summary/paper_critical/ccrp_component_inventory/ccrp_component_inventory_20260604.{json,md}`
   covers 12 items: the executable score modes/LOO handles/hyperparameters plus
   two conceptual-not-currently-executable risks (`raw_vs_calibrated_posterior`
   and `temperature_prompt_variants`). The inventory records that all
   paper-facing execution remains blocked by
   `missing_full_scale_uncertainty_or_recomputable_signal_rows`, and that
   temperature/prompt variants or raw-vs-calibrated posterior must not be
   claimed as completed LOO ablations without new audited handles.
3. Hyperparameter analysis: sweep actual method controls rather than cosmetic
   knobs. Candidate controls include eta, confidence weight, C-CRP weight
   triples, uncertainty thresholds/gates, anchor conflict penalties, and any
   real learning-rate/lambda controls used by trainable SRPD or related modules.
   Use sensible ranges, plot curves with matplotlib, and label validation-only
   selection separately from test reporting. Script entry for C-CRP sweep plots:
   `scripts/analysis/main_plot_ccrp_hyperparameter_sweep.py`, consuming
   `valid_ccrp_sweep.csv` from the validation selector and failing by default
   when a requested curve has fewer than three values. It supports
   `--test_sweep_csv` so validation and test curves can be reported separately;
   valid-only output is not sufficient for a paper stability claim.
   Guard hardening on 2026-06-04 09:21 CST: the plotter now requires
   `audit_ok` and `degeneracy_audit_ok` columns by default, labels valid-only
   output as `validation_only_hyperparameter_selection_curve` with
   `paper_claim_scope=validation_only_not_stability_claim`, labels
   `--no-require_audit_ok` or incomplete curves as diagnostic-only, and reserves
   `paper_critical_hyperparameter_curve_ready` for audited valid+test curves
   that meet the minimum value count. This is plotting/provenance guard
   progress only; actual curves remain blocked by missing full-scale signal
   rows.
4. Framework overview figure: prepare a clean paper figure showing the full
   pipeline, where task-grounded uncertainty is estimated, which components
   enter C-CRP, and where risk-adjusted ranking is applied. Script entry:
   `scripts/analysis/main_build_framework_overview_figure.py`; generated SVG
   is the editable source, with PDF/PNG/caption/provenance exports. Draft
   local package generated at 2026-06-04 04:43 CST under
   `outputs/summary/paper_critical/framework_overview/` with SVG, PDF, PNG,
   caption, provenance, and `framework_overview_manifest.sha256`. The draft
   now labels the risk score formula as
   `risk_score = base_score * (1 - uncertainty)^eta`, matching
   `src/shadow/ccrp.py`, and includes counterevidence in the C-CRP uncertainty
   box; it still needs final paper-layout/reviewer polish before camera-ready use. On
   2026-06-04 17:31 CST, the package was regenerated from
   `scripts/analysis/main_build_framework_overview_figure.py` against git
   commit `9badd19` with the code-matched multiplicative risk formula;
   `python -m pytest tests\test_framework_overview_figure.py
   tests\test_audit_paper_critical_modules.py` passed (`4 passed`), and the
   PNG was visually checked as nonblank/readable. This completes the current
   draft scaffold for the framework-overview paper-critical module, but not
   final camera-ready reviewer polish.
   Framework review-ready checkpoint 2026-06-05 23:20 CST: the figure builder
   and consolidated module audit now require
   `status_label=paper_critical_framework_overview_review_ready`,
   `paper_claim_ready=true`, the controlled same-candidate claim boundary,
   the code-matched multiplicative risk formula, required SVG labels, valid
   PNG dimensions, and a matching sha256 manifest. The regenerated package
   under `outputs/summary/paper_critical/framework_overview/` passes those
   gates with PNG dimensions `2559x1378` and keeps
   `module_scope=framework_figure_only_not_substitute_for_observation_ablation_or_hyperparameter_evidence`.
   Fresh audit snapshot:
   `outputs/summary/paper_critical/paper_critical_module_audit_post_framework_review_20260605.{json,md,sha256}`.
   It reports framework overview `status=review_ready` and
   `paper_claim_ready=true`, while overall `paper_ready=false` and
   `signal_rows_available=false`; observation, ablation, and hyperparameter
   modules remain blocked by missing full-scale valid/test uncertainty signal
   rows. Verification:
   `python -m pytest tests\test_framework_overview_figure.py
   tests\test_audit_paper_critical_modules.py
   tests\test_uncertainty_observation_study.py
   tests\test_ccrp_hyperparameter_sweep_plot.py
   tests\test_build_ccrp_component_inventory.py` (`19 passed`). Visual PNG
   inspection was nonblank/readable. The framework-figure module is now
   review-ready except for final venue-template sizing/polish.

Evidence packaging for these modules follows the same lightweight-but-complete
standard as official rows: keep selected metrics CSV/JSON, final tables, plots,
configs, commands, seeds, git commit, row counts, provenance notes, key logs,
and only minimal checkpoints needed for verification or resume. Compare server
and local manifests so missing files are caught. Do not package huge redundant
raw outputs by default.

Official row gate helper: after `server_final` passes and before local-light
sync or any cleanup decision, run
`scripts/audit/main_build_server_large_artifact_manifest.py` on the server
evidence directory. It writes `server_large_artifact_manifest.sha256/json` for
`scores.csv`, `predictions/rank_predictions.jsonl`, and method-specific
model/checkpoint artifacts such as `*_official_model.pt`, avoiding brittle
manual `sha256sum` commands that guess the model filename.
Operational note from the 2026-06-04 10:35 CST monitor: while Home LLM-ESR is
active, do not `git pull` the server checkout. The server working tree may lag
behind local/GitHub helpers, so when the active run finishes, either perform a
safe `git pull --ff-only` after confirming no active process and no conflicting
worktree state, or run
`scripts/audit/main_remote_server_large_artifact_manifest.py` from the local
checkout before local-light sync; that wrapper sends the current local helper
through SSH stdin and does not depend on the server checkout version.
Guarded completion-gate plan helper:
`scripts/audit/main_plan_official_completion_gates.py` writes a planning-only
JSON plus guarded PowerShell script for a completed official row. The generated
plan fixes the required order as server-final audit, server large-artifact
manifest, local-light sync, and local-light audit; it begins with a PowerShell
`throw` and does not run by default. The current checked-in plan artifact still
targets the completed Home RLMRec row; generate a fresh Home LLM-ESR plan only
after `llmesr_sasrec` completes.

Remote monitor helper:
`scripts/audit/main_remote_baseline_monitor_snapshot.py` captures active-row
state through SSH stdin so it does not depend on the stale server checkout or
fragile shell quoting. It reports tracked PID liveness, matching Python
processes with the monitor helper itself filtered out, log progress,
completion/failure markers, GPU, disk, output sizes, and `should_notify` only
for completion, failure, disk danger, duplicate-run risk, or dead tracked PIDs.
For recurring heartbeats, pass `--output_json_on_notify_only` so quiet checks
print status without dirtying the tracked snapshot file.
Current heartbeat target: Home official-baseline monitoring is complete. Any
remaining Home runner heartbeat should be deleted or retargeted to the next
bounded Tools storage/preflight checkpoint.

Tools LLM2Rec disk-full failure checkpoint: at 2026-06-05 15:23 CST, the
relaunch of Tools `llm2rec_sasrec` from runner PID `3413921` and adapter PID
`3413930` was confirmed failed, not completed. Both tracked PIDs had exited,
GPU was idle, and log `baselines_new_domains_tools_llm2rec_20260605_134355.log`
ended with `OSError: [Errno 28] No space left on device` after the embedding
pass had reached `[hf_mean_pool] encoded 345622/345622`. The method-specific
training log also showed disk exhaustion during `torch.save`, so the partial
SASRec checkpoint was corrupt. No `fairness_provenance.json`, no `scores.csv`,
and no `llm2rec_official_score_audit.json` existed, so the row is not
table-eligible and must not be imported.

Emergency cleanup preserved the reusable upstream embedding at
`/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`
(`5.3G`) and removed only failed or already-repaired artifacts: the corrupt
validation CSV backup
`outputs/baselines/external_tasks/tools_large10000_100neg_valid_same_candidate/candidate_items.csv.corrupt_20260605T051943Z`
(`467M`, sha256 already recorded in the repair manifest), the failed adapter
copy `llm2rec_item_embeddings.npy` (`5.3G`), regenerated adapter CSVs
`candidate_items_mapped.csv` (`818M`) and `item_text_seed.csv` (`233M`), and
the corrupt partial Tools LLM2Rec checkpoint (`1.3G`). Two abandoned diagnostic
grep/bash trees from timed-out monitor commands were also killed after
`ps`/`pstree` verification as non-experiment processes. Post-cleanup state:
`/` is about `8.1G` free / `96%` used, adapter dir is `57M`, final dir is
`24K`, and the upstream embedding cache is `5.3G`. This remains below the
`10G` monitor threshold, so do not launch another baseline yet. The next safe
LLM2Rec recovery should first free additional audited space, then relaunch a
single Tools `llm2rec_sasrec` row with the preserved embedding passed as
`--llm2rec_item_embedding_path /home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`
so Qwen3 embedding is not regenerated.

Tools LLM2Rec recovery relaunch checkpoint: after a sidecar GPT-5.5 xhigh
ARIS storage audit judged completed-checkpoint cleanup conditionally acceptable,
the completed Sports LLM2Rec full checkpoint was removed with sha256/size
manifests
`outputs/summary/sports_llm2rec_checkpoint_deletion_manifest_20260605.json`
and row-local
`outputs/sports_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate/checkpoint_deletion_manifest_20260605_sports_llm2rec.json`.
The deleted checkpoint was `4,652,329,391` bytes with sha256
`acddfcb8680ec7b98f96bbeac67a23d97e63c5cc862cabcea1aa29b5dd7b713c`.
Sports LLM2Rec `scores.csv`, provenance, score audits, run summary, imported
tables, server-final audit, and local-light package were preserved; live
post-delete checks still found `scores.csv` at `1,010,001` lines and
`tables/ranking_eval_records.csv` at `10,001` lines. When the active Tools
run then saved its own full checkpoint and disk again reached warning level,
the completed Toys LLM2Rec full checkpoint was removed under the same
manifest discipline:
`outputs/summary/toys_llm2rec_checkpoint_deletion_manifest_20260605.json` and
row-local
`outputs/toys_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate/checkpoint_deletion_manifest_20260605_toys_llm2rec.json`.
The deleted Toys checkpoint was `4,178,091,523` bytes with sha256
`4f9d087f50116aee1a4dace21b514920f565921659a7c18d9f4e9ee222d9b7d9`; Toys
scores/provenance/audits/run summary/tables/local-light evidence were preserved
and live checks still found `scores.csv` at `1,010,001` lines and
`tables/ranking_eval_records.csv` at `10,001` lines. These deletions are
storage-emergency archive decisions only; they do not downgrade the completed
Sports/Toys LLM2Rec table rows, but future reproduction of their full SASRec
checkpoints would require rerunning from the recorded provenance and scores.

The local and server `scripts/run_baselines_new_domains.sh` now support
`LLM2REC_ITEM_EMBEDDING_PATH_OVERRIDE` for the LLM2Rec case only, forwarding it
as `--llm2rec_item_embedding_path` so recovery can reuse a precomputed
embedding without `--force_embeddings`. Server `bash -n` passed after copying
the patched wrapper. A single Tools `llm2rec_sasrec` recovery run launched at
2026-06-05 15:59 CST from
`baselines_new_domains_tools_llm2rec_recovery_20260605_155904.log` with runner
PID `3423029`, adapter PID `3423037`, official training PID `3423221`, and
heartbeat `monitor-tools-llm2rec-recovery`. The adapter command includes
`--llm2rec_item_embedding_path /home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`,
and the training command uses that same embedding path. At 2026-06-05 16:05
CST it was active around epoch `30+`, had saved a Tools checkpoint, and had no
completion or fatal markers; disk was `10,692,530,176` bytes free (`10G` /
`95%`), adapter dir `1.1G`, output dir `5.3G`, and preserved upstream embedding
dir `5.3G`. The row remains active monitor-only evidence, not table-eligible,
until provenance, score audit, import, server-final, server manifest,
local-light sync, local-light audit, full metrics, and row-count gates pass.

Execution specification: `docs/paper_critical_experiment_plan_2026-06-03.md`.
Do not start Tools official baselines until the Home domain gate artifacts are
committed and a fresh process/GPU/disk preflight plus storage decision passes.

## Current Server State

- Server: `pony-rec-gpu`
- Server repo: `~/projects/pony-rec-rescue-shadow-v6`
- Disk policy update: after a completed official row has passed server-final
  audit and local-light sync, the huge server-side
  `predictions/rank_predictions.jsonl` may be deleted with sha256 manifest and
  docs/memory note if disk pressure threatens active progress. Domain gate and
  comparison scripts now accept the row only if `server_final_evidence_audit.json`
  certifies the missing prediction file's original line count. This exception
  does not cover `scores.csv`, provenance, score audits, run summaries,
  imported `tables/`, models, checkpoints, or local evidence packages.
- Internal C-CRP disk policy update: after a domain gate passes and local light
  evidence has raw `report.json`, raw `user_ranks.jsonl`, and imported
  `tables/`, the server-only imported
  `predictions/rank_predictions.jsonl` may be deleted only with a row-local
  `prediction_deletion_manifest.json` recording sha256, byte size, and 10,000
  lines. The patched domain gate and comparison builder accept this manifest
  only for internal C-CRP imports; official rows still require
  `server_final_evidence_audit.json`.
- Previous completed home row: `irllrec_intent`, completed 2026-06-03 20:05 CST
  with `implementation_status=official_completed`, `blockers=[]`, exact
  `score_coverage_rate=1.0`, server-final audit PASS, lightweight sync PASS,
  local-light audit PASS, and no local forbidden large files. Full metrics over
  10,000 users and 101 candidates are HR@5/10/20
  `0.0821 / 0.1443 / 0.2878`, NDCG@5/10/20
  `0.05108975290090002 / 0.07089752436733526 / 0.10662011242627113`, and MRR
  `0.07424325424843974`. Row counts passed: `scores.csv` `1,010,001` lines,
  predictions `10,000` lines, and `tables/ranking_eval_records.csv` `10,001`
  lines. The local lightweight package is
  `outputs/baselines/official_adapters/home_large10000_100neg_irllrec_intent_official_qwen3base_same_candidate/`.
  Server-only large artifacts are `scores.csv`,
  `predictions/rank_predictions.jsonl`, and `irllrec_official_model.pt`,
  covered by `server_large_artifact_manifest.sha256`. After final/server/local
  gates passed, the completed intermediate adapter
  `outputs/baselines/paper_adapters/home_large10000_100neg_irllrec_official_adapter`
  was removed with cleanup manifest
  `outputs/summary/home_irllrec_completed_adapter_cleanup_manifest_20260603.sha256`.
  A post-cleanup server-final audit remained `ok=true`; final scores,
  provenance, audits, predictions, imported tables, model, and local
  lightweight evidence were preserved. Disk recovered from about `4.7G` to
  `12G` free, and no Pony/C-CRP/baseline/uncertainty Python process remained
  active. At that checkpoint, Home had 5/8 completed official baseline rows.
- Latest completed home row: `rlmrec_graphcl`, completed 2026-06-04 06:49 CST
  as `implementation_status=official_completed`, `blockers=[]`, and exact
  `score_coverage_rate=1.0`. It ran from
  `baselines_new_domains_home_rlmrec_20260603_2028.log`, reached
  `[rlmrec-official] epoch=3000 train_loss=1.546134`, exported final scores,
  predictions, and imported tables, and then passed the required row gates:
  server-final evidence audit PASS, server large-artifact sha256 manifest
  written, lightweight local sync PASS, and local-light audit PASS. Full
  metrics over 10,000 users and 101 candidates are HR@5/10/20
  `0.0685 / 0.1268 / 0.2451`, NDCG@5/10/20
  `0.04126795456098191 / 0.059869974785318684 / 0.08932211717259728`, and MRR
  `0.06397404670906748`. Row counts passed: `scores.csv` `1,010,001` lines,
  predictions `10,000` lines, and `tables/ranking_eval_records.csv` `10,001`
  lines. The local lightweight package is
  `outputs/baselines/official_adapters/home_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate/`.
  Server-only large artifacts are `scores.csv`
  (`sha256=002255fb767705fc2c7a30428269dd5f53cc6268647c04f5c26a4a923ce29c51`),
  `predictions/rank_predictions.jsonl`
  (`sha256=0204ca3e4491440d5d00a84a5fff911ea0d7af225d04574a851912a629148fdc`),
  and `rlmrec_official_model.pt`
  (`sha256=87b900951174eeefc7daa4dc6c15e33bf8709264017bb94b7bfe092d6ac2a32c`),
  covered by `server_large_artifact_manifest.sha256`. Home now has 6/8
  completed official baseline rows. No next baseline was launched; runner PID
  `3178395` had exited, GPU was idle, and `/` was about `12G` free / `94%`
  used after the row gates. Domain-wide Home gates remain pending until all
  eight official rows complete. Before launching the next storage-heavy row,
  the completed non-final intermediate adapter
  `outputs/baselines/paper_adapters/home_large10000_100neg_rlmrec_official_adapter`
  was audited with cleanup manifest
  `outputs/summary/home_rlmrec_completed_adapter_cleanup_manifest_20260604.sha256`
  and removed after exact realpath checks. This recovered `/` from about `12G`
  free / `94%` used to about `19G` free / `91%` used without touching final
  RLMRec scores, provenance, audits, imported tables, model, or local-light
  evidence. A post-cleanup server-final RLMRec audit remained PASS.
- Latest completed home row: `llm2rec_sasrec`, launched 2026-06-04
  07:19 CST after confirming no active experiment process, no existing Home
  LLM2Rec final output directory, no existing Home LLM2Rec adapter directory,
  Home RLMRec final evidence still protected, and disk recovered to about `19G`
  free. Runner PID `3236678`, adapter PID `3236688`, PID file
  `baselines_new_domains_home_llm2rec_20260604_071902.pid`, log
  `baselines_new_domains_home_llm2rec_20260604_071902.log`. At the first stable
  check the row had entered Qwen3 `hf_mean_pool` embedding, progress was about
  `944/568891`, GPU was about `96%` with `15945 MiB / 49140 MiB`, the
  intermediate adapter directory was about `1.3G`, the final output directory
  was still a placeholder, and `/` was about `18G` free / `91%` used. This row
  is not table-eligible until final score/provenance/import/server-final,
  server large-artifact manifest, local-light sync, local-light audit, and full
  metric/row-count gates pass. Do not start `llmesr_sasrec` while this runner
  is active. At the 2026-06-04 08:01 CST robust monitor snapshot, both tracked
  PIDs were alive, exactly one matching Home LLM2Rec Python adapter process was
  present, Qwen3 `hf_mean_pool` progress reached `182792/568891` (`0.3213`),
  GPU was `98%` with `16285 MiB / 49140 MiB`, `/` had about `17.17G` free and
  `91%` used, the intermediate adapter directory was `1.3G`, the final output
  directory remained `4.0K`, and `should_notify=false` with no completion,
  failure, disk danger, or duplicate-run reason. At the 2026-06-04 08:40 CST
  continuation monitor, the same tracked PIDs remained alive, exactly one
  matching adapter Python process was present, Qwen3 embedding progress reached
  `283088/568891` (`0.4976`), GPU was `96%` with `16285 MiB / 49140 MiB`, `/`
  had about `17.17G` free and `91%` used, output sizes remained `1.3G` adapter
  and `4.0K` final directory, and `should_notify=false`. The original run then
  completed Qwen3 embedding but failed while copying the 9.3G upstream item
  embedding because `/` reached 100% used. The partial upstream copy was
  removed and replaced by a symlink to the complete project embedding with
  cleanup manifest
  `outputs/summary/home_llm2rec_failed_partial_upstream_embedding_cleanup_20260604.txt`.
  After deleting only already gated Home IRLLRec/RLMRec prediction JSONLs under
  the documented emergency policy, recovery PID `3244540` launched from
  `baselines_new_domains_home_llm2rec_recovery_20260604_094007.log` and
  completed official SASRec training with early stopping at epoch 50, best
  epoch 30, final provenance
  `implementation_status=official_completed`, `blockers=[]`, and exact
  `score_coverage_rate=1.0`. Score audit, same-candidate import, server-final
  evidence audit, server large-artifact sha256 manifest, lightweight local
  sync, and local-light audit all passed. Full metrics over 10,000 users and
  101 candidates are HR@5/10/20 `0.0577 / 0.1101 / 0.2153`, NDCG@5/10/20
  `0.034207889197971464 / 0.05094946457092549 / 0.07719596686909931`, and
  MRR `0.0563865859396318`. Row counts passed: `scores.csv` `1,010,001`
  lines, predictions `10,000` lines before post-gate deletion, and
  `tables/ranking_eval_records.csv` `10,001` lines. The local lightweight
  package is
  `outputs/baselines/official_adapters/home_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate/`.
  Server-only large artifacts are `scores.csv`
  (`sha256=bf2b30c35815894dd3aa7c6da09fc75746115ea371f23d6473fc152a0cab51f1`)
  and the 9.3G SASRec checkpoint
  (`sha256=07bcfdba02475e76f0b0dfc0f02e64444b881d21f6ccf1cc5276502a25e58d08`),
  covered by `server_large_artifact_manifest.sha256`. Because disk remained
  dangerous after import (`/` about `466M` free / `100%` used), the completed
  server-side prediction JSONL was removed only after server-final and
  local-light gates passed, with sha256/line-count manifest
  `outputs/summary/home_llm2rec_prediction_deleted_after_gates_20260604.sha256`.
  Scores, provenance, score audits, run summary, imported tables, checkpoint,
  embedding, and local evidence were preserved. The completed non-final
  LLM2Rec intermediate adapter
  `outputs/baselines/paper_adapters/home_large10000_100neg_llm2rec_official_adapter`
  and its upstream symlink were then removed after exact realpath/symlink
  checks and sha256 manifest
  `outputs/summary/home_llm2rec_completed_adapter_cleanup_manifest_20260604.sha256`.
  This recovered `/` from about `1.2G` free / `100%` used to about `12G` free /
  `95%` used without touching final scores, provenance, audits, imported
  tables, final checkpoint, or local evidence.
- Latest completed home row: `llmesr_sasrec`, launched 2026-06-04
  10:14 CST after confirming no active experiment process, no existing Home
  LLM-ESR final output directory, no existing Home LLM-ESR adapter directory,
  and disk recovered to about `12G` free / `95%` used. Wrapper bash PID
  `3248921`, adapter PID `3248934`, log
  `baselines_new_domains_home_llmesr_20260604_1015.log`. The script correctly
  skipped completed Home `llmemb`, `proex_profile`, and `promax_profile`, then
  started only `llmesr_sasrec`. It completed at 2026-06-04 13:09 CST with
  `implementation_status=official_completed`, `blockers=[]`, exact
  `score_coverage_rate=1.0`, and no traceback/OOM/no-space/killed markers.
  Full metrics over 10,000 users and 101 candidates are HR@5/10/20
  `0.0621 / 0.1163 / 0.2139`, NDCG@5/10/20
  `0.037993209299003045 / 0.055376101596196485 / 0.0797502336556021`, and
  MRR `0.059737054548523474`. Row counts passed: `scores.csv` has
  `1,010,001` lines, predictions had `10,000` lines before post-gate
  deletion, and `tables/ranking_eval_records.csv` has `10,001` lines. The
  server-final audit, server large-artifact sha256 manifest, lightweight local
  sync, and local-light audit all passed. The local lightweight package is
  `outputs/baselines/official_adapters/home_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/`.
  Server-only protected artifacts still include `scores.csv` and
  `llmesr_official_model.pt`; the local package keeps provenance, score audit,
  run summary, imported tables, server-final audit, server large-artifact
  manifest, sync manifest, and local-light audit.
- Disk rescue during active Home `llmesr_sasrec`: after Qwen3 embedding
  completed and LLM-ESR training began, the active adapter directory grew to
  about `7.0G` and `/` dropped to about `4.1G` free / `98%` used. Read-only
  large-file audit classified the active `llm_esr/handled/itm_emb_np.pkl`,
  final official checkpoints/models, final scores/provenance/audits/imported
  tables, and same-candidate task packages as protected. Cleanup therefore
  removed only old transfer/generated diagnostic artifacts with sha256
  manifests: ignored server `outputs/exports/*.tar.gz` transfer bundles
  (`outputs/summary/server_exports_tarball_cleanup_20260604.sha256`), old
  generated SRPD formal `train.jsonl`/`valid.jsonl` files while preserving
  configs/summaries
  (`outputs/summary/week8_srpd_formal_data_cleanup_for_home_llmesr_disk_20260604.sha256`),
  and old Beauty Shadow diagnostic JSONL inputs/gate-sweep intermediates while
  preserving the C-CRP selected scored rows
  (`outputs/summary/week8_shadow_diagnostic_jsonl_cleanup_for_home_llmesr_disk_20260604.sha256`).
  This recovered `/` to about `8.1G` free / `96%` used, enough to avoid the
  immediate no-space cliff but still below the `<10G` warning threshold. Do
  not delete final method checkpoints, active LLM-ESR files, task data, scores,
  provenance, audits, or imported tables without a separate explicit archive
  decision.
- Post-gate Home `llmesr_sasrec` cleanup: after the row completed and the
  server-final audit, server large-artifact manifest, lightweight local sync,
  and local-light audit passed, `/` had dropped to about `962M` free / `100%`
  used. The completed row's `predictions/rank_predictions.jsonl` and the
  temporary adapter staging directory
  `outputs/baselines/paper_adapters/home_large10000_100neg_llmesr_official_adapter`
  were removed only after exact realpath checks and sha256/line/size manifest
  `outputs/summary/home_llmesr_post_gate_cleanup_20260604.sha256`. Final
  `scores.csv`, provenance, score audits, imported tables,
  `llmesr_official_model.pt`, server large-artifact manifest, and the local
  lightweight evidence package were preserved. Disk recovered to about `8.7G`
  free / `96%` used.
- Disk rescue during active home `irllrec_intent`: at the 2026-06-03 17:21 CST
  heartbeat, the row was active after completing Qwen3 embedding and had
  reached official training epoch `1220`, but disk had fallen to about `30M`
  free. Safe cleanup first removed the completed, non-final Toys LLM2Rec
  intermediate adapter
  `outputs/baselines/paper_adapters/toys_large10000_100neg_llm2rec_official_adapter`
  after confirming Toys LLM2Rec server-final and local-light audits were PASS
  and writing
  `outputs/summary/toys_llm2rec_completed_adapter_cleanup_manifest_20260603_irllrec_disk.sha256`.
  Because this only recovered disk to about `410M`, two server-only prediction
  JSONLs from already gated toys rows were removed after a sha256 manifest:
  `outputs/toys_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate/predictions/rank_predictions.jsonl`
  and
  `outputs/toys_large10000_100neg_irllrec_intent_official_qwen3base_same_candidate/predictions/rank_predictions.jsonl`.
  Scores, provenance, audits, imported tables, models, and local-light packages
  were preserved; the deletion manifest is
  `outputs/summary/toys_predictions_deleted_for_home_irllrec_disk_20260603.sha256`.
  Disk recovered to about `2.0G`; the active home IRLLRec process continued and
  reached epoch `1280` by 17:27 CST. At the 17:43 CST follow-up, disk had
  slipped to about `1.7G` while the same runner/adapter remained active, so the
  already gated Toys ProEx server-only prediction JSONL was removed after
  confirming server-final and local-light audits and recording sha256
  `outputs/summary/toys_proex_prediction_deleted_for_home_irllrec_disk_20260603.sha256`.
  Final scores, provenance, audits, imported tables, models, and local-light
  packages were preserved; disk recovered to about `2.5G`, and the active home
  IRLLRec process reached epoch `1480` by 17:45 CST. At the 18:08 CST
  follow-up, the same active row had reached epoch `1740` with no fatal/OOM/no
  space markers, but disk remained tight. After confirming sports ProMax
  server-final audit `ok=true` and local-light audit `ok=true`, only the
  already gated server-side Sports ProMax prediction JSONL was removed with
  sha256 manifest
  `outputs/summary/sports_promax_prediction_deleted_for_home_irllrec_disk_20260603.sha256`.
  Sports ProMax `scores.csv`, provenance, audits, imported tables, model, and
  local-light package were preserved; disk recovered to about `3.1G`. At the
  18:45 CST continuation check, the same active row had reached epoch `2140`
  with no fatal/OOM/no-space markers, but disk was still tight. After verifying
  server-final audit `ok=true`, local-light audit `ok=true`, provenance
  `implementation_status=official_completed`, `blockers=[]`, and
  `score_coverage_rate=1.0` for the completed home `proex_profile`,
  `promax_profile`, `elmrec_graph`, and `llmemb` rows, only their server-side
  `predictions/rank_predictions.jsonl` files were removed. Original sha256
  values are recorded in
  `outputs/summary/home_completed_predictions_deleted_for_irllrec_disk_20260603.sha256`.
  Their `scores.csv`, provenance, audits, imported tables, models, and local
  lightweight packages were preserved. Disk recovered to about `6.0G`, and the
  active Home IRLLRec runner/adapter remained active.
- Previous completed home row: `llmemb`, completed 2026-06-03 09:55 CST after a
  disk-full checkpoint/import recovery. The first 2026-06-03 06:08 CST LLMEmb
  run reached exact score export but failed during `torch.save` with the
  filesystem at `100%` used. The orphaned `scores.csv` had `1,010,001` lines
  and a read-only exact-key audit passed (`1,010,000/1,010,000` finite keys, no
  duplicates, no missing/extra keys), but provenance/import/audit files were
  missing, so it was never table-eligible. Recovery removed only failed-run or
  generated staging storage, patched the trainer to symlink the large handled
  `itm_emb_np.pkl` into the pinned upstream repo, and relaunched a true symlink
  rerun at 2026-06-03 09:16 CST with log
  `baselines_new_domains_home_llmemb_symlink_rerun_20260603_0920.log`. The
  rerun wrote final provenance, scores, score audit, run summary, predictions,
  imported tables, server-final audit, and local-light package. Full metrics
  over 10,000 users and 101 candidates are HR@5/10/20
  `0.1079 / 0.1856 / 0.3169`, NDCG@5/10/20
  `0.06899578967097944 / 0.09390612986107003 / 0.12674255822842873`, and MRR
  `0.09012268660291177`. Row counts passed: `scores.csv` `1,010,001` lines,
  predictions `10,000` lines, and `tables/ranking_eval_records.csv` `10,001`
  lines. The local lightweight package is
  `outputs/baselines/official_adapters/home_large10000_100neg_llmemb_official_qwen3base_same_candidate/`.
  The server large-artifact manifest records `scores.csv`,
  `predictions/rank_predictions.jsonl`, and `llmemb_official_model.pt` while
  keeping those files server-only. At that checkpoint, Home had 4/8 completed
  official baseline rows before IRLLRec. After local evidence was refreshed,
  the completed
  intermediate adapter
  `outputs/baselines/paper_adapters/home_large10000_100neg_llmemb_official_adapter`
  was removed with cleanup manifest
  `outputs/summary/home_llmemb_completed_adapter_cleanup_manifest_20260603.sha256`;
  a post-cleanup server-final audit remained `ok=true`, and disk recovered to
  about `7.1G` free.
- Previous completed home row: `elmrec_graph`, completed 2026-06-03 05:47 CST
  with `implementation_status=official_completed`, `blockers=[]`, exact
  `score_coverage_rate=1.0`, server-final audit PASS, lightweight sync PASS,
  local-light audit PASS, and no local forbidden large files. Full metrics over
  10,000 users and 101 candidates are HR@5/10/20
  `0.0509 / 0.1021 / 0.2018`, NDCG@5/10/20
  `0.029717257242599254 / 0.0460440741915887 / 0.0708856096588022`, and MRR
  `0.05195852255617441`. Row counts passed: `scores.csv` `1,010,001` lines,
  predictions `10,000` lines, and `tables/ranking_eval_records.csv` `10,001`
  lines. The local lightweight package is
  `outputs/baselines/official_adapters/home_large10000_100neg_elmrec_graph_official_qwen3base_same_candidate/`.
  The server large-artifact manifest records `scores.csv`,
  `predictions/rank_predictions.jsonl`, and `elmrec_official_model.pt` while
  keeping those files server-only. After final/server/local gates passed, the
  completed intermediate adapter
  `outputs/baselines/paper_adapters/home_large10000_100neg_elmrec_official_adapter`
  was removed after exact realpath checks and a 16-file sha256 cleanup
  manifest:
  `outputs/summary/home_elmrec_completed_adapter_cleanup_manifest_20260603.sha256`.
  A post-cleanup server-final audit remained `ok=true`; final scores,
  provenance, audits, predictions, imported tables, model, and local
  lightweight evidence were preserved. Disk recovered from about `6.5G` to
  `14G` free.
- Previous completed home row: `promax_profile`, completed 2026-06-03 02:53 CST
  with `implementation_status=official_completed`, `blockers=[]`, exact
  `score_coverage_rate=1.0`, server-final audit PASS, lightweight sync PASS,
  local-light audit PASS, and no local forbidden large files. Full metrics over
  10,000 users and 101 candidates are HR@5/10/20
  `0.0514 / 0.1019 / 0.2076`, NDCG@5/10/20
  `0.030788292596664168 / 0.04691808776215203 / 0.07326077825489297`, and MRR
  `0.053474908740382465`. Row counts passed: `scores.csv` `1,010,001` lines,
  predictions `10,000` lines, and `tables/ranking_eval_records.csv` `10,001`
  lines. The local lightweight package is
  `outputs/baselines/official_adapters/home_large10000_100neg_promax_profile_official_qwen3base_same_candidate/`.
  The server large-artifact manifest records `scores.csv`,
  `predictions/rank_predictions.jsonl`, and `promax_official_model.pt` while
  keeping those files server-only. After final/server/local gates passed, the
  completed intermediate adapter
  `outputs/baselines/paper_adapters/home_large10000_100neg_promax_official_adapter`
  was removed after exact realpath checks and a 22-file sha256 cleanup
  manifest:
  `outputs/summary/home_promax_completed_adapter_cleanup_manifest_20260602.sha256`.
  Final scores, provenance, audits, predictions, imported tables, model, and
  local lightweight evidence were preserved. Disk recovered from about `7.5G`
  to `15G` free.
- Previous completed home row: `proex_profile`, completed 2026-06-02 22:00 CST
  with `implementation_status=official_completed`, `blockers=[]`, exact
  `score_coverage_rate=1.0`, server-final audit PASS, lightweight sync PASS,
  local-light audit PASS, and no local forbidden large files. Full metrics over
  10,000 users and 101 candidates are HR@5/10/20
  `0.0606 / 0.1177 / 0.2296`, NDCG@5/10/20
  `0.03662857786324662 / 0.054867449700296195 / 0.08290060869107069`, and MRR
  `0.05933326491258513`. Row counts passed: `scores.csv` `1,010,001` lines,
  predictions `10,000` lines, and `tables/ranking_eval_records.csv` `10,001`
  lines. The local lightweight package is
  `outputs/baselines/official_adapters/home_large10000_100neg_proex_profile_official_qwen3base_same_candidate/`.
  The server large-artifact manifest records `scores.csv`,
  `predictions/rank_predictions.jsonl`, and `proex_official_model.pt` while
  keeping those files server-only. After final/server/local gates passed, the
  completed intermediate adapter
  `outputs/baselines/paper_adapters/home_large10000_100neg_proex_official_adapter`
  was removed after exact realpath checks and a 27-file sha256 cleanup
  manifest:
  `outputs/summary/home_proex_completed_adapter_cleanup_manifest_20260602.sha256`.
  Final scores, provenance, audits, predictions, imported tables, model, and
  local lightweight evidence were preserved. Disk recovered from about `8.2G`
  to `16G` free.
- Storage cleanup before home launch: after sports/toys LLMEmb and LLM-ESR
  final server audits and local-light packages were verified as PASS, three
  completed upstream staging directories were removed after exact realpath
  allowlist checks:
  `/home/ajifang/projects/LLMEmb/data/sports_llmemb_same_candidate_100neg`,
  `/home/ajifang/projects/LLMEmb/data/toys_llmemb_same_candidate_100neg`, and
  `/home/ajifang/projects/LLM-ESR/data/sports_same_candidate_100neg`. The
  cleanup manifest is
  `outputs/summary/upstream_completed_sports_toys_llmemb_llmesr_cleanup_manifest_20260602.sha256`.
  Final server evidence (`outputs/*official*` scores, provenance, audits,
  predictions, imported tables, and models) and local lightweight packages were
  not deleted. Disk recovered from about `5.9G` free to `17G` free before the
  home launch.
- Latest completed toys row: `llmesr_sasrec`, completed 2026-06-02 18:59 CST
  after the disk-full recovery. It passed with
  `implementation_status=official_completed`, `blockers=[]`, exact
  `score_coverage_rate=1.0`, server-final audit PASS, lightweight sync PASS,
  local-light audit PASS, and no local forbidden large files. Full metrics over
  10,000 users and 101 candidates: HR@5/10/20
  `0.0637 / 0.1172 / 0.2203`, NDCG@5/10/20
  `0.037504900117522603 / 0.05456849726033091 / 0.08036871527121744`, and MRR
  `0.05844977379835533`. Row counts passed: `scores.csv` `1,010,001` lines,
  predictions `10,000` lines, and `tables/ranking_eval_records.csv`
  `10,001` lines. The local lightweight evidence package is
  `outputs/baselines/official_adapters/toys_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/`;
  sync manifest `ok=true`, 11 allowed files matched by size/sha256, and 3
  server-only large files were intentionally excluded: `scores.csv` (100M),
  `predictions/rank_predictions.jsonl` (779M), and
  `llmesr_official_model.pt` (3.6G). The server large-artifact manifest records
  sha256 for all three excluded files. After final server/local gates passed,
  the completed intermediate adapter
  `outputs/baselines/paper_adapters/toys_large10000_100neg_llmesr_official_adapter`
  was removed after writing
  `outputs/summary/toys_llmesr_completed_adapter_cleanup_manifest_20260602.sha256`;
  final scores, provenance, audits, predictions, imported tables, and model
  were preserved. Disk recovered from about `1.6G` free to `5.9G` free.
- Previous completed toys row: `llm2rec_sasrec`, completed 2026-06-02 16:18 CST
  after a disk-full recovery. It passed with `implementation_status=official_completed`,
  `blockers=[]`, exact `score_coverage_rate=1.0`, server-final audit PASS,
  lightweight sync PASS, local-light audit PASS, and no local forbidden large
  files. Full metrics over 10,000 users and 101 candidates:
  HR@5/10/20 `0.2202 / 0.3172 / 0.4652`,
  NDCG@5/10/20
  `0.1475691807818137 / 0.17887285724512209 / 0.21609262826220665`,
  MRR `0.15921596430464027`. Row counts passed: `scores.csv` `1,010,001`
  lines, predictions `10,000` lines, `tables/ranking_eval_records.csv`
  `10,001` lines, and metrics/coverage/summary tables each have one data row.
  The local lightweight evidence package is
  `outputs/baselines/official_adapters/toys_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate/`.
  It contains 11 server-matched lightweight evidence files plus local audit and
  sync manifest; server-only excluded files are `scores.csv` (101M),
  `predictions/rank_predictions.jsonl` (780M), and the 4.18G official SASRec
  checkpoint. To recover space without touching final evidence, the completed
  LLM2Rec adapter intermediate CSVs `candidate_items_mapped.csv` and
  `item_text_seed.csv` were gzip-compressed in place, and the old sports
  upstream LLM2Rec item-info embedding cache was removed after recording
  sha256 `41e968bc31de1454eb3deab08eff6e06e1d68308d7ed2b25137f0b377f6b9a2c`
  in `outputs/summary/sports_llm2rec_upstream_embedding_cache_cleanup_manifest_20260602.sha256`.
  Final scores, provenance, audits, predictions, imported tables, checkpoints,
  and the toys adapter embedding were not deleted.
- Previous completed toys row: `rlmrec_graphcl`, completed 2026-06-02 12:00 CST
  with `implementation_status=official_completed`, `blockers=[]`, exact
  `score_coverage_rate=1.0`, server-final audit PASS, lightweight sync PASS,
  local-light audit PASS, and no local forbidden large files. Full metrics over
  10,000 users and 101 candidates:
  HR@5/10/20 `0.1281 / 0.1885 / 0.3050`,
  NDCG@5/10/20
  `0.08716027936492049 / 0.10650871155525234 / 0.1353997243006345`,
  MRR `0.1058452782968119`. Row counts passed: `scores.csv` `1,010,001`
  lines, predictions `10,000` lines, `tables/ranking_eval_records.csv`
  `10,001` lines, and metrics/coverage/summary tables each have one data row.
  The local lightweight evidence package is
  `outputs/baselines/official_adapters/toys_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate/`.
  It contains 11 useful evidence files: final provenance, server-final audit,
  local-light audit, sync manifest, score audit JSON/TXT, run summary, imported
  metric/coverage/exposure/eval/summary tables, and the server-side
  large-artifact sha256 manifest. Server-only large files are protected by
  `server_large_artifact_manifest.sha256`: `scores.csv` (102M),
  `predictions/rank_predictions.jsonl` (781M), and
  `rlmrec_official_model.pt` (60M). After final evidence and local backup gates
  passed, the completed intermediate adapter
  `outputs/baselines/paper_adapters/toys_large10000_100neg_rlmrec_official_adapter`
  was removed, recovering disk from about `4.5G` free to `8.8G` free without
  deleting final scores, provenance, audits, predictions, imported tables, or
  model. A follow-up server/local hash check passed for the five earlier toys
  lightweight packages as well: each had 11 allowed files matching by size and
  sha256 and 4 server-only excluded large files.
- Toys C-CRP v3 evidence note: the core result directory
  `outputs/toys_large10000_100neg_ccrp_v3` exists on the server with complete
  full metrics in `report.json`, `scores.csv` line count `1,010,001`, and
  `user_ranks.jsonl` line count `10,000`. Metrics are HR@5/10/20
  `0.3172 / 0.3964 / 0.5059`, NDCG@5/10/20
  `0.2451904009717959 / 0.27079859856897753 / 0.298341205798594`, and MRR
  `0.2503049488607351` over `n_users=10000` and `n_prompts=1010000`. At
  2026-06-02 17:09 CST the C-CRP scores were imported through the existing
  same-candidate importer into
  `outputs/toys_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate`
  without `--allow_partial_scores`; the import reported
  `score_coverage_rate=1.000000`. A follow-up domain gate wrote
  `outputs/summary/toys_official_gate_after_ccrp_import_pending_llmesr_20260602_1709.{json,csv}`
  while LLM-ESR was still pending. After LLM-ESR completion, the final toys
  domain gate wrote `outputs/summary/toys_official_gate_final_20260602_1900.{json,csv}`
  and returned `ccrp_ok=true`, `official_ok_count=8`,
  `official_all_ok=true`, and `gate_ok=true`. The local lightweight C-CRP
  import package contains the five imported tables under
  `outputs/toys_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate/tables/`
  plus the two gate summary files under `outputs/summary/`; server/local
  sha256 checks matched. The 806M imported prediction JSONL remains
  server-only.
- Toys comparison/statistical gate follow-up: at 2026-06-02 19:07 CST,
  `scripts/experiments/main_build_domain_official_comparison.py` built the
  toys C-CRP-vs-8-official comparison and paired-test package under
  `outputs/summary/toys_official_ccrp_20260602_1900_*`. The comparison table
  has 9 methods and full HR@5/@10/@20, NDCG@5/@10/@20, and MRR. C-CRP ranks
  first and is observed-best on all seven metrics. The paired-test table has
  56 tests (8 official baselines x 7 metrics), all with `n_paired_events=10000`,
  positive deltas, bootstrap CIs above zero, and Holm-significant p-values.
  The closest official row is `llmemb`; the smallest margin is HR@20 delta
  `0.0193` with 95% CI `[0.008298, 0.029900]` and Holm p
  `0.0004061119980698498`. This is a toys-domain statistical gate only; do not
  generalize to paper-wide SOTA until the declared domain set is complete.
- Previous sports runner: Sports `llmesr_sasrec`
  launched 2026-06-01 16:13 CST as a single-row production run with runner PID
  `2877443` and adapter PID `2877452`; it finished at 2026-06-01 18:31 CST.
- Latest checked state: 2026-06-01 19:08 CST, sports official baselines are
  8/8 complete. LLM-ESR completed the Qwen3 `hf_mean_pool` embedding pass
  (`233470/233470`), ran the default 200-epoch official LLM-ESR training, and
  exported/imported exact same-candidate scores. The log ended with
  `[2026-06-01 18:31:16] DONE llmesr_sasrec on sports` and
  `=== All baseline runs complete ===`. No active Pony/baseline Python process
  remained at the cleanup preflight. After safe cleanup of the completed
  LLM-ESR intermediate adapter directory, disk recovered from `9.4G` free
  (`95%` used) to `14G` free (`93%` used). A follow-up read-only domain gate
  with `scripts/audit/main_audit_domain_official_gate.py` passed:
  `official_ok_count=8`, `ccrp_ok=true`, `gate_ok=true`, and no stray
  official-like sports output directories remained after removing the confirmed
  empty malformed directory. A follow-up sports comparison/statistical gate
  with `scripts/experiments/main_build_domain_official_comparison.py` also
  passed: C-CRP ranks first by NDCG@10 and is observed-best on all seven full
  metrics against the eight official baselines; all 56 C-CRP-vs-official paired
  tests are positive and Holm-significant. The closest official row is
  `llmemb` for all seven metrics; the smallest margin is HR@20 delta `0.0272`
  with 95% paired-bootstrap CI `[0.0164, 0.0386]` and Holm p
  `1.219129314796352e-06`.
- Storage preflight for next-domain baselines: at 2026-06-01 19:41 CST, no
  experiment process was active, but disk was only `14G` free (`93%` used).
  A read-only storage audit identified sports final evidence as protected and
  user-level caches as disposable. The caches
  `/home/ajifang/.cache/vllm`, `/home/ajifang/.cache/torch`,
  `/home/ajifang/.cache/google-chrome`, `/home/ajifang/.cache/mozilla`, and
  `/home/ajifang/.cache/JetBrains` were removed after path verification under
  `/home/ajifang/.cache`, recovering disk to about `19G` free. No final
  scores, provenance, imported tables, predictions, checkpoints, external task
  packages, or project outputs were deleted.
- Resolved LLM2Rec recovery: the full embedding artifact completed. Both
  `outputs/baselines/paper_adapters/sports_large10000_100neg_llm2rec_official_adapter/llm2rec_item_embeddings.npy`
  and upstream
  `/home/ajifang/projects/LLM2Rec/item_info/SportsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`
  are `4,649,140,352` bytes. Metadata records `items=283760`,
  `embedding_rows=283761`, `embedding_dim=4096`,
  `embedding_text_coverage=1.0`, and
  `valid_history_source=valid_task_train_interactions` from the earlier
  adapter audit. Local runner fix: `src/baselines/official_runner/llm2rec.py`
  now starts the official entrypoint with `sys.executable` instead of bare
  `python`. Targeted tests passed:
  `tests/test_llm2rec_upstream_adapter.py` (`5 passed`) and
  `tests/test_llm2rec_same_candidate_export.py` (`3 passed`).
- Latest completed row: sports `llm2rec_sasrec`, completed 2026-06-01
  15:56 CST with `implementation_status=official_completed`, `blockers=[]`,
  `score_coverage_rate=1.0`, server-final audit PASS, lightweight sync PASS,
  and local-light audit PASS.
- Resolved blocker/recovery: sports `llm2rec_sasrec` previously failed during adapter
  export because the exporter mapped validation candidate events through the
  test-task `train_interactions.csv`; at least one validation user exists only
  in `sports_large10000_100neg_valid_same_candidate/train_interactions.csv`.
  Local fix: `main_export_llm2rec_same_candidate_task.py` now uses
  `valid_task_dir/train_interactions.csv` for validation histories when
  present and keeps the test-task train history for test events. Targeted unit
  test `test_llm2rec_export_uses_separate_validation_task` now covers disjoint
  valid/test users and passed with
  `PYTHONPATH=scripts/build;scripts/audit;scripts/adapters;. python -m pytest
  tests/test_llm2rec_same_candidate_export.py -q`. The fix was copied to the
  server after commit `657929e`; server lacks `pytest`, but `py_compile`
  passed and the resumed official row completed all final gates.
- Storage cleanup: after verifying RLMRec server-final audit `ok=true` and
  absolute target path under
  `~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/paper_adapters/`,
  the completed RLMRec intermediate adapter directory
  `outputs/baselines/paper_adapters/sports_large10000_100neg_rlmrec_official_adapter`
  was removed. This recovered about `4.5G` and did not touch final RLMRec
  scores, provenance, audits, imported tables, predictions, or local
  lightweight evidence.
- Storage cleanup follow-up: after LLM2Rec server-final and local-light audits
  passed, the completed LLM2Rec intermediate adapter directory
  `outputs/baselines/paper_adapters/sports_large10000_100neg_llm2rec_official_adapter`
  was removed. This recovered about `5.3G` and did not touch final LLM2Rec
  scores, final provenance, audits, imported tables, predictions, checkpoints,
  or the upstream embedding under `/home/ajifang/projects/LLM2Rec/item_info/`.
- Evidence backfill follow-up: at 2026-06-01 16:50 CST, the four earlier
  completed sports official rows (`llmemb`, `proex_profile`, `promax_profile`,
  and `elmrec_graph`) were rechecked because their server directories had full
  metrics, score audits, provenance, scores, predictions, and imported tables
  but lacked `server_final_evidence_audit.json`. Server-final audits were run
  in place and all four returned `ok=true`, `failures=[]`, complete
  HR@5/@10/@20, NDCG@5/@10/@20, MRR, `sample_count=10000`,
  `avg_candidates=101.0`, and `score_coverage_rate=1.0`. The new audit JSONs
  were copied into the local lightweight packages, and local-light audits
  passed for all four. No scores or experiment processes were changed.
- Evidence package verification follow-up: at 2026-06-01 17:45 CST, the
  allowlist sync verifier compared all seven completed sports local
  lightweight packages against the server evidence directories by size and
  sha256. All seven returned `failures=0`. The completed rows each had 11
  allowed lightweight files and 3 excluded server-only large files, except
  `llm2rec_sasrec`, which had 12 allowed lightweight files and 3 excluded
  server-only large files. The four earliest packages now also include tracked
  `light_evidence_sync_manifest.json` files. Excluded server-only files remain
  protected on the server (`scores.csv`, predictions, checkpoints/large
  binaries); no experiment output scores were changed.
- LLM-ESR completion and cleanup follow-up: at 2026-06-01 18:31 CST, sports
  `llmesr_sasrec` completed as the eighth sports official row with
  `implementation_status=official_completed`, `blockers=[]`, and
  `score_coverage_rate=1.0`. Server-final audit PASS, lightweight sync PASS,
  and local-light audit PASS. Full metrics over 10,000 users and 101
  candidates:
  HR@5/10/20 `0.0916 / 0.1564 / 0.2650`,
  NDCG@5/10/20 `0.054919833257876506 / 0.0758115528438973 / 0.10310478593304104`,
  MRR `0.0751149958885503`. Row counts passed: `scores.csv` `1,010,001`
  lines, predictions `10,000` lines, `tables/ranking_eval_records.csv`
  `10,001` lines, and summary table `2` lines. The local lightweight package
  is under
  `outputs/baselines/official_adapters/sports_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/`.
  After verifying no active LLM-ESR process, server-final audit `ok=true`, and
  final score/provenance files present, the completed intermediate adapter
  directory
  `outputs/baselines/paper_adapters/sports_large10000_100neg_llmesr_official_adapter`
  was removed; final server outputs and local lightweight evidence were
  preserved.
- Sports domain gate follow-up: at 2026-06-01 19:08 CST, the new read-only
  domain gate generated
  `outputs/summary/sports_official_ccrp_gate_20260601.json` and `.csv` on the
  server and copied both light summaries locally. The gate verified all eight
  official rows plus `ccrp_v3_qwen3base_pointwise` have complete
  HR@5/@10/@20, NDCG@5/@10/@20, MRR, `sample_count=10000`,
  `avg_candidates=101.0`, `score_coverage_rate=1.0`, expected row counts
  (`scores.csv` `1,010,001`, predictions `10,000`,
  `ranking_eval_records.csv` `10,001`), and no failures. The stale non-experiment
  bash diagnostic process from an earlier malformed grep command was cleaned,
  and the confirmed empty malformed directory
  `outputs/sports_large10000_100neg_TRAIN_METHODS_OVERRIDE=_official_qwen3base_same_candidate`
  was removed. No experiment process, final score, provenance, or imported
  table was touched.
- Sports comparison/statistical gate follow-up: at 2026-06-01 19:20 CST,
  `scripts/experiments/main_build_domain_official_comparison.py` generated and
  synced
  `outputs/summary/sports_official_ccrp_20260601_comparison.csv`,
  `outputs/summary/sports_official_ccrp_20260601_comparison.md`,
  `outputs/summary/sports_official_ccrp_20260601_paired_tests.csv`, and
  `outputs/summary/sports_official_ccrp_20260601_paired_summary.json`. The
  comparison table has 9 methods (C-CRP + 8 official baselines) and full
  HR@5/@10/@20, NDCG@5/@10/@20, and MRR. C-CRP is rank 1 and observed-best on
  all seven metrics. The paired-test table has 56 tests (8 baselines x 7
  metrics), all with `n_paired_events=10000`, positive deltas, 95% paired
  bootstrap CIs above zero, and Holm-significant p-values. This supports a
  sports-domain claim only; paper-wide SOTA wording still requires the
  declared domain set and ARIS review.
- Toys official-baseline launch follow-up: at 2026-06-01 19:44 CST, after the
  cache cleanup and a no-active-experiment preflight, toys `proex_profile` was
  launched as the next official row with:
  `nohup env DOMAINS_OVERRIDE=toys FAST_METHODS_OVERRIDE=proex_profile TRAIN_METHODS_OVERRIDE= bash scripts/run_baselines_new_domains.sh`.
  This intentionally runs one low-space-risk row rather than the whole toys
  domain, because disk remains tight and LLM2Rec/LLM-ESR/LLMEmb create large
  protected checkpoints. At the 19:48 CST check, runner PID `2893793` and
  adapter PID `2893803` were active, embedding progress was about
  `7088/215034`, GPU was `95%`, and disk was about `18G` free. Do not start
  another baseline until this row finishes or fails and has been audited.
- Toys ProEx monitoring follow-up: at 2026-06-01 20:17 CST, runner PID
  `2893793` and adapter PID `2893803` were still active. The log had advanced
  to about `113608/215034` Qwen3 `hf_mean_pool` embeddings, GPU was `96%`
  with `16285 MiB / 49140 MiB`, disk was `18G` free (`91%` used), and the
  error scan showed only the known model-loading `UNEXPECTED` notes, not
  traceback/OOM/no-space/fatal markers. The final toys ProEx evidence directory
  still had no final `scores.csv`, `fairness_provenance.json`, score audit,
  imported tables, or row-countable predictions, so the row remains running
  and not table-eligible. At that checkpoint, sports was still the only domain
  with all eight official final provenance packages and toys/home/tools had no
  completed official baseline rows; this historical status has since been
  superseded by the toys 8/8 completion and toys domain/statistical gate.
- Toys ProEx completion/package follow-up: at 2026-06-01 21:11 CST, toys
  `proex_profile` completed as `implementation_status=official_completed`,
  `blockers=[]`, and `score_coverage_rate=1.0`. Server-final evidence audit
  passed with full metrics over 10,000 users and 101 candidates:
  HR@5/10/20 `0.0895 / 0.1615 / 0.3017`, NDCG@5/10/20
  `0.058141214365017416 / 0.0810170703641553 / 0.11607709818340411`, and MRR
  `0.08121671352544663`. Row counts passed: `scores.csv` `1,010,001` lines,
  predictions `10,000` lines, and `tables/ranking_eval_records.csv` `10,001`
  lines. Local lightweight evidence sync and local-light audit passed under
  `outputs/baselines/official_adapters/toys_large10000_100neg_proex_profile_official_qwen3base_same_candidate/`;
  the sync manifest has `allowed_file_count=11`, `excluded_file_count=4`, and
  `failures=0`, including a server-side large-artifact sha256 manifest for
  `scores.csv`, predictions, and `proex_official_model.pt`. After verifying no
  active toys ProEx process and protected final outputs, the intermediate
  adapter directory
  `outputs/baselines/paper_adapters/toys_large10000_100neg_proex_official_adapter`
  was removed, recovering disk from about `14G` to `18G` free. Final server
  scores, provenance, audits, predictions, imported tables, and model were not
  deleted.
- Toys ProMax launch follow-up: at 2026-06-01 21:28 CST, after a no-active
  experiment preflight and the toys ProEx package/cleanup gate, toys
  `promax_profile` was launched as the next low-space-risk official row with:
  `nohup env DOMAINS_OVERRIDE=toys FAST_METHODS_OVERRIDE=promax_profile TRAIN_METHODS_OVERRIDE= bash scripts/run_baselines_new_domains.sh`.
  The first launch command timed out locally but did start the intended row; a
  follow-up process check found adapter PID `2899998` under runner PID
  `2899989`, and PID files were corrected to
  `baselines_new_domains_toys_promax_adapter.pid` and
  `baselines_new_domains_toys_promax_runner.pid`. Log path:
  `baselines_new_domains_toys_promax_20260601_212808.log`. At the 21:31 CST
  check it was in Qwen3 `hf_mean_pool` embedding at about `1312/215034`, with
  only the known model-loading `UNEXPECTED` note and no fatal/OOM/no-space
  markers. Do not start another baseline until this row finishes or fails and
  has been audited.
- Toys ProMax completion/package follow-up: at 2026-06-02 00:02 CST, toys
  `promax_profile` completed as `implementation_status=official_completed`,
  `blockers=[]`, and `score_coverage_rate=1.0`. Server-final evidence audit
  passed with full metrics over 10,000 users and 101 candidates:
  HR@5/10/20 `0.0920 / 0.1435 / 0.2416`, NDCG@5/10/20
  `0.06289618254810064 / 0.07937554863319267 / 0.10387644003990415`, and MRR
  `0.08184625622431366`. Row counts passed: `scores.csv` `1,010,001` lines,
  predictions `10,000` lines, and `tables/ranking_eval_records.csv` `10,001`
  lines. Local lightweight evidence sync and local-light audit passed under
  `outputs/baselines/official_adapters/toys_large10000_100neg_promax_profile_official_qwen3base_same_candidate/`;
  the sync manifest has `allowed_file_count=11`, `excluded_file_count=4`, and
  `failures=0`, including a server-side large-artifact sha256 manifest for
  `scores.csv`, predictions, and `promax_official_model.pt`. After verifying
  no active toys ProMax process and protected final outputs, the intermediate
  adapter directory
  `outputs/baselines/paper_adapters/toys_large10000_100neg_promax_official_adapter`
  was removed, recovering disk from about `13G` to `17G` free. Final server
  scores, provenance, audits, predictions, imported tables, and model were not
  deleted.
- Toys ElmRec launch follow-up: at 2026-06-02 00:07 CST, after a no-active
  experiment preflight and the toys ProMax package/cleanup gate, toys
  `elmrec_graph` was launched as the next official row with:
  `nohup env DOMAINS_OVERRIDE=toys FAST_METHODS_OVERRIDE= TRAIN_METHODS_OVERRIDE=elmrec_graph bash scripts/run_baselines_new_domains.sh`.
  The first launch command timed out locally but did start the intended row; a
  follow-up process check found adapter PID `2906455` under runner PID
  `2906447`, and PID files were corrected to
  `baselines_new_domains_toys_elmrec_adapter.pid` and
  `baselines_new_domains_toys_elmrec_runner.pid`. Log path:
  `baselines_new_domains_toys_elmrec_20260602_000729.log`. At the 00:10 CST
  check it was in Qwen3 `hf_mean_pool` embedding at about `3624/215034`, with
  only the known model-loading `UNEXPECTED` note and no fatal/OOM/no-space
  markers. Do not start another baseline until this row finishes or fails and
  has been audited.
- Toys ElmRec monitor follow-up: at 2026-06-02 00:15 CST, runner PID `2906447`
  and adapter PID `2906455` were still active. The log had advanced to about
  `21872/215034` Qwen3 `hf_mean_pool` embeddings, GPU was `95%` with
  `16213 MiB / 49140 MiB`, disk was `16G` free (`92%` used), and the error
  scan showed only the known model-loading `UNEXPECTED` note, not
  traceback/OOM/no-space/fatal markers. The final toys ElmRec evidence
  directory still had no final score/provenance/audit/import package, so the
  row remains running and not table-eligible.
- Toys ElmRec completion/package follow-up: at 2026-06-02 01:36 CST, toys
  `elmrec_graph` completed as `implementation_status=official_completed`,
  `blockers=[]`, and `score_coverage_rate=1.0`. Server-final evidence audit
  passed with full metrics over 10,000 users and 101 candidates:
  HR@5/10/20 `0.0545 / 0.1043 / 0.2013`, NDCG@5/10/20
  `0.03259298673054038 / 0.04856005753116525 / 0.07278039157879498`, and MRR
  `0.05431081812612059`. Row counts passed: `scores.csv` `1,010,001` lines,
  predictions `10,000` lines, and `tables/ranking_eval_records.csv` `10,001`
  lines. Local lightweight evidence sync and local-light audit passed under
  `outputs/baselines/official_adapters/toys_large10000_100neg_elmrec_graph_official_qwen3base_same_candidate/`;
  the sync manifest has `allowed_file_count=11`, `excluded_file_count=4`, and
  `failures=0`, including a server-side large-artifact sha256 manifest for
  `scores.csv`, predictions, and `elmrec_official_model.pt`. After verifying no
  active toys ElmRec process, server-final audit `ok=true`, and protected final
  outputs, the intermediate adapter directory
  `outputs/baselines/paper_adapters/toys_large10000_100neg_elmrec_official_adapter`
  was removed, recovering disk from about `12G` to `16G` free. Final server
  scores, provenance, audits, predictions, imported tables, and model were not
  deleted.
- Toys LLMEmb launch follow-up: at 2026-06-02 01:43 CST, after a no-active
  experiment preflight and the toys ElmRec package/cleanup gate, toys `llmemb`
  was launched as the next disk-aware single-row official baseline with:
  `nohup env DOMAINS_OVERRIDE=toys FAST_METHODS_OVERRIDE=llmemb TRAIN_METHODS_OVERRIDE= bash scripts/run_baselines_new_domains.sh`.
  The local SSH command timed out while the remote process continued, so PID
  files were corrected after a process check. Runner PID is `2915438`, adapter
  PID is `2915450`, and log path is
  `baselines_new_domains_toys_llmemb_20260602_014334.log`. At the 01:46 CST
  check it was in Qwen3 `hf_mean_pool` embedding at about `1592/215034`, GPU
  was `95%` with `15945 MiB / 49140 MiB`, disk was `15G` free (`92%` used),
  and no final score/provenance/audit/import package existed yet. Do not start
  another baseline until this row finishes or fails and has been audited.
- Toys LLMEmb completion/package follow-up: at 2026-06-02 03:04 CST, toys
  `llmemb` completed as `implementation_status=official_completed`,
  `blockers=[]`, and `score_coverage_rate=1.0`. Server-final evidence audit
  passed with full metrics over 10,000 users and 101 candidates:
  HR@5/10/20 `0.2499 / 0.3505 / 0.4866`, NDCG@5/10/20
  `0.17252113274887534 / 0.20485045979333913 / 0.23905481091819092`, and MRR
  `0.1813804118284203`. Row counts passed: `scores.csv` `1,010,001` lines,
  predictions `10,000` lines, and `tables/ranking_eval_records.csv` `10,001`
  lines. Local lightweight evidence sync and local-light audit passed under
  `outputs/baselines/official_adapters/toys_large10000_100neg_llmemb_official_qwen3base_same_candidate/`;
  the sync manifest has `allowed_file_count=11`, `excluded_file_count=4`, and
  `failures=0`. The local package keeps provenance, run summary, score audits,
  server-final audit, server large-artifact manifest, ranking metrics, coverage,
  exposure, and per-user ranking eval records; it intentionally excludes
  server-only `scores.csv`, predictions, `llmemb_official_model.pt`, and
  `server_large_artifact_sizes.txt`. After verifying no active LLMEmb Python
  process, server-final audit `ok=true`, local-light audit `ok=true`, and
  protected final outputs, the completed intermediate adapter directory
  `outputs/baselines/paper_adapters/toys_large10000_100neg_llmemb_official_adapter`
  was removed, recovering disk from about `4.0G` to `8.3G` free. Final server
  scores, provenance, audits, predictions, imported tables, and model were not
  deleted.
- Toys IRLLRec launch follow-up: at 2026-06-02 03:16 CST, after a no-active
  official Python process check and the toys LLMEmb package/cleanup gate, toys
  `irllrec_intent` was launched as the next single-row official baseline with:
  `nohup env DOMAINS_OVERRIDE=toys FAST_METHODS_OVERRIDE= TRAIN_METHODS_OVERRIDE=irllrec_intent bash scripts/run_baselines_new_domains.sh`.
  The local SSH command timed out while the remote process continued, so PID
  files were corrected after a process check. Runner PID is `2923429`, adapter
  PID is `2923437`, and log path is
  `baselines_new_domains_toys_irllrec_20260602_031623.log`. At the 03:19 CST
  check it was in Qwen3 `hf_mean_pool` embedding at about `1400/215034`, GPU
  was `96%` with `15945 MiB / 49140 MiB`, disk was `7.3G` free (`97%` used),
  and no final score/provenance/audit/import package existed yet. Do not start
  another baseline until this row finishes or fails and has been audited.
- IRLLRec storage-risk follow-up: at 2026-06-02 04:12 CST, toys IRLLRec had
  entered official training (`epoch=30`) after completing Qwen3 embeddings.
  Its active intermediate adapter directory had grown to about `4.3G`, final
  evidence still had no score/provenance package, and disk was down to about
  `4.0G` free (`98%` used). To reduce no-space risk without touching project
  evidence, final outputs, active adapters, or other projects, only clearly
  disposable pip cache/temp paths were removed after realpath/scope checks:
  `/home/ajifang/.cache/pip` (`117M`) and `/tmp/pip-unpack-920865s3` (`314M`).
  Disk recovered to about `4.4G` free. The old
  `outputs/baselines/paper_adapters/books_large10000_100neg_llmesr_adapter`
  directory was observed at about `1.3G` but was not removed because its final
  evidence/backup status was not audited in this cycle.
- IRLLRec monitoring/cleanup decision follow-up: at 2026-06-02 04:35 CST,
  toys IRLLRec was still healthy under runner PID `2923429` and adapter PID
  `2923437`. The log had reached official training `epoch=500` of the default
  `3000` epochs, with no traceback/OOM/killed/no-space/fatal markers. GPU was
  active (`94%`, `16295 MiB / 49140 MiB`) and disk remained tight at about
  `4.4G` free (`98%` used). The final evidence directory still had no
  `scores.csv`, provenance, audit JSON, imported tables, or row-countable
  predictions, so no lightweight sync or table claim is allowed yet. A
  read-only cleanup audit inspected the old
  `outputs/baselines/paper_adapters/books_large10000_100neg_llmesr_adapter`
  directory: it contains `adapter_metadata.json`, `candidate_items_mapped.csv`
  (`950M`), `item_text_seed.csv` (`350M`), maps, and LLM-ESR handled data,
  while the corresponding final books directory is table-only and no local
  lightweight evidence package was found. Therefore it is not classified as
  disposable garbage in this cycle. The active IRLLRec adapter and all final
  evidence directories were left untouched.
- IRLLRec cache-cleanup follow-up: at 2026-06-02 04:47 CST, toys IRLLRec was
  still running and had reached `epoch=760/3000`; the final evidence directory
  still had no score/provenance/audit/import package. Because disk remained
  tight, three user-level cache directories were removed after realpath
  allowlist checks: `/home/ajifang/.vscode-server/data/CachedExtensionVSIXs`
  (`148M`), `/home/ajifang/.config/google-chrome/component_crx_cache` (`44M`),
  and `/home/ajifang/.config/Code/CachedData` (`11M`). Disk recovered from
  about `4.4G` to `4.6G` free. No project outputs, evidence directories,
  active adapters, models, Python site-packages, or other projects were
  removed.
- IRLLRec IDE-cache cleanup follow-up: at 2026-06-02 04:56 CST, toys IRLLRec
  was still running and had reached `epoch=940/3000`; no final evidence files
  existed yet and the error scan remained clean. To give the final
  scores/predictions export more disk headroom, five inactive VSCode remote
  server cache directories under
  `/home/ajifang/.vscode-server/cli/servers/Stable-*` were removed after a
  no-process check and realpath prefix verification. This recovered disk from
  about `4.6G` to `6.4G` free. The cleanup did not touch project outputs,
  final evidence, active adapters, models, conda/Python environments, or other
  projects. VSCode remote server binaries can be reinstalled by VSCode if
  needed later.
- Toys IRLLRec completion/package follow-up: at 2026-06-02 06:35 CST, toys
  `irllrec_intent` completed as `implementation_status=official_completed`,
  `blockers=[]`, and `score_coverage_rate=1.0`. Server-final audit passed with
  `ok=true`, `failures=[]`, full metrics over 10,000 users and 101 candidates:
  HR@5/10/20 `0.1565 / 0.2293 / 0.4098`, NDCG@5/10/20
  `0.11049209461545026 / 0.13380144693674725 / 0.1785851471792316`, and MRR
  `0.1311986744710446`. Row counts passed for `scores.csv` (`1,010,001`
  lines), predictions (`10,000` lines), and
  `tables/ranking_eval_records.csv` (`10,001` lines). A server-side sha256
  manifest records server-only `scores.csv`,
  `predictions/rank_predictions.jsonl`, and `irllrec_official_model.pt`.
  Lightweight sync and local-light audit passed under
  `outputs/baselines/official_adapters/toys_large10000_100neg_irllrec_intent_official_qwen3base_same_candidate/`
  with 11 allowlist files matched by size and sha256 and 4 excluded server-only
  files. After verifying no active IRLLRec Python process, protected final
  evidence, and realpath scope under `outputs/baselines/paper_adapters/`, the
  completed intermediate adapter directory
  `outputs/baselines/paper_adapters/toys_large10000_100neg_irllrec_official_adapter`
  was removed. Disk recovered from about `5.4G` to `9.7G` free; final scores,
  provenance, audits, predictions, imported tables, and model were not deleted.
- Toys RLMRec launch follow-up: at 2026-06-02 06:44 CST, after the IRLLRec
  gate/package/cleanup preflight, toys `rlmrec_graphcl` was launched as the
  next single-row official baseline with:
  `nohup env DOMAINS_OVERRIDE=toys FAST_METHODS_OVERRIDE= TRAIN_METHODS_OVERRIDE=rlmrec_graphcl bash scripts/run_baselines_new_domains.sh`.
  The local SSH command timed out while the remote job continued, so PID files
  were corrected after process inspection. Runner PID is `2937284`, adapter PID
  is `2937292`, and log path is
  `baselines_new_domains_toys_rlmrec_20260602_064443.log`. At the 06:47 CST
  check it was active in Qwen3 `hf_mean_pool` embedding at about `1664/215034`,
  GPU was `99%` with `15945 MiB / 49140 MiB`, and disk was about `8.8G` free.
  Do not start another baseline until this row finishes or fails and has been
  audited.
- Warning note: graph normalization emitted the same zero-degree
  `divide by zero encountered in power` warning pattern seen in prior completed
  graph baselines; the implementation immediately maps `inf` inverse degrees
  to `0.0`, so this is recorded as a non-fatal warning unless later audit or
  outputs show score/provenance corruption.
- Follow-up preflight: 2026-06-01 07:21 CST server code still has real
  `run` dispatch branches for `rlmrec`, `llm2rec`, and `llmesr`; the sports
  inspect provenance for `rlmrec_graphcl`, `llm2rec_sasrec`, and
  `llmesr_sasrec` is `official_inspection_ready` with `blockers=[]`, pinned
  official repo commits, and aligned train/valid/test/candidate task sources.
  No server git pull or cleanup was performed while the active runner is live.
- Local preparation: `scripts/audit/main_sync_official_evidence_package.py`
  now provides a safe allowlist-based lightweight evidence sync. It was
  compiled and verified against the completed sports `llmemb` package:
  10 allowed lightweight files matched server/local size and sha256, while
  `scores.csv`, `predictions/rank_predictions.jsonl`, and the large checkpoint
  were excluded.
- Runner hygiene note: a read-only server scan found one empty malformed output
  directory,
  `outputs/sports_large10000_100neg_TRAIN_METHODS_OVERRIDE=_official_qwen3base_same_candidate/`.
  It contains no files and is not one of the eight official rows. The local
  runner now validates method tokens before creating output directories so a
  misquoted override cannot create another malformed method directory. No server
  pull or cleanup was performed while RLMRec is running.

## Sports Official Baselines

| Method | Status | Evidence status |
| --- | --- | --- |
| `llmemb` | complete | local lightweight package PASS; server-final package PASS |
| `proex_profile` | complete | local lightweight package PASS; server-final package PASS |
| `promax_profile` | complete | local lightweight package PASS; server-final package PASS |
| `elmrec_graph` | complete | local lightweight package PASS; server-final package PASS |
| `irllrec_intent` | complete | local lightweight package PASS; server-final package PASS |
| `rlmrec_graphcl` | complete | local lightweight package PASS; server-final package PASS |
| `llm2rec_sasrec` | complete | local lightweight package PASS; server-final package PASS |
| `llmesr_sasrec` | complete | local lightweight package PASS; server-final package PASS |

Completed sports rows have server-side `scores.csv` line count `1,010,001`,
`predictions/rank_predictions.jsonl` line count `10,000`, final provenance,
score audits, full metric tables, coverage/exposure tables, and
`tables/ranking_eval_records.csv`.
RLMRec, LLM2Rec, and LLM-ESR are now completed rows. Sports official baselines
are 8/8 complete, and the sports domain gate plus comparison/paired-test gate
passed for the eight official rows plus C-CRP. Sports can be labeled a
sports-domain passed gate. Paper-wide SOTA wording remains blocked until the
declared domain set has aligned baseline results, comparison tables, paired
tests, and ARIS review.

## Toys Official Baselines

| Method | Status | Evidence status |
| --- | --- | --- |
| `proex_profile` | complete | server-final package PASS; local lightweight package PASS; full @5/@10/@20 + MRR metrics and row counts recorded |
| `llmemb` | complete | server-final package PASS; local lightweight package PASS; full @5/@10/@20 + MRR metrics and row counts recorded |
| `promax_profile` | complete | server-final package PASS; local lightweight package PASS; full @5/@10/@20 + MRR metrics and row counts recorded |
| `elmrec_graph` | complete | server-final package PASS; local lightweight package PASS; full @5/@10/@20 + MRR metrics and row counts recorded |
| `irllrec_intent` | complete | server-final package PASS; local lightweight package PASS; full @5/@10/@20 + MRR metrics and row counts recorded |
| `rlmrec_graphcl` | complete | server-final package PASS; local lightweight package PASS; full @5/@10/@20 + MRR metrics and row counts recorded |
| `llm2rec_sasrec` | complete | server-final package PASS; local lightweight package PASS; full @5/@10/@20 + MRR metrics and row counts recorded |
| `llmesr_sasrec` | complete | server-final package PASS; local lightweight package PASS; full @5/@10/@20 + MRR metrics and row counts recorded; final toys domain/comparison/paired-test gate PASS |

Toys official baselines are now 8/8 complete (`proex_profile`, `llmemb`,
`promax_profile`, `elmrec_graph`, `irllrec_intent`, `rlmrec_graphcl`,
`llm2rec_sasrec`, and `llmesr_sasrec`). Toys C-CRP imported evidence and all
eight official rows pass the domain gate, and the toys comparison/paired-test
gate passes with C-CRP rank 1 on all seven metrics and all 56 C-CRP-vs-official
paired tests positive and Holm-significant. This supports a toys-domain passed
gate only; paper-wide SOTA wording remains blocked until the declared domain
set is complete.

## Home Official Baselines

| Method | Status | Evidence status |
| --- | --- | --- |
| `proex_profile` | complete | server-final package PASS; local lightweight package PASS; full @5/@10/@20 + MRR metrics and row counts recorded |
| `promax_profile` | complete | server-final package PASS; local lightweight package PASS; full @5/@10/@20 + MRR metrics and row counts recorded |
| `elmrec_graph` | complete | server-final package PASS; local lightweight package PASS; full @5/@10/@20 + MRR metrics and row counts recorded |
| `llmemb` | complete | server-final package PASS; local lightweight package PASS; full @5/@10/@20 + MRR metrics and row counts recorded after disk-full recovery |
| `irllrec_intent` | complete | server-final package PASS; local lightweight package PASS; full @5/@10/@20 + MRR metrics and row counts recorded after disk-full recovery and adapter cleanup |
| `rlmrec_graphcl` | complete | server-final package PASS; server large-artifact sha256 manifest PASS; local lightweight package PASS; local-light audit PASS; full @5/@10/@20 + MRR metrics and row counts recorded |
| `llm2rec_sasrec` | complete | server-final package PASS; server large-artifact sha256 manifest PASS; local lightweight package PASS; local-light audit PASS; full @5/@10/@20 + MRR metrics and row counts recorded after disk-full recovery; post-gate server prediction JSONL deleted with sha256 manifest while preserving scores/provenance/audits/tables/checkpoint |
| `llmesr_sasrec` | complete | server-final package PASS; server large-artifact sha256 manifest PASS; local lightweight package PASS; local-light audit PASS; full @5/@10/@20 + MRR metrics and row counts recorded after disk-full finish; post-gate prediction JSONL and temporary adapter staging directory deleted with sha256 manifest while preserving scores/provenance/audits/tables/final model |

Home official baselines are now 8/8 row-gated complete (`proex_profile`,
`promax_profile`, `elmrec_graph`, `llmemb`, `irllrec_intent`,
`rlmrec_graphcl`, `llm2rec_sasrec`, `llmesr_sasrec`). All completed rows
passed final provenance, exact score coverage, server-final package audit,
lightweight local sync, local-light audit, full metrics, and row-count gates.

Home domain gate follow-up 2026-06-04 13:50 CST: C-CRP raw scores were imported
into
`outputs/home_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate`
with exact `score_coverage_rate=1.0`; the import reported NDCG@10
`0.13239420796539653`, wrote `predictions/rank_predictions.jsonl` with 10,000
rows and `tables/ranking_eval_records.csv` with 10,001 lines, and local light
sync copied raw `report.json`, `user_ranks.jsonl`, and the imported tables
while keeping `scores.csv` and the imported prediction JSONL server-side. The
read-only domain gate wrote
`outputs/summary/home_official_ccrp_gate_20260604.{json,csv}` and passed with
`official_ok_count=8`, `official_all_ok=true`, `ccrp_ok=true`, and
`gate_ok=true`. The comparison/statistical gate wrote
`outputs/summary/home_official_ccrp_20260604_*`; C-CRP ranks first on all
seven metrics, all 56 C-CRP-vs-official paired tests are positive and
Holm-significant, and `home_official_ccrp_20260604_paired_summary.json`
records `claim_gate=home_domain_pass`, `min_delta=0.0336`, `min_ci_low=0.0216`,
and `max_holm_p_value=1.0216927255359559e-08`. This supports a Home-domain
passed gate only; paper-wide SOTA wording remains blocked until Tools is
complete and all paper-critical modules/review gates pass.

Tools storage preflight cleanup 2026-06-04 14:20 CST: after confirming no
active Pony/C-CRP/baseline/uncertainty Python process, idle GPU, and Home gate
evidence present, disk was only about `7.9G` free. A read-only large-file audit
identified final models/checkpoints/task data as protected. Cleanup removed
only three server-only C-CRP imported prediction JSONLs for already
domain-gated sports/toys/home after writing and syncing row-local
`prediction_deletion_manifest.json` files:
`outputs/{sports,toys,home}_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate/prediction_deletion_manifest.json`.
The deleted files were sports `685174767` bytes
(`6ed1b78fbd5b056e4ee50c13bdcbaaa08c533bc9a6c1395188c97cc61390d783`),
toys `801921148` bytes
(`653af0d46c253bb84e58b83e22847b42c4bd8831b0490092417666fd1a5390db`),
and home `744081868` bytes
(`f70fd500113dfe3465e0dc373860fe4a690ec7c58badd63c64941207d668688a`).
Post-cleanup domain gates for sports/toys/home passed using the new manifest
certificate, and a Home comparison-builder smoke with `n_bootstrap=0` also
passed. Four obsolete home-root transfer archives were then deleted with
manifest `outputs/summary/server_home_root_archive_cleanup_for_tools_20260604.{json,sha256}`.
Final scores, raw C-CRP outputs, imported `tables/`, official evidence,
models/checkpoints, task data, source/config/docs, and other project
directories were preserved. Final cleanup preflight: no matching experiment
process, GPU idle, and `/` at `11,504,844,800` bytes free (`11G`, `95%` used).

Tools `proex_profile` launch: at 2026-06-04 14:25 CST, after the Home domain
gate was committed and pushed, a fresh duplicate-output check found no existing
Tools ProEx final directory, adapter directory, log, or PID file. The first
Tools official row was launched as a single-domain, single-method run:

```bash
nohup env DOMAINS_OVERRIDE=tools FAST_METHODS_OVERRIDE=proex_profile TRAIN_METHODS_OVERRIDE= bash scripts/run_baselines_new_domains.sh
```

The SSH launch timed out while backgrounding, so a clean adapter PID file was
written manually after verifying the active process. Current monitor target:
adapter PID `3269572`, PID file
`baselines_new_domains_tools_proex_20260604_142548.adapter.pid`, log
`baselines_new_domains_tools_proex_20260604_142548.log`. Stable checkpoint at
2026-06-04 14:31 CST: exactly one matching ProEx adapter process was active,
Qwen3 `hf_mean_pool` had reached `11104/269711`, GPU was about `95%` with
`16091 MiB / 49140 MiB`, no fatal/OOM/no-space/failed markers were present,
the active adapter directory was about `1005M`, and final output was still a
placeholder. Disk fell under the warning line during staging, so a bounded
cleanup removed only inactive old `outputs/baselines/paper_adapters/*`
directories while preserving the active Tools ProEx adapter. Manifest:
`outputs/summary/inactive_paper_adapters_cleanup_for_tools_proex_20260604.{json,sha256}`.
Post-cleanup `/` was `10,951,237,632` bytes free (`95%` used). Do not start
another baseline while this row is active.

Tools ProEx monitor checkpoint 2026-06-04 15:02 CST: adapter PID `3269572`
remained the only matching active experiment process, Qwen3 `hf_mean_pool`
progress had reached `138656/269711`, GPU was `96%` with
`16213 MiB / 49140 MiB`, `/` had `10,950,553,600` bytes free (`95%` used),
the active adapter directory was still about `1005M`, and the final output
directory was still placeholder-empty. No `DONE`, Traceback, OOM, no-space,
blocker, or failure markers were present. A guarded, non-executing completion
gate plan for this row was generated at
`outputs/summary/official_completion_gate_plan/tools_proex_profile_completion_gates_20260604.{json,ps1}`.
The PowerShell file starts with a `throw` and records the required order:
server-final audit, server large-artifact sha256 manifest, local-light sync,
then local-light audit. It must not be used until the active runner exits
normally and no duplicate process is present.

Tools ProEx completion/gate checkpoint 2026-06-04 16:40 CST:
`proex_profile` completed at 2026-06-04 16:08 CST as the first Tools official
row. Final provenance records `implementation_status=official_completed`,
`blockers=[]`, and exact `score_coverage_rate=1.0`. Server-final audit, server
large-artifact sha256 manifest, lightweight local sync, and local-light audit
all passed. Full metrics over 10,000 users and 101 candidates are HR@5/10/20
`0.0602 / 0.1177 / 0.2329`, NDCG@5/10/20
`0.037281705859706714 / 0.055676376797898205 / 0.08437492971571317`, and MRR
`0.06071849976691817`. Row counts passed for `scores.csv` (`1,010,001`
lines), predictions (`10,000` lines before post-gate deletion), and
`tables/ranking_eval_records.csv` (`10,001` lines). The local lightweight
evidence package is
`outputs/baselines/official_adapters/tools_large10000_100neg_proex_profile_official_qwen3base_same_candidate/`
and includes final provenance, score audits, run summary, server-final audit,
server large-artifact manifest, imported metric/coverage/exposure/eval-record
tables, local sync manifest, local-light audit, and the post-gate prediction
deletion manifest. Under disk pressure (`4.9G` free / `98%` used after
completion), the server-only prediction JSONL was removed only after the
server-final and local-light gates passed; its deletion manifest records
`805939218` bytes, `10,000` lines, and sha256
`9985f5fe3e99efed02d9e111ec53d3daf52e91800ff30b58f2d33ef089e4b312`. The
completed intermediate adapter
`outputs/baselines/paper_adapters/tools_large10000_100neg_proex_official_adapter`
was then removed after resolved-path checks. Final scores, provenance, audits,
imported `tables/`, and `proex_official_model.pt` were preserved. Post-cleanup
checks show no matching Python experiment process, GPU idle, and `/` at about
`12G` free / `95%` used. At this checkpoint, Tools was 1/8 official rows
complete and the next action was a fresh single-row ProMax preflight.

Tools ProMax launch checkpoint 2026-06-04 16:50 CST: after the ProEx gates,
local package, commit/push, and cleanup completed, a fresh preflight found no
matching experiment process, GPU idle, `/` at about `12G` free / `95%` used,
and no existing Tools ProMax final output, adapter directory, PID, or log.
Tools `promax_profile` was launched as the second Tools single-row official
baseline with:

```bash
nohup env DOMAINS_OVERRIDE=tools FAST_METHODS_OVERRIDE=promax_profile TRAIN_METHODS_OVERRIDE= bash scripts/run_baselines_new_domains.sh
```

The SSH wrapper timed out while backgrounding, so PID files were written after
process inspection. Initial monitor target: runner PID `3279573`, adapter PID
`3279582`, PID files
`baselines_new_domains_tools_promax_20260604_164630.runner.pid` and
`baselines_new_domains_tools_promax_20260604_164630.adapter.pid`, and log
`baselines_new_domains_tools_promax_20260604_164630.log`. Stable launch
checkpoint: exactly one matching ProMax adapter process was active, Qwen3
`hf_mean_pool` had reached `1336/269711`, GPU was `96%` with
`15947 MiB / 49140 MiB`, disk was about `11G` free / `95%` used, the adapter
directory was about `1005M`, and final output was still placeholder-only. No
`DONE`, Traceback, OOM, no-space, blocker, or failure markers were present.
The guarded completion-gate plan
`outputs/summary/official_completion_gate_plan/tools_promax_profile_completion_gates_20260604.{json,ps1}`
was verified by `python -m pytest tests\test_plan_official_completion_gates.py`
(`3 passed`) before the row completed.

Tools ProMax completion/gate checkpoint 2026-06-04 20:15 CST:
`promax_profile` completed at 2026-06-04 19:59 CST as the second Tools
official row. Final provenance records `implementation_status=official_completed`,
`blockers=[]`, and exact `score_coverage_rate=1.0`. Server-final audit,
server large-artifact sha256 manifest, lightweight local sync, and local-light
audit all passed. Full metrics over 10,000 users and 101 candidates are
HR@5/10/20 `0.056 / 0.1046 / 0.2018`, NDCG@5/10/20
`0.03468275603534166 / 0.05029722685396016 / 0.07458228366305956`, and MRR
`0.056527355267188224`. Row counts passed for `scores.csv` (`1,010,001`
lines), predictions (`10,000` lines before post-gate deletion), and
`tables/ranking_eval_records.csv` (`10,001` lines). The local lightweight
evidence package is
`outputs/baselines/official_adapters/tools_large10000_100neg_promax_profile_official_qwen3base_same_candidate/`
and includes final provenance, score audits, run summary, server-final audit,
server large-artifact manifest, imported metric/coverage/exposure/eval-record
tables, local sync manifest, local-light audit, and the post-gate prediction
deletion manifest. The large-artifact manifest records server-side `scores.csv`
sha256 `2a9797b945fef73e76a1db18efe7ef037f2b47732c29ebaaf44ece01b33ac781`,
prediction JSONL sha256
`b2123ea945285b9ee7ca940819382191fbae6af945cee09fb741b7c5ca95c717`, and
`promax_official_model.pt` sha256
`c0e17d003ba1e055e65d38dfac4dc96483f4a8744e201e3daf168a71f31890fc`.
Under disk pressure (`4.9G` free / `98%` used after completion), the
server-only prediction JSONL was removed only after the server-final and
local-light gates passed; its row-local deletion manifest records
`796026115` bytes, `10,000` lines, and the prediction sha256 above. The
completed intermediate adapter
`outputs/baselines/paper_adapters/tools_large10000_100neg_promax_official_adapter`
was then removed after resolved-path checks and manifesting:
`outputs/summary/tools_promax_completed_adapter_cleanup_manifest_20260604.sha256`
and `outputs/summary/tools_promax_completed_adapter_cleanup_du_20260604.txt`.
Final scores, provenance, audits, imported `tables/`, and
`promax_official_model.pt` were preserved. Post-cleanup checks show no
matching Python experiment process, GPU idle, and `/` at about `11G` free /
`95%` used. Tools is now 2/8 official rows complete. Next action: fresh
process/GPU/disk/duplicate-output preflight, then launch the next Tools
single-row official baseline; do not batch multiple rows.

Tools ElmRec launch checkpoint 2026-06-04 20:55 CST: after the ProMax gates,
local package, commit/push, cleanup, and heartbeat retarget completed, a fresh
preflight found no matching experiment process, GPU idle, `/` and
`/home/ajifang` at about `11G` free / `95%` used, no existing Tools
`elmrec_graph` final output, and no existing ElmRec adapter path, log, or PID
file. Server worktree is dirty and behind local `main`, so no server pull was
attempted; the server-side runner was inspected directly and has the
`elmrec_graph` branch plus override validation. Tools `elmrec_graph` was
launched as the third Tools single-row official baseline with:

```bash
nohup env DOMAINS_OVERRIDE=tools FAST_METHODS_OVERRIDE= TRAIN_METHODS_OVERRIDE=elmrec_graph bash scripts/run_baselines_new_domains.sh
```

The SSH wrapper timed out while backgrounding, so PID files were written after
process inspection. Current monitor target: runner PID `3301337`, adapter PID
`3301345`, PID files
`baselines_new_domains_tools_elmrec_20260604_204602.runner.pid` and
`baselines_new_domains_tools_elmrec_20260604_204602.adapter.pid`, and log
`baselines_new_domains_tools_elmrec_20260604_204602.log`. Stable launch
snapshot `outputs/summary/tools_elmrec_launch_monitor_20260604.json` reports
exactly one matching ElmRec adapter process, Qwen3 `hf_mean_pool` progress
`7880/269711`, GPU `96%` with `16091 MiB / 49140 MiB`, adapter directory
`1005M`, and no `DONE`, Traceback, OOM, no-space, blocker, or failure markers.
Disk is already below the warning line at about `9.85G` free / `95%` used, so
monitor disk closely. Do not start another baseline while this row is active.
If it completes, run server-final audit, server large-artifact manifest,
local-light sync, local-light audit, docs/memory update, and related-only
commit/push before the next row.

Tools ElmRec monitor checkpoint 2026-06-04 21:15 CST: robust SSH-stdin monitor
snapshot `outputs/summary/tools_elmrec_monitor_snapshot_latest.json` reports
the same single matching adapter PID `3301345`, Qwen3 `hf_mean_pool` progress
`116952/269711` (`0.43361968922290894`), GPU `96%` with
`16213 MiB / 49140 MiB`, adapter directory `1005M`, no completion marker, and
no Traceback/OOM/no-space/blocker/failure markers. C-CRP all-domain log still
ends in `ALL C-CRP v3 INFERENCE COMPLETE`. Disk remains warning-level at
about `9.85G` free / `95%` used, which is the only `should_notify` reason. A
read-only large-file audit found the largest directories/files are mostly
protected final official evidence, task packages, model/checkpoint files, or
active ElmRec adapter files; no cleanup was performed while the row is active.
GPT-5.5 xhigh sidecar Noether independently confirmed the monitor-only
sequence, the post-completion gate order, and that the server ELMRec checkout
matches pinned commit `b28c4f786d89fb8473ab358e12a882b30259f627`.

Tools ElmRec monitor checkpoint 2026-06-04 22:05 CST: the row is still active
and no gates have been run. The latest robust SSH-stdin monitor snapshot shows
Qwen3 embedding finished at `269711/269711`, then the adapter entered official
ELMRec training. The log contains the expected sparse tensor invariant
`UserWarning` followed by `[elmrec-official] epoch=1 train_loss=0.189466`,
`epoch=5 train_loss=0.110476`, `epoch=10 train_loss=0.065513`,
`epoch=15 train_loss=0.041795`, and `epoch=20 train_loss=0.046937`; this is
healthy progress, not a failure marker. Runner PID `3301337` and adapter PID
`3301345` remain alive, GPU is active at about `96%` with `16245 MiB / 49140
MiB`, the final evidence directory is still placeholder-only, and there is no
`scores.csv`, completion marker, Traceback, OOM, no-space, blocker, or failure
marker. Disk is now danger-level at about `5.67G` free / `97%` used. The
active adapter directory is `5.2G`; recent writes include
`llm_esr/handled/itm_emb_np.pkl` (`4,418,945,194` bytes),
`pca64_itm_emb_np.pkl` (`69,046,181` bytes), `llmesr_embedding_metadata.json`,
and `elmrec/handled/elmrec_bridge_manifest.json`. A read-only storage audit
found the largest space consumers are protected completed official evidence,
same-candidate task packages, final checkpoints/embeddings, and the active
ElmRec adapter. No cleanup was performed. Since the ELMRec official adapter
defaults to `100` epochs with logs every `5`, `epoch=20` is an early training
checkpoint; continue monitoring and do not start another baseline. If disk
drops further or a no-space marker appears, propose an exact cleanup candidate
with artifact-protection reasoning before deleting anything.

Tools ElmRec completion/gate checkpoint 2026-06-04 23:05 CST: Tools
`elmrec_graph` completed as the third Tools official row. The wrapper log
`baselines_new_domains_tools_elmrec_20260604_204602.log` ended with
`DONE elmrec_graph on tools` and `=== All baseline runs complete ===`.
Final provenance records `implementation_status=official_completed`,
`blockers=[]`, `official_repo_commit=b28c4f786d89fb8473ab358e12a882b30259f627`,
and exact `score_coverage_rate=1.0`. Codex ran the required sequence:
server-final audit PASS, server large-artifact manifest PASS, lightweight
local sync PASS, and local-light audit PASS. Full metrics over 10,000 users
and 101 candidates are HR@5/10/20 `0.0501 / 0.101 / 0.2101`,
NDCG@5/10/20 `0.029656030656687697 / 0.045870649973376774 /
0.07316592297455926`, and MRR `0.05237582779698271`; `scores.csv` has
`1,010,001` lines, predictions had `10,000` lines before post-gate deletion,
and `tables/ranking_eval_records.csv` has `10,001` lines. The local
lightweight package is
`outputs/baselines/official_adapters/tools_large10000_100neg_elmrec_graph_official_qwen3base_same_candidate/`.
Server-only `scores.csv`, deleted prediction metadata, and
`elmrec_official_model.pt` are covered by `server_large_artifact_manifest.*`
and `prediction_deletion_manifest.json`; the manifest records sha256 values
`13e8aa52ed5c69fa8c9b04006907b907043dd5fbc26b1829e3542d2ed58b050c` for
scores, `d49457a2d7fc7a5877f565f51913f7110fac438bd6cadd871a8e2da68237c4fd`
for the deleted prediction JSONL, and
`637e253d09007a93dfa1fc3f78ba4209b2d052b76a22f629e6e1ad8bf375a22d` for the
final model. After all gates and local sync passed, Codex removed only the
server-side prediction JSONL and completed intermediate adapter
`outputs/baselines/paper_adapters/tools_large10000_100neg_elmrec_official_adapter`.
Cleanup manifests were synced locally as
`outputs/summary/tools_elmrec_completed_adapter_cleanup_manifest_20260604.sha256`
and `outputs/summary/tools_elmrec_completed_adapter_cleanup_du_20260604.txt`.
Final scores, provenance, audits, imported tables, and model were preserved.
Post-cleanup checks show no matching Python experiment process and `/` back to
about `11G` free / `95%` used. Tools is now 3/8 official rows complete; the
remaining Tools official rows are `irllrec_intent`, `rlmrec_graphcl`,
`llm2rec_sasrec`, and `llmesr_sasrec`.

Tools LLMEmb completion/gate checkpoint 2026-06-05 00:55 CST: after the ElmRec
gates, local package, commit/push, and post-gate cleanup, Tools `llmemb`
launched as the fourth Tools single-row official baseline with:

```bash
nohup env DOMAINS_OVERRIDE=tools FAST_METHODS_OVERRIDE=llmemb TRAIN_METHODS_OVERRIDE= bash scripts/run_baselines_new_domains.sh
```

The wrapper log `baselines_new_domains_tools_llmemb_20260604_231030.log`
ended with `DONE llmemb on tools` and `=== All baseline runs complete ===`.
Final provenance records `implementation_status=official_completed`,
`blockers=[]`, pinned LLMEmb commit
`3458a5e225062e94b4f1a01e41f3ec82089f0407`, and exact
`score_coverage_rate=1.0`. The row passed server-final audit, server
large-artifact manifest, lightweight local sync, and local-light audit. Full
metrics over 10,000 users and 101 candidates are HR@5/10/20
`0.1365 / 0.2257 / 0.3637`, NDCG@5/10/20
`0.087457824217457 / 0.11594350972806679 / 0.15050644138929892`, and MRR
`0.10649354669900822`; `scores.csv` has `1,010,001` lines, predictions had
`10,000` lines before post-gate deletion, and
`tables/ranking_eval_records.csv` has `10,001` lines. The local lightweight
package is
`outputs/baselines/official_adapters/tools_large10000_100neg_llmemb_official_qwen3base_same_candidate/`.
Server-only `scores.csv`, deleted prediction metadata, and
`llmemb_official_model.pt` are covered by `server_large_artifact_manifest.*`
and `prediction_deletion_manifest.json`; sha256 values are
`bcbf84afd2af8c43b943e97464ab21060bc84fce8601a047698057cd63042e4c` for
scores, `a8301ae1337551f1502c751dad66e6d9e8e69c50e22764b251e1a3fdbf2b38c2`
for the deleted prediction JSONL, and
`a2cbeab06b437b09c7d4ed66f052b54baf7de04923cc4bdb84bf9df91c661c52` for the
final model. During the run, disk fell to about `4.9G` free after the active
adapter grew to `5.2G`, so the completed Home LLM2Rec checkpoint was removed
under emergency approval after server-final/local-light evidence and
sha256/size manifest verification; local record:
`outputs/summary/home_llm2rec_checkpoint_deletion_manifest_20260604.json`.
After all LLMEmb gates and local sync passed, Codex removed only the
server-side prediction JSONL and completed intermediate adapter
`outputs/baselines/paper_adapters/tools_large10000_100neg_llmemb_official_adapter`.
Cleanup manifests were synced locally as
`outputs/summary/tools_llmemb_prediction_deletion_manifest_20260605.json`,
`outputs/summary/tools_llmemb_completed_adapter_cleanup_manifest_20260605.sha256`,
and `outputs/summary/tools_llmemb_completed_adapter_cleanup_du_20260605.txt`.
Final scores, provenance, audits, imported tables, and model were preserved.
Post-cleanup `/` returned to about `15G` free / `93%` used. Tools is now 4/8
official rows complete; next action is a fresh preflight and one single next
Tools official row, likely `irllrec_intent`, not a batch. The stale
`Monitor Tools LLMEmb` heartbeat (`monitor-home-llm2rec`) was deleted after
completion.

Tools IRLLRec launch checkpoint 2026-06-05 01:07 CST: after the Tools LLMEmb
commit/push and a fresh preflight, Codex verified no matching experiment
Python process, GPU idle, `/home/ajifang` at about `15G` free / `93%` used, no
existing Tools `irllrec_intent` final output, no existing Tools IRLLRec adapter
path, and no existing Tools IRLLRec log/PID files. Launched exactly one row:

```bash
nohup env DOMAINS_OVERRIDE=tools FAST_METHODS_OVERRIDE= TRAIN_METHODS_OVERRIDE=irllrec_intent bash scripts/run_baselines_new_domains.sh
```

The SSH launch wrapper timed out while backgrounding, so Codex did not retry;
it inspected the process tree and wrote PID files after confirming the active
run. Current monitor target: runner PID `3326805`, adapter PID `3326813`, PID
files `baselines_new_domains_tools_irllrec_20260605_0058.runner.pid` and
`baselines_new_domains_tools_irllrec_20260605_0058.adapter.pid`, and log
`baselines_new_domains_tools_irllrec_20260605_0058.log`. Clean launch snapshot
`outputs/summary/tools_irllrec_launch_monitor_20260605.json` reports one
matching IRLLRec adapter process, Qwen3 `hf_mean_pool` progress `2056/269711`
(`0.007622974220554594`), GPU `95%` with `15947 MiB / 49140 MiB`, adapter dir
`1005M`, final output placeholder-only, disk `13.34G` free / `93%` used, and
no completion/failure markers. Heartbeat `monitor-tools-irllrec` is active.
Do not start another Tools row while IRLLRec is active.
Guarded completion-gate plan prepared at 2026-06-05 01:59 CST:
`outputs/summary/official_completion_gate_plan/tools_irllrec_intent_completion_gates_20260605.{json,ps1}`.
The plan is `planning_only_not_executed`, starts with a PowerShell `throw`,
records the required server-final audit -> server large-artifact manifest ->
local-light sync -> local-light audit order, and must not be used until the
active runner exits normally.
Monitor update 2026-06-05 02:14 CST: Qwen3 embedding completed
`269711/269711` and IRLLRec official training began; latest observed line was
`[irllrec-official] epoch=1 train_loss=1.467340`. Runner PID `3326805` and
adapter PID `3326813` remained alive, with no completion marker and final
output still placeholder-only. Disk crossed the warning threshold at about
`9.2G` free / `96%` used. Read-only storage audit found the active IRLLRec
adapter at `5.2G` (`llm_esr/handled/itm_emb_np.pkl` about `4.1G`) and no
obvious safe large deletion outside active/protected evidence, models,
checkpoints, task packages, or final artifacts, so no cleanup was performed.
Continue monitoring disk; do not delete active adapter files.
Monitor update 2026-06-05 04:00 CST: runner PID `3326805` and adapter PID
`3326813` remained alive, GPU was active at about `100%` with
`16247 MiB / 49140 MiB`, and IRLLRec official training had advanced to
`epoch=1740` with latest loss `0.657816`. Log scans still showed no
completion marker and no fatal/OOM/no-space marker, and the final output
directory remained placeholder-only. Disk remained in warning state at about
`9.1G` free / `96%` used. This is still monitor-only evidence, not an official
completed row; wait for normal runner exit and final artifacts before running
the prepared completion gates.
Monitor update 2026-06-05 04:10 CST: runner PID `3326805` and adapter PID
`3326813` remained alive and there was exactly one matching IRLLRec adapter
process. Qwen3 embedding was complete (`269711/269711`), GPU stayed active at
about `100%` with `16247 MiB / 49140 MiB`, and official training advanced to
`epoch=1970` with latest loss `0.657435`. Log scans still showed no completion
marker and no fatal/OOM/no-space marker. The final evidence directory remained
placeholder-only (`4.0K`), active adapter size was about `5.2G`, and disk stayed
below the notification threshold at about `9.1G` free / `96%` used. No gates,
cleanup, or new experiments were run.
Monitor update 2026-06-05 04:22 CST: runner PID `3326805` and adapter PID
`3326813` remained alive, with one matching adapter process and GPU still active
(`68-99%`, `16247 MiB / 49140 MiB` across samples). Official training advanced
to `epoch=2090` with latest loss `0.659265`. Log scans still showed no
completion marker and no fatal/OOM/no-space marker, and the final evidence
directory remained placeholder-only (`4.0K`). Disk remained in warning state at
about `9.1G` free / `96%` used. A read-only large-file audit found no safe
cleanup target: the multi-GB files were protected final models/checkpoints,
same-candidate task packages, the active IRLLRec adapter, or artifacts requiring
separate gate review before any deletion. User cache was only about `58M`, and
no project archive/temp candidate was found. No gates, cleanup, or new
experiments were run.
Monitor update 2026-06-05 04:28 CST: runner PID `3326805` and adapter PID
`3326813` remained alive, with one matching adapter process and active GPU
(`66-99%`, `16247 MiB / 49140 MiB` across samples). Official training advanced
to `epoch=2200` with latest loss `0.656818`. Log scans still showed no
completion marker and no fatal/OOM/no-space marker, and the final evidence
directory remained placeholder-only (`4.0K`). Disk stayed in warning state at
about `9.1G` free / `96%` used. No gates, cleanup, or new experiments were run.
Monitor update 2026-06-05 04:34 CST: runner PID `3326805` and adapter PID
`3326813` remained alive, with one matching adapter process and active GPU
(`64-99%`, `16247 MiB / 49140 MiB` across samples). Official training advanced
to `epoch=2300` with latest loss `0.658179`. Log scans still showed no
completion marker and no fatal/OOM/no-space marker, and the final evidence
directory remained placeholder-only (`4.0K`). Disk stayed in warning state at
about `9.1G` free / `96%` used. No gates, cleanup, or new experiments were run.
Monitor update 2026-06-05 04:41 CST: runner PID `3326805` and adapter PID
`3326813` remained alive, with one matching adapter process and active GPU
(`96-99%`, `16247 MiB / 49140 MiB` across samples). Official training advanced
to `epoch=2410` with latest loss `0.656903`. Log scans still showed no
completion marker and no fatal/OOM/no-space marker, and the final evidence
directory remained placeholder-only (`4.0K`). Disk stayed in warning state at
about `9.1G` free / `96%` used. No gates, cleanup, or new experiments were run.
Monitor update 2026-06-05 04:46 CST: runner PID `3326805` and adapter PID
`3326813` remained alive, with one matching adapter process and active GPU
(`69-70%`, `16247 MiB / 49140 MiB` across samples). Official training advanced
to `epoch=2500` with latest loss `0.658937`. Log scans still showed no
completion marker and no fatal/OOM/no-space marker, and the final evidence
directory remained placeholder-only (`4.0K`). Disk stayed in warning state at
about `9.1G` free / `96%` used. No gates, cleanup, or new experiments were run.
Monitor update 2026-06-05 04:53 CST: runner PID `3326805` and adapter PID
`3326813` remained alive, with one matching adapter process and active GPU
(`70-99%`, `16247 MiB / 49140 MiB` across samples). Official training advanced
to `epoch=2600` with latest loss `0.656922`. Log scans still showed no
completion marker and no fatal/OOM/no-space marker, and the final evidence
directory remained placeholder-only (`4.0K`). Disk stayed in warning state at
about `9.1G` free / `96%` used. No gates, cleanup, or new experiments were run.
Monitor update 2026-06-05 05:02 CST: runner PID `3326805` and adapter PID
`3326813` remained alive, with one matching adapter process and active GPU
(`73-100%`, `16247 MiB / 49140 MiB` across samples). Qwen3 embedding was
complete (`269711/269711`) and official training advanced to `epoch=2720` with
latest loss `0.656919`. Log scans still showed no completion marker and no
fatal/OOM/no-space marker, and the final evidence directory remained
placeholder-only (`4.0K`). Disk stayed in warning state at about `9.1G` free /
`96%` used. No gates, cleanup, or new experiments were run.
Monitor update 2026-06-05 05:12 CST: runner PID `3326805` and adapter PID
`3326813` remained alive, with one matching adapter process and active GPU
(`67-100%`, `16247 MiB / 49140 MiB` across samples). Qwen3 embedding stayed
complete (`269711/269711`) and official training advanced to `epoch=2890` with
latest loss `0.657135`. Log scans still showed no completion marker and no
fatal/OOM/no-space marker, and the final evidence directory remained
placeholder-only (`4.0K`). Disk stayed in warning state at about `9.1G` free /
`96%` used. No gates, cleanup, or new experiments were run.
Tools IRLLRec completion/gate checkpoint 2026-06-05 05:29 CST:
`irllrec_intent` completed normally at 2026-06-05 05:19:06 CST with log marker
`DONE irllrec_intent on tools` and wrapper marker `All baseline runs complete`.
The adapter reached `epoch=3000` with latest loss `0.656947`, exported
`implementation_status=official_completed`, `blockers=[]`, exact
`score_coverage_rate=1.0`, score audit/imported tables, server-final audit,
server large-artifact manifest, lightweight local sync, and local-light audit
all passing. Full metrics over 10,000 users and 101 candidates are
HR@5/10/20 `0.102 / 0.1651 / 0.3095`, NDCG@5/10/20
`0.06504709833452535 / 0.08525707170530923 / 0.12100829406900945`, and MRR
`0.08670154590435823`; `scores.csv` has `1,010,001` lines, predictions have
`10,000` lines, and `tables/ranking_eval_records.csv` has `10,001` lines.
The local lightweight package is
`outputs/baselines/official_adapters/tools_large10000_100neg_irllrec_intent_official_qwen3base_same_candidate/`.
Server-only `scores.csv`, predictions, and `irllrec_official_model.pt` are
covered by `server_large_artifact_manifest.*`; the provenance records the
IRLLRec scalability bridge `deterministic_node_cap` for the all-node
contrastive term. After gates and local-light backup passed, the completed
intermediate adapter
`outputs/baselines/paper_adapters/tools_large10000_100neg_irllrec_official_adapter`
was removed with sha256 manifest
`outputs/summary/tools_irllrec_completed_adapter_cleanup_manifest_20260605.sha256`,
recovering `/` to about `14G` free / `93%` used while preserving all final
scores, provenance, audits, imported tables, predictions, and model. Tools is
now 5/8 official rows gated; remaining rows are `rlmrec_graphcl`,
`llm2rec_sasrec`, and `llmesr_sasrec`.
Tools RLMRec launch checkpoint 2026-06-05 05:44 CST: after the IRLLRec
gate/package/cleanup commit and a fresh preflight, Codex verified no active
project Python process, GPU idle (`0%`, `15 MiB / 49140 MiB`), `/` about `14G`
free / `93%` used, and no existing Tools `rlmrec_graphcl` final output,
adapter, log, or PID path. Launched exactly one row:

```bash
nohup env DOMAINS_OVERRIDE=tools FAST_METHODS_OVERRIDE= TRAIN_METHODS_OVERRIDE=rlmrec_graphcl bash scripts/run_baselines_new_domains.sh
```

Current monitor target: runner PID `3347729`, adapter PID `3347738`, log
`baselines_new_domains_tools_rlmrec_20260605_054158.log`, and heartbeat
`monitor-tools-rlmrec`. The launch wrapper printed a trailing shell syntax
error after the background launch, but a follow-up process check confirmed the
runner and adapter are live and unique, so Codex did not relaunch. Guarded
completion-gate plan:
`outputs/summary/official_completion_gate_plan/tools_rlmrec_graphcl_completion_gates_20260605.{json,ps1}`.
Initial robust monitor snapshot
`outputs/summary/tools_rlmrec_launch_monitor_20260605.json` reports one
matching adapter process, no completion/failure markers, adapter dir about
`1000M`, final output placeholder-only (`4.0K`), and disk about `12.35G` free /
`94%` used. Do not start another Tools row while RLMRec is active.
Monitor update 2026-06-05 05:53 CST: runner PID `3347729` and adapter PID
`3347738` remained alive, with one matching RLMRec adapter process and active
GPU (`96%`, `16189 MiB / 49140 MiB`). The row is still in Qwen3 embedding at
`35360/269711`; no RLMRec training epoch has started yet. Log scans showed no
completion marker and no fatal/OOM/no-space marker, the active adapter
directory was about `1005M`, and the final evidence directory remained
placeholder-only (`4.0K`). Disk was about `12.35G` free / `94%` used, above
the 10G warning threshold. No gates, cleanup, or new experiments were run.
Monitor update 2026-06-05 06:03 CST: runner PID `3347729` and adapter PID
`3347738` remained alive, with one matching RLMRec adapter process and active
GPU (`95%`, `16213 MiB / 49140 MiB`). The row is still in Qwen3 embedding at
`79528/269711`; no RLMRec training epoch has started yet. Log scans showed no
completion marker and no fatal/OOM/no-space marker, the active adapter
directory remained about `1005M`, and the final evidence directory remained
placeholder-only (`4.0K`). Disk was about `12.35G` free / `94%` used, above
the 10G warning threshold. No gates, cleanup, or new experiments were run.
Monitor update 2026-06-05 06:15 CST: runner PID `3347729` and adapter PID
`3347738` remained alive, with one matching RLMRec adapter process and active
GPU (`100%`, `16213 MiB / 49140 MiB`). The row is still in Qwen3 embedding at
`128944/269711`; no RLMRec training epoch has started yet. Log scans showed no
completion marker and no fatal/OOM/no-space marker, the active adapter
directory remained about `1005M`, and the final evidence directory remained
placeholder-only (`4.0K`). Disk was about `12.35G` free / `94%` used, above
the 10G warning threshold. No gates, cleanup, or new experiments were run.
Monitor update 2026-06-05 06:25 CST: runner PID `3347729` and adapter PID
`3347738` remained alive, with one matching RLMRec adapter process and active
GPU (`96%`, `16237 MiB / 49140 MiB`). The row is still in Qwen3 embedding at
`172352/269711`; no RLMRec training epoch has started yet. Log scans showed no
completion marker and no fatal/OOM/no-space marker, the active adapter
directory remained about `1005M`, and the final evidence directory remained
placeholder-only (`4.0K`). Disk was about `12.35G` free / `94%` used, above
the 10G warning threshold. No gates, cleanup, or new experiments were run.
Monitor update 2026-06-05 06:33 CST: runner PID `3347729` and adapter PID
`3347738` remained alive, with one matching RLMRec adapter process and active
GPU (`96%`, `16237 MiB / 49140 MiB`). The row is still in Qwen3 embedding at
`209840/269711`; no RLMRec training epoch has started yet. Log scans showed no
completion marker and no fatal/OOM/no-space marker, the active adapter
directory remained about `1005M`, and the final evidence directory remained
placeholder-only (`4.0K`). Disk was about `12.35G` free / `94%` used, above
the 10G warning threshold. No gates, cleanup, or new experiments were run.
Monitor/storage update 2026-06-05 07:02 CST: runner PID `3347729` and adapter
PID `3347738` remained alive, with exactly one matching RLMRec adapter process.
Qwen3 embedding completed (`269711/269711`) and official RLMRec training
entered the graph stage; latest observed training line was
`[rlmrec-official] epoch=110 train_loss=1.519527`. GPU remained active with
`24679 MiB / 49140 MiB` resident; the active adapter grew to about `5.2G`
(`llm_esr/handled/itm_emb_np.pkl` about `4.1G`), final RLMRec evidence was
still placeholder-only (`4.0K`), and no completion/failure marker or final
scores/provenance/tables existed yet. Disk crossed the warning line, first
falling to about `8.6G` free / `96%` used. A large-file audit found no safe deletion
inside the active RLMRec adapter, protected same-candidate task packages, or
final model/checkpoint artifacts. The only policy-eligible emergency cleanup
was the completed Tools IRLLRec server prediction JSONL: server-final audit
proved it existed with `10,000` lines, local-light audit had already passed
with `tables/ranking_eval_records.csv` preserved locally, and final scores,
provenance, audits, tables, and model stayed untouched. Codex wrote sha256
manifest `outputs/summary/tools_irllrec_prediction_cleanup_manifest_20260605.sha256`
and removed only
`outputs/tools_large10000_100neg_irllrec_intent_official_qwen3base_same_candidate/predictions/rank_predictions.jsonl`,
recovering disk to about `9.3G` free / `96%` used. This remains below the
10G warning threshold, so continue monitoring closely and do not start another
baseline. An attempted Electronics ELMRec prediction cleanup was rejected
because that older directory lacked server-final provenance/scores proof.
Monitor update 2026-06-05 08:01 CST: runner PID `3347729` and adapter PID
`3347738` remained alive and unique, with exactly one matching RLMRec adapter
process. The trainer/default adapter path confirms the official RLMRec run uses
the default `3000` epochs (`--rlmrec_epochs=3000` / trainer `--epochs=3000`);
the active log reached `[rlmrec-official] epoch=590 train_loss=1.509551`, so
this remains expected long training rather than a suspected stall. GPU memory
remained resident at `24679 MiB / 49140 MiB`, final RLMRec evidence stayed
placeholder-only (`4.0K`), and no final `scores.csv`, provenance, score audit,
imported tables, predictions, completion marker, OOM, no-space, killed, or
traceback marker existed yet. Disk remained about `9.3G` free / `96%` used,
below the 10G warning threshold. A repeat large-file/cache/temp/archive audit
found no safe cleanup candidate: the large files were active RLMRec
intermediates, protected same-candidate task packages, completed evidence or
checkpoints, or an older prediction file without deletion proof. Do not start
another Tools row while this active RLMRec row is training.
Monitor update 2026-06-05 08:46 CST: runner PID `3347729` and adapter PID
`3347738` remained alive and unique, with exactly one matching RLMRec adapter
process. The active log crossed `[rlmrec-official] epoch=1000
train_loss=1.505931` on the default `3000`-epoch official path. This is a
material monitor checkpoint only, not a completed row: final RLMRec evidence
remained placeholder-only (`4.0K`), with no final `scores.csv`, provenance,
score audit, imported tables, predictions, completion marker, OOM/no-space,
killed, traceback, or error marker. Disk remained warning-level at about
`9.3G` free / `96%` used, and no new safe cleanup candidate was identified.
Continue monitoring; do not start another Tools row while RLMRec is active.

Monitor update 2026-06-05 09:45 CST: runner PID `3347729` and adapter PID
`3347738` remained alive and unique, with exactly one matching RLMRec adapter
process. Official RLMRec training passed the halfway checkpoint on the default
`3000`-epoch path: latest observed line was `[rlmrec-official] epoch=1510
train_loss=1.506936`, after epoch `1500` loss `1.507642`. This is still active
monitor-only evidence, not a completed or table-eligible row: final RLMRec
evidence remained placeholder-only (`4.0K`) with no final scores, provenance,
score audit, imported tables, predictions, completion marker, OOM/no-space,
killed, traceback, or error marker. Disk remained warning-level at about
`9.3G` free / `96%` used. A repeat large-file/cache/temp/archive and prediction
cleanup audit found no safe meaningful deletion candidate: visible large files
were active RLMRec intermediates, protected same-candidate task packages,
retained completed checkpoints/evidence, or older predictions lacking the
required server-final/local-light proof. Continue monitoring; do not start
another Tools row while RLMRec is active.

Monitor update 2026-06-05 10:36 CST: runner PID `3347729` and adapter PID
`3347738` remained alive and unique, with exactly one matching RLMRec adapter
process and zero matching IRLLRec adapter processes. Official RLMRec training
crossed the two-thirds monitor checkpoint on the default `3000`-epoch path:
latest observed line was `[rlmrec-official] epoch=2000 train_loss=1.506144`.
This remains active monitor-only evidence, not a completed or table-eligible
row: final RLMRec evidence stayed placeholder-only (`4.0K`) with no final
scores, provenance, score audit, imported tables, predictions, completion
marker, OOM/no-space, killed, traceback, or error marker. Disk remained
warning-level at about `9.3G` free / `96%` used. A fresh cleanup audit found no
safe deletion target: caches/tmp were small, no temp/archive/part/core files
were present, and the largest files were active RLMRec intermediates,
protected task splits, retained completed checkpoints/evidence, or historical
C-CRP/fusion summaries that are not disposable without a separate archive
decision. Continue monitoring; do not start another Tools row while RLMRec is
active.

Monitor update 2026-06-05 11:49 CST: runner PID `3347729` and adapter PID
`3347738` remained alive and unique, with exactly one matching RLMRec adapter
process and zero matching IRLLRec/LLM2Rec/LLM-ESR adapter processes. Official
RLMRec training crossed the five-sixths monitor checkpoint on the default
`3000`-epoch path: latest observed epoch `2610/3000` had train loss
`1.504630`, after epoch `2600` loss `1.505511`. This remains active
monitor-only evidence, not a completed or table-eligible row: final RLMRec
evidence stayed placeholder-only (`4.0K`) with no final scores, provenance,
score audit, imported tables, predictions, completion marker, OOM/no-space,
killed, traceback, or error marker. Disk remained warning-level at about
`9.3G` free / `96%` used. A repeat cleanup audit found no safe deletion target:
caches/tmp were small, no temp/archive/part/core files were present, and the
largest files were active RLMRec intermediates, protected task splits, retained
completed checkpoints/evidence, or the legacy Electronics ELMRec prediction
JSONL without server-final/local-light deletion proof. Continue monitoring; do
not start another Tools row while RLMRec is active.

Tools RLMRec completion/gate checkpoint 2026-06-05 12:49 CST: Tools
`rlmrec_graphcl` completed as the sixth Tools official row. The wrapper log
`baselines_new_domains_tools_rlmrec_20260605_054158.log` reached
`[rlmrec-official] epoch=3000 train_loss=1.505858`, then wrote
`implementation_status=official_completed`, `blockers=0`, saved predictions
and tables, recorded `score_coverage_rate=1.000000`, and ended with
`DONE rlmrec_graphcl on tools` / `All baseline runs complete`. Server-final
audit, server large-artifact manifest, lightweight local sync, and local-light
audit all passed. Full metrics over 10,000 users and 101 candidates are
HR@5/10/20 `0.0784 / 0.1354 / 0.2465`, NDCG@5/10/20
`0.05017501611537314 / 0.06838865570840932 / 0.09599330874161652`, and MRR
`0.07220064580885768`. Row counts: `scores.csv` has `1,010,001` lines,
predictions had `10,000` lines before post-gate deletion, and
`tables/ranking_eval_records.csv` has `10,001` lines. The local lightweight
package is
`outputs/baselines/official_adapters/tools_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate/`;
it includes provenance, score audits, run summary, server-final audit, server
large-artifact manifest, imported tables, ranking records, and the wrapper log
copied from the server. Server-only `scores.csv`, deleted prediction metadata,
and `rlmrec_official_model.pt` are covered by `server_large_artifact_manifest.*`
and `prediction_deletion_manifest.json`; the deleted prediction sha256 is
`73ba3bf4adffe3efd49bb59967ea5a692e44a5f796e743afc50d2dd830b4f6b7`.
After gates and local backup passed, Codex removed only the server prediction
JSONL and the completed intermediate adapter directory
`outputs/baselines/paper_adapters/tools_large10000_100neg_rlmrec_official_adapter`
with manifests
`outputs/summary/tools_rlmrec_completed_adapter_cleanup_manifest_20260605.sha256`
and `outputs/summary/tools_rlmrec_completed_adapter_cleanup_du_20260605.txt`.
Final scores, provenance, audits, imported tables, model, server manifests, and
local evidence were preserved; `/` recovered to about `15G` free / `93%` used.
Tools is now 6/8 official rows gated; remaining rows are `llm2rec_sasrec` and
`llmesr_sasrec`, followed by the Tools domain/comparison/paired-test gates.

Tools LLM2Rec launch/validation-task repair checkpoint 2026-06-05 13:22 CST:
after the RLMRec completion gates and cleanup, a single-row Tools
`llm2rec_sasrec` launch was attempted with runner PID `3405735` and log
`baselines_new_domains_tools_llm2rec_20260605_130248.log`. It failed before a
stable adapter process or score file existed. The traceback was in
`scripts/build/main_export_llm2rec_same_candidate_task.py` validation export:
`source_event_id='AGEY75LYLXUAHG3KW5KF5ICMKA4A' has 0 positive rows; expected
exactly 1`. Audit found that
`outputs/baselines/external_tasks/tools_large10000_100neg_valid_same_candidate/candidate_items.csv`
was truncated: the original file had `587252` logical rows, ended in a partial
event plus a malformed user-id-only row, and had sha256
`4302712bb7dbe0a8cfde99b0a2727c8de0818d250b65e7cc3bc0b8ad01fa6f2b`. The file
was rebuilt from the canonical `ranking_valid.jsonl`, preserving the
`source_event_id,user_id,timestamp,split_name,candidate_index,item_id,label,is_positive,popularity_group,candidate_title,candidate_text`
schema. Repair manifest:
`outputs/summary/tools_valid_candidate_items_repair_20260605T051943Z.json`.
Independent validation manifest:
`outputs/summary/tools_valid_candidate_items_repair_validation_20260605T0520Z.json`
with PASS, `1,010,000` rows, `10,000` events, exactly `101` candidates/event,
and exactly one positive/label per event. The corrupt original remains
server-side as
`candidate_items.csv.corrupt_20260605T051943Z`. No Tools LLM2Rec row is active
or table-eligible. Disk is about `14G` free / `93%` used after the repair, so
do not relaunch until a fresh process/GPU/disk preflight and storage-margin
decision pass.

Tools LLM2Rec relaunch checkpoint 2026-06-05 13:46 CST: after the repaired
validation CSV passed, a GPT-5.5 xhigh sidecar ARIS auditor reviewed the
storage/relaunch risk and found a conditional relaunch acceptable: no matching
baseline process, stale failed PID dead, GPU idle, repaired validation manifest
PASS, failed final output placeholder-only, and disk above the hard alert floor
at roughly `13G` free / `94%` used. No large automatic cleanup was performed:
the only clearly safe file was the corrupt validation CSV backup
`candidate_items.csv.corrupt_20260605T051943Z` (`467M`), but deleting it would
not materially change the LLM2Rec storage risk; completed scores, tables,
models/checkpoints, and task splits remained protected. Launched exactly one
row:

```bash
nohup env DOMAINS_OVERRIDE=tools FAST_METHODS_OVERRIDE= TRAIN_METHODS_OVERRIDE=llm2rec_sasrec bash scripts/run_baselines_new_domains.sh > baselines_new_domains_tools_llm2rec_20260605_134355.log 2>&1 &
```

Runner PID `3413921`, adapter PID `3413930`, heartbeat
`monitor-tools-llm2rec`, and launch snapshot
`outputs/summary/tools_llm2rec_launch_monitor_20260605.json`. The first stable
snapshot found exactly one adapter process, no failure markers, GPU active at
about `95%` / `16089 MiB`, adapter directory `1.1G`, final output
placeholder-only, disk about `12G` free / `94%` used, and embedding progress
around `[hf_mean_pool] encoded 3888/345622`. This is active monitor-only
evidence, not a completed or table-eligible row. Do not start another baseline
while this row is active; alert if disk falls below `10G` free or reaches
`97%` used.

Tools LLM2Rec monitor checkpoint 2026-06-05 13:57 CST: runner PID `3413921`
and adapter PID `3413930` remained alive. A duplicate-process audit initially
matched three stale shell diagnostics from earlier grep/find checks
(`3412219/3412220`, `3413516/3413517`, `3413819/3413820`); after verifying via
`ps`/`pstree` that they were abandoned diagnostics and not experiment
processes, those six stale diagnostic PIDs were killed to avoid false duplicate
alerts. Exactly one real LLM2Rec adapter process remained. The active log
showed Qwen3 embedding progress around `[hf_mean_pool] encoded 53296/345622`,
with no `Traceback`, OOM, no-space, killed, exact-one-positive, completion, or
blocked markers. GPU was about `95%` / `16199 MiB`, adapter dir `1.1G`, final
output placeholder-only, and disk `12G` free / `94%` used
(`12,711,534,592` bytes free). Snapshot:
`outputs/summary/tools_llm2rec_monitor_checkpoint_20260605_1355.json`. This is
still active monitor-only evidence.

Read-only toys domain gate checkpoint 2026-06-02 07:18 CST: server-side
official rows `llmemb`, `proex_profile`, `promax_profile`, `elmrec_graph`, and
`irllrec_intent` each passed the compact gate with `sample_count=10000`,
`avg_candidates=101.0`, `score_coverage_rate=1.0`, `scores.csv` line count
`1,010,001`, predictions `10,000`, and
`tables/ranking_eval_records.csv` `10,001`. A later 2026-06-02 17:09 CST
import reconciled the toys C-CRP gate path and produced imported same-candidate
tables with exact coverage. The final 2026-06-02 19:00 CST toys domain gate
then passed with `official_ok_count=8`, `ccrp_ok=true`, and `gate_ok=true`.

RLMRec training checkpoint 2026-06-02 07:48 CST: toys `rlmrec_graphcl`
finished the Qwen3 embedding pass (`215034/215034`) and entered official
training. The latest training line was `[rlmrec-official] epoch=90
train_loss=1.496428`; the same non-fatal graph normalization warnings seen in
previous RLMRec runs appeared before training. No final
`scores.csv`, provenance, score audit, imported tables, or predictions exist
yet, so no local package or table row is allowed. Disk dropped to about `5.4G`
free because the active adapter directory grew to about `4.3G`. A read-only
cleanup audit did not find a safe large deletion: active RLMRec artifacts must
stay, the old `books_large10000_100neg_llmesr_adapter` remains unverified, and
`.vscode-server` has live Code-related processes. Monitor disk closely through
final export.

## Completed Checkpoints

### IRLLRec Completed Checkpoint

At 2026-06-01 08:10 CST, sports `irllrec_intent` completed as
`implementation_status=official_completed`, `blockers=[]`, and
`score_coverage_rate=1.0`. Full metrics over 10,000 users and 101 candidates
per user:

- HR@5/10/20: `0.1573 / 0.2215 / 0.4016`
- NDCG@5/10/20: `0.10642150916142634 / 0.12691703149297534 / 0.17128490034441315`
- MRR: `0.12444202662842994`

Server row counts: `scores.csv` `1,010,001` lines including header,
`predictions/rank_predictions.jsonl` `10,000` lines, and local
`tables/ranking_eval_records.csv` `10,001` lines. Server-final package audit,
lightweight sync, and local-light package audit all passed. Local sync copied
11 lightweight files with matching size/sha256, including the server-final
evidence audit JSON, while excluding server-only `scores.csv`, predictions,
and `irllrec_official_model.pt`. After local verification, the server
intermediate adapter directory
`outputs/baselines/paper_adapters/sports_large10000_100neg_irllrec_official_adapter`
was removed, recovering disk from `28G` free to `32G` free; final IRLLRec
outputs remain on the server.

### RLMRec Completed Checkpoint

At 2026-06-01 09:53 CST, sports `rlmrec_graphcl` passed the 500-epoch
checkpoint under child PID `2851207`; at 2026-06-01 10:41 CST it passed the
1000-epoch checkpoint; at 2026-06-01 12:10 CST it passed the 2000-epoch
checkpoint; at 2026-06-01 13:43 CST it completed 3000 epochs, final score
export, and same-candidate metric import. Logged training loss:

- epoch 500: `1.480699`
- epoch 510: `1.482085`
- epoch 1000: `1.477797`
- epoch 1030: `1.478778`
- epoch 2000: `1.476514`
- epoch 3000: `1.476057`

Final evidence status: `implementation_status=official_completed`,
`blockers=[]`, `score_coverage_rate=1.0`, server-final audit PASS,
lightweight sync PASS, local-light audit PASS. Full metrics over 10,000 users
and 101 candidates:

- HR@5/10/20: `0.1212 / 0.1879 / 0.3009`
- NDCG@5/10/20: `0.078580389191345 / 0.10001773336299705 / 0.12818232277286493`
- MRR: `0.09720456858848743`

Server row counts: `scores.csv` `1,010,001` lines including header,
`predictions/rank_predictions.jsonl` `10,000` lines, and
`tables/ranking_eval_records.csv` `10,001` lines. Local sync copied
lightweight provenance/audit/run-summary/imported-table evidence under
`outputs/baselines/official_adapters/sports_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate/`
and excluded server-only `scores.csv`, predictions, and
`rlmrec_official_model.pt`.

### LLM2Rec Export Blocker and Fix

At 2026-06-01 13:43 CST, the sports runner advanced to `llm2rec_sasrec` and
failed before training with:

```text
ValueError: source_event_id='AE226MX6WSMZ33PVVTMN4LAOMIAA::1330139774000'
has empty mapped history for user_id='AE226MX6WSMZ33PVVTMN4LAOMIAA'
```

Server data check showed the user has one row in
`outputs/baselines/external_tasks/sports_large10000_100neg_valid_same_candidate/train_interactions.csv`
and no rows in the test task train/candidate files. This is a real
adapter-export bug, not a valid reason to skip validation. The local exporter
fix keeps test histories and validation histories split-aligned. Next recovery
step completed: commit `657929e` was pushed, the fixed exporter was copied to
the server, `py_compile` passed, and the real sports LLM2Rec run reached
embedding generation. The active log is
`baselines_new_domains_sports_llm2rec_resume.log`, and the corrected PID file is
`baselines_new_domains_sports_llm2rec_resume.pid`.

The exported LLM2Rec adapter package was audited on the server and is
`ready_for_llm2rec_upstream_wrapper`: users `19912`, items `283760`,
train interaction rows `41142`, validation train interaction rows `31417`,
candidate events `10000`, candidate rows `1010000`, validation candidate rows
`1010000`, missing mapped candidates `0`, and
`valid_history_source=valid_task_train_interactions`.

### LLM2Rec Embedding Completion and Training-Launch Blocker

At 2026-06-01 15:14 CST, sports `llm2rec_sasrec` had completed the full Qwen3
item embedding step (`283760/283760`) but stopped before official SASRec
training. The failure was:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'python'
```

The traceback points to `_train_with_official_entrypoint` in
`src/baselines/official_runner/llm2rec.py`. This is an execution-environment
bug in the wrapper command, not a metric result and not a reason to skip
LLM2Rec. The local fix switches the command to `sys.executable` while
preserving the official `evaluate_with_seqrec.py` entrypoint and SASRec
arguments. Resume should reuse the existing upstream embedding path
`/home/ajifang/projects/LLM2Rec/item_info/SportsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`
instead of regenerating the 4.65GB embedding.

### LLM2Rec Training Resume

At 2026-06-01 15:51 CST, the fixed runner was confirmed on the server:
`/home/ajifang/miniconda3/bin/python -m py_compile
src/baselines/official_runner/llm2rec.py` passed, and the file contains
`sys.executable`. A first direct launch command timed out locally but did start
the intended single LLM2Rec row. The follow-up safety launcher detected the
active process and refused to duplicate it. Active processes:

- adapter process PID `2875446`;
- upstream official `evaluate_with_seqrec.py` PID `2875559`.

The upstream command uses the existing embedding path
`/home/ajifang/projects/LLM2Rec/item_info/SportsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`
and writes checkpoints under
`outputs/sports_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate/seqrec_ckpt`.
The log `llm2rec_official_training.log` reached epoch 15 validation and saved
checkpoints at epochs 5 and 10. No final `scores.csv`,
`fairness_provenance.json`, score audit, imported tables, or row-count gates
exist yet.

### LLM2Rec Completed Checkpoint

At 2026-06-01 15:56 CST, sports `llm2rec_sasrec` completed as
`implementation_status=official_completed`, `blockers=[]`, and
`score_coverage_rate=1.0`. The official training early-stopped at epoch 45,
loaded the best epoch 25 checkpoint, exported `scores.csv`, and the unified
same-candidate importer produced full metrics over 10,000 users and 101
candidates:

- HR@5/10/20: `0.1105 / 0.206 / 0.3657`
- NDCG@5/10/20: `0.06514778914391295 / 0.09566791850988236 / 0.13561659669926907`
- MRR: `0.08828933028385053`

Server row counts: `scores.csv` `1,010,001` lines including header,
`predictions/rank_predictions.jsonl` `10,000` lines, and
`tables/ranking_eval_records.csv` `10,001` lines. Server-final evidence audit,
lightweight sync, and local-light evidence audit all passed. Local lightweight
evidence is under
`outputs/baselines/official_adapters/sports_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate/`.

### Tools LLM2Rec Recovery Monitor Checkpoint

At 2026-06-05 17:04 CST, the active Tools `llm2rec_sasrec` recovery row was
still running under runner PID `3423029`, adapter PID `3423037`, and official
training PID `3423221`, with log
`baselines_new_domains_tools_llm2rec_recovery_20260605_155904.log` and
heartbeat `monitor-tools-llm2rec-recovery`. Duplicate audit found exactly one
Tools LLM2Rec adapter and one `ToolsSameCandidate100Neg` training child. The
row is not complete or table-eligible yet: no `fairness_provenance.json`, no
`scores.csv`, and no imported ranking table exist. Validation metrics were
still improving at epoch `330`, latest observed validation HR@5/10/20
`0.26969999074935913 / 0.3407999873161316 / 0.414900004863739` and
NDCG@5/10/20
`0.20475752651691437 / 0.22773440182209015 / 0.2465011030435562`; these are
training-log validation metrics only, not final same-candidate table metrics.
Disk briefly crossed the monitor threshold at about `9.9G` free / `95%` used.
A read-only large-file/cache audit found no safe project-output deletion:
visible large artifacts were either active, final evidence/model artifacts,
same-candidate task data, or an older Electronics ELMRec prediction whose
server-final audit is `ok=false`. After approval, only conda package cache
cleanup was run; no project outputs, evidence, checkpoints, embeddings, task
splits, or other projects were deleted. Post-cleanup `/` was about `11G` free
(`10,773,983,232` bytes) / `95%` used.

At 2026-06-05 18:07-18:11 CST, the same Tools `llm2rec_sasrec` recovery row
remained active with runner PID `3423029`, adapter PID `3423037`, and training
PID `3423221`; duplicate counts stayed exactly one adapter and one
`ToolsSameCandidate100Neg` training child. No final provenance, `scores.csv`,
score audit, or imported ranking table existed yet. Training-log validation
progress had reached epoch `715`, latest observed validation HR@5/10/20
`0.8866999745368958 / 0.9272000193595886 / 0.9584000110626221` and
NDCG@5/10/20
`0.8021516799926758 / 0.8153268098831177 / 0.8232611417770386`; these remain
validation-only progress numbers, not final table metrics. Disk was only about
34 MiB above the 10 GiB guard (`10,773,143,552` bytes free), so non-project
cache cleanup was run: conda reported no unused tarballs/packages/tempfiles,
pip cache purge removed `17.0 MB`, and npm cache clean was attempted. No
project outputs, evidence, checkpoints, embeddings, task splits, or other
projects were deleted. A post-cleanup safety check at 18:11 CST confirmed the
same PIDs alive, duplicate counts `1/1`, no final evidence, and disk
`10,820,177,920` bytes free / `95%` used. Continue monitoring disk through the
final export.

Tools LLM2Rec official completion/gate checkpoint: at 2026-06-05 18:38 CST,
Tools `llm2rec_sasrec` completed normally under the recovery run. Wrapper log
`baselines_new_domains_tools_llm2rec_recovery_20260605_155904.log` contains
`implementation_status=official_completed`, `[2026-06-05 18:38:04] DONE
llm2rec_sasrec on tools`, and `=== All baseline runs complete ===`; the
tracked runner, adapter, and training PIDs exited, with duplicate counts `0/0`.
Server-final audit passed with `ok=true`, `failures=[]`, `warnings=[]`;
server large-artifact manifest passed and records the hashes for
`predictions/rank_predictions.jsonl`, `scores.csv`, and the SASRec checkpoint.
Local-light evidence sync and local-light audit both passed. Local lightweight
evidence is under
`outputs/baselines/official_adapters/tools_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate/`.
Full imported metrics over `10,000` users and `101` candidates are HR@5/10/20
`0.0957 / 0.1625 / 0.2954`, NDCG@5/10/20
`0.060227320850546905 / 0.08147852827156639 / 0.11481218118869342`, and MRR
`0.08101396652891538`. Score audit is PASS with `candidate_rows=1010000`,
`score_rows=1010000`, `matched_keys=1010000`, no missing/extra/duplicate keys,
finite scores `1010000`, and `score_coverage_rate=1.0`. Row counts:
`scores.csv` `1,010,001` lines, `predictions/rank_predictions.jsonl` `10,000`
lines before post-gate cleanup, and `tables/ranking_eval_records.csv` `10,001`
lines. Provenance has `implementation_status=official_completed`,
`blockers=[]`, `score_coverage_rate=1.0`, comparison variant
`official_code_qwen3base_default_hparams_declared_adaptation`, no test-set
model selection, and no extra baseline tuning.

Post-gate disk cleanup for the completed Tools LLM2Rec row preserved final
scores, provenance, audits, imported tables, server-final audit, large-artifact
manifest, SASRec checkpoint, upstream embedding, and compact adapter metadata.
Because final export left disk below the 10 GiB guard, the server-side
prediction JSONL was removed only after server-final and local-light gates
passed and after sha256/line-count manifesting:
`outputs/summary/tools_llm2rec_prediction_deleted_post_gate_20260605.{json,sha256}`.
The removed file had `10,000` lines, size `804,958,794` bytes, and sha256
`211a037a71020955e3488fdcd53f8d6710505bdad23e13c60e7a869d83e99148`.
Disk was still slightly below the strict 10 GiB guard, so the completed-row
adapter staging CSVs `candidate_items_mapped.csv` and `item_text_seed.csv`
were removed with manifest
`outputs/summary/tools_llm2rec_completed_adapter_staging_cleanup_20260605.{json,sha256}`;
removed total size `1,100,523,516` bytes. Post-cleanup disk is
`11,772,899,328` bytes free / `95%` used, and preserved row files were
rechecked: `scores.csv` `1,010,001` lines, `ranking_eval_records.csv`
`10,001` lines, provenance and score audit present.

Tools LLM-ESR launch checkpoint: after Tools LLM2Rec gates and cleanup, a fresh
process/GPU/disk/duplicate-output preflight at 2026-06-05 18:52 CST found no
active experiment Python process, GPU idle, disk `11,647,037,440` bytes free /
`95%` used, and no existing Tools LLM-ESR output or adapter directory. Exactly
one final Tools row was launched with:

```bash
nohup env DOMAINS_OVERRIDE=tools FAST_METHODS_OVERRIDE= TRAIN_METHODS_OVERRIDE=llmesr_sasrec bash scripts/run_baselines_new_domains.sh > baselines_new_domains_tools_llmesr_20260605_185250.log 2>&1 &
```

Runner PID is `3440278`, adapter PID is `3440287`, log is
`baselines_new_domains_tools_llmesr_20260605_185250.log`, pidfile is
`outputs/summary/tools_llmesr_launch_20260605_185250.pid`, launch snapshot is
`outputs/summary/tools_llmesr_launch_monitor_20260605_185250.json`, and
heartbeat is `monitor-tools-llm-esr`. A stable 18:57 CST follow-up confirmed
the runner and adapter were alive, duplicate counts were exactly one LLM-ESR
adapter and zero `ToolsSameCandidate100Neg` training child, GPU was active at
`99%` / `16091 MiB`, embedding progress had reached about `11288/269711`, no
completion/failure/OOM/no-space markers were present, the final output
directory was placeholder-only, and the adapter directory was about `1005M`.
Because adapter staging pushed disk below the strict 10 GiB guard, only
non-project caches were cleaned: `.cache`, `.codex/.tmp`, Google Chrome GPU
cache, and inactive `.cursor-server/bin` after verifying no Cursor process.
No project outputs, evidence, checkpoints, embeddings, task splits, source
code, configs, or other project files were deleted. Cache-cleanup record:
`outputs/summary/tools_llmesr_launch_cache_cleanup_20260605.txt`. Post-cleanup
disk was `10,866,053,120` bytes free / `95%` used at the 18:57 follow-up.

Tools LLM-ESR monitor/cache checkpoint: at 2026-06-05 19:04-19:06 CST, the
active Tools `llmesr_sasrec` row remained alive with runner PID `3440278` and
adapter PID `3440287`; duplicate counts stayed exactly one LLM-ESR adapter and
zero `ToolsSameCandidate100Neg` training child. GPU was active around
`91-96%` / `16189 MiB`, embedding progress advanced to about `47824/269711`,
and no final provenance, scores, score audit, imported ranking table,
completion marker, failure marker, OOM, no-space, or killed marker was present.
Disk dipped just below the strict 10 GiB guard (`10,726,035,456` bytes free).
A read-only cache audit found only small non-project cache/log candidates, so
only Chrome cache/GPU cache and VSCode-server logs were cleared; no project
outputs, evidence, checkpoints, embeddings, task splits, source code, configs,
or other project files were deleted. Cleanup record:
`outputs/summary/tools_llmesr_monitor_cache_cleanup_20260605_1904.txt`.
Post-cleanup disk was `10,757,963,776` bytes free / `95%` used.

Tools LLM-ESR disk emergency checkpoint: at 2026-06-05 19:09-19:13 CST, the
active Tools `llmesr_sasrec` row remained alive with runner PID `3440278` and
adapter PID `3440287`; duplicate counts stayed exactly one LLM-ESR adapter and
zero `ToolsSameCandidate100Neg` training child. GPU was active around `96%` /
`16213 MiB`, embedding progress advanced from about `59448/269711` to about
`77056/269711`, and no final evidence or failure markers were present. Disk
fell below the strict 10 GiB guard (`10,633,687,040` bytes free). A completed
prediction audit found no safe prediction deletion: the only remaining large
prediction was the legacy Electronics ELMRec file whose server-final audit is
`ok=false`. Re-running conda/pip/npm cache cleanup freed `0` bytes; record:
`outputs/summary/tools_llmesr_monitor_conda_cache_cleanup_20260605_1909.txt`.
After explicit destructive-action approval, the completed Tools LLM2Rec SASRec
checkpoint was deleted only after verifying `server_final_evidence_audit.ok=true`,
`implementation_status=official_completed`, `blockers=[]`, and
`score_coverage_rate=1.0`, and after writing sha256/size manifests:
`outputs/summary/tools_llm2rec_completed_checkpoint_cleanup_for_llmesr_disk_20260605.{json,sha256}`.
Deleted checkpoint size was `5,665,876,357` bytes with sha256
`8ad7ce0316befeb8ee6b3482546ffe3e301e42e9a6b1e10ee608689ea5ece414`. Preserved
Tools LLM2Rec artifacts were rechecked: `scores.csv` `1,010,001` lines,
`tables/ranking_eval_records.csv` `10,001` lines, provenance, score audit,
run summary, server-final audit, large-artifact manifest, and local-light
package. Post-cleanup disk was `16,293,425,152` bytes free / `92%` used, and
the active LLM-ESR row remained alive.

Tools LLM-ESR embedding/training transition checkpoint: at 2026-06-05
19:55-20:01 CST, the active Tools `llmesr_sasrec` row remained alive with
runner PID `3440278` and adapter PID `3440287`; duplicate counts stayed exactly
one LLM-ESR adapter, zero `ToolsSameCandidate100Neg` training child, and one
relevant Python process. The Qwen3 `hf_mean_pool` embedding pass completed
(`269711/269711`), the row entered LLM-ESR training with log marker
`[llmesr] epoch=1 train_loss=1.398057`, and final evidence was still absent:
no `scores.csv`, `fairness_provenance.json`, score audit, run summary,
`tables/ranking_eval_records.csv`, or prediction JSONL. Disk tightened to
`11,804,352,512` bytes free / `95%` used after the active adapter grew to
about `5.2G`; GPU was active again at `99%` / `21783 MiB`. A read-only storage
audit showed the new pressure is mainly the active adapter embedding
`outputs/baselines/paper_adapters/tools_large10000_100neg_llmesr_official_adapter/llm_esr/handled/itm_emb_np.pkl`
(`4.12G`), which must not be deleted while training is active. Other largest
visible candidates were protected completed models/checkpoints/task splits or
the legacy Electronics ELMRec prediction without a passing server-final
deletion gate. No cleanup was performed. Continue monitoring; do not start
another Tools row while LLM-ESR is active.

Tools LLM-ESR bounded monitor and gate-plan checkpoint: at 2026-06-05
20:23 CST, the active Tools `llmesr_sasrec` row was still training normally:
runner PID `3440278` and adapter PID `3440287` were alive, duplicate counts were
exactly one LLM-ESR adapter, zero `ToolsSameCandidate100Neg` training child, and
one relevant Python process, GPU was active at `100%` / `21785 MiB`, and disk
was `11,804,286,976` bytes free / `95%` used. The latest log progress was
`[llmesr] epoch=60 train_loss=0.013176`; no completion marker, traceback, OOM,
no-space, killed marker, or final evidence files were present, so the row is
still not table-eligible. A guarded, non-executing completion gate plan was
created at
`outputs/summary/official_completion_gate_plan/tools_llmesr_sasrec_completion_gates_20260605.{json,ps1}`.
It records the required order: server-final audit, server large-artifact
manifest, local-light sync, and local-light audit; the generated PowerShell
script throws before any gate command and must not be used until the active run
exits normally and final evidence exists.

Tools LLM-ESR completion/domain-gate checkpoint: at 2026-06-05 21:19 CST,
the final Tools official row, `llmesr_sasrec`, completed normally. The wrapper
log recorded `implementation_status=official_completed`, `blockers=0`,
`DONE llmesr_sasrec on tools`, and `All baseline runs complete`. Server-final
audit, server large-artifact manifest, local-light sync, and local-light audit
all passed. Full metrics over 10,000 users and 101 candidates are HR@5/10/20
`0.0711 / 0.1270 / 0.2219`, NDCG@5/10/20
`0.042728964614829223 / 0.060602849768892623 / 0.08433244535733923`, and MRR
`0.06334161303438132`; row counts are `scores.csv` `1,010,001` lines,
`predictions/rank_predictions.jsonl` `10,000` lines before cleanup, and
`tables/ranking_eval_records.csv` `10,001` lines. The local lightweight
package is
`outputs/baselines/official_adapters/tools_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/`.
Post-gate cleanup removed only the completed server prediction JSONL and two
adapter staging CSVs with manifest
`outputs/summary/tools_llmesr_post_gate_cleanup_20260605.sha256`; final
`scores.csv`, provenance, audits, imported tables, server-final certificate,
large-artifact manifest, `llmesr_official_model.pt`, and upstream embedding
artifact were preserved.

Tools domain gate/comparison checkpoint: after restoring the valid pre-cleanup
Tools IRLLRec `server_final_evidence_audit.json` from the local-light package
(the overwritten post-cleanup failed audit was preserved as
`server_final_evidence_audit_post_cleanup_failed_20260605.json`), the read-only
Tools official+C-CRP gate passed. C-CRP raw scores were imported into
`outputs/tools_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate`
with `score_coverage_rate=1.0`; the post-cleanup domain gate
`outputs/summary/tools_official_ccrp_gate_post_cleanup_20260605.{json,csv}`
records `official_ok_count=8`, `official_all_ok=true`, `ccrp_ok=true`, and
`gate_ok=true`. Tools C-CRP metrics are HR@5/10/20
`0.1937 / 0.2696 / 0.3931`, NDCG@5/10/20
`0.14186375906171483 / 0.16611553052793934 / 0.19703986741872317`, and MRR
`0.15585924577949772`, with raw `scores.csv` `1,010,001` lines,
`user_ranks.jsonl` `10,000` lines, imported prediction `10,000` lines before
cleanup, and imported `tables/ranking_eval_records.csv` `10,001` lines.
The comparison/statistical package
`outputs/summary/tools_official_ccrp_20260605_*` records
`claim_gate=tools_domain_pass`: C-CRP is observed-best on all seven metrics,
all 56 C-CRP-vs-8-official paired tests have positive deltas and are
Holm-significant, `min_delta=0.0294`, `min_ci_low=0.0173`, and
`max_holm_p_value=9.7172307634557e-07`. The closest official baseline is
`llmemb` on every metric. After the passed domain/comparison gates and local
table sync, the server-only imported C-CRP prediction JSONL was removed under
the internal-method cleanup exception with
`outputs/tools_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate/prediction_deletion_manifest.json`
and summary manifest
`outputs/summary/tools_ccrp_imported_prediction_cleanup_20260605.sha256`;
the post-cleanup domain gate still passes by certificate. Disk remains tight
at about `7.4G` free / `97%` used, so do not start new server jobs without a
fresh storage preflight.

Four-new-domain compact evidence checkpoint: at 2026-06-05 22:15 CST, Codex
used ARIS experiment-audit discipline and a read-only GPT-5.5 xhigh sidecar
audit to consolidate the Sports/Toys/Home/Tools official+C-CRP gates into
`outputs/summary/new_domains_official_ccrp_cross_domain_20260605_*`. The
bundle includes a JSON summary, domain-summary CSV, method-row CSV, audit MD,
and sha256 manifest. Blocking checks pass: all four domains have `gate_ok`,
8/8 completed official rows, one completed C-CRP internal-method row,
C-CRP rank 1 on all seven metrics, and 56/56 C-CRP-vs-official paired tests
with positive Holm-significant deltas. This supports the narrow wording that
C-CRP ranks first against eight official-code-level baselines on the four new
10k-user/101-candidate same-candidate domains. It does not support
paper-ready SOTA, full-catalog SOTA, a large practical-effect claim, universal
winner wording, or a proven uncertainty mechanism. Non-blocking gaps remain:
local raw C-CRP event-level restat is thin for Sports/Toys, so the compact
bundle relies on copied gate/comparison/paired-test certificates rather than a
self-contained local raw reproduction package.

Four-new-domain local/server evidence consistency backfill: at 2026-06-06
01:44 CST, a fresh server preflight found no matching project Python process,
GPU idle at `0%` / `15 MiB`, and `/` at `12,406,677,504` free bytes / `94%`
used. No baseline or experiment was launched. Codex regenerated and synced the
small `server_large_artifact_manifest.{json,sha256}` files for older
Sports/Toys/Home official local packages, using
`--allow_certified_missing_prediction_jsonl` only for rows whose
`server_final_evidence_audit.json` proves the deleted server-only prediction
JSONL existed with `10,000` lines before approved post-gate cleanup. The
post-backfill audit
`outputs/summary/paper_critical/local_server_evidence_consistency_new_domains_post_backfill_20260606.{json,md,sha256}`
reports `ok=true`, `row_count=32`, `ok_count=32`, and `failure_count=0` across
Sports/Toys/Home/Tools. This resolves the local evidence-packaging gap recorded
at 00:50 CST; it does not alter metrics, row eligibility, or the Phase 2.5
signal-row blocker.

Consolidated paper-critical go/no-go checkpoint: at 2026-06-06 01:55 CST,
Codex extended `scripts/audit/main_audit_paper_critical_modules.py` so it
integrates the four-domain local/server evidence consistency artifact and a
current Phase 2.5 storage audit. Fresh storage audit:
`outputs/summary/paper_critical/server_storage_phase2_5_retention_audit_current_20260606_0155.{json,md,sha256}`.
Consolidated module audit:
`outputs/summary/paper_critical/paper_critical_module_audit_post_evidence_backfill_20260606_0155.{json,md,sha256}`.
The audit reports `ok=true`, `paper_ready=false`,
`four_domain_evidence_consistent=true`, `framework_overview_scaffold_ready=true`,
`component_inventory_ready=true`, `guarded_plan_ready=true`,
`signal_rows_available=false`, and `phase2_5_storage_launch_allowed=false`.
Server state in the storage audit: no active project Python process, GPU idle,
`12,406,644,736` free bytes, `3,699,482,624` bytes deficit to the 15GiB floor,
zero safe-now recoverable bytes, and eight high-yield candidates requiring
explicit archive/retention approval. The current lowest-risk high-yield
candidate is the completed Tools LLM2Rec upstream embedding cache:
`/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`.
No deletion was performed and no experiment was launched.

Current retention decision packet: at 2026-06-06 02:00 CST, after another
clean server preflight (no matching project Python process, GPU idle, `/` at
`12,406,620,160` free bytes / `94%` used), Codex refreshed the read-only
storage audit and generated a non-destructive approval-decision packet for the
same lowest-risk high-yield candidate. Artifacts:
`outputs/summary/paper_critical/server_storage_phase2_5_retention_audit_current_20260606_0200.{json,md,sha256}`
and
`outputs/summary/paper_critical/retention_cleanup_plan_20260606_current/tools_llm2rec_upstream_embedding_current_retention_decision_plan_20260606_0200.{json,sh,md,sha256}`.
The packet is tied to the current storage audit via `--retention_audit_json`,
reports `will_delete=false`, `will_start_experiment=false`,
`requires_explicit_approval=true`, expected post-delete free bytes
`18,069,307,520`, and `expected_to_clear_min_free_gate=true`. Its generated
shell keeps `exit 2` before `sha256sum` and `rm --`; the Markdown memo is only
a decision surface, not approval. No deletion, manifesting, or experiment
launch occurred.

Retention packet audit checkpoint: at 2026-06-06 02:05 CST, Codex added
`scripts/audit/main_audit_phase2_5_retention_decision_packet.py`, a local-only
auditor for Phase 2.5 retention packets. It verifies plan JSON safety fields,
the shell `exit 2` guard ordering before `sha256sum` and `rm --`, Markdown
safety text, packet sha256 manifest consistency, and agreement with the
referenced storage-audit recommendation. Current audit artifact:
`outputs/summary/paper_critical/retention_cleanup_plan_20260606_current/tools_llm2rec_upstream_embedding_current_retention_decision_packet_audit_20260606_0205.{json,md,sha256}`.
It reports `ok=true`, `read_only=true`, `will_delete=false`,
`will_start_experiment=false`, and no failures. This strengthens the approval
surface but does not authorize deletion or launch.

Live retention pre-approval audit: at 2026-06-06 02:12 CST, Codex added and ran
`scripts/audit/main_remote_phase2_5_retention_preapproval_audit.py`, a
read-only SSH audit against the current server state and the 02:00 retention
packet. Artifact:
`outputs/summary/paper_critical/retention_cleanup_plan_20260606_current/tools_llm2rec_upstream_embedding_preapproval_audit_20260606_0212.{json,md,sha256}`.
It reports `ok=false` only because `disk_below_min_free_before_cleanup`, and
`preapproval_checks_ready_except_disk=true`. Verified live state: active process
count `0`, target size `5,662,687,360`, target sha256 matches
`306618d974eb4133d9cda87bae3251e17d793aa6f5a8cb38d558b549ed31d56e`,
provenance has `implementation_status=official_completed`, `blockers=[]`,
`score_coverage_rate=1.0`, and server-final audit is `ok=true` with scores,
prediction, and ranking eval records present. This is still not approval to
delete; it proves the approval packet is live-state consistent except for the
disk condition it is meant to resolve.

Guarded cleanup dry-run action: at 2026-06-06 02:25 CST, a fresh server
preflight again found no matching project Python process, GPU idle, and `/` at
about `12.41GB` free / `94%` used. Codex added
`scripts/audit/main_execute_phase2_5_retention_cleanup.py`, which validates the
02:00 plan, the 02:05 packet audit, and a fresh 02:25 live preapproval audit
before rendering the ordered cleanup steps. Default mode is dry-run and runs no
remote command. Current artifacts:
`outputs/summary/paper_critical/retention_cleanup_plan_20260606_current/tools_llm2rec_upstream_embedding_preapproval_audit_20260606_0225.{json,md}`,
`outputs/summary/paper_critical/retention_cleanup_plan_20260606_current/tools_llm2rec_upstream_embedding_cleanup_action_dry_run_20260606_0225.{json,md}`,
and
`outputs/summary/paper_critical/retention_cleanup_plan_20260606_current/tools_llm2rec_upstream_embedding_cleanup_action_dry_run_20260606_0225.sha256`.
The dry-run reports `ok=true`, `will_delete=false`, and
`execution_status=dry_run_no_remote_commands`; no deletion, manifesting,
post-delete gate run, or experiment launch occurred.

Paper-facing comparison ledger checkpoint: at 2026-06-06 02:35-02:40 CST,
Codex used a GPT-5.5 xhigh read-only sidecar reviewer to audit the compact
Sports/Toys/Home/Tools official+C-CRP evidence. The sidecar confirmed the
compact certificate is supported but identified a missing strict paper-facing
ledger with provenance columns. Codex added
`scripts/audit/main_audit_cross_domain_official_ccrp_certificate.py` and
`scripts/audit/main_build_new_domains_paper_facing_evidence_ledger.py`.
Artifacts:
`outputs/summary/paper_critical/cross_domain_official_ccrp_certificate_audit_20260606_0235.{json,md,sha256}`,
`outputs/summary/paper_critical/new_domains_paper_facing_full_metric_evidence_ledger_20260606_0240.{csv,json,md}`,
and
`outputs/summary/paper_critical/new_domains_paper_facing_full_metric_evidence_ledger_20260606_0240.sha256`.
The certificate audit reports `ok=true`, `comparison_certificate_ready=true`,
and `paper_ready=false`. The ledger reports `ok=true`,
`comparison_ledger_ready=true`, `row_count=36`, `official_row_count=32`,
`ccrp_row_count=4`, no failures, and the expected warnings that Sports/Toys
C-CRP compact rows are not self-contained for event-level restatement because
local `user_ranks.jsonl` is missing. This closes the strict full-metric
provenance-ledger gap for the four-domain comparison table; it does not unblock
Phase 2.5 signal-row modules.

## Required Next Actions

1. Treat Phase 2 official new-domain baselines as complete for
   Sports/Toys/Home/Tools under the same-candidate gate. Do not launch more
   official baseline rows unless a later audit finds a concrete failed or
   invalid row.
2. Use `outputs/summary/new_domains_official_ccrp_cross_domain_20260605_*` as
   the compact four-new-domain comparison-gate certificate, and use
   `outputs/summary/paper_critical/new_domains_paper_facing_full_metric_evidence_ledger_20260606_0240.*`
   as the full-metric provenance ledger for paper-table drafting. Do not use
   stale partial drafts that still say Home/Tools are incomplete.
3. Prioritize Phase 2.5 paper-critical modules before any paper-readiness
   claim: uncertainty observation/motivation figure or table, full C-CRP
   leave-one-component-out ablation, real hyperparameter curves, and the clean
   framework overview figure.
4. Before any new server work, run a fresh process/GPU/disk preflight. Current
   disk remains around `12.41GB` free / `94%` used and below the Phase 2.5
   launch floor. Either expand disk or make an explicit archive/retention
   decision for the audited lowest-risk high-yield candidate, then use the
   guarded action script only with `--execute` and the exact approval token.
   Preserve scores, provenance, audits, imported tables, C-CRP raw reports/ranks,
   checkpoints/models, and embeddings unless a separate archive decision
   explicitly allows deletion.
5. Only after comparisons, paper-critical modules, provenance, statistical
   tests, and figure checks are complete, move to ARIS paper writing and
   GPT-5.5/Codex xhigh review. The review loop must reach at least 8/10 before
   submission-level readiness is claimed.

## Evidence Gate Commands

Local lightweight package:

```powershell
python scripts\audit\main_sync_official_evidence_package.py `
  --remote_evidence_dir outputs/<EXP>_<METHOD>_official_qwen3base_same_candidate `
  --local_evidence_dir outputs\baselines\official_adapters\<EXP>_<METHOD>_official_qwen3base_same_candidate `
  --copy `
  --quiet

python scripts\audit\main_audit_official_evidence_package.py `
  --evidence_dir outputs\baselines\official_adapters\<EXP>_<METHOD>_official_qwen3base_same_candidate `
  --mode local_light `
  --quiet
```

Server final package:

```powershell
python scripts\audit\main_remote_official_evidence_audit.py `
  --remote_evidence_dir outputs/<EXP>_<METHOD>_official_qwen3base_same_candidate `
  --mode server_final `
  --quiet

python scripts\audit\main_remote_server_large_artifact_manifest.py `
  --remote_evidence_dir outputs/<EXP>_<METHOD>_official_qwen3base_same_candidate `
  --quiet
```

Passing these gates is necessary but not sufficient for paper readiness; final
paper claims still require full cross-baseline comparison tables, paired tests,
claim audit, citation audit, and GPT-5.5/Codex review.
