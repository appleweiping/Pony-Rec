# Server Runbook

This file is the stable entry point for server-side work. It avoids relying on
chat memory.

## Durability Rule

Use `agentmemory` for shared recall, but never treat it as the only source of
truth. After restarts or context loss, recover state from GitHub `main`, the
canonical docs, local evidence packages/manifests, and fresh server audits. For
each important experiment/code/doc checkpoint, push the related local change to
GitHub and mirror key server evidence locally before any cleanup.

## Always Start Here

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
git pull --ff-only
python -m scripts.audit.main_project_readiness_check
```

Read these files before launching heavy work:

```text
AGENTS.md
docs/milestones/README.md
docs/top_conference_review_gate.md
docs/archive/legacy_root_reports/CODEX_HANDOFF_WEEK8_2026-05-06.md
docs/archive/legacy_root_reports/WEEK8_LARGE_SCALE_10K_100NEG_PLAN_2026-05-06.md
docs/archive/legacy_root_reports/OFFICIAL_EXTERNAL_BASELINE_UPGRADE_PLAN_2026-05-07.md
```

The agent normally cannot see this server. Do not assume server state from
local files. Paste back command outputs when something is run, especially logs,
PIDs, audit summaries, and missing-file errors.

## Current Priority Order (2026-06-13)

```text
1. Pull latest repo state.
1.5. The current Codex thread has heartbeat automation
   `uncertainty-2h-goal-heartbeat` active at a two-hour interval. Treat each
   wakeup as one bounded monitor/progress cycle: verify server/git/memory/docs,
   then advance one final-readiness gate without starting or stopping
   experiments blindly.
2. Do not launch a new default experiment. Phase 2.5 evidence is ready for
   strict manuscript-level claim/citation review, not final submission. Citation
   repair for `paper/references.bib` is complete enough for audit:
   `Paper/main.blg` reports `warning$ -- 0`,
   `outputs/summary/paper_critical/citation_audit_repair_20260612.{json,md}`
   reports `must_add_count=0`, and `Paper/main.pdf` compiles. The first
   structural expansion pass is complete:
   `outputs/summary/paper_critical/manuscript_structural_expansion_audit_20260612.{json,md}`
   records an 8-page PDF, explicit uncertainty-stratification table, expanded
   method/protocol/result text, and `66 passed` paper-critical tests. The
   expanded-manuscript claim-text audit at
   `outputs/summary/paper_critical/manuscript_claim_audit_after_structural_expansion_20260612.{json,md}`
   reports `READY_WITH_SCOPE_GUARDS` with no unsupported, overclaimed, or
   contradicted claims. Final citation spot-check is now refreshed at
   `outputs/summary/paper_critical/final_citation_spot_check_20260613.{json,md}`
   with `cited_key_count=21`, `bibliography_entry_count=21`,
   `must_add_count=0`, all eight official baselines cited, no missing or
   uncited keys, and `Paper/main.blg` still at `warning$ -- 0`. GPT-5.5 xhigh
   section-level review returned `8.0/10` conditional pass but not
   submission-ready; the applied fixes are recorded in
   `outputs/summary/paper_critical/section_level_review_20260612.{json,md}`.
   The post-handoff GPT-5.5 xhigh review also returned `8.0/10`
   `CONDITIONAL_PASS`; Codex fixed the abstract `C-CRPranks` spacing issue,
   restored
   `outputs/summary/paper_critical/final_paper_claim_audit_post_section_review_20260612.csv`,
   and recorded the handoff at
   `outputs/summary/paper_critical/review_continuation_packet_20260613.{json,md}`.
   That packet reports `ok=true`, `review_continuation_ready=true`, score floor
   `8.0`, and `final_submission_ready=false`, while keeping
   `final_panel_coverage_complete=false` because explicit Claude Opus reviewer
   output is still missing. The attempted Claude Opus review job failed with
   `Claude CLI did not return JSON output`; the failed attempt is recorded at
   `outputs/summary/paper_critical/claude_opus_review_attempt_20260613.json`
   and does not count as reviewer coverage. A second retry and a third
   tool-discovered retry failed with the same error and are recorded at
   `outputs/summary/paper_critical/claude_opus_review_attempt_retry_20260613.json`
   and
   `outputs/summary/paper_critical/claude_opus_review_attempt_third_20260613.json`.
   `scripts/audit/main_build_review_continuation_packet.py` records failed
   reviewer attempts separately from valid reviewer JSONs. A fourth synchronous
   no-tools Claude review call failed with the same error and is recorded at
   `outputs/summary/paper_critical/claude_opus_review_attempt_sync_notools_20260613.json`;
   a fifth minimal JSON-oriented no-tools call failed at the same
   CLI/connector layer and is recorded at
   `outputs/summary/paper_critical/claude_opus_review_attempt_minimal_json_20260613.json`.
   A sixth synchronous JSON-only call with `model=opus` and tools disabled
   failed with the same connector-layer error and is recorded at
   `outputs/summary/paper_critical/claude_opus_review_attempt_sixth_20260613.json`.
   A seventh synchronous JSON-only call with `model=opus`, tools disabled, and
   a shorter structured JSON-only prompt failed with the same connector-layer
   error and is recorded at
   `outputs/summary/paper_critical/claude_opus_review_attempt_seventh_20260613.json`.
   An eighth asynchronous call with `mcp__claude_review.review_start`,
   `review_status`, `model=opus`, tools disabled, and a short JSON-only prompt
   failed with the same connector-layer error and is recorded at
   `outputs/summary/paper_critical/claude_opus_review_attempt_eighth_20260613.json`.
   A ninth asynchronous call with `mcp__claude_review.review_start`, job
   `a3863723466147e9b9b849cf994ca8fd`, again failed with
   `Claude CLI did not return JSON output` and is recorded at
   `outputs/summary/paper_critical/claude_opus_review_attempt_ninth_20260613.json`.
   A tenth asynchronous call with `mcp__claude_review.review_start`, job
   `b6e19654680c457d8be4845e168ce251`, again failed with the same connector
   error and is recorded at
   `outputs/summary/paper_critical/claude_opus_review_attempt_tenth_20260613.json`.
   An eleventh asynchronous call with `mcp__claude_review.review_start`, job
   `bf4b6b8145404ffa881cd99ed3c73429`, used `model=opus`, tools disabled, and
   a short schema-aligned ARIS hostile-review prompt; it again failed with
   `Claude CLI did not return JSON output` and is recorded at
   `outputs/summary/paper_critical/claude_opus_review_attempt_eleventh_20260613.json`.
   A twelfth direct `mcp__claude_review.review` call used the newly exposed
   tool surface with `model=opus`, tools disabled, and a schema-aligned prompt,
   but failed at the CLI/shell layer because prompt enum pipe characters were
   interpreted by the shell; it is recorded at
   `outputs/summary/paper_critical/claude_opus_review_attempt_twelfth_20260613.json`.
   A thirteenth safe-prompt direct call removed those shell-sensitive
   characters, but returned to `Claude CLI did not return JSON output`; it is
   recorded at
   `outputs/summary/paper_critical/claude_opus_review_attempt_thirteenth_20260613.json`.
   The current packet reports failed Claude attempts `13` and still keeps
   `explicit_claude_opus_present=false`. Use the public-safe request packet
   at
   `outputs/summary/paper_critical/claude_opus_review_request_packet_20260613.{json,md}`
   before trying another Claude route; it was refreshed at
   `2026-06-13T07:13Z`, records failed Claude attempts `13`, contains the
   exact prompt/schema for a valid additional review JSON, dynamically includes
   all recorded failed-attempt artifacts in the follow-up command, and is not
   itself reviewer coverage. The
   local connector-health audit
   `outputs/summary/paper_critical/claude_review_connector_health_20260613.{json,md}`
   reports failed attempt count `13`, valid review evidence count `0`,
   `same_error_tail_streak=1`, `connector_unhealthy=false`, and
   `same_route_retry_recommended=true`. Treat this as diagnostic, not coverage:
   if retrying this route again, keep the shell-safe prompt style and still use
   the request packet to obtain an external Claude Opus JSON if the connector
   returns no substantive JSON. Run
   `scripts/audit/main_validate_claude_opus_review_json.py` before attaching
   it.
   The
   review-continuation builder now validates additional reviewer JSONs before
   counting them for panel coverage; a Claude/Opus JSON must be complete
   substantive evidence (`valid_review_evidence=true`, scoped claim boundary,
   final-ready claim disallowed, kill-argument, concerns, required changes, and
   acknowledged remaining blockers), so a name+score shell cannot close the
   explicit Claude gap.
   The final submission gate now consumes this review-continuation packet
   directly. The refreshed
   `outputs/summary/paper_critical/final_submission_gate_20260613.{json,md}`
   reports `review_continuation_ready=true`,
   `review_panel_coverage_complete=false`, verdict
   `LOCAL_PACKAGE_READY_BUT_EXTERNAL_MANUAL_OR_REVIEW_BLOCKED`, and
   `final_submission_ready=false`; the release-candidate stack now reports
   `blocking_status=external_manual_or_review_blocked`, and the closure packet
   includes a separate `review_panel_coverage` closure group. Do not treat a
   locally ready source/package stack as final ready until ProMax metadata,
   private manual submission-system confirmation, and explicit Claude Opus
   coverage all close.
   A GPT-5.5 xhigh sidecar audit found no veto but identified stale-input risk:
   because the final gate reads the review-continuation packet, the
   pre-submission refresh must fingerprint that packet and its builder. This is
   now fixed in `scripts/audit/main_refresh_pre_submission_gates.py`; the
   refreshed 2026-06-13 freshness artifact records
   `refresh_artifact_fresh=true`. The review-continuation builder also now
   accepts future closed ProMax/closure/release-candidate states instead of
   requiring the current ProMax-blocked shape.
   The latest `Paper/main.pdf` compiles to 9 pages /
   546716 bytes with visible
   official-baseline provenance, all-metric rank-first, and four-domain
   ablation summary tables. Codex has already rerun the evidence-to-claim gate at
   `outputs/summary/paper_critical/final_paper_claim_audit_post_section_review_20260612.{json,md,csv}`
   (`ok=true`, `final_submission_ready=false`) and added a claim-text audit at
   `outputs/summary/paper_critical/manuscript_claim_audit_after_section_review_20260612.{json,md}`
   (`READY_WITH_SCOPE_GUARDS`, 12 supported manuscript claims, no overclaims).
   The current local handoff index is
   `outputs/summary/paper_critical/submission_release_candidate_stack_refresh_20260613.{json,md}`:
   it reports `ok=true`, `local_release_candidate_ready=true`,
   `refresh_artifact_fresh=true`, `blocking_status=external_manual_or_review_blocked`,
   and `final_submission_ready=false` because ProMax public proceedings
   metadata, private manual submission-system confirmation, and explicit Claude
   Opus review coverage remain open. The ProMax
   audit now records passing arXiv HTML ACM-metadata evidence for the expected
   DOI/ISBN/venue/location and exposes BibTeX `isbn`/`location`, but final ACM
   page range plus Crossref/DOI resolver visibility are still unresolved.
   For a compact closure view, read
   `outputs/summary/paper_critical/final_submission_blocker_closure_packet_20260613.{json,md}`;
   it now uses the 2026-06-13 final gate, external metadata audit, manual
   checklist, full release-candidate stack, and public probe; it groups local
   artifact, review-panel coverage, external metadata, and private manual
   submission blockers and
   gives the exact next commands without storing private fields.
   The latest ProMax public probe is
   `outputs/summary/paper_critical/promax_public_metadata_probe_20260613.{json,md}`:
   Crossref remains `404`, DOI resolver remains `404`, ACM DL returns `403`,
   and `promax_public_metadata_ready=false`; arXiv HTML ACM metadata, the
   SIGIR accepted-paper page, the UQ author-profile page, the author Google
   Sites publications page, and the UQ Experts profile source probes all pass.
   Follow-up live probes at `2026-06-13T00:49:05Z`,
   `2026-06-13T01:59:36Z`, `2026-06-13T02:32:01Z`,
   `2026-06-13T03:16:11Z`, `2026-06-13T04:11:37Z`,
   `2026-06-13T04:30:39Z`, `2026-06-13T04:49:27Z`,
   `2026-06-13T05:11:34Z`, `2026-06-13T05:31:00Z`,
   `2026-06-13T05:42:19Z`, `2026-06-13T06:04:37Z`, and
   `2026-06-13T06:52:21Z` again found Crossref `404`, DOI resolver `404`, ACM
   DL `403`, and all `5/5` public source probes passing. After the
   latest probe, the closure packet was refreshed again at
   `2026-06-13T06:53:47Z`;
   the closure Markdown now lists those latest public source probes and
   explicitly keeps the review-panel blockers. This is
   stronger provenance evidence, not a readiness upgrade. The complete local release-candidate
   stack was also refreshed at `2026-06-13T06:06:09Z`: it reports `ok=true`,
   `local_release_candidate_ready=true`, `refresh_artifact_fresh=true`,
   `failures=[]`, and `final_submission_ready=false`; the freshness audit has
   zero input or generated-gate mismatches, and the independent source-package
   rebuild produces a `9`-page PDF with zero BibTeX and overfull hbox warnings.
   The release-stack warning aggregators now normalize known nested gate
   prefixes before adding the current layer prefix, preventing recursive
   warning growth in review-continuation, final-gate, pre-submission-refresh,
   release-candidate, stack, and closure packet outputs. The closure-packet
   builder also now infers the input stamp from a dated `--output-json` or
   `--output-md` path when `--stamp` is omitted, preventing a 20260613 closure
   artifact from silently reading 20260612 inputs. It also defaults to the
   same-stamp `promax_public_metadata_probe_<stamp>.json` when available; the
   final-blocker consistency audit now fails if the closure packet omits that
   probe or records direct status codes inconsistent with the standalone probe.
   The Claude review handoff is also stricter now: explicit Claude Opus
   coverage requires a reviewer identity containing both `claude` and `opus`,
   and `scripts/audit/main_validate_claude_opus_review_json.py` must be run on
   any external Claude Opus JSON before attaching it through
   `--additional-review-json`. The refreshed
   `outputs/summary/paper_critical/claude_opus_review_request_packet_20260613.{json,md}`
   includes the exact validation command, a fillable `response_template` whose
   `valid_review_evidence=false` placeholder prevents accidental attachment,
   plus the follow-up review-continuation command. The current validator and
   review-continuation builder require any returned Claude Opus JSON to
   acknowledge the open ProMax public metadata blocker and the private manual
   submission-system blocker before it can count for explicit Claude Opus
   coverage; the refreshed review-continuation packet exposes
   `required_claude_blocker_ack_groups=["manual_submission_system",
   "promax_public_metadata"]`.
   `scripts/audit/main_audit_final_blocker_consistency.py` now uses schema
   `2026-06-13.final_blocker_consistency_audit.v3` and fails if those Claude
   intake safeguards disappear from the request packet or review-continuation
   packet, or if the private manual-confirmation validator is no longer routed
   before the public manual checklist. Rerun it after any Claude
   request/review, final-gate, closure, manual-request, or release-stack
   refresh.
   The current priority is to capture explicit Claude Opus reviewer output
   using the request packet if available, then keep monitoring the ProMax public
   metadata and private manual submission-system blockers. Do not claim final
   readiness until the final submission gate reports true.
3. Treat performance/table evidence as complete for the current same-candidate
   claim: C-CRP v3 has eight-domain reports; Sports/Toys/Home/Tools each have
   all eight official-code-level baseline rows complete; the four-new-domain
   paper-facing ledger has `official_row_count=32`, `ccrp_row_count=4`, and
   per-domain 56/56 positive Holm-significant paired tests.
4. Do not launch more official baseline rows unless a later audit finds a
   concrete failed or invalid row.
5. Phase 2.5 component ablation is now closed as supplementary diagnostic
   evidence: Sports/Toys/Home/Tools component packages and the four-domain
   aggregation pass their audits, with table eligibility limited to
   `supplementary_diagnostic_only`. Do not claim every component is necessary.
6. Phase 2.5 observation/motivation is now closed as descriptive
   motivation-only evidence: Sports/Toys/Home/Tools observation packages and
   the four-domain aggregate pass their hardened gates, with table eligibility
   limited to `motivation_only_not_main_table_sota`. Use wording that C-CRP
   event-level uncertainty stratifies ranking reliability; do not claim
   causality, statistical significance, exhaustive baseline behavior, or
   main-table SOTA evidence from this module.
7. Phase 2.5 hyperparameter analysis is now closed as supplementary
   stability/sensitivity evidence: Sports/Toys/Home/Tools single-domain
   packages and the four-domain aggregate pass their hardened gates. The
   aggregate lives at
   `outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/ccrp_hyperparameter_four_domain/`
   and reports `all_controls_stable=true` for `eta` and
   `weight_grid_label` on NDCG@10. Do not call this main-table SOTA evidence,
   all-metric robustness, test-selected tuning, or proof that the risk penalty
   is necessary.
8. If a future audit finds no saved signal rows pass audit, use the guarded
   signal-row runner
   `experiments/rsc/run_ccrp_v3_signal_rows.py` to generate valid/test
   recomputable signal rows on the server, then audit them before selector use.
   Use `/home/ajifang/miniconda3/envs/qwen_vllm/bin/python`; the base
   `/home/ajifang/miniconda3/bin/python` vLLM import failed on 2026-06-06 with
   `libcudart.so.13` missing.
   Monitor progress as chunk-local: with the default `chunk_users=5000` and
   `expected_candidates_per_event=101`, each full chunk logs
   `505000/505000`, while a 10k-user valid/test split still expects
   `1,010,000` final signal rows. Do not treat one `505000/505000` progress
   line as completion; require process exit plus final CSV/provenance files.
9. Keep SETRec excluded while blocked by upstream `tokenize_all` CUDA OOM
   failures; do not include it in the main official block unless a future
   memory-stable run passes all gates.
10. Build Signal/Decision/Generative LoRA artifacts only after teacher data and
   validation gates exist.
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `run_ccrp_v3_all_new_domains.sh` | C-CRP v3 on sports/toys/home/tools (sequential) |
| `scripts/run_baselines_new_domains.sh` | 8 baselines × 4 new domains |
| `scripts/run_ccrp_v3_new_domains.sh` | C-CRP v3 on home/tools only |
| `scripts/audit/main_audit_domain_official_gate.py` | Read-only domain gate for the eight official baselines plus C-CRP full metrics, row counts, coverage, provenance, and stray official-like directories |
| `scripts/experiments/main_build_domain_official_comparison.py` | Read-only full comparison table and paired tests for C-CRP vs eight official baselines |
| `scripts/audit/main_plan_phase2_5_retention_cleanup.py` | Guarded, non-executing Phase 2.5 storage-retention plan; generated shell exits before manifest or delete commands, emits a packet `.sha256`, and requires explicit approval |
| `scripts/audit/main_audit_phase2_5_storage_retention.py` | Read-only SSH storage audit for Phase 2.5 disk gates; classifies safe-now, protected, and approval-required high-yield artifacts and emits a ranked approval recommendation without deleting |
| `scripts/audit/main_cleanup_phase2_5_safe_now_remnants.py` | Exact-target cleanup helper for audited low-yield Phase 2.5 staging/temp remnants; manifests every file before deletion and never targets final evidence |
| `scripts/audit/main_audit_local_server_evidence_consistency.py` | Local-only consistency audit comparing lightweight evidence packages against copied server large-artifact manifests; catches missing local files and accidental local bulk artifacts |
| `scripts/audit/main_audit_paper_critical_modules.py` | Local paper-critical go/no-go audit; integrates signal-row readiness, guarded plan, framework figure, component inventory, observation/component/hyperparameter execution support, four-domain evidence consistency, and the current Phase 2.5 storage gate when artifact paths are supplied |
| `scripts/audit/main_audit_phase2_5_module_package.py` | Local-only package audit for completed Phase 2.5 observation, component-ablation, and hyperparameter modules; enforces command/log/config/hash/row-count/plot/provenance/local-server evidence before paper claims |
| `scripts/audit/main_sync_ccrp_signal_evidence_package.py` | Copies and audits completed Phase 2.5 C-CRP signal-row packages from pony-rec-gpu; verifies local/server size and sha256, row counts, provenance counts, parse-failure rate, and source-audit candidate-key coverage |
| `scripts/audit/main_plan_ccrp_signal_generation.py` | Guarded, non-executing Phase 2.5 C-CRP signal-row plan; records discovery, valid/test signal generation, valid/test source audits, local signal-package sync templates, validation selection, component-ablation summary, observation, hyperparameter, and module-package audit commands while generated shell exits before execution |
| `scripts/analysis/main_build_ccrp_component_ablation_summary.py` | Builds the paper-critical C-CRP leave-one-component-out summary from a frozen validation-selected selector package; writes summary/provenance/figures and fails closed on missing validation metadata, non-full score mode by default, missing ablations, coverage, or audit failures |
| `scripts/analysis/main_aggregate_ccrp_component_ablation.py` | Aggregates package-audited Sports/Toys/Home/Tools component-ablation summaries; fails closed on missing package audits, row counts, coverage, or ablations; emits `removal_minus_full` deltas, `tie_epsilon=1e-12`, figures, and supplementary/diagnostic-only provenance |
| `scripts/analysis/main_build_uncertainty_observation_study.py` | Builds the paper-critical uncertainty observation/motivation package from real C-CRP uncertainty rows plus representative same-candidate ranking records; fails closed on missing uncertainty columns, duplicate/missing events, candidate-count mismatches, and incomplete 101-row/event uncertainty coverage |
| `scripts/analysis/main_aggregate_uncertainty_observation_study.py` | Aggregates package-audited Sports/Toys/Home/Tools observation packages; requires the exact four-domain set for paper-claim readiness, downgrades subsets to diagnostic-only, emits high-minus-low and five-bin trend figures, and keeps table eligibility `motivation_only_not_main_table_sota` |
| `scripts/analysis/main_build_ccrp_hyperparameter_sweep.py` | Builds paper-critical C-CRP valid/test hyperparameter sweep inputs from audited saved signal rows only; no LLM re-query, score import, retained per-grid score dumps, prediction JSONL, or checkpoints. Main controls are `eta` and `weight_grid_label`; `confidence_weight` is diagnostic-only under `confidence_plus_evidence`. |
| `scripts/analysis/main_plot_ccrp_hyperparameter_sweep.py` | Builds hyperparameter curve summaries/figures from valid/test sweep CSVs and embeds sweep-source provenance; package audit requires `test_not_used_for_selection=true`, exact row counts, cleanup status, coverage `1.0`, and audit/degeneracy flags before paper use. |
| `scripts/analysis/main_aggregate_ccrp_hyperparameter_analysis.py` | Aggregates package-audited Sports/Toys/Home/Tools C-CRP hyperparameter curve packages; fails closed on missing package audits, invalid row counts/coverage, unstable controls, or test-selection provenance; emits four-domain sensitivity/stability tables and figures with supplementary-only eligibility. |
| `scripts/audit/main_audit_phase2_5_retention_decision_packet.py` | Local-only audit for a Phase 2.5 retention decision packet; verifies the plan, shell guard, markdown memo, sha256 manifest, and storage-audit target agreement without SSH or deletion |
| `scripts/audit/main_execute_phase2_5_retention_cleanup.py` | Fail-closed retention cleanup action renderer/executor; defaults to dry-run, validates the plan, packet audit, and live preapproval audit, and requires `--execute` plus the exact approval token before any remote command runs |
| `scripts/audit/main_audit_cross_domain_official_ccrp_certificate.py` | Local-only audit for the compact Sports/Toys/Home/Tools official+C-CRP comparison certificate; verifies domain gates, method rows, paired-test counts, evidence consistency, and paper-scope disclaimers |
| `scripts/audit/main_build_new_domains_paper_facing_evidence_ledger.py` | Local-only builder for the 36-row paper-facing full-metric evidence ledger; joins metrics, row counts, provenance/status, gate paths, paired-test paths, score audits, server-final audits, and local-light evidence paths |
| `scripts/audit/main_build_final_paper_claim_audit.py` | Local-only final evidence-to-claim gate for the current paper-facing evidence package; emits supported/contradicted/unsupported claim rows and citation/manuscript readiness status without SSH or experiment execution |
| `scripts/audit/main_build_final_citation_spot_check.py` | Local-only ARIS citation spot-check over the current TeX aux/BibTeX files; verifies cited/bibliography key counts, missing/uncited keys, BibTeX warnings, official-baseline citation coverage, and keeps final readiness false while metadata blockers remain |
| `scripts/audit/main_build_review_continuation_packet.py` | Local-only review-continuation handoff over the full-panel review, claim audit, submission package audit, release-candidate stack, closure packet, ProMax probe, and optional fresh reviewer JSONs; validates additional reviewer JSONs before coverage, records score floor, missing reviewer perspectives, and final-blocked status |
| `scripts/audit/main_build_claude_review_request_packet.py` | Local-only request-packet builder for the missing explicit Claude Opus review; emits a public-safe prompt, expected JSON schema, failed-attempt summary, and follow-up command without counting as reviewer coverage |
| `scripts/analysis/main_build_paper_result_tables.py` | Local-only paper-table builder for the visible complete NDCG@10 ranking over all eight official baselines plus C-CRP and the per-domain paired-test summary; reads existing local CSV evidence only |
| `scripts/audit/main_audit_submission_package.py` | Local read-only package/source/PDF gate for the anonymous ACM submission package; verifies source closure, local figures, BibTeX/build health, profile constraints, source manifest, final-panel evidence, external metadata blockers, and privacy-preserving anonymous source leak scan counts |
| `scripts/audit/main_build_submission_source_package.py` | Local-only anonymous source-package staging builder; copies exactly the audited `source_package_manifest` files into ignored `artifacts/`, validates hashes/sizes/git allowlist/private-path exclusions/exact tree replacement, and writes a compact JSON/MD manifest without changing final submission readiness |
| `scripts/audit/main_audit_submission_source_package_rebuild.py` | Local-only rebuildability audit for the staged anonymous source package; verifies exact staged tree/hash match, rebuilds in ignored `artifacts/` with `pdflatex -> bibtex -> pdflatex -> pdflatex`, checks PDF/log/BibTeX/overfull gates, and remains separate from final submission readiness |
| `scripts/audit/main_refresh_pre_submission_gates.py` | Preferred one-command local artifact refresh for final submission status; regenerates external proceedings metadata, submission package, source-package staging, source-package rebuild, metadata packet, manual checklist, and final gate in dependency order, and passes the current review-continuation packet into the final gate while recording Git HEAD, tracked dirty state, input sha256 fingerprints, and generated gate hashes |
| `scripts/audit/main_audit_pre_submission_refresh_freshness.py` | Local read-only freshness audit for a pre-submission refresh artifact; verifies recorded input fingerprints and generated gate JSON/MD hashes against the current worktree, treating Git HEAD as generation provenance rather than a strict post-commit equality gate |
| `scripts/audit/main_build_submission_release_candidate_packet.py` | Local read-only release-candidate handoff index over the final gate, freshness audit, source package, rebuild audit, metadata packet, manual checklist, and external metadata audit; distinguishes `local_release_candidate_ready` from `final_submission_ready` and surfaces `external_manual_or_review_blocked` when review coverage is still incomplete |
| `scripts/audit/main_refresh_submission_release_candidate_stack.py` | Preferred sequential local handoff wrapper; runs pre-submission refresh, freshness audit, and release-candidate packet generation in order, then emits a compact stack artifact while preserving external/manual/review blockers and `final_submission_ready=false` |
| `scripts/audit/main_audit_final_blocker_consistency.py` | Local read-only consistency audit over the final gate, release stack, closure packet, review-continuation packet, Claude request packet, ProMax probe, and manual request packet; catches stale failed-Claude counts, missing open blockers, unexpected final-ready states, recursive warning-prefix regressions, closure packets that omit or mismatch the same-stamp ProMax probe, and manual handoffs that skip the private-confirmation validator |
| `scripts/audit/main_audit_final_blocker_doc_status.py` | Local read-only canonical-doc status audit over active TODO, claim/status, milestones, and server runbook current sections; compares them with the final-blocker consistency audit and catches stale current failed-Claude counts, old two-class blocker taxonomy, missing ProMax/manual/Claude blocker wording, or accidental final-ready wording |
| `scripts/audit/main_build_final_submission_blocker_closure_packet.py` | Local read-only closure packet over the final gate, external metadata audit, manual checklist, release-candidate stack, and same-stamp ProMax public probe; groups final blockers into local artifact, review-panel coverage, public external metadata, and private manual-submission closure paths; infers the stamp from dated output paths when `--stamp` is omitted |
| `scripts/audit/main_build_final_submission_gate.py` | Local read-only final pre-submission aggregator over package audit, metadata packet, source-package rebuild, external proceedings metadata, manual checklist, and review-continuation coverage; keeps final readiness false while external DOI/page-range, private submission-system, or explicit Claude Opus review blockers remain |
| `scripts/audit/main_build_manual_submission_checklist.py` | Local read-only checklist builder for submission-system actions; safely pre-fills public metadata and can optionally consume an untracked `--private-confirmation-json` while keeping authors, conflicts, reviewer preferences, declarations, and private account metadata out of git |
| `scripts/audit/main_build_manual_submission_private_confirmation_request_packet.py` | Local read-only public-safe request-packet builder for the private manual submission-system confirmation; emits the current source-manifest hash, safe confirmation skeleton, forbidden private fields/keys, recommended ignored path, and follow-up commands without consuming private data or closing final readiness |
| `scripts/audit/main_validate_manual_submission_private_confirmation_json.py` | Local read-only preflight for an untracked private manual confirmation JSON; checks schema/profile/checklist/source-manifest agreement, ignored-path placement, forbidden private keys, item IDs, and currently blocked items before the public manual checklist consumes the file |
| `scripts/audit/main_audit_external_proceedings_metadata.py` | Local read-only ARIS citation/proceedings metadata recheck for ProEx/ProMax; records BibTeX, DOI/Crossref, arXiv, DBLP/SIGIR source visibility, and advisory Crossref title-discovery candidates while keeping final submission blocked until exact public page-range/registry evidence is present |
| `scripts/audit/main_probe_promax_public_metadata.py` | Lightweight local live probe for the ProMax public metadata blocker; checks BibTeX pages, Crossref `/works`, DOI resolver, ACM DL, arXiv HTML ACM metadata, SIGIR accepted-paper source, UQ author-profile announcement, author Google Sites publications, and UQ Experts profile without running the full submission stack |
| `outputs/summary/paper_critical/citation_audit_repair_20260612.{json,md}` | ARIS-style citation repair audit for the active manuscript; records all eight official baseline citations, recency counts, and `bibtex main` warning status |
| `experiments/rsc/run_ccrp_v3_domain.py` | Single-domain C-CRP v3 runner |
| `experiments/rsc/run_ccrp_v3_signal_rows.py` | Phase 2.5 server-side C-CRP signal-row runner; emits recomputable rows with identity, raw/calibrated relevance probability, evidence support, counterevidence strength, provenance, and parse-failure audit; do not use outputs until `main_audit_ccrp_uncertainty_sources.py` passes |

Pre-submission freshness check after a refresh:

```bash
python -m scripts.audit.main_audit_pre_submission_refresh_freshness \
  --refresh-json outputs/summary/paper_critical/pre_submission_gate_refresh_YYYYMMDD.json \
  --output-json outputs/summary/paper_critical/pre_submission_gate_refresh_freshness_YYYYMMDD.json \
  --output-md outputs/summary/paper_critical/pre_submission_gate_refresh_freshness_YYYYMMDD.md
```

This command is local-only. A changed Git HEAD is not a failure by itself; file
fingerprint mismatches are.

Preferred one-command local release-candidate handoff refresh:

```bash
python -m scripts.audit.main_refresh_submission_release_candidate_stack \
  --stamp YYYYMMDD \
  --output-json outputs/summary/paper_critical/submission_release_candidate_stack_refresh_YYYYMMDD.json \
  --output-md outputs/summary/paper_critical/submission_release_candidate_stack_refresh_YYYYMMDD.md
```

This command is local-only except for the live external metadata HTTP checks
inside the refresh step. It regenerates the refresh/freshness/release-candidate
stack in sequence and should be the default before reporting local
pre-submission handoff status.

Final blocker consistency audit after any final-gate, stack, closure, review,
ProMax, or manual-request refresh:

```bash
python -m scripts.audit.main_audit_final_blocker_consistency \
  --output-json outputs/summary/paper_critical/final_blocker_consistency_audit_YYYYMMDD.json \
  --output-md outputs/summary/paper_critical/final_blocker_consistency_audit_YYYYMMDD.md
```

This command is local-only and must not close readiness by itself. It verifies
that the current expected blockers are still consistently represented across
the handoff artifacts.

Canonical-doc status audit after any current-status documentation update:

```bash
python -m scripts.audit.main_audit_final_blocker_doc_status \
  --output-json outputs/summary/paper_critical/final_blocker_doc_status_audit_YYYYMMDD.json \
  --output-md outputs/summary/paper_critical/final_blocker_doc_status_audit_YYYYMMDD.md
```

This command is local-only and must not close readiness by itself. It verifies
that first-read docs match the latest consistency audit and that old
eight-attempt/two-blocker chronology is either historical or removed.

Private manual submission confirmation, after a human has completed the
submission-system fields:

```bash
python -m scripts.audit.main_build_manual_submission_private_confirmation_request_packet \
  --output-json outputs/summary/paper_critical/manual_submission_private_confirmation_request_packet_YYYYMMDD.json \
  --output-md outputs/summary/paper_critical/manual_submission_private_confirmation_request_packet_YYYYMMDD.md
```

Use the request packet first to get the current source manifest sha256, the full
safe `completed_item_ids` skeleton, the recommended ignored confirmation path,
and the forbidden private fields/JSON keys. The request packet is public-safe
and does not close any gate.

```bash
python -m scripts.audit.main_validate_manual_submission_private_confirmation_json \
  --private-confirmation-json path/to/untracked_private_confirmation.json \
  --manual-request-packet-json outputs/summary/paper_critical/manual_submission_private_confirmation_request_packet_YYYYMMDD.json \
  --output-json outputs/summary/paper_critical/manual_private_confirmation_validation_YYYYMMDD.json \
  --output-md outputs/summary/paper_critical/manual_private_confirmation_validation_YYYYMMDD.md
```

Run this validator before the checklist consumes the private confirmation JSON.
It must pass without storing private field values; if it reports a currently
blocked item such as `confirm_external_proceedings_metadata`, refresh/close the
public ProMax metadata gate first instead of marking that item complete early.

```bash
python -m scripts.audit.main_build_manual_submission_checklist \
  --private-confirmation-json path/to/untracked_private_confirmation.json \
  --output-json outputs/summary/paper_critical/manual_submission_checklist_YYYYMMDD.json \
  --output-md outputs/summary/paper_critical/manual_submission_checklist_YYYYMMDD.md
```

Use `configs/paper_manual_submission_private_confirmation.template.json` as the
safe shape, but keep the filled confirmation file untracked. It must contain no
author names, COI details, reviewer preferences, declarations, account
metadata, or other private payload.

## Monitoring

```bash
# Check progress
tail -5 ~/projects/pony-rec-rescue-shadow-v6/ccrp_v3_all_domains.log
grep 'DONE:\|FAILED:\|C-CRP v3:' ccrp_v3_all_domains.log

# GPU status
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader

# Disk
df -h /home/ajifang

# Running processes
ps aux | grep python | grep -v grep
```

For `experiments/rsc/run_ccrp_v3_signal_rows.py`, parse both `Processed
prompts:` and `Adding requests:` lines. Progress bars are chunk-local. A
completed Sports valid/test package must have `1,010,000` signal rows plus the
CSV header, matching provenance counts, and a passing
`main_audit_ccrp_uncertainty_sources.py` source audit before local sync or
selector use.

For Phase 2.5 disk gates, run the read-only storage retention audit from the
local checkout:

```powershell
python scripts\audit\main_audit_phase2_5_storage_retention.py `
  --remote pony-rec-gpu `
  --project ~/projects/pony-rec-rescue-shadow-v6 `
  --output_json outputs\summary\paper_critical\server_storage_phase2_5_retention_audit_current_20260605.json `
  --output_md outputs\summary\paper_critical\server_storage_phase2_5_retention_audit_current_20260605.md
```

If the audit recommends an approval-required artifact, generate a guarded
planning artifact before any retention decision is acted on. The generated
shell exits with `exit 2`; it is a decision surface, not an execution script:

```powershell
python scripts\audit\main_plan_phase2_5_retention_cleanup.py `
  --candidate tools_llm2rec_upstream_embedding `
  --current_free_bytes 12407414784 `
  --output_dir outputs\summary\paper_critical\retention_cleanup_plan_20260606 `
  --plan_id tools_llm2rec_upstream_embedding_ranked_retention_cleanup_plan_20260606
```

The 2026-06-06 ranked plan artifact records the completed Tools LLM2Rec
upstream embedding as `recommended_by_ranked_audit=true` with risk tier
`approval_required_external_embedding_cache`, but deletion still requires an
explicit archive/retention approval, exact sha256/size manifesting, and
post-delete domain/comparison gates.

For a current approval-decision packet tied to a fresh storage audit, pass the
audit artifact and request a Markdown memo:

```powershell
python scripts\audit\main_plan_phase2_5_retention_cleanup.py `
  --retention_audit_json outputs\summary\paper_critical\server_storage_phase2_5_retention_audit_current_20260606_0200.json `
  --output_dir outputs\summary\paper_critical\retention_cleanup_plan_20260606_current `
  --plan_id tools_llm2rec_upstream_embedding_current_retention_decision_plan_20260606_0200 `
  --output_json outputs\summary\paper_critical\retention_cleanup_plan_20260606_current\tools_llm2rec_upstream_embedding_current_retention_decision_plan_20260606_0200.json `
  --output_sh outputs\summary\paper_critical\retention_cleanup_plan_20260606_current\tools_llm2rec_upstream_embedding_current_retention_decision_plan_20260606_0200.sh `
  --output_md outputs\summary\paper_critical\retention_cleanup_plan_20260606_current\tools_llm2rec_upstream_embedding_current_retention_decision_plan_20260606_0200.md
```

The generated shell is still intentionally non-runnable as generated: `exit 2`
precedes sha256 manifesting and `rm --`. The Markdown packet is the user-facing
retention-decision surface, not an approval by itself. As of the 2026-06-06
03:00 CST refresh, the planner also writes the packet `.sha256` manifest
expected by the packet auditor.

Before any future approval or action, audit the packet locally:

```powershell
python scripts\audit\main_audit_phase2_5_retention_decision_packet.py `
  --plan_json outputs\summary\paper_critical\retention_cleanup_plan_20260606_current\tools_llm2rec_upstream_embedding_current_retention_decision_plan_20260606_0200.json `
  --plan_sh outputs\summary\paper_critical\retention_cleanup_plan_20260606_current\tools_llm2rec_upstream_embedding_current_retention_decision_plan_20260606_0200.sh `
  --plan_md outputs\summary\paper_critical\retention_cleanup_plan_20260606_current\tools_llm2rec_upstream_embedding_current_retention_decision_plan_20260606_0200.md `
  --packet_sha256 outputs\summary\paper_critical\retention_cleanup_plan_20260606_current\tools_llm2rec_upstream_embedding_current_retention_decision_plan_20260606_0200.sha256 `
  --retention_audit_json outputs\summary\paper_critical\server_storage_phase2_5_retention_audit_current_20260606_0200.json `
  --output_json outputs\summary\paper_critical\retention_cleanup_plan_20260606_current\tools_llm2rec_upstream_embedding_current_retention_decision_packet_audit_20260606_0205.json `
  --output_md outputs\summary\paper_critical\retention_cleanup_plan_20260606_current\tools_llm2rec_upstream_embedding_current_retention_decision_packet_audit_20260606_0205.md
```

The 2026-06-06 02:05 CST packet audit reports `ok=true`, `read_only=true`,
`will_delete=false`, `will_start_experiment=false`, and no failures. This still
does not authorize deletion.

For the final read-only pre-approval check against live server state, run:

```powershell
python scripts\audit\main_remote_phase2_5_retention_preapproval_audit.py `
  --plan_json outputs\summary\paper_critical\retention_cleanup_plan_20260606_current\tools_llm2rec_upstream_embedding_current_retention_decision_plan_20260606_0200.json `
  --output_json outputs\summary\paper_critical\retention_cleanup_plan_20260606_current\tools_llm2rec_upstream_embedding_preapproval_audit_20260606_0212.json `
  --output_md outputs\summary\paper_critical\retention_cleanup_plan_20260606_current\tools_llm2rec_upstream_embedding_preapproval_audit_20260606_0212.md
```

The 2026-06-06 02:12 CST pre-approval audit reports
`preapproval_checks_ready_except_disk=true`: no active process, target size and
sha256 match, row provenance is `official_completed` with `blockers=[]` and
`score_coverage_rate=1.0`, and server-final evidence remains `ok=true`. Its
only failure is `disk_below_min_free_before_cleanup`, which is the condition the
approved cleanup would address.

Before executing any approved cleanup, render the guarded action in dry-run mode
against the latest plan, packet audit, and live preapproval audit:

```powershell
python scripts\audit\main_execute_phase2_5_retention_cleanup.py `
  --plan_json outputs\summary\paper_critical\retention_cleanup_plan_20260606_current\tools_llm2rec_upstream_embedding_current_retention_decision_plan_20260606_0200.json `
  --packet_audit_json outputs\summary\paper_critical\retention_cleanup_plan_20260606_current\tools_llm2rec_upstream_embedding_current_retention_decision_packet_audit_20260606_0205.json `
  --preapproval_audit_json outputs\summary\paper_critical\retention_cleanup_plan_20260606_current\tools_llm2rec_upstream_embedding_preapproval_audit_20260606_0225.json `
  --output_json outputs\summary\paper_critical\retention_cleanup_plan_20260606_current\tools_llm2rec_upstream_embedding_cleanup_action_dry_run_20260606_0225.json `
  --output_md outputs\summary\paper_critical\retention_cleanup_plan_20260606_current\tools_llm2rec_upstream_embedding_cleanup_action_dry_run_20260606_0225.md
```

The 2026-06-06 02:25 CST dry-run action reports `ok=true`,
`will_delete=false`, and `execution_status=dry_run_no_remote_commands`. Execute
mode is only allowed after explicit archive/retention approval for the exact
target, with `--execute --approval_token <exact token>`, and still performs the
ordered pre-delete manifest, exact-target delete, disk check, domain gate, and
comparison gate sequence. Do not use execute mode merely because the dry-run
passes.

Current refreshed decision surface: the 2026-06-06 03:00 CST checkpoint is
under
`outputs/summary/paper_critical/retention_cleanup_plan_20260606_current_0300/`
with checkpoint manifest
`outputs/summary/paper_critical/phase2_5_retention_decision_checkpoint_20260606_0300.sha256`.
It reports packet audit `ok=true`, live preapproval
`preapproval_checks_ready_except_disk=true` with only
`disk_below_min_free_before_cleanup`, and dry-run action rendering `ok=true`,
`will_delete=false`, `execution_status=dry_run_no_remote_commands`. Server disk
remains below the `15GiB` Phase 2.5 launch floor, so do not start signal-row
generation until disk is expanded or the exact approval-required cleanup is
explicitly approved and then passes post-delete gates.

After local-light packages are copied, run a local/server evidence consistency
audit over the completed domain before relying on the package for later paper
work. This reads local files and copied server manifests only:

```powershell
python scripts\audit\main_audit_local_server_evidence_consistency.py `
  --root . `
  --domain tools `
  --output_json outputs\summary\paper_critical\local_server_evidence_consistency_tools_20260606.json `
  --output_md outputs\summary\paper_critical\local_server_evidence_consistency_tools_20260606.md
```

For the four new domains together:

```powershell
python scripts\audit\main_audit_local_server_evidence_consistency.py `
  --root . `
  --domain sports `
  --domain toys `
  --domain home `
  --domain tools `
  --output_json outputs\summary\paper_critical\local_server_evidence_consistency_new_domains_20260606.json `
  --output_md outputs\summary\paper_critical\local_server_evidence_consistency_new_domains_20260606.md
```

The audit should report all expected official rows ok. It fails if a required
local-light file is missing or hash-mismatched, if the copied server large
artifact manifest lacks `scores.csv`, `predictions/rank_predictions.jsonl`, or
a model/checkpoint record, or if server-only bulk files were accidentally copied
into the local package. For older gated rows whose server-only prediction JSONL
was already removed under the approved post-gate cleanup exception, regenerate
the server manifest with:

```powershell
python scripts\audit\main_remote_server_large_artifact_manifest.py `
  --remote_evidence_dir outputs\<ROW> `
  --allow_certified_missing_prediction_jsonl `
  --quiet
```

This accepts the missing prediction JSONL only when
`server_final_evidence_audit.json` proves it existed with the expected `10,000`
lines. As of 2026-06-06 01:45 CST, the post-backfill four-domain audit
`outputs/summary/paper_critical/local_server_evidence_consistency_new_domains_post_backfill_20260606.{json,md,sha256}`
passes for Sports/Toys/Home/Tools: `row_count=32`, `ok_count=32`, and
`failure_count=0`.

For the consolidated paper-critical go/no-go checkpoint after evidence backfill
and storage audit, pass the post-cleanup storage artifact explicitly when
reproducing the current gate. Historical `current_*` or `ranked_*` storage
audits from before cleanup are valid provenance, but they should not be passed
as the current `--storage_audit_json` because they will intentionally report
the old blocked disk gate. Since commit `346f9cc`, the audit's default storage
selection prefers `after_cleanup_final_*` and `after_cleanup_*` before stale
`current_*` artifacts.

```powershell
python scripts\audit\main_audit_paper_critical_modules.py `
  --evidence_consistency_json outputs\summary\paper_critical\local_server_evidence_consistency_new_domains_post_backfill_20260606.json `
  --storage_audit_json outputs\summary\paper_critical\server_storage_phase2_5_retention_audit_after_cleanup_final_20260606_0650.json `
  --output_json outputs\summary\paper_critical\paper_critical_module_audit_current.json `
  --output_md outputs\summary\paper_critical\paper_critical_module_audit_current.md
```

The current go/no-go state should report four-domain evidence consistent,
`phase2_5_storage_launch_allowed=true`, full-scale uncertainty signal rows still
absent, and therefore `paper_ready=false`. If the storage gate is reported as
blocked while the post-cleanup audit is available, check whether an older
`current_*` storage file was passed explicitly.

For active official-baseline rows, prefer the local robust SSH-stdin monitor
when the current local checkout has newer audit helpers than the server:

```powershell
python scripts\audit\main_remote_baseline_monitor_snapshot.py `
  --log_path baselines_new_domains_home_llm2rec_20260604_071902.log `
  --pid 3236678 --pid 3236688 `
  --process_token llm2rec_sasrec --process_token home `
  --size_path outputs/baselines/paper_adapters/home_large10000_100neg_llm2rec_official_adapter `
  --size_path outputs/home_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate `
  --output_json outputs\summary\home_llm2rec_monitor_snapshot_20260604.json `
  --output_json_on_notify_only
```

The helper reads the remote log and status directly without shell regex
pipelines. It reports tracked PID liveness, matching Python processes, latest
`hf_mean_pool` progress, completion/failure markers, GPU, disk, output sizes,
and `should_notify`. With `--output_json_on_notify_only`, quiet checks print to
stdout but do not dirty the working tree by overwriting the snapshot file.
The failure detector ignores generic `error` prose inside warning lines, but it
still treats hard markers such as `Traceback`, exceptions, OOM/no-space,
`killed`, and `failed` as failures.
Treat `should_notify=true` as a handoff trigger: inspect the listed
`notify_reasons` before deciding whether to run official gates, recover from
failure, clean disk, or stop a duplicate launch.

## Safe Nohup Pattern

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
mkdir -p outputs/summary/logs
LOG=outputs/summary/logs/week8_fourdomain_100neg_full_external_$(date +%F_%H%M%S).log
nohup bash scripts/run_week8_large_scale_10k_100neg.sh > "$LOG" 2>&1 &
echo $! > outputs/summary/logs/week8_fourdomain_100neg_full_external.pid
echo "$LOG"
```

Monitor:

```bash
tail -f "$LOG"
ps -p $(cat outputs/summary/logs/week8_fourdomain_100neg_full_external.pid) -o pid=,etime=,cmd=
```

Paste-back template for the user:

```bash
git status --short --branch
echo "LOG=$LOG"
tail -n 80 "$LOG"
ps -p $(cat outputs/summary/logs/week8_fourdomain_100neg_full_external.pid) -o pid=,etime=,cmd=
ls -lh outputs/summary outputs/baselines 2>/dev/null | head -80
```

If a command fails, paste the full traceback plus the command that produced it.
The next agent should patch locally, push to GitHub, and give a `git pull
--ff-only` recovery command.

Stop:

```bash
kill $(cat outputs/summary/logs/week8_fourdomain_100neg_full_external.pid)
```

## Official Baseline Audit

```bash
python -m scripts.audit.main_audit_official_external_repos
python -m scripts.audit.main_audit_official_fairness_policy
```

The old all-row adapter-plan generator is not a current Phase 2.5 entry point.
Sports/Toys/Home/Tools already have eight official baseline rows gated under
the same-candidate protocol. If a later audit finds a concrete invalid row, use
the corresponding method adapter under `scripts/adapters/` and re-run that row
through final provenance, score coverage, import, server-final, local-light,
domain-gate, and paired-test checks.

Do not import official-baseline rows into main comparison tables until:

```text
comparison variant recorded
implementation_status=official_completed for main-table official rows
official repo commit pinned
official training/scoring entrypoint recorded
Qwen3-8B base path recorded
method-declared adaptation mode recorded
baseline hyperparameter source and overrides recorded
method-specific adapter/checkpoint path recorded
source_event_id,user_id,item_id,score score file emitted
exact score-key coverage verified
finite numeric scores verified
paired-test inputs generated
```

The unified runner entry point is:

```bash
python main_run_official_same_candidate_adapter.py \
  --method llm2rec \
  --stage inspect \
  --domain books \
  --task_dir outputs/baselines/external_tasks/books_large10000_100neg_test_same_candidate \
  --output_scores_path outputs/baselines/official_adapters/books_large10000_100neg_llm2rec_official/llm2rec_official_scores.csv \
  --provenance_output_path outputs/baselines/official_adapters/books_large10000_100neg_llm2rec_official/fairness_provenance.json \
  --fairness_policy_id official_code_qwen3base_default_hparams_declared_adaptation_v1 \
  --comparison_variant official_code_qwen3base_default_hparams_declared_adaptation \
  --backbone_path /home/ajifang/models/Qwen/Qwen3-8B \
  --allow_blocked_exit_zero
```

## Lightweight Evidence Sync

For a completed official row, you can first generate a guarded local gate plan
so the required sequence is explicit before running anything:

```powershell
python scripts\audit\main_plan_official_completion_gates.py `
  --domain <domain> `
  --method <method> `
  --output_dir outputs\summary\official_completion_gate_plan `
  --plan_id <domain>_<method>_completion_gates_<date>
```

The generated PowerShell file starts with a `throw`, so it is documentation
until the active runner has exited normally and the preconditions have been
checked. It must not be used to start the next baseline.
For the active Tools IRLLRec row, the current local plan artifact is
`outputs/summary/official_completion_gate_plan/tools_irllrec_intent_completion_gates_20260605.{json,ps1}`.

After a row has passed the server-final evidence audit, first record the
server-only large artifacts on the server. From the local repository, prefer
the SSH-stdin wrapper so it uses the current local helper even if the server
checkout is behind:

```powershell
python scripts\audit\main_remote_server_large_artifact_manifest.py `
  --remote_evidence_dir outputs/<EXP>_<METHOD>_official_qwen3base_same_candidate `
  --quiet
```

If the server checkout is confirmed up to date and idle, the equivalent
server-side command is:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
python scripts/audit/main_build_server_large_artifact_manifest.py \
  --evidence_dir outputs/<EXP>_<METHOD>_official_qwen3base_same_candidate \
  --quiet
```

This writes both `server_large_artifact_manifest.sha256` and
`server_large_artifact_manifest.json` inside the evidence directory. It records
`scores.csv`, `predictions/rank_predictions.jsonl`, and model/checkpoint
artifacts such as `*_official_model.pt` without requiring the operator to guess
the method-specific model filename.

Then sync only the useful local evidence package from the local repository:

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

The sync tool uses an allowlist and verifies server/local size and sha256. It
copies final and inspect provenance, score audits, run summaries, imported
`tables/`, logs/manifests if present, and compact metadata. It excludes
`scores.csv`, `predictions/`, checkpoints, embeddings, and other large binary
artifacts by default, so those remain server-side unless a separate archive
decision is recorded.

For completed Phase 2.5 C-CRP signal-row packages, first run the source audit on
the server against the matching split's `candidate_items.csv`, then sync the
signal package locally with the dedicated helper:

```powershell
python scripts\audit\main_sync_ccrp_signal_evidence_package.py `
  --remote_signal_dir outputs/summary/paper_critical/<PLAN>/ccrp_signal_rows_<domain> `
  --remote_extra_file ccrp_signal_rows_<domain>_<split>_<timestamp>.log `
  --local_package_dir outputs\summary\paper_critical\<PLAN>\ccrp_signal_rows_<domain> `
  --copy `
  --quiet
```

The helper intentionally treats `*_ccrp_signal_rows.csv` as primary Phase 2.5
evidence rather than disposable bulk output. It still enforces a size ceiling,
records excluded files, verifies copied file hashes, and writes
`signal_evidence_sync_manifest.json` plus
`signal_evidence_package_audit.json`. A package is not selector-, observation-,
ablation-, or hyperparameter-eligible unless the source audit and package audit
both pass.

If disk pressure threatens the next active row after `server_final` and
`local_light` audits have passed, a completed row's
`predictions/rank_predictions.jsonl` may be removed only after recording its
sha256 in `outputs/summary/` and documenting the exact path. The domain gate and
comparison builder can use `server_final_evidence_audit.json` as the line-count
certificate for a missing prediction JSONL. Do not remove `scores.csv`, final
provenance, score audits, run summaries, imported `tables/`, or models/checkpoints
under this shortcut.

For imported internal C-CRP rows, the analogous disk-pressure exception is
allowed only after the domain gate has passed, the raw C-CRP `report.json` and
`user_ranks.jsonl` plus imported `tables/` have been copied locally, and a
row-local `prediction_deletion_manifest.json` records the deleted
`predictions/rank_predictions.jsonl` sha256, byte size, and 10,000-line count.
The domain gate and comparison builder accept that manifest only for internal
C-CRP imported predictions; official rows still require
`server_final_evidence_audit.json`.

## LLM2Rec Single-Domain Production Loop

LLM2Rec is the first official-code-level adapter wired for execution. Its run
stage is not a toy scorer: it exports the same-candidate task into LLM2Rec's
native `data/<alias>/downstream` layout, patches the pinned official checkout's
dataset maps, generates Qwen3-8B item text embeddings, invokes the official
`evaluate_with_seqrec.py` / `seqrec.runner.Runner` SASRec training path, then
exports exact same-candidate scores with the shared schema.

For large domains, the official LLM2Rec SASRec state dict includes a duplicate
copy of the precomputed item embedding table. The runner compacts the checkpoint
after official training by removing that duplicated `item_embedding.weight` and
records the operation in provenance; scoring injects the same external Qwen3
`.npy` table before loading the model. Pass `--llm2rec_keep_full_checkpoint`
only when you intentionally want the much larger original checkpoint.

For recovery after a failed LLM2Rec run that already produced a valid upstream
item embedding, the production wrapper accepts
`LLM2REC_ITEM_EMBEDDING_PATH_OVERRIDE`. Set it to the preserved
`.../item_info/<DatasetAlias>/pony_qwen3_8b_title_item_embs.npy` path before
launching the single-domain loop; the wrapper forwards it as
`--llm2rec_item_embedding_path` and must not be combined with
`--force_embeddings`.

Example Tools recovery launch:

```bash
nohup env \
  DOMAINS_OVERRIDE=tools \
  FAST_METHODS_OVERRIDE= \
  TRAIN_METHODS_OVERRIDE=llm2rec_sasrec \
  LLM2REC_ITEM_EMBEDDING_PATH_OVERRIDE=/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy \
  bash scripts/run_baselines_new_domains.sh \
  > baselines_new_domains_tools_llm2rec_recovery_<STAMP>.log 2>&1 &
```

Before using this override, verify no active project experiment process,
GPU/disk health, the existence and size of the embedding file, and absence of
final `scores.csv`/completed provenance for the target row. The row is still
not complete until score audit, import, server-final audit, large-artifact
manifest, local-light sync/audit, full metrics, and row-count gates pass.

Large domains are storage-heavy. The default production policy is one domain at
a time:

```text
run one domain
-> verify implementation_status=official_completed, blockers=[], audit_ok=True
-> import the row with --ks 5,10,20 and rebuild the comparison table
-> package the evidence artifact
-> copy it to local storage with scp
-> verify the local archive exists
-> delete only documented server-side intermediates, not imported summary tables
-> start the next domain
```

This is also the template for future official external baselines. Do not call a
baseline complete after a single raw score file. A method-level official
baseline is complete only after all declared domains have unblocked provenance,
exact score coverage, local evidence backup, server intermediates cleaned, and
same-candidate imports included in a method-level summary table. If the runner
prints `run_stage_not_implemented_for_method`, stop and implement the pinned
official adapter before launching expensive jobs.

For LLM-rec official large domains, use the completed LLM2Rec large-domain
packages as the default archive standard. The lightweight evidence package must
include the score CSV, fairness provenance, score audit, run summary,
training/server log, imported same-candidate predictions/tables, comparison
summary tables, and checkpoint/embedding sha256 manifests when the checkpoint
or embedding file is too large to archive immediately. Do not build a huge
checkpoint tarball by default. Full checkpoints can be archived separately only
when time and storage allow.

Important: comparison tables are rebuilt from imported
`outputs/*_same_candidate/tables/same_candidate_external_baseline_summary.csv`
files. After a domain is imported, keep that `tables/` summary available until
the method-level and final cross-baseline comparison tables have been rebuilt
and archived. If storage is tight, delete predictions or large model/adapter
intermediates first, but do not remove the imported summary directory in a way
that makes a completed domain disappear from later comparisons.
For deleted official-row prediction JSONLs, keep the completed row's
`server_final_evidence_audit.json`; it is the accepted certificate that the
prediction file existed and had the expected 10,000-line count. For deleted
internal C-CRP imported prediction JSONLs, keep
`prediction_deletion_manifest.json` beside the imported `tables/` directory.

After copying a lightweight local evidence package, run the package audit before
recording the row as safely backed up:

```powershell
python scripts\audit\main_audit_official_evidence_package.py `
  --evidence_dir outputs\baselines\official_adapters\<EXP>_<METHOD>_official_qwen3base_same_candidate `
  --mode local_light `
  --quiet
```

Run the same audit in `server_final` mode on the server output directory when
verifying the full server-side result. From the local repository, prefer:

```powershell
python scripts\audit\main_remote_official_evidence_audit.py `
  --remote_evidence_dir outputs/<EXP>_<METHOD>_official_qwen3base_same_candidate `
  --mode server_final `
  --quiet
```

This sends the current local audit helper through SSH stdin, so it does not
depend on the server checkout version. `local_light` intentionally permits
`scores.csv` and `predictions/rank_predictions.jsonl` to stay server-only, but
still requires final provenance, score audits, run summary, full `@5/@10/@20`
and MRR metrics, coverage/exposure/summary tables, and
`tables/ranking_eval_records.csv`.

Single-domain command template:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
DOMAIN=books
EXP=books_large10000_100neg
mkdir -p outputs/summary/logs
LOG=outputs/summary/logs/week8_llm2rec_official_${DOMAIN}_$(date +%F_%H%M%S).log
PID=outputs/summary/logs/week8_llm2rec_official_${DOMAIN}.pid
nohup python main_run_llm2rec_official_same_candidate_adapter.py \
  --stage run \
  --domain "$DOMAIN" \
  --task_dir "outputs/baselines/external_tasks/${EXP}_test_same_candidate" \
  --valid_task_dir "outputs/baselines/external_tasks/${EXP}_valid_same_candidate" \
  --output_scores_path "outputs/baselines/official_adapters/${EXP}_llm2rec_official/llm2rec_official_scores.csv" \
  --provenance_output_path "outputs/baselines/official_adapters/${EXP}_llm2rec_official/fairness_provenance.json" \
  --fairness_policy_id official_code_qwen3base_default_hparams_declared_adaptation_v1 \
  --comparison_variant official_code_qwen3base_default_hparams_declared_adaptation \
  --backbone_path /home/ajifang/models/Qwen/Qwen3-8B \
  --llm_adaptation_mode frozen_base_embedding \
  --hparam_policy official_default_or_recommended \
  --embedding_backend hf_mean_pool \
  --embedding_max_length 128 \
  --hf_device_map auto > "$LOG" 2>&1 &
echo $! > "$PID"
disown
echo "log=$LOG"
echo "pid_file=$PID"
```

Monitor:

```bash
tail -f "$LOG"
ps -p $(cat "$PID") -o pid=,etime=,stat=,cmd=
```

Package after the domain completes. Prefer this lightweight evidence package on
large domains:

```bash
DOMAIN=books
EXP=books_large10000_100neg
BASE=llm2rec_official_qwen3base_sasrec
OUTDIR="outputs/baselines/official_adapters/${EXP}_llm2rec_official"
LOG=$(ls -t outputs/summary/logs/week8_llm2rec_official_${DOMAIN}_*.log 2>/dev/null | head -1)
STAMP=$(date +%F_%H%M%S)
mkdir -p outputs/exports

find "$OUTDIR" -maxdepth 3 -type f \( -name "*.pt" -o -name "*.pth" \) -print0 \
  | xargs -0 -r sha256sum > "${OUTDIR}/checkpoint_manifest.sha256"
ADAPTER_DIR="outputs/baselines/paper_adapters/${EXP}_llm2rec_official_adapter"
if [ -d "$ADAPTER_DIR" ]; then
  find "$ADAPTER_DIR" -maxdepth 3 -type f \( -name "*.npy" -o -name "*.pkl" \) -print0 \
    | xargs -0 -r sha256sum > "${OUTDIR}/embedding_manifest.sha256"
else
  echo "missing_adapter_dir $ADAPTER_DIR" > "${OUTDIR}/embedding_manifest.sha256"
fi

FILES=(
  "${OUTDIR}/fairness_provenance.json"
  "${OUTDIR}/llm2rec_official_score_audit.json"
  "${OUTDIR}/llm2rec_official_run_summary.json"
  "${OUTDIR}/llm2rec_official_scores.csv"
  "${OUTDIR}/checkpoint_manifest.sha256"
  "${OUTDIR}/embedding_manifest.sha256"
  "outputs/${EXP}_${BASE}_same_candidate"
  "outputs/summary/week8_official_external_qwen3base_multik_comparison.csv"
  "outputs/summary/week8_official_external_qwen3base_multik_comparison.md"
)
[ -n "$LOG" ] && [ -f "$LOG" ] && FILES+=("$LOG")
for f in \
  "${ADAPTER_DIR}/adapter_metadata.json" \
  "${ADAPTER_DIR}/llm2rec_embedding_metadata.json" \
  "${ADAPTER_DIR}/llm2rec_upstream_prepare_summary.json"
do
  [ -f "$f" ] && FILES+=("$f")
done

tar -czf "outputs/exports/llm2rec_${DOMAIN}_official_qwen3base_evidence_${STAMP}.tar.gz" \
  "${FILES[@]}"
sha256sum "outputs/exports/llm2rec_${DOMAIN}_official_qwen3base_evidence_${STAMP}.tar.gz"
```

Copy that archive from the local machine, then confirm the local file exists
before deleting server intermediates:

```powershell
scp pony-rec-gpu:~/projects/pony-rec-rescue-shadow-v6/outputs/exports/llm2rec_books_official_qwen3base_evidence_<STAMP>.tar.gz .
Get-Item .\llm2rec_books_official_qwen3base_evidence_<STAMP>.tar.gz
```

Only after local confirmation, clean the completed domain on the server:

```bash
DOMAIN=books
EXP=books_large10000_100neg
rm -rf "outputs/baselines/official_adapters/${EXP}_llm2rec_official"
rm -rf "outputs/baselines/paper_adapters/${EXP}_llm2rec_official_adapter"
rm -rf /home/ajifang/projects/LLM2Rec/item_info/BooksLarge10000_100Neg
df -h /
```

Do not delete final scores, provenance, audits, compact checkpoints, Qwen3
embedding artifacts, or method checkpoints before the evidence archive has
been copied off the server and confirmed by the user. If the archive check
prints `gzip: unexpected end of file` or `tar: unexpected EOF`, delete only the
bad archive and keep all domain outputs.

## LLM2Rec Four-Domain Convenience Wrapper

The four-domain wrapper is not the default production path on storage-limited
servers. Use it only when disk space is sufficient and the user explicitly wants
one batch job:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
mkdir -p outputs/summary/logs
LOG=outputs/summary/logs/week8_llm2rec_official_fourdomain_$(date +%F_%H%M%S).log
PID=outputs/summary/logs/week8_llm2rec_official_fourdomain.pid
nohup bash scripts/run_week8_llm2rec_official_fourdomain.sh > "$LOG" 2>&1 &
echo $! > "$PID"
disown
echo "log=$LOG"
echo "pid_file=$PID"
```

Only after each domain writes `implementation_status=official_completed`,
`blockers=[]`, and the score audit prints `audit_ok=True`, import its row into
the same-candidate summary table. Failed domains can be rerun after fixing the
reported blocker; the adapter package and embeddings are deterministic and can
be reused unless `--force_embeddings` is passed.

LLM2Rec official Qwen3-base status as of 2026-05-09:

```text
beauty supplementary smaller-N: completed/imported
books large10000 100neg: completed/imported
electronics large10000 100neg: completed/imported
movies large10000 100neg: completed/imported
summary:
  outputs/summary/week8_llm2rec_official_qwen3base_fourdomain_summary.csv
  outputs/summary/week8_llm2rec_official_qwen3base_fourdomain_summary.md
```

LLM-ESR also has run-stage support wired in the unified runner. It imports the
pinned repo's `models.LLMESR.LLMESR_SASRec` class and preserves the official
SASRec-style architecture/loss/predict path while local code only adapts the
same-candidate handled files, Qwen3 item embeddings, and exact score export.
As with LLM2Rec, it is not a completed result until a server run writes
`implementation_status=official_completed`, `blockers=[]`, and exact score
coverage.

LLM-ESR single-domain smoke/production template:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
DOMAIN=beauty
EXP=beauty_supplementary_smallerN_100neg
mkdir -p outputs/summary/logs
LOG=outputs/summary/logs/week8_llmesr_official_${DOMAIN}_$(date +%F_%H%M%S).log
PID=outputs/summary/logs/week8_llmesr_official_${DOMAIN}.pid
nohup python main_run_llmesr_official_same_candidate_adapter.py \
  --stage run \
  --domain "$DOMAIN" \
  --task_dir "outputs/baselines/external_tasks/${EXP}_test_same_candidate" \
  --valid_task_dir "outputs/baselines/external_tasks/${EXP}_valid_same_candidate" \
  --output_scores_path "outputs/baselines/official_adapters/${EXP}_llmesr_official/llmesr_official_scores.csv" \
  --provenance_output_path "outputs/baselines/official_adapters/${EXP}_llmesr_official/fairness_provenance.json" \
  --fairness_policy_id official_code_qwen3base_default_hparams_declared_adaptation_v1 \
  --comparison_variant official_code_qwen3base_default_hparams_declared_adaptation \
  --backbone_path /home/ajifang/models/Qwen/Qwen3-8B \
  --llm_adaptation_mode frozen_base_embedding \
  --hparam_policy official_default_or_recommended \
  --embedding_backend hf_mean_pool \
  --embedding_max_length 128 \
  --hf_device_map auto > "$LOG" 2>&1 &
echo $! > "$PID"
disown
echo "log=$LOG"
echo "pid_file=$PID"
```

LLM-ESR follows the same one-domain archive-and-clean loop as LLM2Rec. Its
large-domain checkpoints can be several GB, so the default archive is a light
evidence package plus a model sha256 manifest. After the run and score audit
complete, import the domain with `--ks 5,10,20`, rebuild
`week8_official_external_qwen3base_multik_comparison`, then package:

```bash
DOMAIN=books
EXP=books_large10000_100neg
BASE=llmesr_official_qwen3base_sasrec
OUTDIR="outputs/baselines/official_adapters/${EXP}_llmesr_official"
LOG=$(ls -t outputs/summary/logs/week8_llmesr_official_${DOMAIN}_*.log 2>/dev/null | head -1)
STAMP=$(date +%F_%H%M%S)
mkdir -p outputs/exports

if [ -f "${OUTDIR}/llmesr_official_model.pt" ]; then
  sha256sum "${OUTDIR}/llmesr_official_model.pt" > "${OUTDIR}/llmesr_official_model.pt.sha256"
else
  echo "missing_model ${OUTDIR}/llmesr_official_model.pt" > "${OUTDIR}/llmesr_official_model.pt.sha256"
fi

FILES=(
  "${OUTDIR}/fairness_provenance.json"
  "${OUTDIR}/llmesr_official_run_summary.json"
  "${OUTDIR}/llmesr_official_score_audit.json"
  "${OUTDIR}/llmesr_official_scores.csv"
  "${OUTDIR}/llmesr_official_model.pt.sha256"
  "outputs/${EXP}_${BASE}_same_candidate"
  "outputs/summary/week8_official_external_qwen3base_multik_comparison.csv"
  "outputs/summary/week8_official_external_qwen3base_multik_comparison.md"
)
[ -n "$LOG" ] && [ -f "$LOG" ] && FILES+=("$LOG")

tar -czf "outputs/exports/llmesr_${DOMAIN}_official_qwen3base_evidence_${STAMP}.tar.gz" \
  "${FILES[@]}"
sha256sum "outputs/exports/llmesr_${DOMAIN}_official_qwen3base_evidence_${STAMP}.tar.gz"
ls -lh "outputs/exports/llmesr_${DOMAIN}_official_qwen3base_evidence_${STAMP}.tar.gz"
```

Copy the archive to local storage and verify it before deleting any server
intermediates. Only after confirmation, clean the completed LLM-ESR domain:

```bash
DOMAIN=books
EXP=books_large10000_100neg
BASE=llmesr_official_qwen3base_sasrec
rm -rf "outputs/baselines/official_adapters/${EXP}_llmesr_official"
rm -rf "outputs/baselines/paper_adapters/${EXP}_llmesr_official_adapter"
# Keep outputs/${EXP}_${BASE}_same_candidate/tables/ so comparison builders
# continue to include this completed domain. If space is tight, prune only
# large/non-table files after confirming the archive:
find "outputs/${EXP}_${BASE}_same_candidate" -mindepth 1 -maxdepth 1 ! -name tables -exec rm -rf {} +
df -h /
```

Keep this sequence domain-by-domain. Do not start cleanup for a domain whose
evidence archive has not been confirmed locally, and do not let a slow full
checkpoint archive block the next domain unless the user explicitly asks for
full checkpoint preservation. Beauty must be restored or retained in the same
way as the large domains; do not publish a method-level comparison table where
an official baseline is missing a completed domain because its imported summary
was cleaned or left only in a local archive.

The next official external LLM-rec baselines after LLM2Rec/LLM-ESR/LLMEmb are
RLMRec, IRLLRec, and the replacement expansion baselines ELMRec, ProEx, and
ProMax. RLMRec imports the pinned repo's
`encoder.models.general_cf.simgcl_plus.SimGCL_plus` and preserves the official
BPR, graph contrastive, and semantic alignment losses. IRLLRec imports the
pinned repo's `encoder.models.general_cf.lightgcn_int.LightGCN_int` and
preserves the official BPR, semantic alignment, and intent representation
losses while supplying same-candidate graph data, Qwen3 item embeddings, and
Qwen3-PCA64 intent artifacts. On large domains, IRLLRec's official
`ssl_con_loss` would materialize an all-node N x N matrix; the runner applies a
documented deterministic node cap (`--irllrec_ssl_con_max_nodes`, default
4096) for that term and records the bridge in provenance. ELMRec uses the
pinned `WangXFng/ELMRec` repo and preserves its high-order LightGCN
whole-word interaction bridge plus official Beauty sequential `alpha/sigma/L`
defaults while replacing the T5 text backbone with the unified Qwen3-8B item
representation bridge and exporting exact same-candidate scores. ProEx
(KDD 2026) and ProMax (SIGIR 2026) are selected from the official ProRec
repository as 2026 expansion baselines. ProEx run-stage support is implemented
through the pinned ProRec `LightGCN_proex` model while adapting only
same-candidate graph data, Qwen3-derived profile arrays, and exact score
export. ProMax run-stage support is implemented through the pinned ProRec
`LightGCN_promax` model while preserving BPR, SDR, and S2DR losses; local code
adapts only same-candidate graph data, Qwen3-derived profile arrays,
Qwen3-profile-retrieval `new_trn_rag_mat.pkl`, and exact score export. SETRec
remains `official_blocked_replaced` and any partial/failed SETRec outputs are
not table eligible. Use the same one-domain archive-and-clean loop. Do not
import blocked scaffold rows.

ELMRec Beauty full-domain command:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
DOMAIN=beauty
EXP=beauty_supplementary_smallerN_100neg
mkdir -p outputs/summary/logs
LOG=outputs/summary/logs/week8_elmrec_official_${DOMAIN}_$(date +%F_%H%M%S).log
PID=outputs/summary/logs/week8_elmrec_official_${DOMAIN}.pid
nohup python main_run_elmrec_official_same_candidate_adapter.py \
  --stage run \
  --domain "$DOMAIN" \
  --task_dir "outputs/baselines/external_tasks/${EXP}_test_same_candidate" \
  --valid_task_dir "outputs/baselines/external_tasks/${EXP}_valid_same_candidate" \
  --output_scores_path "outputs/baselines/official_adapters/${EXP}_elmrec_official/elmrec_official_scores.csv" \
  --provenance_output_path "outputs/baselines/official_adapters/${EXP}_elmrec_official/fairness_provenance.json" \
  --fairness_policy_id official_code_qwen3base_default_hparams_declared_adaptation_v1 \
  --comparison_variant official_code_qwen3base_default_hparams_declared_adaptation \
  --backbone_path /home/ajifang/models/Qwen/Qwen3-8B \
  --llm_adaptation_mode representation_learner \
  --hparam_policy official_default_or_recommended \
  --embedding_backend hf_mean_pool \
  --embedding_batch_size 1 \
  --embedding_max_length 64 \
  --embedding_max_text_chars 512 \
  --torch_dtype bfloat16 \
  --hf_device_map auto \
  --elmrec_epochs 100 \
  --elmrec_train_batch_size 64 > "$LOG" 2>&1 &
echo $! > "$PID"
disown
echo "log=$LOG"
echo "pid_file=$PID"
```

After completion, import only if provenance reports
`implementation_status=official_completed`, `blockers=[]`, and
`score_coverage_rate=1.0`.

ProMax follows the same one-domain loop. Start with Beauty or a single large
domain, then verify provenance, import, package, copy locally, and clean before
the next domain:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
DOMAIN=beauty
EXP=beauty_supplementary_smallerN_100neg
mkdir -p outputs/summary/logs
LOG=outputs/summary/logs/week8_promax_official_${DOMAIN}_$(date +%F_%H%M%S).log
PID=outputs/summary/logs/week8_promax_official_${DOMAIN}.pid
nohup python main_run_promax_official_same_candidate_adapter.py \
  --stage run \
  --domain "$DOMAIN" \
  --task_dir "outputs/baselines/external_tasks/${EXP}_test_same_candidate" \
  --valid_task_dir "outputs/baselines/external_tasks/${EXP}_valid_same_candidate" \
  --output_scores_path "outputs/baselines/official_adapters/${EXP}_promax_official/promax_official_scores.csv" \
  --provenance_output_path "outputs/baselines/official_adapters/${EXP}_promax_official/fairness_provenance.json" \
  --fairness_policy_id official_code_qwen3base_default_hparams_declared_adaptation_v1 \
  --comparison_variant official_code_qwen3base_default_hparams_declared_adaptation \
  --backbone_path /home/ajifang/models/Qwen/Qwen3-8B \
  --llm_adaptation_mode representation_learner \
  --hparam_policy official_default_or_recommended \
  --embedding_backend hf_mean_pool \
  --embedding_batch_size 1 \
  --embedding_max_length 64 \
  --embedding_max_text_chars 512 \
  --torch_dtype bfloat16 \
  --hf_device_map auto \
  --promax_epochs 3000 \
  --promax_train_batch_size 4096 \
  --promax_rag_topk 20 > "$LOG" 2>&1 &
echo $! > "$PID"
disown
echo "log=$LOG"
echo "pid_file=$PID"
```

For baseline comparison tables, keep the main reading order at
`NDCG@5`, `NDCG@10`, `HR@5`, `HR@10`, then use `@20` as the extended-check
column when the exporter provides it. The working target for the official
external block is eight baselines, not six: the current six are the floor, and
two additional current-year recommendation baselines from DBLP/GitHub should be
added as separate official-code-level rows when they are ready.

## Output Interpretation

- `*_style_*` rows are paper-style supplementary diagnostics.
- `*_official_qwen3base_*` rows are the target official external-baseline
  family.
- Full fine-tuning and retuned-baseline variants are supplementary/sensitivity
  rows unless a new experiment-wide policy is explicitly declared.
- Beauty is supplementary smaller-N unless the eligible user count reaches the
  main-domain target.
- Week7.7 compact six-candidate results and Week8 101-candidate results must
  not be mixed as direct row-level comparisons.

## C-CRP And SRPD Formal Internal Methods

C-CRP and SRPD are our internal method lines, but they still use the same
candidate-score gate as external baselines. Do not call a prompt-only Shadow
run, a local C-CRP CSV, or an SRPD training artifact paper-facing until it has
exact score coverage and an imported same-candidate summary.

C-CRP production flow:

```text
build pointwise shadow rows
-> run Qwen3-8B shadow_v1 on valid/test
-> calibrate on valid only
-> select C-CRP mode/weights/eta/ablation on valid only
-> export test source_event_id,user_id,item_id,score
-> import with status_label=same_schema_internal_method
```

Generate and run the server command script:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
python main_make_week8_future_framework_commands.py \
  --stage shadow \
  --domains books,electronics,movies \
  --output_path outputs/summary/week8_large10000_100neg_ccrp_shadow_commands.sh

mkdir -p outputs/summary/logs
LOG=outputs/summary/logs/week8_ccrp_formal_$(date +%F_%H%M%S).log
PID=outputs/summary/logs/week8_ccrp_formal.pid
nohup bash outputs/summary/week8_large10000_100neg_ccrp_shadow_commands.sh > "$LOG" 2>&1 &
echo $! > "$PID"
disown
echo "log=$LOG"
echo "pid_file=$PID"
```

Monitor:

```bash
tail -f "$LOG"
ps -p $(cat "$PID") -o pid=,etime=,stat=,cmd=
```

SRPD production flow is stricter because it is trainable. The formal configs
live at:

```text
configs/srpd/{books,electronics,movies}_large10000_100neg_srpd_v6_formal.yaml
configs/lora/{books,electronics,movies}_large10000_100neg_srpd_v6_formal.yaml
```

They are intentionally fail-fast: they require validation-side teacher files,
reject test-derived teacher paths, require leakage audit against final test
events, require `training.use_sample_weights=true`, and default to
`status_label=same_schema_internal_ablation`.

```text
teacher data must come from train/valid-compatible sources
-> leakage audit must pass against final eval events
-> sample weights must be enabled if the variant claims weighting
-> LoRA train/eval
-> export exact candidate scores
-> import as same_schema_internal_ablation unless native candidate scores and
   all main gates are complete
```

If SRPD predictions are already available, export/import them with:

```bash
python main_export_srpd_scores_from_predictions.py \
  --ranking_input_path outputs/baselines/external_tasks/books_large10000_100neg_test_same_candidate/ranking_test.jsonl \
  --candidate_items_path outputs/baselines/external_tasks/books_large10000_100neg_test_same_candidate/candidate_items.csv \
  --prediction_path outputs/books_srpd_formal/predictions/rank_predictions.jsonl \
  --output_scores_path outputs/summary/week8_srpd_formal/books/srpd_scores.csv \
  --provenance_output_path outputs/summary/week8_srpd_formal/books/srpd_internal_provenance.json \
  --method_variant SRPD-formal

python scripts/misc/main_import_same_candidate_baseline_scores.py \
  --baseline_name books_srpd_formal \
  --exp_name books_srpd_formal_same_candidate \
  --domain books \
  --ranking_input_path outputs/baselines/external_tasks/books_large10000_100neg_test_same_candidate/ranking_test.jsonl \
  --scores_path outputs/summary/week8_srpd_formal/books/srpd_scores.csv \
  --method_provenance_path outputs/summary/week8_srpd_formal/books/srpd_internal_provenance.json \
  --status_label same_schema_internal_ablation \
  --artifact_class completed_result
```

For internal C-CRP/SRPD rows whose selector/evaluator used seeded
order-independent tie-breaking, pass the same seed to the importer, e.g.
`--tie_break_seed 20260607`. External official-baseline imports keep the
default candidate-order tie-break unless their own protocol declares a seeded
tie policy. Do not compare selector-side metrics against a default importer run
when tied scores are common.

Generate SRPD formal commands:

```bash
python main_make_srpd_formal_commands.py \
  --domains books,electronics,movies \
  --stage all \
  --output_path outputs/summary/week8_srpd_formal_commands.sh
```

Before GPU training, run startup checks after the validation-side teacher files
exist:

```bash
python main_make_srpd_formal_commands.py \
  --domains books,electronics,movies \
  --stage train \
  --startup_check_only \
  --output_path outputs/summary/week8_srpd_formal_startup_commands.sh
bash outputs/summary/week8_srpd_formal_startup_commands.sh
```

This startup script intentionally runs the SRPD data-build step first, so the
teacher existence checks and leakage audit execute before LoRA startup checks.

The formal teacher files are required inputs, not generated by fallback:

```text
outputs/summary/week8_srpd_formal_teachers/books/valid_teacher_rank_reranked.jsonl
outputs/summary/week8_srpd_formal_teachers/electronics/valid_teacher_rank_reranked.jsonl
outputs/summary/week8_srpd_formal_teachers/movies/valid_teacher_rank_reranked.jsonl
```

If any of those paths are missing, SRPD formal should fail. Do not point SRPD
formal configs at `ranking_test.jsonl` teachers to make it run.

## Legacy Entry Points

These remain in the tree for history and compatibility, but they are not the
preferred first-read files:

- `docs/archive/legacy_root_reports/CODEX_HANDOFF_WEEK8_2026-05-06.md`
- `docs/archive/legacy_root_reports/WEEK8_FUTURE_FRAMEWORK_ROADMAP_2026-05-06.md`
- `docs/archive/legacy_root_reports/WEEK8_LARGE_SCALE_10K_100NEG_PLAN_2026-05-06.md`
- `docs/archive/legacy_root_reports/WEEK8_FUSION_EXTERNAL_ONLY_CONTRIBUTION_UPDATE_2026-05-06.md`
- `docs/archive/legacy_root_reports/WEEK8_OURS_EXTERNAL_COMBO_AND_EXTERNAL_ONLY_PLAN_2026-05-06.md`

Use them only when you need historical detail for that specific stage.

## GitHub Push Convention

After code/config/doc changes that affect server commands or project status:

```bash
python -m scripts.audit.main_project_readiness_check
python -m scripts.audit.main_audit_official_fairness_policy
git status --short
git add <related files only>
git commit -m "<milestone/status message>"
git push origin main
```

Do not push bulk `outputs/`, raw logs, model weights, local datasets, or keys.
Push source, configs, provenance schemas, manifests, and concise docs. If a
server artifact is too large or ignored by git, record its path and regeneration
command in the final answer or runbook.
