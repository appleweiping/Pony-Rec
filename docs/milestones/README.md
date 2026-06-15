# Milestone Map

This directory is the canonical navigation layer for the project. It preserves
the important historical milestones without treating every experiment log as an
equal paper claim.

## Project Spine

```text
M0 Week1-4 / pony12 observation
-> M1 Pony framework
-> M2 Light series boundary test
-> M3 Shadow series task-grounded signals
-> M4 Baseline system and fairness contract
-> M5 Small-domain to four-domain validation
-> M6 Complete recommendation-system roadmap
```

The defended paper story is narrower than the full roadmap:

```text
A task-grounded pointwise LLM relevance posterior is a strong same-candidate
reranking signal under a controlled recommendation protocol.
The calibrated-uncertainty/risk decomposition is a characterized negative
result: it stratifies reliability but does not improve ranking over the
posterior.
```

For the current cumulative execution checklist, see
`docs/active_todo_pony_uncertainty.md`. Keep that file updated after each
completed official row, blocker, evidence-package decision, comparison-table
build, or review cycle.

The larger system story remains staged:

```text
observation -> framework -> reproducible baseline system
-> four-domain validation -> official baselines
-> signal/decision/generative recommendation modules
```

## Evidence Levels

| level | meaning | main-claim eligible |
| --- | --- | --- |
| L0 | observation and diagnostic evidence | context only |
| L1 | small-domain controlled experiments | supporting evidence |
| L2 | Pony12 / Light / Shadow internal framework progression | supporting or ablation evidence |
| L3 | classical same-candidate baselines | main evidence if complete |
| L4 | four-domain same-candidate 100neg validation | main robustness evidence if complete |
| L5 | official-code-level external LLM-rec baselines | main external-baseline evidence only after provenance passes |
| L6 | artifact-ready reproducibility package | submission/artifact evidence |

## Milestone Index

- [M0 Week1-4 / pony12 Observation](M0_week1_4_pony12_observation.md)
- [M1 Pony Framework](M1_pony_framework_week5_6.md)
- [M2 Light Series](M2_light_series.md)
- [M3 Shadow Series](M3_shadow_series.md)
- [M4 Baseline System](M4_baseline_system.md)
- [M5 Four-Domain Same-Candidate Validation](M5_four_domain_same_candidate.md)
- [M6 Complete Recommendation-System Roadmap](M6_complete_recommender_system.md)

## Current Working Position

The repository is currently between M4 and M5:

- the same-candidate baseline system exists;
- the official external-baseline contract is written;
- the paper-style external rows are supplementary, not official reproductions;
- the four-domain 100neg protocol is the active robustness gate;
- the complete recommendation system remains a roadmap until the official
  baselines, Shadow large-scale diagnostics, and LoRA/generative modules are
  completed under the same protocol.

## Current Evidence Integrity (updated 2026-06-15)

Phase 2.5 evidence is ready for strict manuscript-level claim and citation
review, but not final submission. The current paper branch
`paper/reframe-major-revision` uses the reframed 8-domain claim: pointwise
posterior first in 6/8 domains, with uncertainty/risk adjustment reported as a
negative result. The 2026-06-14 package-polish smoke audit
`outputs/summary/paper_critical/submission_package_tikz_smoke_20260614.{json,md}`
confirms `Paper/main.blg` is clean (`warning$ -- 0`) and the framework
overview is accepted as `inline_tikz`, but keeps `final_submission_ready=false`
because the PDF is 15 pages against the 9-page target profile and has 8
overfull hbox warnings. The later 2026-06-14 live ProMax/final-stack refresh
`outputs/summary/paper_critical/submission_release_candidate_stack_refresh_20260614.{json,md}`
keeps `local_release_candidate_ready=false` and
`final_submission_ready=false`: the stale title-mismatch failure was removed by
aligning `configs/paper_submission_metadata.json` with the current paper title,
but ProMax page-range/Crossref/DOI visibility, manual submission-system
confirmation, explicit Claude Opus coverage, and target formatting remain open.
The 2026-06-15 ProMax/final-stack refresh
`outputs/summary/paper_critical/promax_public_metadata_probe_20260615.{json,md}`,
`outputs/summary/paper_critical/external_proceedings_metadata_recheck_20260615.{json,md}`,
and
`outputs/summary/paper_critical/submission_release_candidate_stack_refresh_20260615.{json,md}`
keeps the same state: ProMax Crossref and DOI resolver are still `404`, ACM DL
is still `403`, the ProMax BibTeX still has no final ACM page range, and the
stack still reports `local_release_candidate_ready=false`,
`blocking_status=local_artifact_repair_required`, and
`final_submission_ready=false`. The same-stamp final-blocker consistency audit
now separates consistency from readiness:
`outputs/summary/paper_critical/final_blocker_consistency_audit_20260615.{json,md}`
is `ok=true` and `final_blocker_consistency_ok=true` because the blocked-state
handoff is internally consistent, while `final_submission_ready=false` remains
unchanged until target formatting, ProMax metadata, manual submission, and
Claude Opus coverage all close. The 2026-06-15
private manual-confirmation request refresh
`outputs/summary/paper_critical/manual_submission_private_confirmation_request_packet_20260615.{json,md}`
is public-safe and `ok=true`: it recommends the ignored path
`artifacts/private/manual_submission_private_confirmation_20260615.json`, pins
the current source-manifest hash, forbids private author/COI/reviewer/account
payloads in git, and keeps `manual_submission_system_ready=false` plus
`final_submission_ready=false` until a validated untracked private JSON is
provided. The 2026-06-15 Claude Opus coverage refresh repaired the request-packet
builder to use this current dual claim and stamp-specific 20260615 paths, then
recorded two additional failed direct Opus attempts:
`outputs/summary/paper_critical/claude_opus_review_attempt_fourteenth_20260615.json`
(`The command line is too long`) and
`outputs/summary/paper_critical/claude_opus_review_attempt_fifteenth_20260615.json`
(`Claude CLI did not return JSON output`). The refreshed
`outputs/summary/paper_critical/claude_review_connector_health_20260615.{json,md}`
reports `failed_attempt_count=15`, `valid_review_evidence_count=0`,
`connector_unhealthy=true`, `same_route_retry_recommended=false`, and
`recommended_next_route=external_claude_opus_json_via_request_packet_and_validator`.
This is connector-failure evidence only; explicit Claude Opus coverage and
`final_submission_ready` remain false.
`scripts/audit/main_build_final_paper_claim_audit.py`
generates
`outputs/summary/paper_critical/final_paper_claim_audit_20260612.{json,md,csv}`
with `paper_evidence_ready_for_drafting=true`, `final_submission_ready=false`,
and verdict `READY_FOR_MANUSCRIPT_LEVEL_CLAIM_AND_CITATION_AUDIT`. The stale
pre-submission status can now be regenerated and freshness-checked locally:
`scripts/audit/main_refresh_pre_submission_gates.py` writes the ordered
submission gate stack, and
`scripts/audit/main_audit_pre_submission_refresh_freshness.py` verifies the
recorded input fingerprints and generated gate hashes. The current freshness
artifact
`outputs/summary/paper_critical/pre_submission_gate_refresh_freshness_20260612.{json,md}`
reports `ok=true`, `refresh_artifact_fresh=true`, `21` input fingerprints,
`14` generated gate files, zero mismatches, and
`final_submission_ready=false`. The local release-candidate handoff packet
`outputs/summary/paper_critical/submission_release_candidate_20260612.{json,md}`
now aggregates that freshness audit, the final gate, source package, independent
rebuild, metadata packet, manual checklist, and external metadata audit; it
reports `local_release_candidate_ready=true`,
`readiness_scope=local_artifacts_only`, `blocking_status=external_or_manual_blocked`,
and still `final_submission_ready=false`. The sequential wrapper
`scripts/audit/main_refresh_submission_release_candidate_stack.py` now runs the
refresh, freshness audit, and local release-candidate packet in order and
writes
`outputs/summary/paper_critical/submission_release_candidate_stack_refresh_20260612.{json,md}`;
the current stack artifact reports `ok=true`,
`local_release_candidate_ready=true`, zero failures, and the same
`final_submission_ready=false` external/manual blockers. The 2026-06-13 full
local release-candidate refresh
`outputs/summary/paper_critical/submission_release_candidate_stack_refresh_20260613.{json,md}`
also reports `ok=true`, `local_release_candidate_ready=true`,
`refresh_artifact_fresh=true`, `blocking_status=external_manual_or_review_blocked`,
and `final_submission_ready=false`; this is now the latest stack artifact for
handoff. The new closure packet
`outputs/summary/paper_critical/final_submission_blocker_closure_packet_20260612.{json,md}`
was the compact first-read artifact for the then-two public/manual blocker
classes: external ProMax proceedings metadata and private manual
submission-system confirmation. The current final blocker taxonomy is
three-class: explicit Claude Opus review coverage, ProMax public metadata, and
private manual submission confirmation. It reports `closure_packet_ready=true`,
`ready_for_human_handoff=true`, local artifact handoff `ready`, external
metadata `blocked`, and manual submission `manual_private_pending`, while
preserving `final_submission_ready=false`. The refreshed external metadata audit also
passes a required official SIGIR 2026 accepted-papers source check for ProMax,
in addition to arXiv `2604.26231`, and now reports advisory Crossref
title-discovery candidates. The ProMax metadata audit also requires and passes
an arXiv HTML ACM-metadata source check for DOI `10.1145/3805712.3809600`,
ISBN `979-8-4007-2599-9/2026/07`, the SIGIR venue string, and Melbourne
location; the local BibTeX entry records ISBN `979-8-4007-2599-9` and location
`Melbourne, VIC, Australia`, and the audit JSON/Markdown now exposes those
fields. The discovery layer found `5` ProMax title-search candidates but `0`
matching the expected DOI, so it does not close any final metadata gate; ProMax
final page range and DOI/Crossref visibility remain unresolved. The lightweight
ProMax public metadata probe
`outputs/summary/paper_critical/promax_public_metadata_probe_20260612.{json,md}`
now checks BibTeX pages, Crossref, DOI resolver, ACM DL, arXiv HTML ACM
metadata, the SIGIR accepted-paper source, and the UQ author-profile
announcement source without running the full
submission stack; it reports Crossref `404`, DOI resolver `404`, ACM DL `403`,
source probes passing, and `promax_public_metadata_ready=false`, so it is
monitoring evidence rather than a readiness upgrade. The 2026-06-13 refresh
`outputs/summary/paper_critical/promax_public_metadata_probe_20260613.{json,md}`
continues to show Crossref `404`, DOI resolver `404`, ACM DL `403`, source
probes passing, and `promax_public_metadata_ready=false`. Follow-up live probes
at `2026-06-13T00:49:05Z`, `2026-06-13T01:59:36Z`,
`2026-06-13T02:32:01Z`, `2026-06-13T03:16:11Z`,
`2026-06-13T04:11:37Z`, `2026-06-13T04:30:39Z`,
`2026-06-13T04:49:27Z`, `2026-06-13T05:11:34Z`,
`2026-06-13T05:31:00Z`, `2026-06-13T05:42:19Z`,
`2026-06-13T06:04:37Z`, `2026-06-13T06:52:21Z`, and
`2026-06-13T07:37:42Z` found the same direct
blocker state with five source probes passing: arXiv HTML ACM metadata, SIGIR
accepted papers, UQ author profile, author Google Sites publications, and UQ
Experts profile. The closure packet refreshed after the latest probes and now
lists those source probes plus the explicit review-panel blockers in Markdown;
this strengthens provenance but is not a readiness upgrade.
The public-safe private manual submission confirmation request packet was
refreshed at `2026-06-13T08:10:35Z`, and the closure/consistency handoff was
refreshed at `2026-06-13T08:11Z`; the manual submission-system gate remains
private-pending and `final_submission_ready=false`.
The complete release-candidate stack was also
refreshed at `2026-06-13T04:31:25Z` and reports `ok=true`,
`local_release_candidate_ready=true`, `refresh_artifact_fresh=true`,
`failures=[]`, and `final_submission_ready=false`; its freshness audit checks
`23` input fingerprints and `14` generated gate files with zero mismatches,
and the independent source-package rebuild produces a `9`-page PDF with zero
BibTeX and overfull hbox warnings. The matching
`outputs/summary/paper_critical/final_submission_blocker_closure_packet_20260613.{json,md}`
now uses the 2026-06-13 final gate, external metadata audit, manual checklist,
full release-candidate stack, and public probe while keeping
`final_submission_ready=false`. `scripts/audit/main_build_final_submission_blocker_closure_packet.py`
now infers the input stamp from a dated output path when `--stamp` is omitted,
preventing a 20260613 closure artifact from reading stale 20260612 inputs. A
later hardening makes the same script default to the same-stamp
`promax_public_metadata_probe_<stamp>.json` when available, while
`scripts/audit/main_audit_final_blocker_consistency.py` now fails if the
closure packet omits that probe or records mismatched direct status codes. A
fresh GPT-5.5 xhigh post-handoff review rated
the current manuscript `8.0/10` with `CONDITIONAL_PASS` and no new GPU
experiment requirement, but still forbids a final-submission-ready claim.
Codex fixed the reviewer-caught abstract spacing issue (`\method{} ranks`),
restored the post-section-review claim-audit `.csv` compatibility artifact, and
refreshed
`outputs/summary/paper_critical/final_citation_spot_check_20260613.{json,md}`,
which now reports `21` cited keys, `21` bibliography entries, no missing or
uncited keys, zero BibTeX warnings, and `must_add_count=0`. The new
`outputs/summary/paper_critical/review_continuation_packet_20260613.{json,md}`
reports `ok=true`, `review_continuation_ready=true`, score floor `8.0`,
`local_release_candidate_ready=true`, `ready_for_human_handoff=true`, and
`final_submission_ready=false`; explicit Claude Opus reviewer output remains
missing, so `final_panel_coverage_complete=false`. The attempted Claude Opus
review job failed with `Claude CLI did not return JSON output` and is recorded
at `outputs/summary/paper_critical/claude_opus_review_attempt_20260613.json`;
it does not count as reviewer coverage. A second retry and a third
tool-discovered retry failed with the same error and are recorded at
`outputs/summary/paper_critical/claude_opus_review_attempt_retry_20260613.json`
and
`outputs/summary/paper_critical/claude_opus_review_attempt_third_20260613.json`.
At that point, the review-continuation packet recorded three failed Claude
attempts separately from valid reviewer JSONs. A fourth synchronous no-tools Claude
review call failed with the same error and is recorded at
`outputs/summary/paper_critical/claude_opus_review_attempt_sync_notools_20260613.json`;
the fifth minimal JSON-oriented no-tools Claude call failed at the same
CLI/connector layer and is recorded at
`outputs/summary/paper_critical/claude_opus_review_attempt_minimal_json_20260613.json`.
The sixth synchronous JSON-only Claude call with `model=opus` and tools
disabled failed with the same connector-layer error and is recorded at
`outputs/summary/paper_critical/claude_opus_review_attempt_sixth_20260613.json`.
The seventh synchronous JSON-only Claude call with `model=opus`, tools disabled,
and a shorter structured JSON-only prompt failed with the same connector-layer
error and is recorded at
`outputs/summary/paper_critical/claude_opus_review_attempt_seventh_20260613.json`.
The eighth asynchronous Claude call with `mcp__claude_review.review_start` plus
`review_status`, `model=opus`, tools disabled, and a short JSON-only prompt
failed with the same connector-layer error and is recorded at
`outputs/summary/paper_critical/claude_opus_review_attempt_eighth_20260613.json`.
The ninth asynchronous Claude call through `mcp__claude_review.review_start`
also failed with `Claude CLI did not return JSON output`; job
`a3863723466147e9b9b849cf994ca8fd` is recorded at
`outputs/summary/paper_critical/claude_opus_review_attempt_ninth_20260613.json`.
At that point, the refreshed packet recorded nine failed attempts and still kept
`explicit_claude_opus_present=false`. Codex also added
`scripts/audit/main_build_claude_review_request_packet.py`, generating
`outputs/summary/paper_critical/claude_opus_review_request_packet_20260613.{json,md}`
as a public-safe prompt/schema handoff for a future valid Claude Opus review.
Codex then retried a tenth asynchronous Claude Opus call through
`mcp__claude_review.review_start`; job
`b6e19654680c457d8be4845e168ce251` again failed with
`Claude CLI did not return JSON output` and is recorded at
`outputs/summary/paper_critical/claude_opus_review_attempt_tenth_20260613.json`.
After the tenth failed connector attempt, the request packet was refreshed at
`2026-06-13T04:15:35Z` and then recorded ten failed Claude attempts. The
request packet is not reviewer coverage. The review-continuation gate now
validates additional reviewer JSONs before counting them for panel coverage:
Claude/Opus JSON must explicitly identify a Claude Opus reviewer and include
`valid_review_evidence=true`,
`claim_boundary_ok=true`, `final_submission_ready_claim_allowed=false`,
substantive kill-argument/concerns/required changes, and acknowledged
remaining blockers; a name+score shell cannot close the explicit Claude gap.
Codex later tightened this path so explicit Claude Opus coverage requires a
reviewer identity containing both `claude` and `opus`, and added
`scripts/audit/main_validate_claude_opus_review_json.py` as a read-only
preflight for future external Claude JSONs before they are attached to the
review-continuation packet.
Codex then retried an eleventh asynchronous Claude Opus call through
`mcp__claude_review.review_start`; job
`bf4b6b8145404ffa881cd99ed3c73429` again failed with
`Claude CLI did not return JSON output` and is recorded at
`outputs/summary/paper_critical/claude_opus_review_attempt_eleventh_20260613.json`.
That eleventh-attempt refresh still kept
`explicit_claude_opus_present=false`,
`final_panel_coverage_complete=false`, and `final_submission_ready=false`.
Codex also added
`scripts/audit/main_audit_claude_review_connector_health.py`, which reports
the then-same connector error repeated across all `11` failed attempts,
`connector_unhealthy=true`, `same_route_retry_recommended=false`, and the
recommended route `external_claude_opus_json_via_request_packet_and_validator`.
This health audit is advisory and does not close review coverage.
Codex later tested the changed direct `mcp__claude_review.review` tool surface:
the twelfth attempt exposed a prompt shell-escaping failure caused by pipe
characters in the verdict enum, and the thirteenth shell-safe retry returned to
`Claude CLI did not return JSON output`. The current request, health,
review-continuation, final-gate, release-stack, closure, and consistency
artifacts now record `13` failed Claude attempts with no valid Claude Opus
review evidence.
Codex then hardened that external JSON intake route: the refreshed
`claude_opus_review_request_packet_20260613.{json,md}` includes a fillable
response template with `valid_review_evidence=false` by default, and both the
review-continuation packet builder and standalone Claude JSON validator now
require a current Claude Opus JSON to acknowledge the open ProMax public
metadata blocker and the private manual submission-system blocker before it can
count as explicit Claude Opus coverage. The refreshed
`review_continuation_packet_20260613.json` exposes
`required_claude_blocker_ack_groups=["manual_submission_system",
"promax_public_metadata"]` while keeping `final_submission_ready=false`.
The final-blocker consistency audit is now schema
`2026-06-13.final_blocker_consistency_audit.v3` and checks that these Claude
intake safeguards plus the private manual-confirmation validator path remain
present before any final-readiness report can pass the local consistency layer.
The final submission gate has also been hardened to consume the
review-continuation packet directly: the refreshed
`outputs/summary/paper_critical/final_submission_gate_20260613.{json,md}`
reports `review_panel_coverage_complete=false`, verdict
`LOCAL_PACKAGE_READY_BUT_EXTERNAL_MANUAL_OR_REVIEW_BLOCKED`, and
`final_submission_ready=false`; the blocker closure packet now includes a
separate `review_panel_coverage` group for `explicit_claude_opus_review`.
After a GPT-5.5 xhigh sidecar audit, Codex also fixed the freshness edge:
`scripts/audit/main_refresh_pre_submission_gates.py` now fingerprints the
review-continuation packet and its builder because the final gate reads that
packet. The refreshed freshness artifact reports `refresh_artifact_fresh=true`.
`scripts/audit/main_build_review_continuation_packet.py` now accepts future
closed ProMax/closure/release-candidate states instead of requiring the current
blocked ProMax state as the only valid handoff shape.
Codex also fixed recursive warning-prefix growth across the release-stack
handoff artifacts by normalizing known aggregator prefixes before adding a new
layer prefix in review-continuation, final-gate, pre-submission-refresh,
release-candidate, stack, and closure packet builders. The refreshed artifacts
now have compact warning lists and the stack still reports
`blocking_status=external_manual_or_review_blocked` and
`final_submission_ready=false`.
Codex then added
`scripts/audit/main_audit_final_blocker_consistency.py`,
`tests/test_audit_final_blocker_consistency.py`, and
`outputs/summary/paper_critical/final_blocker_consistency_audit_20260613.{json,md}`
to audit cross-packet consistency after blocker refreshes. The current audit
reports `ok=true`, `final_blocker_consistency_ok=true`, failed Claude attempts
`13`, `explicit_claude_opus_present=false`,
`promax_public_metadata_ready=false`, manual confirmation still needed,
recursive warning regressions `0`, and `final_submission_ready=false`. This is
a handoff consistency guard, not a final-readiness upgrade.
Codex then added
`scripts/audit/main_audit_final_blocker_doc_status.py`,
`tests/test_audit_final_blocker_doc_status.py`, and
`outputs/summary/paper_critical/final_blocker_doc_status_audit_20260613.{json,md}`
to audit the canonical current-status docs against the consistency audit. It
reports `ok=true`, `final_blocker_doc_status_ok=true`, failed Claude attempts
`13`, `explicit_claude_opus_present=false`,
`promax_public_metadata_ready=false`, private manual confirmation still needed,
recursive warning regressions `0`, and `final_submission_ready=false`, while
rejecting current/handoff wording that still presents eight-attempt or
two-blocker states as live truth.
The submission package
audit now includes a privacy-preserving anonymous source leak scan over the TeX closure and the
current package has zero email, ORCID, acknowledgment, local-path,
non-anonymous-author, or non-anonymous-affiliation hits. The manual submission
package can also be staged locally from the audited source manifest via
`scripts/audit/main_build_submission_source_package.py`; the current staging
manifest reports 21 copied files and a copied manifest sha256 matching the
source audit manifest under ignored `artifacts/submission_source_package_20260612/`.
`scripts/audit/main_audit_submission_source_package_rebuild.py` now verifies
that staged package in an independent ignored worktree; the current rebuild
audit reports four successful LaTeX/BibTeX commands, a 9-page PDF, zero BibTeX
warnings, and zero overfull hbox warnings. This is an artifact handoff and
local rebuildability aid, not final submission readiness. The manual submission
gate now has a
privacy-preserving closure path:
`configs/paper_manual_submission_private_confirmation.template.json` plus the
optional `--private-confirmation-json` argument to
`scripts/audit/main_build_manual_submission_checklist.py`; a future untracked
confirmation file can prove submission-system items are done by hash and item
IDs without storing author/COI/reviewer/account payloads in git. The current
committed state has no private confirmation and remains
`manual_submission_system_ready=false`. The recorded Git HEAD is generation
provenance, not a strict post-commit freshness condition. Codex then added
`scripts/audit/main_build_manual_submission_private_confirmation_request_packet.py`
and
`outputs/summary/paper_critical/manual_submission_private_confirmation_request_packet_20260613.{json,md}`
as a public-safe request packet for that private manual step. It reports
`ok=true`, `request_packet_ready=true`, `manual_confirmation_needed=true`,
`manual_submission_system_ready=false`, and `final_submission_ready=false`;
records the current source manifest sha256
`91d1d6495fe3fa85608d7711fb5873730d907237242b3b3fa489c6f1ed516424`,
the safe confirmation skeleton, recommended ignored path
`artifacts/private/manual_submission_private_confirmation_20260613.json`,
forbidden private fields/keys, and follow-up commands that validate the
private confirmation JSON before the manual checklist consumes it; and is
linked from the
refreshed
`outputs/summary/paper_critical/final_submission_blocker_closure_packet_20260613.{json,md}`.
This request packet is a handoff artifact only and does not close manual,
ProMax, or Claude review blockers. Codex later added
`scripts/audit/main_validate_manual_submission_private_confirmation_json.py`
and
`tests/test_validate_manual_submission_private_confirmation_json.py` as the
local read-only preflight for any future untracked private confirmation JSON.
The refreshed request packet and closure packet now place this validator before
the manual checklist command, and the v3 consistency audit records
`manual_request_has_private_confirmation_validator=true` and
`closure_manual_group_has_private_confirmation_validator=true`. The validator
rejects forbidden private keys, source-manifest/profile/checklist mismatches,
in-repo paths outside ignored `artifacts/private/`, unknown/duplicate/missing
item IDs, and early completion of currently blocked IDs such as
`confirm_external_proceedings_metadata`.
The stale
`paper/` draft was rewritten to the current C-CRP same-candidate
official-baseline spine, stale calibration table removed, current main/module
tables added, and `scripts/analysis/main_build_paper_result_tables.py`
generates visible full-baseline and paired-test summary tables from the local
36-row evidence ledger. `paper/references.bib` has been repaired to 19 used
references with no `Anonymous` placeholders, `Paper/main.blg` reports
`warning$ -- 0`,
`outputs/summary/paper_critical/citation_audit_repair_20260612.{json,md}`
reports `must_add_count=0` and recency verdict `Good`, and the first structural
expansion pass now produces `Paper/main.pdf` (8 pages, 533021 bytes). The
expansion added detailed method notation, protocol/fairness text, an explicit
uncertainty-stratification table, numeric diagnostic summaries, and updated
limitations; the audit lives at
`outputs/summary/paper_critical/manuscript_structural_expansion_audit_20260612.{json,md}`.
The remaining blockers are no longer missing citations or the initial
compressed draft; the expanded-manuscript claim-text audit at
`outputs/summary/paper_critical/manuscript_claim_audit_after_structural_expansion_20260612.{json,md}`
reports `READY_WITH_SCOPE_GUARDS` with no unsupported, overclaimed, or
contradicted claims. Final citation spot-check is now complete at
`outputs/summary/paper_critical/final_citation_spot_check_20260612.{json,md}`
with `must_add_count=0`, all 19 cited keys resolved, all eight official
baselines cited, and `Paper/main.blg` still at `warning$ -- 0`. A GPT-5.5 xhigh
section-level review returned `8.0/10` conditional pass and requested more
visible provenance/rank/diagnostic detail and risk-penalty wording discipline;
the applied-fix audit lives at
`outputs/summary/paper_critical/section_level_review_20260612.{json,md}`. The
latest draft adds visible official-baseline provenance, all-metric rank-first,
and ablation summary tables and compiles to `Paper/main.pdf` (9 pages, 541654
bytes). The evidence-to-claim gate was rerun at
`outputs/summary/paper_critical/final_paper_claim_audit_post_section_review_20260612.{json,md,csv}`
and remains `ok=true`, `paper_evidence_ready_for_drafting=true`,
`final_submission_ready=false`. A fresh ARIS claim-text pass at
`outputs/summary/paper_critical/manuscript_claim_audit_after_section_review_20260612.{json,md}`
reports `READY_WITH_SCOPE_GUARDS` with 12 supported manuscript claims and no
unsupported, overclaimed, or contradicted manuscript claims. Remaining blockers
are paper-critical tests/readiness checks, another section-level review on this
latest draft, missing Claude Opus reviewer perspective for the final panel, and
pre-submission ProEx/ProMax proceedings metadata recheck.

Historical execution detail follows.

Phase 2.5 is in active execution, not writing-ready closure. As of
2026-06-11, Tools test signal-row generation completed, passed source/package
sync audits, and the server has no matching active Pony/C-CRP/baseline Python
process. Home component-ablation packaging has advanced from provisional to
package-audited:
`outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/ccrp_ablation_home/phase2_5_component_ablation_package_audit.{json,md}`
reports `ok=true`, `paper_claim_ready=true`, and no failures. The accepted Home
package uses the preregistered main C-CRP config (`eta=1.0`) and
tie-break-aware same-candidate import (`--tie_break_seed 20260607`), avoiding
the earlier mismatch where component summary rows froze the validation
sensitivity row (`eta=0.5`). The scientific interpretation remains cautious:
Home, Sports, Toys, and Tools all show near-inert or redundant uncertainty/risk
components in leave-one-component-out diagnostics, so the component story must
be framed honestly as weak/redundant component evidence rather than a strong
component-necessity claim. Tools package evidence lives at
`outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/ccrp_ablation_tools/`
and `phase2_5_component_ablation_package_audit.{json,md}` reports `ok=true`,
`paper_claim_ready=true`, and `failures=[]`.
Post-module GPT-5.5 xhigh sidecar review rated Tools component-ablation
**CONDITIONAL PASS, 7.5/10**, with supplementary/diagnostic table eligibility
only; Claude Opus reviewer tooling was unavailable in this session.
Sports and Toys were then backfilled from existing full-scale server signal
rows without LLM re-query. Their component package audits now report `ok=true`,
`paper_claim_ready=true`, and `failures=[]`, and the four-domain component
aggregation at
`outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/ccrp_component_ablation_four_domain/`
reports `ok=true`, `paper_claim_ready=true`, `delta_convention=removal_minus_full`,
`tie_epsilon=1e-12`, and `table_eligibility=supplementary_diagnostic_only`.
On NDCG@10, removing boundary uncertainty is exactly inert across all four
domains, removing counterevidence or risk penalty is nonworse/better in all
four, calibration gap is mixed, and evidence support is only directionally
supportive with a small mean delta. A GPT-5.5 xhigh post-module review rated
the completed component module **CONDITIONAL PASS, 8.1/10**. This closes the
Phase 2.5 component-ablation module as supplementary diagnostic evidence, not
as a component-necessity claim and not as main-table SOTA evidence.
The Phase 2.5 observation/motivation module is also closed as descriptive
motivation evidence. Per-domain packages at
`outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/observation_{sports,toys,home,tools}/`
pass the hardened package audit with `ok=true`, `paper_claim_ready=true`, and
`failures=[]`. The audit independently verifies `1,010,000` finite uncertainty
rows per domain, exactly `101` uncertainty rows/event, zero invalid uncertainty
rows, exact event joins, package-local `same_candidate_alignment.json`, same
candidate key count `1,010,000`, score coverage `1.0`, zero missing/extra/
duplicate/invalid score keys, and local/server hash evidence. The four-domain
aggregate at
`outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/observation_four_domain/`
reports `ok=true`, `paper_claim_ready=true`,
`claim_status=uncertainty_stratifies_reliability`, and
`table_eligibility=motivation_only_not_main_table_sota`; high-uncertainty
C-CRP bins degrade versus low-uncertainty bins in all four domains on
`NDCG@10`, `MRR`, and `HR@10`. GPT-5.5 xhigh review rated this module
**PASS, 8.4/10**; a separate engineering audit veto was lifted after gate
hardening with **8.6/10**. This supports only motivation wording, not causal,
statistically significant, exhaustive-baseline, or main-table SOTA claims.
On 2026-06-11 the selector/import command surface was also repaired and synced
to the server: C-CRP selector imports now call
`scripts/misc/main_import_same_candidate_baseline_scores.py` and pass
`--tie_break_seed`, preventing a missing-root-script failure when Tools reaches
the selector stage.

Regression evidence for this gate:
`python -m pytest tests\test_uncertainty_observation_study.py tests\test_aggregate_uncertainty_observation_study.py tests\test_audit_phase2_5_module_package.py tests\test_audit_paper_critical_modules.py tests\test_plan_ccrp_signal_generation.py -q`
passed with `58 passed`; `python -m scripts.audit.main_project_readiness_check`
and `python scripts\audit\main_project_bootstrap.py` report
`project_readiness_ok=True`.

## Current Evidence Integrity (updated 2026-06-06)

Sports, Toys, Home, and Tools now have all eight new-domain official-code-level
baseline rows plus C-CRP domain/comparison/paired-test gates complete under the
10k-user/101-candidate same-candidate protocol. The local/server evidence
consistency backfill
`outputs/summary/paper_critical/local_server_evidence_consistency_new_domains_post_backfill_20260606.{json,md,sha256}`
passes for all `32/32` official rows. This resolves the local packaging gap for
older Sports/Toys/Home rows whose small copied server-large manifests were
missing; it does not change the Phase 2.5 blocker that full-scale uncertainty
signal rows are still needed for the observation, component ablation, and
hyperparameter-curve modules.

The consolidated 2026-06-06 01:55 CST paper-critical go/no-go artifact
`outputs/summary/paper_critical/paper_critical_module_audit_post_evidence_backfill_20260606_0155.{json,md,sha256}`
records a historical boundary: evidence consistency, framework overview,
component inventory, and guarded planning were ready, while paper readiness
remained false because full-scale uncertainty signal rows were absent and the
Phase 2.5 storage gate was still closed at that time. The storage condition was
later superseded by the 2026-06-06 06:23 CST completed-row model-checkpoint
cleanup below; the full-scale signal-row blocker remains.

The 2026-06-06 02:00 CST retention decision packet
`outputs/summary/paper_critical/retention_cleanup_plan_20260606_current/tools_llm2rec_upstream_embedding_current_retention_decision_plan_20260606_0200.{json,sh,md,sha256}`
is the current non-destructive approval surface for clearing the storage gate:
it cites the fresh storage audit, exits before `sha256sum` or `rm --`, and
requires explicit archive/retention approval before any cleanup can occur.
The follow-up packet audit
`outputs/summary/paper_critical/retention_cleanup_plan_20260606_current/tools_llm2rec_upstream_embedding_current_retention_decision_packet_audit_20260606_0205.{json,md,sha256}`
passes with `ok=true` and no failures, but it is still not deletion approval.
The live read-only pre-approval audit
`outputs/summary/paper_critical/retention_cleanup_plan_20260606_current/tools_llm2rec_upstream_embedding_preapproval_audit_20260606_0212.{json,md,sha256}`
then verifies the current server target size/hash and completed-row evidence;
its only failure is the expected `disk_below_min_free_before_cleanup`.
The follow-up guarded action dry-run
`outputs/summary/paper_critical/retention_cleanup_plan_20260606_current/tools_llm2rec_upstream_embedding_cleanup_action_dry_run_20260606_0225.{json,md,sha256}`
validates the plan, packet audit, and a fresh 02:25 live preapproval audit, but
reports `will_delete=false` and
`execution_status=dry_run_no_remote_commands`. No cleanup or experiment launch
occurred.

At 2026-06-06 06:23-06:25 CST, after explicit user direction to clear server
storage pressure, Codex performed a bounded cleanup under
`~/projects/pony-rec-rescue-shadow-v6` only. Two completed Home official-row
model checkpoints were deleted after verifying server-final audits,
large-artifact manifests, provenance, scores, score audits, run summaries, and
imported tables remained present:
`outputs/home_large10000_100neg_llmemb_official_qwen3base_same_candidate/llmemb_official_model.pt`
and
`outputs/home_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/llmesr_official_model.pt`.
Corrected cleanup record:
`outputs/summary/server_cleanup/cleanup_large_completed_model_checkpoints_20260605T222339Z.corrected.{summary.txt,tsv}`.
The cleanup freed `13,259,007,313` bytes and moved `/` to
`25,656,266,752` bytes free / `88%` used. No data splits, final scores,
provenance, audits, imported tables, source code, configs, active outputs, or
other-project files were deleted. Future server work still requires a fresh
process/GPU/disk preflight.

The 2026-06-06 06:36-06:38 CST post-cleanup launch audit separates the two
remaining gates. Storage/process is now open:
`outputs/summary/paper_critical/server_storage_phase2_5_retention_audit_after_cleanup_final_20260606_0650.{json,md}`
reports zero active project Python processes, idle GPU, `/` at
`25,656,160,256` free bytes / `88%` used, and
`experiment_launch_allowed=true`. The scientific signal-row gate remains
closed: fresh discovery
`outputs/summary/paper_critical/ccrp_signal_source_discovery_after_cleanup_20260606_0640.{json,csv}`
and per-domain audits
`outputs/summary/paper_critical/ccrp_signal_source_audit_{sports,toys,home,tools}_after_cleanup_20260606_0645.{json,csv}`
still find only the four formal C-CRP `scores.csv` files, all classified as
`score_only_not_uncertainty` with candidate-key coverage `1.0` and
`missing_uncertainty_column`. No Phase 2.5 experiment was launched.

The 2026-06-06 07:20 CST signal-runner wiring checkpoint confirms the
performance/table gate is already closed for the current same-candidate claim:
server reports show C-CRP v3 `report.json` present for all eight domains, and
Sports/Toys/Home/Tools each have `8/8` official completed baselines with empty
blockers. The local paper-facing ledger has `official_row_count=32`,
`ccrp_row_count=4`, and per-domain `8 official + 1 C-CRP` rows. Codex added
`experiments/rsc/run_ccrp_v3_signal_rows.py` and refreshed the guarded signal
plan so a future server run can generate auditable valid/test C-CRP signal
rows before selector, observation, component-ablation, and hyperparameter
steps. Consolidated audit
`outputs/summary/paper_critical/paper_critical_module_audit_post_signal_runner_plan_20260606_0720.{json,md}`
reports `ok=true`, `four_domain_evidence_consistent=true`,
`phase2_5_storage_launch_allowed=true`, and `guarded_plan_ready=true`, while
keeping `paper_ready=false` because no completed full-scale signal rows have
passed audit. No Phase 2.5 experiment was launched.

The 2026-06-06 07:19 CST Sports-valid signal-row launch is the first bounded
Phase 2.5 execution after the signal-runner checkpoint. Before launch, Codex
hardened `experiments/rsc/run_ccrp_v3_signal_rows.py` with a strict
generation-count guard and pushed commit `70f2f0d`. A first attempt with base
`/home/ajifang/miniconda3/bin/python` failed before writing rows because vLLM
could not import `libcudart.so.13`. The corrected run uses
`/home/ajifang/miniconda3/envs/qwen_vllm/bin/python`, PID `3543564`, log
`ccrp_signal_rows_sports_valid_20260606_071906.log`, pidfile
`outputs/summary/paper_critical/ccrp_signal_rows_sports_valid_20260606_071906.pid`,
and output dir
`outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/ccrp_signal_rows_sports`.
Local launch manifest:
`outputs/summary/paper_critical/ccrp_signal_rows_sports_valid_launch_20260606_071906.{json,md}`.
At launch audit the process was unique, GPU was active, disk remained safe,
and the fatal log scan was clean. This is not yet paper evidence: after
completion, the valid signal rows must pass candidate-key coverage/provenance
audit against Sports valid `candidate_items.csv` before any selector,
observation, ablation, hyperparameter, or test use.

At the 2026-06-06 07:40 CST monitor, the Sports valid signal-row job was still
active and unique under PID `3543564`; GPU was active, root disk was safe at
`25,982,668,800` bytes free / `87%` used, fatal scan was clean, and the first
chunk had reached `25149/505000` processed prompts with no final output files
yet. During this wait, Codex added
`scripts/audit/main_sync_ccrp_signal_evidence_package.py` plus
`tests/test_sync_ccrp_signal_evidence_package.py` to make the post-completion
local package path explicit. The helper copies Phase 2.5 signal evidence from
pony-rec-gpu, verifies server/local hashes, and audits signal row counts,
provenance, parse failures, source-audit candidate-key coverage, and local
hash evidence before selector/observation/ablation/hyperparameter consumption.
Focused verification passed with `40 passed`; no experiment was stopped or
launched, and the running valid-split row is still not paper-ready evidence.

At the 2026-06-06 07:52 CST monitor, PID `3543564` was still active and unique,
GPU was `93%`, root disk remained safe at `25,982,087,168` bytes free / `87%`
used, fatal scan was clean, and the first chunk had reached `41214/505000`
processed prompts. Codex then hardened the guarded signal-generation plan
surface: `scripts/audit/main_plan_ccrp_signal_generation.py` now records
generated valid/test signal paths, split-specific source-audit commands, and
local post-completion `main_sync_ccrp_signal_evidence_package.py` sync/package
audit templates. The regenerated tracked plan JSON/shell artifacts remain
non-executing (`exit 2` before any command) and the shell does not include local
sync commands. Focused verification passed with `51 passed`; no new experiment
was launched.

At the 2026-06-06 08:37 CST monitor, PID `3543564` was still active and unique.
GPU was `100%` with `42863 MiB / 49140 MiB`, root disk remained safe at
`25,979,711,488` bytes free / `87%` used, fatal scan was clean, and the first
chunk had reached `103908/505000` processed prompts. The signal output
directory still had zero files and no `valid_ccrp_signal_rows.csv`,
`valid_ccrp_signal_rows_provenance.json`, or source-audit artifact, so no
completion gate or local package sync was run.

At the 2026-06-06 10:04 CST monitor, PID `3543564` remained active and unique,
elapsed `02:45:07`, with duplicate signal-runner process count `1`. GPU was
`100%` with `42863 MiB / 49140 MiB`, root disk remained safe at
`25,975,029,760` bytes free / `87%` used, fatal scan stayed clean, and progress
had reached `222693/505000` prompts (`44%`). The output directory still had zero
files and no `valid_ccrp_signal_rows.csv`, provenance, or source-audit artifact,
so no completion gate, source audit, cleanup, or local package sync was run.

At the 2026-06-06 10:24 CST monitor, PID `3543564` remained active and unique,
elapsed `03:05:36`, with duplicate signal-runner process count `1`. GPU was
`100%` with `42863 MiB / 49140 MiB`, root disk remained safe at
`25,973,907,456` bytes free / `87%` used, fatal scan stayed clean, and progress
had reached `250729/505000` prompts (`50%`). The output directory still had zero
files and no `valid_ccrp_signal_rows.csv`, provenance, or source-audit artifact,
so no completion gate, source audit, cleanup, or local package sync was run.

At the 2026-06-06 10:57 CST monitor, PID `3543564` remained active and unique,
elapsed `03:38:49`, with duplicate signal-runner process count `1`. GPU was
`100%` with `42863 MiB / 49140 MiB`, root disk remained safe at
`25,972,076,544` bytes free / `87%` used, fatal scan stayed clean, and progress
had reached `296362/505000` prompts (`59%`). The output directory still had zero
files and no `valid_ccrp_signal_rows.csv`, provenance, or source-audit artifact,
so no completion gate, source audit, cleanup, or local package sync was run.

At the 2026-06-06 11:38 CST monitor, PID `3543564` remained active and unique,
elapsed `04:19:20`, with duplicate signal-runner process count `1`. GPU was
`100%` with `42863 MiB / 49140 MiB`, root disk remained safe at
`25,969,831,936` bytes free / `87%` used, fatal scan stayed clean, and progress
had reached `351871/505000` prompts (`70%`). The output directory still had zero
files and no `valid_ccrp_signal_rows.csv`, provenance, or source-audit artifact,
so no completion gate, source audit, cleanup, or local package sync was run.

At the 2026-06-06 16:11 CST monitor, PID `3543564` remained active and unique,
elapsed `08:52:39`, with duplicate signal-runner process count `1`. GPU was
`93%` with `42863 MiB / 49140 MiB`, root disk remained safe at
`25,946,750,976` bytes free / `87%` used, fatal scan stayed clean, and the
second chunk had reached `221282/505000` prompts after the first chunk had
completed, about `71.9%` overall for the `1,010,000` expected Sports-valid
signal rows. The output directory still had zero files and no
`valid_ccrp_signal_rows.csv`, provenance, or source-audit artifact, so no
completion gate, source audit, cleanup, or local package sync was run. Focused
sync-helper verification passed with `python -m pytest
tests\test_sync_ccrp_signal_evidence_package.py
tests\test_ccrp_uncertainty_source_audit.py -q` (`8 passed`); the helper keeps
the full signal CSV allowed by default while enforcing row-count, sha256,
provenance, parse-failure, and candidate-key coverage checks after completion.

At the 2026-06-06 16:16 CST monitor, PID `3543564` remained active and unique,
elapsed `08:57:33`, with duplicate signal-runner process count `1`. GPU was
`92%` with `42863 MiB / 49140 MiB`, root disk remained safe at
`25,946,492,928` bytes free / `87%` used, fatal scan stayed clean, and the
second chunk had reached `227991/505000` prompts after the first chunk had
completed, about `72.6%` overall for the `1,010,000` expected Sports-valid
signal rows. The output directory still had zero files and no
`valid_ccrp_signal_rows.csv`, provenance, or source-audit artifact, so no
completion gate, source audit, cleanup, local package sync, or new experiment
was run.

At the 2026-06-06 16:20 CST monitor, PID `3543564` remained active and unique,
elapsed `09:01:10`, with duplicate signal-runner process count `1`. GPU was
`93%` with `42863 MiB / 49140 MiB`, root disk remained safe at
`25,946,288,128` bytes free / `87%` used, fatal scan stayed clean, and the
second chunk had reached `232950/505000` prompts after the first chunk had
completed, about `73.1%` overall. The output directory still had zero files and
no signal CSV, provenance, or source-audit artifact, so no completion gate,
source audit, cleanup, local package sync, or new experiment was run.

At the 2026-06-06 16:23 CST monitor, PID `3543564` remained active and unique,
elapsed `09:04:09`, with duplicate signal-runner process count `1`. GPU was
`91%` with `42863 MiB / 49140 MiB`, root disk remained safe at
`25,946,136,576` bytes free / `87%` used, fatal scan stayed clean, and the
second chunk had reached `236988/505000` prompts after the first chunk had
completed, about `73.5%` overall. The output directory still had zero files and
no signal CSV, provenance, or source-audit artifact, so no completion gate,
source audit, cleanup, local package sync, or new experiment was run.

At the 2026-06-06 16:26 CST monitor, PID `3543564` remained active and unique,
elapsed `09:07:10`, with duplicate signal-runner process count `1`. GPU was
`92%` with `42863 MiB / 49140 MiB`, root disk remained safe at
`25,945,972,736` bytes free / `87%` used, fatal scan stayed clean, and the
second chunk had reached `241182/505000` prompts after the first chunk had
completed, about `73.9%` overall. The output directory still had zero files and
no signal CSV, provenance, or source-audit artifact, so no completion gate,
source audit, cleanup, local package sync, or new experiment was run.

The 2026-06-06 02:35-02:40 CST paper-facing comparison ledger
`outputs/summary/paper_critical/new_domains_paper_facing_full_metric_evidence_ledger_20260606_0240.{csv,json,md,sha256}`
joins the compact comparison method rows, domain gate summaries, paired-test
artifacts, local/server evidence consistency audit, and official provenance
paths into a 36-row ledger. It reports `ok=true`,
`comparison_ledger_ready=true`, `official_row_count=32`, and `ccrp_row_count=4`.
This closes the paper-table evidence-ledger gap for the four-domain official
comparison, while paper readiness remains blocked by missing full-scale
uncertainty signal rows. The storage-floor violation present at ledger-build
time was later cleared by the 2026-06-06 06:23 CST completed-row
model-checkpoint cleanup.

The 2026-06-06 02:55 CST C-CRP local evidence backfill copied the missing
Sports/Toys `user_ranks.jsonl` files and missing Sports imported C-CRP tables
from `pony-rec-gpu` without running experiments or modifying server files. The
post-backfill certificate audit and ledger
`outputs/summary/paper_critical/*post_ccrp_backfill_20260606_0255*` both report
`ok=true`, zero failures, and zero warnings. The comparison evidence ledger is
now locally self-contained for the four-domain C-CRP event/ranking records, but
paper readiness remains blocked by missing full-scale uncertainty signal rows.
The storage-floor violation present at this checkpoint was later cleared by
the 2026-06-06 06:23 CST completed-row model-checkpoint cleanup.

The 2026-06-06 03:00 CST retention-decision checkpoint refreshed the live
server storage audit and module audit without launching work. Server state was
idle with `/` at `12,406,411,264` free bytes / `94%` used, so signal-row
generation remains blocked by the `15GiB` floor. Codex fixed the retention
planner to emit the packet `.sha256` manifest expected by the packet auditor
and regenerated
`outputs/summary/paper_critical/retention_cleanup_plan_20260606_current_0300/`.
The refreshed packet audit passes with `ok=true`; live preapproval is ready
except for `disk_below_min_free_before_cleanup`; and the action renderer is
dry-run only with `will_delete=false` and
`execution_status=dry_run_no_remote_commands`. No cleanup, deletion,
post-delete gate, baseline launch, or Phase 2.5 experiment launch occurred.
Checkpoint manifest:
`outputs/summary/paper_critical/phase2_5_retention_decision_checkpoint_20260606_0300.sha256`.

The 2026-06-06 04:47 CST retention-decision refresh regenerated the same
approval-required packet against the latest 04:32 storage audit. Fresh artifacts
are in
`outputs/summary/paper_critical/retention_cleanup_plan_20260606_current_0447/`,
with checkpoint manifest
`outputs/summary/paper_critical/phase2_5_retention_decision_checkpoint_20260606_0447.sha256`.
The packet audit passes with no failures. Live preapproval remains ready except
for `disk_below_min_free_before_cleanup`: no active project Python process,
target size and sha256 match provenance, completed Tools LLM2Rec evidence is
present, and disk remains below the 15GiB launch floor at `12397662208` free
bytes / `94%` used. The action audit is dry-run only
(`will_delete=false`, `execution_status=dry_run_no_remote_commands`). No
cleanup, deletion, post-delete gate, baseline launch, or Phase 2.5 experiment
launch occurred.

The 2026-06-06 05:05 CST framework overview refresh tightened the figure after
GPT-5.5 xhigh sidecar review by replacing the potentially overclaiming
`Paper-critical method evidence` label with `Required method-evidence gates`.
It regenerated the figure package at generator commit
`e81d1a8404b2a90467e1912c53e082e1e3c46dae`:
`outputs/summary/paper_critical/framework_overview/framework_overview.{svg,pdf,png}`,
caption, provenance, and manifest. Visual inspection found the exported PNG
nonblank and correctly carrying the same-candidate task, LLM signal extraction,
calibration layer, C-CRP uncertainty decomposition, multiplicative risk formula,
official baseline block, required method-evidence gates, shared evidence gates,
and claim boundary. The post-refresh consolidated audit
`outputs/summary/paper_critical/paper_critical_module_audit_post_framework_gate_wording_20260606_0505.{json,md,sha256}`
passes with framework status `review_ready` but keeps `paper_ready=false`
because signal rows and storage remain blocked. No experiment or cleanup was
run.

The 2026-06-06 05:18 CST framework overclaim guard hardening extended the
consolidated paper-critical audit so the framework figure cannot pass if its
caption, SVG text, provenance caption, claim limits, or `evidence_gate_status`
claim that the still-blocked observation, component-ablation, or hyperparameter
modules are complete. Fresh audit artifact:
`outputs/summary/paper_critical/paper_critical_module_audit_post_framework_overclaim_guards_20260606_0518.{json,md,sha256}`.
It passes with framework status `review_ready` while preserving
`paper_ready=false`, `signal_rows_available=false`, and
`phase2_5_storage_launch_allowed=false`.

The 2026-06-06 05:23 CST producer-test hardening tightened
`tests/test_framework_overview_figure.py` to verify the conservative caption
phrase, caption/provenance equality, and the claim limit that the framework
figure does not complete observation, component-ablation, or hyperparameter
evidence. This keeps the figure generator aligned with the stricter audit.

The 2026-06-06 05:29 CST Phase 2.5 package-audit hardening tightened
`scripts/audit/main_audit_phase2_5_module_package.py` so future completed
module packages must include seed provenance, clean log snippets, no nested
bulk score/prediction dumps, and finite in-range metric values. Regression
coverage now rejects missing seeds, fatal log markers, nested prediction JSONL,
nonfinite full metrics, and out-of-range hyperparameter metric values.

The 2026-06-06 05:36 CST component-ablation package table-count hardening added
checks for exact `ranking_eval_records.csv` event rows, same-candidate coverage
totals, matched-candidate totals, and allowed same-candidate summary status
labels. This prevents a future component-ablation package from passing with
present but incomplete imported tables.

The 2026-06-06 05:45 CST package manifest-hash hardening tightened the
local/server manifest-comparison gate for future Phase 2.5 module packages.
The audit now rejects presence-only or size-only comparison JSON and requires a
file identity plus valid hash evidence: either checked `sha256` status or
matching local/server or expected/actual SHA-256 values. Regression tests cover
the rejected vague shape and the accepted `manifest_checks` equality shape.

The 2026-06-06 05:54 CST hyperparameter package hardening used a GPT-5.5 xhigh
sidecar audit and tightened the package gate so hyperparameter stability
evidence must be proven by the summary CSV itself. The audit now requires
valid/test rows, at least three distinct values per expected control and split,
metric consistency, row-level audit/degeneracy/coverage/key-count evidence,
one fixed-filter source row per plotted point, no duplicate curve points, real
distinct valid/test sweep SHA-256 hashes, a supported full-metric-set metric,
and consistency between `control_reports.curve_values` and the summary rows.

The 2026-06-06 06:09 CST hyperparameter stability-report hardening extended
both producer and package audit. The plotter now emits a per-control
valid/test stability report and downgrades unstable curves to diagnostic
status; the package audit recomputes the report from the summary CSV, rejects
mismatches and duplicate/extra controls, and enforces a maximum relative drop
tolerance of `0.05`.

Phase 2.5 package-audit hardening on 2026-06-06 added
`scripts/audit/main_audit_phase2_5_module_package.py`, a local read-only gate
for future observation/motivation, component-ablation, and hyperparameter
packages. A module cannot support paper claims until this audit passes. The
gate checks command/log/config/hash records, git commit, row counts, join
counts where applicable, full metrics/tables where required, generated plots,
provenance/status labels, and local/server manifest comparison. For component
ablation it requires a dedicated `component_ablation_summary.csv` with all
expected leave-one-component-out rows; the existing validation sweep alone is
not sufficient evidence for the component study. It also checks selected
valid/test artifacts, imported same-candidate tables, exact score coverage, and
audit/degeneracy flags. For hyperparameter analysis, it requires the main
controls (`eta`, `weight_grid_label`), treats `confidence_weight` as
diagnostic-only under `confidence_plus_evidence`, and requires valid/test
curves, producer audit-summary fields, and figures contained in the package.

## Stable Agent Roles

Use these roles in future multi-agent work:

| role | responsibility |
| --- | --- |
| Milestone Architect | README, milestone map, file index, claim boundaries |
| Server Runner | server commands, nohup logs, resume checks, output validation |
| Baseline Engineer | official repos, provenance, score schema, importer coverage |
| Literature Scout | top-conference sources, related work, protocol expectations |
| Reviewer Agent | overclaim audit, fairness audit, ablations, statistical validity |

## Multi-Agent Handoff Rules

For broad research or implementation tasks, use multi-agent collaboration.
Every agent handoff should state:

- current milestone and files inspected;
- whether the work affects a main claim, supplementary evidence, or diagnostic
  evidence;
- blockers and server commands needed next;
- output/provenance paths;
- whether a result is eligible for a main table.

The Reviewer Agent can veto overclaims, toy shortcuts, and baseline rows that
do not pass the fairness/provenance gates. The Server Runner reports commands,
logs, PIDs, output paths, and failures; it does not change claim boundaries.

Root-level [AGENTS.md](../../AGENTS.md) is the operating contract for future
agents. If a milestone status changes, update this file and the relevant M-file
rather than relying on a chat handoff.

## Current Working Position (updated 2026-06-04)

The repository is now in M5 (multi-domain SOTA validation):

- C-CRP v3 completed on all 8 domains
- Official external baselines completed on original 4 domains (8 methods each)
- New domains (sports/toys/home/tools) official baselines are in Phase 2.
  Sports and toys each have all eight audited official rows plus C-CRP imported
  evidence through domain and paired-test gates. Home has all eight audited official
  rows complete (`proex_profile`, `promax_profile`, `elmrec_graph`, `llmemb`,
  `irllrec_intent`, `rlmrec_graphcl`, `llm2rec_sasrec`, `llmesr_sasrec`) after
  the LLM2Rec and LLM-ESR recovery/evidence gates and local lightweight
  packages passed. Home LLM2Rec
  completed at 2026-06-04 09:49 CST after a disk-full partial-copy recovery,
  with full `@5/@10/@20 + MRR` metrics, exact score coverage, row counts,
  server-final audit, server large-artifact sha256 manifest, local-light audit,
  and post-gate prediction JSONL deletion manifest. The completed LLM2Rec
  intermediate adapter was then removed with sha256 cleanup manifest, restoring
  `/` to about `12G` free / `95%` used. Home LLM-ESR launched at
  2026-06-04 10:14 CST, completed at 2026-06-04 13:09 CST with
  `implementation_status=official_completed`, `blockers=[]`, exact
  `score_coverage_rate=1.0`, and passed server-final audit, server
  large-artifact manifest, lightweight local sync, and local-light audit. Full
  Home LLM-ESR metrics over 10,000 users and 101 candidates are HR@5/10/20
  `0.0621 / 0.1163 / 0.2139`, NDCG@5/10/20
  `0.037993209299003045 / 0.055376101596196485 / 0.0797502336556021`, and
  MRR `0.059737054548523474`. Its post-gate prediction JSONL and temporary
  adapter staging directory were removed under
  `outputs/summary/home_llmesr_post_gate_cleanup_20260604.sha256`, preserving
  scores/provenance/audits/tables/final model and local evidence. The Home
  C-CRP import, domain gate, and comparison/paired-test package then passed at
  2026-06-04 13:50 CST: `outputs/summary/home_official_ccrp_gate_20260604.*`
  records `official_ok_count=8`, `ccrp_ok=true`, and `gate_ok=true`, and
  `outputs/summary/home_official_ccrp_20260604_paired_summary.json` records
  `claim_gate=home_domain_pass`, C-CRP observed-best on all seven metrics, and
  all 56 C-CRP-vs-official paired tests positive and Holm-significant. Tools
  `proex_profile` completed at 2026-06-04 16:08 CST as the first Tools official
  row with `implementation_status=official_completed`, `blockers=[]`, exact
  `score_coverage_rate=1.0`, and passing server-final audit, server
  large-artifact sha256 manifest, lightweight local sync, and local-light
  audit. Full metrics over 10,000 users and 101 candidates are HR@5/10/20
  `0.0602 / 0.1177 / 0.2329`, NDCG@5/10/20
  `0.037281705859706714 / 0.055676376797898205 / 0.08437492971571317`, and
  MRR `0.06071849976691817`; `scores.csv` has `1,010,001` lines, predictions
  had `10,000` lines before post-gate deletion, and
  `tables/ranking_eval_records.csv` has `10,001` lines. The local lightweight
  package is
  `outputs/baselines/official_adapters/tools_large10000_100neg_proex_profile_official_qwen3base_same_candidate/`.
  The post-gate server prediction deletion manifest is
  `outputs/tools_large10000_100neg_proex_profile_official_qwen3base_same_candidate/prediction_deletion_manifest.json`,
  and the completed intermediate adapter
  `outputs/baselines/paper_adapters/tools_large10000_100neg_proex_official_adapter`
  was removed after path checks, restoring `/` to about `12G` free / `95%`
  used while preserving final scores, provenance, audits, imported tables, and
  `proex_official_model.pt`. Tools `promax_profile` launched at
  2026-06-04 16:46 CST as the second Tools row with runner PID `3279573`,
  adapter PID `3279582`, and log
  `baselines_new_domains_tools_promax_20260604_164630.log`; it completed at
  2026-06-04 19:59 CST with `implementation_status=official_completed`,
  `blockers=[]`, exact `score_coverage_rate=1.0`, and passing server-final
  audit, server large-artifact sha256 manifest, lightweight local sync, and
  local-light audit. Full metrics over 10,000 users and 101 candidates are
  HR@5/10/20 `0.056 / 0.1046 / 0.2018`, NDCG@5/10/20
  `0.03468275603534166 / 0.05029722685396016 / 0.07458228366305956`, and MRR
  `0.056527355267188224`; `scores.csv` has `1,010,001` lines, predictions had
  `10,000` lines before post-gate deletion, and
  `tables/ranking_eval_records.csv` has `10,001` lines. The local lightweight
  package is
  `outputs/baselines/official_adapters/tools_large10000_100neg_promax_profile_official_qwen3base_same_candidate/`.
  Server-only `scores.csv`, deleted prediction metadata, and
  `promax_official_model.pt` are covered by `server_large_artifact_manifest.sha256`
  and `prediction_deletion_manifest.json`; the completed ProMax intermediate
  adapter was removed after final evidence and local backup passed, with
  cleanup manifests
  `outputs/summary/tools_promax_completed_adapter_cleanup_manifest_20260604.sha256`
  and `outputs/summary/tools_promax_completed_adapter_cleanup_du_20260604.txt`.
  Final scores, provenance, audits, imported tables, and model were preserved,
  and post-cleanup `/` was about `11G` free / `95%` used. Tools `elmrec_graph`
  launched at 2026-06-04 20:46 CST as the third single-row Tools official
  baseline with runner PID `3301337`, adapter PID `3301345`, and log
  `baselines_new_domains_tools_elmrec_20260604_204602.log`; it completed at
  2026-06-04 22:39 CST with `implementation_status=official_completed`,
  `blockers=[]`, exact `score_coverage_rate=1.0`, and passing server-final
  audit, server large-artifact sha256 manifest, lightweight local sync, and
  local-light audit. Full metrics over 10,000 users and 101 candidates are
  HR@5/10/20 `0.0501 / 0.101 / 0.2101`, NDCG@5/10/20
  `0.029656030656687697 / 0.045870649973376774 / 0.07316592297455926`, and MRR
  `0.05237582779698271`; `scores.csv` has `1,010,001` lines, predictions had
  `10,000` lines before post-gate deletion, and
  `tables/ranking_eval_records.csv` has `10,001` lines. The local lightweight
  package is
  `outputs/baselines/official_adapters/tools_large10000_100neg_elmrec_graph_official_qwen3base_same_candidate/`.
  Server-only `scores.csv`, deleted prediction metadata, and
  `elmrec_official_model.pt` are covered by `server_large_artifact_manifest.*`
  and `prediction_deletion_manifest.json`; the completed ElmRec intermediate
  adapter was removed after final evidence and local backup passed, with
  cleanup manifests
  `outputs/summary/tools_elmrec_completed_adapter_cleanup_manifest_20260604.sha256`
  and `outputs/summary/tools_elmrec_completed_adapter_cleanup_du_20260604.txt`.
  Final scores, provenance, audits, imported tables, and model were preserved,
  and post-cleanup `/` was about `11G` free / `95%` used. Tools `llmemb`
  launched at 2026-06-04 23:10 CST as the fourth single-row Tools official
  baseline with runner PID `3317251`, adapter PID `3317260`, and log
  `baselines_new_domains_tools_llmemb_20260604_231030.log`; it completed at
  2026-06-05 00:43 CST with `implementation_status=official_completed`,
  `blockers=[]`, exact `score_coverage_rate=1.0`, and passing server-final
  audit, server large-artifact sha256 manifest, lightweight local sync, and
  local-light audit. Full metrics over 10,000 users and 101 candidates are
  HR@5/10/20 `0.1365 / 0.2257 / 0.3637`, NDCG@5/10/20
  `0.087457824217457 / 0.11594350972806679 / 0.15050644138929892`, and MRR
  `0.10649354669900822`; `scores.csv` has `1,010,001` lines, predictions had
  `10,000` lines before post-gate deletion, and
  `tables/ranking_eval_records.csv` has `10,001` lines. The local lightweight
  package is
  `outputs/baselines/official_adapters/tools_large10000_100neg_llmemb_official_qwen3base_same_candidate/`.
  Server-only `scores.csv`, deleted prediction metadata, and
  `llmemb_official_model.pt` are covered by `server_large_artifact_manifest.*`
  and `prediction_deletion_manifest.json`; the completed intermediate adapter
  was removed after final evidence and local backup passed, with cleanup
  manifests
  `outputs/summary/tools_llmemb_completed_adapter_cleanup_manifest_20260605.sha256`
  and `outputs/summary/tools_llmemb_completed_adapter_cleanup_du_20260605.txt`.
  A completed Home LLM2Rec checkpoint was also deleted under emergency disk
  approval after sha256/size manifesting; local record:
  `outputs/summary/home_llm2rec_checkpoint_deletion_manifest_20260604.json`.
  Final scores, provenance, audits, imported tables, and model were preserved,
  and post-cleanup `/` was about `15G` free / `93%` used. Tools
  `irllrec_intent` launched at
  2026-06-05 01:04 CST as the fifth single-row Tools official baseline with
  runner PID `3326805`, adapter PID `3326813`, log
  `baselines_new_domains_tools_irllrec_20260605_0058.log`, and heartbeat
  `monitor-tools-irllrec`; launch snapshot
  `outputs/summary/tools_irllrec_launch_monitor_20260605.json` shows Qwen3
  embedding progress `2056/269711`, GPU `95%`, disk `13.34G` free, and no
  failure markers. It completed normally at 2026-06-05 05:19 CST with
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
  (`269711/269711`) and RLMRec official training started, latest observed at
  `epoch=110` with train loss `1.519527`. The same PIDs were alive and unique,
  GPU remained active with `24679 MiB / 49140 MiB` resident, the active
  adapter was about `5.2G`, and final RLMRec evidence remained placeholder-only.
  Disk fell below the warning line to about `8.6G` free / `96%` used. After
  confirming completed Tools IRLLRec server-final and local-light evidence,
  Codex removed only its server-side `predictions/rank_predictions.jsonl` with
  sha256 manifest
  `outputs/summary/tools_irllrec_prediction_cleanup_manifest_20260605.sha256`;
  final scores/provenance/audits/tables/model were preserved. Disk recovered
  only to about `9.3G` free, so the active row still needs close disk
  monitoring and no next baseline should start.
  A 2026-06-05 08:01 CST follow-up confirmed the active Tools RLMRec run uses
  the default official `3000`-epoch setting (`--rlmrec_epochs=3000` / trainer
  `--epochs=3000`) and had reached epoch `590/3000` with train loss
  `1.509551`. The same runner/adapter PIDs were alive and unique, no
  completion/failure/final-artifact markers existed, final evidence remained
  placeholder-only, disk remained about `9.3G` free / `96%` used, and a repeat
  large-file/cache/temp/archive audit found no safe cleanup candidate.
  A 2026-06-05 08:46 CST follow-up found the same runner/adapter PIDs alive
  and unique, with official RLMRec training past the first material monitor
  checkpoint at epoch `1000/3000` and train loss `1.505931`. Final RLMRec
  evidence still remained placeholder-only, with no final scores/provenance,
  score audit, imported tables, predictions, completion marker, OOM/no-space,
  killed, traceback, or error marker. Disk remained warning-level at about
  `9.3G` free / `96%` used and no new safe cleanup candidate was identified.
  This remains active monitor-only evidence, not a completed RLMRec row.
  A 2026-06-05 09:45 CST follow-up found the same runner/adapter PIDs alive
  and unique, with official RLMRec training past the halfway monitor checkpoint
  on the default `3000`-epoch path: latest observed epoch `1510/3000` had train
  loss `1.506936`, after epoch `1500` loss `1.507642`. Final RLMRec evidence
  still remained placeholder-only (`4.0K`) with no final scores, provenance,
  score audit, imported tables, predictions, completion marker, OOM/no-space,
  killed, traceback, or error marker. Disk remained warning-level at about
  `9.3G` free / `96%` used; a repeat large-file/cache/temp/archive and
  prediction cleanup audit found no safe meaningful deletion candidate. This
  remains active monitor-only evidence, not a completed RLMRec row.
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
  audit, imported tables, predictions, completion marker, OOM/no-space, killed,
  traceback, or error marker. Disk remained warning-level at about `9.3G` free
  / `96%` used; a repeat large-file/cache/temp/archive and prediction cleanup
  audit found no safe deletion target because visible reclaimable candidates
  were active RLMRec intermediates, protected task splits, retained completed
  checkpoints/evidence, or the legacy Electronics ELMRec prediction JSONL
  without server-final/local-light deletion proof. This remains active
  monitor-only evidence, not a completed RLMRec row.
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
  A first Tools `llm2rec_sasrec` launch at 2026-06-05 13:02 CST failed before
  a stable adapter process or score file because validation export found a
  malformed/truncated Tools valid candidate CSV. The corrupt file had
  `587252` logical rows and ended with a user-id-only row for
  `AGEY75LYLXUAHG3KW5KF5ICMKA4A`; sha256
  `4302712bb7dbe0a8cfde99b0a2727c8de0818d250b65e7cc3bc0b8ad01fa6f2b`. It was
  rebuilt from `ranking_valid.jsonl` and independently validated with
  `1,010,000` rows, `10,000` events, `101` candidates/event, and exactly one
  positive/label per event. Manifests:
  `outputs/summary/tools_valid_candidate_items_repair_20260605T051943Z.json`
  and
  `outputs/summary/tools_valid_candidate_items_repair_validation_20260605T0520Z.json`.
  No Tools LLM2Rec row is active or table-eligible; disk is about `14G` free /
  `93%` used, so relaunch is pending a fresh preflight and storage-margin
  decision.
  A 2026-06-05 13:46 CST relaunch checkpoint then started exactly one Tools
  `llm2rec_sasrec` row after a GPT-5.5 xhigh sidecar ARIS auditor judged the
  relaunch conditionally acceptable. Runner PID `3413921`, adapter PID
  `3413930`, log `baselines_new_domains_tools_llm2rec_20260605_134355.log`,
  heartbeat `monitor-tools-llm2rec`, and launch snapshot
  `outputs/summary/tools_llm2rec_launch_monitor_20260605.json`. The snapshot
  showed one adapter process, no failure markers, active Qwen3 embedding at
  `[hf_mean_pool] encoded 3888/345622`, GPU about `95%` / `16089 MiB`, adapter
  dir `1.1G`, final output placeholder-only, and disk about `12G` free / `94%`
  used. This is active monitor-only evidence, not a completed row.
  A 2026-06-05 13:57 CST follow-up found the same runner/adapter alive, cleared
  three stale diagnostic grep/bash processes that were creating false duplicate
  matches, and observed embedding progress around
  `[hf_mean_pool] encoded 53296/345622`; disk remained about `12G` free / `94%`
  used and no failure/completion marker was present. Snapshot:
  `outputs/summary/tools_llm2rec_monitor_checkpoint_20260605_1355.json`.
  Every completed row imports full `@5/@10/@20 + MRR` metrics after score audit.
  A 2026-06-05 15:23 CST monitor then confirmed this Tools LLM2Rec relaunch
  failed under disk exhaustion, not completion: runner PID `3413921` and
  adapter PID `3413930` had exited, GPU was idle, the wrapper log ended with
  `OSError: [Errno 28] No space left on device`, and the training log showed
  `torch.save` failed while writing the SASRec checkpoint. The row has no
  provenance, no scores, and no score audit, so it is not table-eligible. Safe
  emergency cleanup removed the already-manifested corrupt validation backup,
  failed adapter-side embedding copy, regenerable adapter CSVs, and corrupt
  partial checkpoint while preserving the upstream embedding cache
  `/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`.
  Post-cleanup disk is about `8.1G` free / `96%` used, still below the `10G`
  monitor threshold. Tools remains 6/8 official rows gated; next action is an
  additional audited storage decision, then exactly one LLM2Rec recovery run
  using `--llm2rec_item_embedding_path` to reuse the preserved embedding.
  A 2026-06-05 16:05 CST recovery/storage checkpoint then removed the completed
  Sports and Toys LLM2Rec full checkpoints under manifest-backed emergency
  archive decisions after confirming each row's completed provenance,
  server-final audit, local-light package, live scores, score audits, run
  summary, and imported tables. Manifests are
  `outputs/summary/sports_llm2rec_checkpoint_deletion_manifest_20260605.json`
  and `outputs/summary/toys_llm2rec_checkpoint_deletion_manifest_20260605.json`;
  final scores/provenance/audits/tables/local packages were preserved. The
  wrapper gained `LLM2REC_ITEM_EMBEDDING_PATH_OVERRIDE`, was syntax-checked on
  the server, and a single Tools `llm2rec_sasrec` recovery launched at
  2026-06-05 15:59 CST with runner PID `3423029`, adapter PID `3423037`,
  training PID `3423221`, log
  `baselines_new_domains_tools_llm2rec_recovery_20260605_155904.log`, and
  heartbeat `monitor-tools-llm2rec-recovery`. The command reuses the preserved
  Tools embedding path and does not force regeneration. At the 16:05 CST
  snapshot it was active around epoch `30+`, had saved its Tools checkpoint,
  and disk was about `10G` free / `95%` used. The row is still monitor-only
  evidence until all completion gates pass.
  A 2026-06-05 17:04 CST follow-up found the same recovery row still active
  with runner PID `3423029`, adapter PID `3423037`, and training PID
  `3423221`; duplicate counts remained exactly one LLM2Rec adapter and one
  `ToolsSameCandidate100Neg` training child. No final provenance, `scores.csv`,
  or imported ranking table existed yet. Training-log validation metrics were
  still improving at epoch `330`, with latest observed validation HR@5/10/20
  `0.26969999074935913 / 0.3407999873161316 / 0.414900004863739` and
  NDCG@5/10/20
  `0.20475752651691437 / 0.22773440182209015 / 0.2465011030435562`; these are
  validation-only progress numbers, not final same-candidate table metrics.
  Disk briefly crossed the monitor threshold at about `9.9G` free / `95%`
  used. A large-file/cache/temp/archive audit found no safe project-output
  deletion, and only approved conda package cache cleanup was run. No project
  outputs, evidence, checkpoints, embeddings, task splits, or other projects
  were deleted; post-cleanup `/` was about `11G` free (`10,773,983,232` bytes)
  / `95%` used.
  A 2026-06-05 18:07-18:11 CST follow-up found the same recovery row still
  active with runner PID `3423029`, adapter PID `3423037`, and training PID
  `3423221`; duplicate counts remained exactly one LLM2Rec adapter and one
  `ToolsSameCandidate100Neg` training child. No final provenance, `scores.csv`,
  score audit, or imported ranking table existed yet. Training-log validation
  progress reached epoch `715`, with latest observed validation HR@5/10/20
  `0.8866999745368958 / 0.9272000193595886 / 0.9584000110626221` and
  NDCG@5/10/20
  `0.8021516799926758 / 0.8153268098831177 / 0.8232611417770386`; these are
  still validation-only progress numbers, not final same-candidate table
  metrics. Because disk was only about 34 MiB above the 10 GiB guard
  (`10,773,143,552` bytes free), only non-project cache cleanup was run: conda
  reported no unused tarballs/packages/tempfiles, pip cache purge removed
  `17.0 MB`, and npm cache clean was attempted. No project outputs, evidence,
  checkpoints, embeddings, task splits, or other projects were deleted. The
  post-cleanup safety check confirmed the same PIDs alive, duplicate counts
  `1/1`, no final evidence, and disk `10,820,177,920` bytes free / `95%` used.
  At 2026-06-05 18:38 CST, Tools `llm2rec_sasrec` completed normally under
  the recovery run: the wrapper reported `implementation_status=official_completed`,
  `[2026-06-05 18:38:04] DONE llm2rec_sasrec on tools`, and `=== All baseline
  runs complete ===`. Server-final audit passed with `ok=true`; server
  large-artifact manifest passed; local-light sync and local-light audit both
  passed. Local lightweight evidence is under
  `outputs/baselines/official_adapters/tools_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate/`.
  Full metrics over `10,000` users and `101` candidates are HR@5/10/20
  `0.0957 / 0.1625 / 0.2954`, NDCG@5/10/20
  `0.060227320850546905 / 0.08147852827156639 / 0.11481218118869342`, and MRR
  `0.08101396652891538`. Score audit/import coverage passed with
  `1,010,000` candidate/score rows, `1,010,000` matched keys, no missing,
  extra, duplicate, or non-finite scores, and `score_coverage_rate=1.0`.
  `scores.csv` has `1,010,001` lines, predictions had `10,000` lines before
  post-gate cleanup, and `tables/ranking_eval_records.csv` has `10,001`
  lines. Because disk remained tight, post-gate cleanup removed only the
  server-side prediction JSONL after sha256/line-count manifesting
  (`outputs/summary/tools_llm2rec_prediction_deleted_post_gate_20260605.{json,sha256}`)
  and the completed-row adapter staging CSVs
  (`outputs/summary/tools_llm2rec_completed_adapter_staging_cleanup_20260605.{json,sha256}`).
  Final scores, provenance, audits, imported tables, server-final audit,
  large-artifact manifest, SASRec checkpoint, upstream embedding, and compact
  adapter metadata were preserved; disk recovered to `11,772,899,328` bytes
  free / `95%` used. Tools is now 7/8 official rows gated; the remaining row
  is `llmesr_sasrec`.
  After a fresh no-process/GPU/disk/duplicate-output preflight, the final Tools
  official row `llmesr_sasrec` launched at 2026-06-05 18:52 CST with runner PID
  `3440278`, adapter PID `3440287`, log
  `baselines_new_domains_tools_llmesr_20260605_185250.log`, pidfile
  `outputs/summary/tools_llmesr_launch_20260605_185250.pid`, launch snapshot
  `outputs/summary/tools_llmesr_launch_monitor_20260605_185250.json`, and
  heartbeat `monitor-tools-llm-esr`. The 18:57 CST stable follow-up showed
  exactly one LLM-ESR adapter process, no training child yet, GPU `99%` /
  `16091 MiB`, embedding progress about `11288/269711`, no failure markers,
  final output placeholder-only, and adapter dir about `1005M`. Because adapter
  staging pushed disk below the strict 10 GiB guard, only non-project caches
  were cleaned (`.cache`, `.codex/.tmp`, Chrome GPU cache, and inactive
  `.cursor-server/bin` after verifying no Cursor process). No project evidence,
  checkpoints, embeddings, task splits, configs, or other project files were
  deleted. Cache-cleanup record:
  `outputs/summary/tools_llmesr_launch_cache_cleanup_20260605.txt`.
  Post-cleanup disk was `10,866,053,120` bytes free / `95%` used.
  A 2026-06-05 19:04-19:06 CST monitor/cache checkpoint found the same row
  still active with runner PID `3440278` and adapter PID `3440287`, duplicate
  counts exactly one adapter and zero training child, GPU about `91-96%` /
  `16189 MiB`, embedding progress about `47824/269711`, no final evidence, and
  no failure/OOM/no-space markers. Disk dipped just below the strict 10 GiB
  guard (`10,726,035,456` bytes free), so only non-project Chrome cache/GPU
  cache and VSCode-server logs were cleared. No project outputs, evidence,
  checkpoints, embeddings, task splits, configs, source code, or other project
  files were deleted. Cleanup record:
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
  SASRec checkpoint was deleted only after verifying server-final/local-light
  gated evidence and writing manifests:
  `outputs/summary/tools_llm2rec_completed_checkpoint_cleanup_for_llmesr_disk_20260605.{json,sha256}`.
  The deleted checkpoint was `5,665,876,357` bytes, sha256
  `8ad7ce0316befeb8ee6b3482546ffe3e301e42e9a6b1e10ee608689ea5ece414`. Final
  LLM2Rec scores, provenance, score audit, run summary, imported tables,
  server-final audit, large-artifact manifest, and local-light package were
  preserved; disk recovered to `16,293,425,152` bytes free / `92%` used.
  A 2026-06-05 19:55-20:01 CST follow-up found the active Tools
  `llmesr_sasrec` row still alive and unique with runner PID `3440278` and
  adapter PID `3440287`. Qwen3 embedding completed (`269711/269711`) and the
  row entered LLM-ESR training at `[llmesr] epoch=1 train_loss=1.398057`.
  Final evidence was still absent, so the row remained non-table-eligible.
  Disk tightened to `11,804,352,512` bytes free / `95%` used; a read-only
  storage audit identified the active adapter embedding
  `outputs/baselines/paper_adapters/tools_large10000_100neg_llmesr_official_adapter/llm_esr/handled/itm_emb_np.pkl`
  (`4.12G`) as the main new pressure, with other large visible candidates
  protected completed models/checkpoints/task splits or non-deletable legacy
  prediction evidence. No cleanup was performed.
  At 2026-06-05 21:19 CST, Tools `llmesr_sasrec` completed normally as the
  eighth Tools official row. Server-final audit, server large-artifact
  manifest, local-light sync, and local-light audit passed. Full metrics over
  10,000 users and 101 candidates are HR@5/10/20
  `0.0711 / 0.1270 / 0.2219`, NDCG@5/10/20
  `0.042728964614829223 / 0.060602849768892623 / 0.08433244535733923`, and
  MRR `0.06334161303438132`; row counts are `scores.csv` `1,010,001`,
  prediction JSONL `10,000` before cleanup, and
  `tables/ranking_eval_records.csv` `10,001`. Post-gate cleanup removed only
  the server prediction JSONL and two adapter staging CSVs under
  `outputs/summary/tools_llmesr_post_gate_cleanup_20260605.sha256`, preserving
  final scores, provenance, audits, imported tables, server-final certificate,
  `llmesr_official_model.pt`, and upstream embedding artifacts.
  Tools C-CRP raw scores were then imported into
  `outputs/tools_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate`
  with `score_coverage_rate=1.0`. The post-cleanup Tools domain gate
  `outputs/summary/tools_official_ccrp_gate_post_cleanup_20260605.{json,csv}`
  records `official_ok_count=8`, `official_all_ok=true`, `ccrp_ok=true`, and
  `gate_ok=true`; the missing imported C-CRP prediction is certified by
  `prediction_deletion_manifest.json` after domain-gate cleanup. The comparison
  package `outputs/summary/tools_official_ccrp_20260605_*` records
  `claim_gate=tools_domain_pass`: C-CRP is observed-best on all seven metrics,
  all 56 C-CRP-vs-8-official paired tests are positive and Holm-significant,
  with `min_delta=0.0294`, `min_ci_low=0.0173`, and
  `max_holm_p_value=9.7172307634557e-07`. Disk remains tight at about `7.4G`
  free / `97%` used, so future server work needs a fresh storage preflight.
  A compact four-new-domain comparison-gate certificate was built locally at
  `outputs/summary/new_domains_official_ccrp_cross_domain_20260605_*`. It
  records that Sports/Toys/Home/Tools each have 8/8 completed official rows,
  passed domain gates, C-CRP rank 1 on all seven metrics, and 56/56 positive
  Holm-significant paired tests. This closes the Phase 2 new-domain
  official+C-CRP gate summary, but it is not a paper-readiness verdict and not
  a full raw local reproduction package.
  A 2026-06-05 Phase 2.5 storage-retention audit then found the server at
  `12,342,841,344` bytes free / `94%` used, below the strict `15GiB` launch
  floor. Routine safe-now cleanup is insufficient; high-yield cleanup now
  requires explicit archive/retention approval for a completed upstream
  embedding or model checkpoint. A guarded non-executing plan now exists for
  the completed Tools LLM2Rec upstream embedding under
  `outputs/summary/paper_critical/retention_cleanup_plan_20260605/`; it exits
  before any delete command and records `will_delete=false`. No Phase 2.5
  experiment was launched. A reusable read-only helper,
  `scripts/audit/main_audit_phase2_5_storage_retention.py`, was added at the
  2026-06-05 23:42 CST checkpoint. Current artifact
  `outputs/summary/paper_critical/server_storage_phase2_5_retention_audit_current_20260605.{json,md,sha256}`
  confirms no active project Python process, GPU idle, `12,342,640,640` free
  bytes, `64,611,717` safe-now recoverable bytes, and
  `experiment_launch_allowed=false`; eight high-yield candidates still require
  explicit retention/archive approval before deletion. The audited safe-now
  cleanup was then executed at 2026-06-06 00:10-00:15 CST using
  `scripts/audit/main_cleanup_phase2_5_safe_now_remnants.py`; it deleted only
  the two completed Tools paper-adapter staging directories and
  `tmp_llm2rec_sync`, recovering `64,574,853` bytes with a file-level sha256
  manifest. Post-cleanup disk is `12,407,840,768` free bytes / `94%` used,
  still below the `15GiB` floor. Post-cleanup Tools domain gate and comparison
  gates still pass, with C-CRP observed-best on all seven metrics and all 56
  paired tests positive and Holm-significant. No Phase 2.5 experiment was
  launched. A ranked retention audit at 2026-06-06 00:20 CST now recommends
  the completed Tools LLM2Rec upstream embedding as the lowest-risk
  approval-required candidate because it is outside final evidence directories
  and would raise expected free space to `18,070,102,144` bytes; it still
  requires explicit archive/retention approval and the guarded manifest/gate
  sequence before deletion. Evidence:
  `outputs/summary/paper_critical/server_storage_phase2_5_retention_audit_ranked_20260606.{json,md,sha256}`.
  The guarded retention plan was refreshed at 2026-06-06 00:30 CST under
  `outputs/summary/paper_critical/retention_cleanup_plan_20260606/` so the
  plan artifact directly cites the ranked audit, records
  `recommended_by_ranked_audit=true`, risk tier
  `approval_required_external_embedding_cache`, rank `20`, and expected
  post-approval free space `18,070,102,144` bytes. The generated shell still
  exits with `exit 2` before any manifest or delete command, and the JSON
  remains `will_delete=false` / `requires_explicit_approval=true`. A fresh
  read-only preflight before refresh found no matching project Python process,
  GPU idle, and `/` at `12,407,390,208` free bytes / `94%` used. No deletion
  was performed and no Phase 2.5 experiment was launched.
  A local/server evidence consistency audit was added at 2026-06-06 00:40 CST:
  `scripts/audit/main_audit_local_server_evidence_consistency.py` checks
  local lightweight packages against copied server large-artifact manifests,
  hashes every sync-manifest checked file, and fails if server-only bulk
  artifacts appear locally. Current Tools artifact
  `outputs/summary/paper_critical/local_server_evidence_consistency_tools_20260606.{json,md,sha256}`
  reports `8/8` Tools official packages ok with `failure_count=0`, without
  SSH, copying, deletion, or experiment launch.
  The same local-only audit run over Sports/Toys/Home/Tools at 2026-06-06
  00:50 CST produced
  `outputs/summary/paper_critical/local_server_evidence_consistency_new_domains_20260606.{json,md,sha256}`
  with `row_count=32`, `ok_count=11`, and `failure_count=51`. The gap is
  evidence packaging, not failed baselines: older Sports/Toys/Home packages
  are missing copied `server_large_artifact_manifest.json` and/or `.sha256`,
  while representative inspected rows still have passing server-final audits,
  full metrics, exact score coverage, and row counts. Next evidence action:
  backfill or regenerate those small server-large manifests and rerun the
  consistency audit before treating all four local packages as artifact-ready.
- Strategy: achieve SOTA only after the new-domain official baselines pass
  same-candidate score/provenance/import gates
- Paper readiness now also requires three paper-critical modules before final
  writing: a representative uncertainty observation/motivation study, a
  leave-one-component-out C-CRP ablation suite, and real hyperparameter
  sensitivity curves, plus a clean framework overview figure. These modules are
  main paper gates, not optional polish. Execution specification:
  `docs/paper_critical_experiment_plan_2026-06-03.md`.
- Observation/motivation script entry now exists:
  `scripts/analysis/main_build_uncertainty_observation_study.py`. It is not a
  completed paper result until real C-CRP uncertainty signal rows are located
  or regenerated without LLM re-query leakage and the Sports/Toys runs pass
  row-count, join-rate, table, figure, and provenance gates.
  Fixed-filter discovery/full audits on 2026-06-04 found no paper-ready or
  recomputable full-scale new-domain C-CRP signal rows under visible server
  artifacts. A 2026-06-05 03:27 CST local verification reran
  `scripts/audit/main_audit_paper_critical_modules.py --root .` and the focused
  paper-critical tests
  (`tests\test_audit_paper_critical_modules.py`,
  `tests\test_framework_overview_figure.py`,
  `tests\test_uncertainty_observation_study.py`,
  `tests\test_ccrp_hyperparameter_sweep_plot.py`,
  `tests\test_build_ccrp_component_inventory.py`), with `19 passed`; the audit
  still reports `paper_ready=false` and `signal_rows_available=false`.
  outputs; Sports/Toys/Home/Tools formal `scores.csv` files are complete for
  ranking import but score-only for uncertainty analysis. A project-root broad
  scan also found no additional matching signal rows outside `outputs/`.
  Guard hardening on 2026-06-04 added duplicate/extra eval-event rejection,
  positive-rank and optional `num_candidates` checks, full-metric provenance,
  and a `motivation_only_not_main_table_sota` claim scope for this script.
- C-CRP hyperparameter curve script entry now exists:
  `scripts/analysis/main_plot_ccrp_hyperparameter_sweep.py`. It consumes
  validation `valid_ccrp_sweep.csv` artifacts and is not a completed paper
  result until Sports/Toys sweeps produce at least three values per plotted
  control with validation/test separation, `--test_sweep_csv` reporting, and
  provenance. Guard hardening on 2026-06-04 now downgrades valid-only,
  incomplete, or audit-not-enforced curves through explicit status labels and
  paper-claim scopes.
- C-CRP component inventory helper now exists:
  `scripts/audit/main_build_ccrp_component_inventory.py`. Current artifact
  `outputs/summary/paper_critical/ccrp_component_inventory/ccrp_component_inventory_20260604.{json,md}`
  covers 12 current C-CRP component/handle/risk entries and records the
  code-matched multiplicative risk formula
  `base_score * ((1 - uncertainty) ** eta)`. This is an audit scaffold, not an
  executed ablation result; paper-facing component claims remain blocked until
  full-scale valid/test signal rows exist.
- C-CRP component-ablation summary builder now exists:
  `scripts/analysis/main_build_ccrp_component_ablation_summary.py`. It freezes
  `selected_valid_config.json`, evaluates expected leave-one-component-out
  ablations on test without test selection, writes
  `component_ablation_summary.{csv,json}`, `component_ablation_provenance.json`,
  and PNG/PDF figures, and fails closed on missing validation-selection
  metadata, non-`full` score mode by default, missing valid-sweep ablations,
  coverage failures, or audit/degeneracy failures. The guarded signal plan was
  refreshed to include the builder and module package audits, but this is
  tooling only; no full-scale component result exists until signal rows and
  package audits pass.
- Framework overview figure builder now exists:
  `scripts/analysis/main_build_framework_overview_figure.py`. It exports SVG
  editable source plus PDF/PNG/caption/provenance. As of 2026-06-04 04:43 CST,
  the draft local package exists at
  `outputs/summary/paper_critical/framework_overview/` with a sha256 manifest.
  It was regenerated at 2026-06-04 17:31 CST against git commit `9badd19` with
  the code-matched multiplicative risk formula; focused figure/audit tests
  passed (`4 passed`) and the PNG was visually checked as nonblank/readable. It
  must keep the claim boundary as controlled same-candidate ranking rather than
  full-catalog or generative-title recommendation. On 2026-06-05 23:20 CST the
  package was upgraded from draft scaffold to figure-module review-ready:
  provenance records
  `status_label=paper_critical_framework_overview_review_ready`,
  `paper_claim_ready=true`, and
  `module_scope=framework_figure_only_not_substitute_for_observation_ablation_or_hyperparameter_evidence`;
  the consolidated audit checks required SVG labels, PNG dimensions
  (`2559x1378` in the regenerated artifact), claim boundary, formula
  alignment, and manifest hashes. Evidence:
  `outputs/summary/paper_critical/paper_critical_module_audit_post_framework_review_20260605.{json,md,sha256}`.
  Overall paper readiness remains false because observation, ablation, and
  hyperparameter modules still need full-scale valid/test uncertainty signal
  rows.
- Consolidated paper-critical module audit now exists:
  `scripts/audit/main_audit_paper_critical_modules.py`. Current artifact
  `outputs/summary/paper_critical/paper_critical_module_audit_20260604.{json,md}`
  reports `paper_ready=false`, `framework_overview_scaffold_ready=true`,
  `component_inventory_ready=true`, `guarded_plan_ready=true`, and
  `signal_rows_available=false`; observation, ablation, and hyperparameter
  modules remain blocked until full-scale
  uncertainty or recomputable signal rows are located or regenerated.
  The 2026-06-06 03:48 CST refresh adds an explicit
  `component_ablation_execution_support_ready` check covering the selector
  full-metric path, component summary builder, module package audit, and
  guarded-plan command templates. Artifact:
  `outputs/summary/paper_critical/paper_critical_module_audit_post_component_execution_support_20260606_0348.{json,md,sha256}`.
  It reports `ok=true`, `paper_ready=false`,
  `component_ablation_execution_support_ready=true`,
  `four_domain_evidence_consistent=true`, `signal_rows_available=false`, and
  `phase2_5_storage_launch_allowed=false`.
  The 2026-06-06 04:32 CST refresh generalizes this to all three
  paper-critical modules. It checks observation/motivation execution support,
  component-ablation execution support, hyperparameter execution support, and
  guarded-plan command coverage. Artifact:
  `outputs/summary/paper_critical/paper_critical_module_audit_post_all_module_execution_support_20260606_0432.{json,md,sha256}`.
  It reports `ok=true`, `paper_ready=false`,
  `observation_execution_support_ready=true`,
  `component_ablation_execution_support_ready=true`,
  `hyperparameter_execution_support_ready=true`,
  `four_domain_evidence_consistent=true`, `signal_rows_available=false`, and
  `phase2_5_storage_launch_allowed=false`.
  The 2026-06-06 04:40 CST refresh makes the disk decision actionable in the
  same audit output. Artifact:
  `outputs/summary/paper_critical/paper_critical_module_audit_storage_actionable_20260606_0440.{json,md,sha256}`.
  It reports no safe-now recoverable bytes, an approval-required high-yield
  candidate of `5662687360` bytes, `storage_cleanup_decision_required=true`,
  and that the candidate would clear the minimum Phase 2.5 disk gate.
- The guarded C-CRP signal-generation plan now uses generic active-row
  preconditions: no active official baseline row or matching baseline Python
  process may be running before the placeholder signal paths are filled and the
  non-executing shell guard is removed. This 2026-06-05 guard hardening removes
  stale Home RLMRec wording from the generated plan without starting any
  experiment.

### C-CRP v3 Results (all domains)

| Domain | Users | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR | Status |
|--------|-------|------|-------|-------|--------|---------|---------|-----|--------|
| beauty | 973 | 0.157 | 0.229 | 0.369 | 0.111 | 0.134 | 0.169 | 0.128 | #2 (ProEx=0.253) |
| books | 10000 | 0.374 | **0.476** | 0.592 | 0.300 | **0.333** | 0.362 | 0.306 | **SOTA** |
| electronics | 10000 | 0.218 | **0.299** | 0.418 | 0.157 | **0.183** | 0.213 | 0.168 | **SOTA** |
| movies | 10000 | 0.145 | 0.208 | 0.331 | 0.108 | 0.128 | 0.159 | 0.127 | #5 |
| sports | 10000 | 0.275 | 0.382 | 0.517 | 0.198 | 0.233 | 0.267 | 0.208 | domain gate PASS |
| toys | 10000 | 0.317 | 0.396 | 0.506 | 0.245 | 0.271 | 0.298 | 0.250 | domain gate PASS |
| home | 10000 | 0.156 | 0.226 | 0.351 | 0.110 | 0.132 | 0.164 | 0.126 | domain gate PASS |
| tools | 10000 | 0.194 | 0.270 | 0.393 | 0.142 | 0.166 | 0.197 | 0.156 | domain gate PASS |

Original-domain C-CRP v3 formal reports are under
`outputs/ccrp_v3_formal/<domain>/report.json`; the four-domain comparison with
the canonical official baseline block is
`outputs/ccrp_v3_formal/main_comparison_table.csv`. New-domain artifact
completeness: each of sports/toys/home/tools has `report.json`, `scores.csv`
with 1,010,000 candidate-score rows plus header, and `user_ranks.jsonl` with
10,000 user-rank rows.

New-domain imported-table note (updated 2026-06-05): sports, toys, home, and
tools C-CRP v3 imported same-candidate evidence now passes domain gates against
all eight official baselines. For toys, raw scores under
`outputs/toys_large10000_100neg_ccrp_v3`
were imported into
`outputs/toys_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate`
with exact coverage (`score_coverage_rate=1.0`). After toys `llmesr_sasrec`
finished, `outputs/summary/toys_official_gate_final_20260602_1900.{json,csv}`
recorded `ccrp_ok=true`, `official_ok_count=8`, `official_all_ok=true`, and
`gate_ok=true`. The follow-up toys comparison/statistical gate records C-CRP
rank 1 on all seven metrics and 56/56 positive Holm-significant paired tests.
For home, raw scores under `outputs/home_large10000_100neg_ccrp_v3` were
imported into
`outputs/home_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate`
with exact coverage, and
`outputs/summary/home_official_ccrp_gate_20260604.{json,csv}` plus
`outputs/summary/home_official_ccrp_20260604_*` record the passed Home domain
gate and 56/56 positive Holm-significant paired tests. For tools, raw scores
under `outputs/tools_large10000_100neg_ccrp_v3` were imported into
`outputs/tools_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate`
with exact coverage, and
`outputs/summary/tools_official_ccrp_gate_post_cleanup_20260605.{json,csv}`
plus `outputs/summary/tools_official_ccrp_20260605_*` record the passed Tools
domain gate and 56/56 positive Holm-significant paired tests.

Artifact audit note (2026-05-31): the old four-domain C-CRP reports are present
under `outputs/ccrp_v3_formal/<domain>/report.json`; they were only missed by
flat-path checks such as `outputs/*ccrp_v3/report.json`. The old four-domain
8-baseline comparison table is metric-complete, but some method-specific
old-domain baseline directories are table-only under the current strict
evidence gate because final provenance/audit files are not co-located with the
imported tables. Treat this as artifact reconciliation before paper submission,
not as a reason to silently rerun completed metric rows.

### Experiment Execution Plan

1. C-CRP v3 on all 8 domains (Phase 1) — complete
2. 8 official baselines on 4 new domains (Phase 2) — complete for sports,
   toys, home, and tools: each has all eight audited official rows plus C-CRP
   import, domain/comparison, and paired-test gates complete.
3. Paper-critical modules (Phase 2.5/3 gate) — observation/motivation figure,
   C-CRP component ablations, hyperparameter curves, and framework overview
   figure.
   2026-06-12 update: observation/motivation, four-domain component-ablation,
   and four-domain hyperparameter stability packages are closed as
   paper-critical supporting evidence under their scoped labels. Use
   `scripts/analysis/main_build_ccrp_hyperparameter_sweep.py` to build
   valid/test sweeps from audited saved signal rows, then
   `scripts/analysis/main_plot_ccrp_hyperparameter_sweep.py` with the sweep
   provenance. Main controls are `eta` and `weight_grid_label`;
   `confidence_weight` is diagnostic-only for `confidence_plus_evidence`.
   The completed aggregate at
   `outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/ccrp_hyperparameter_four_domain/`
   reports `all_controls_stable=true` for NDCG@10 and must stay limited to
   supplementary hyperparameter stability/sensitivity.
4. Full comparison table + statistical tests (Phase 3)
5. Paper writing with ARIS skill (Phase 4)
6. GPT-5.5/Codex review cycle until 8/10 (Phase 5)

### Server State

- Batch script complete: `run_ccrp_v3_all_new_domains.sh` (sports/toys/home/tools)
- Phase 2 sports official-baseline run started 2026-05-31 and is now complete:
  all eight sports official rows passed final provenance/coverage/import gates.
  The last separate LLM-ESR runner (`2877443`/`2877452`) finished at
  2026-06-01 18:31 CST.
- Phase 2 toys official-baseline run is complete for all eight rows. The final
  row, `llmesr_sasrec`, first hit `OSError: [Errno 28] No space left on device`
  while copying the 3.3G LLM-ESR handled item embedding into the upstream repo;
  recovery removed only verified non-final/failed-copy storage and changed the
  adapter to symlink handled files. The recovery finished at 2026-06-02
  18:59 CST with `implementation_status=official_completed`, `blockers=[]`,
  `score_coverage_rate=1.0`, full metrics HR@5/10/20
  `0.0637 / 0.1172 / 0.2203`, NDCG@5/10/20
  `0.037504900117522603 / 0.05456849726033091 / 0.08036871527121744`, MRR
  `0.05844977379835533`, and aligned row counts (`scores.csv` `1,010,001`,
  predictions `10,000`, `tables/ranking_eval_records.csv` `10,001`).
  Server-final audit, lightweight sync, and local-light audit all passed.
  The lightweight evidence package is
  `outputs/baselines/official_adapters/toys_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/`;
  server-only `scores.csv`, predictions, and `llmesr_official_model.pt` are
  covered by `server_large_artifact_manifest.sha256`. After gates passed, the
  completed intermediate adapter was removed with cleanup manifest
  `outputs/summary/toys_llmesr_completed_adapter_cleanup_manifest_20260602.sha256`,
  recovering disk from about `1.6G` to `5.9G` free while preserving final
  evidence. The final toys domain gate and comparison/statistical gate passed
  under `outputs/summary/toys_official_gate_final_20260602_1900.*` and
  `outputs/summary/toys_official_ccrp_20260602_1900_*`.
- Storage/home launch checkpoint 2026-06-02 19:50 CST: after verifying
  sports/toys LLMEmb and LLM-ESR final server audits plus local-light packages,
  three completed upstream staging directories were removed with realpath
  allowlist checks and a sha256 cleanup manifest:
  `outputs/summary/upstream_completed_sports_toys_llmemb_llmesr_cleanup_manifest_20260602.sha256`.
  The removed directories were
  `/home/ajifang/projects/LLMEmb/data/sports_llmemb_same_candidate_100neg`,
  `/home/ajifang/projects/LLMEmb/data/toys_llmemb_same_candidate_100neg`, and
  `/home/ajifang/projects/LLM-ESR/data/sports_same_candidate_100neg`. Final
  evidence directories and local lightweight packages were preserved. Disk
  recovered from about `5.9G` to `17G` free. The safer local
  `scripts/run_baselines_new_domains.sh` was copied to the server to preserve
  method-token validation. Home `proex_profile` then launched as a single-row
  official loop with runner PID `3004208`, adapter PID `3004218`, and log
  `baselines_new_domains_home_proex_20260602_1950.log`. It completed at
  2026-06-02 22:00 CST with `implementation_status=official_completed`,
  `blockers=[]`, exact `score_coverage_rate=1.0`, server-final audit PASS,
  lightweight sync PASS, and local-light audit PASS. Full metrics over 10,000
  users and 101 candidates are HR@5/10/20 `0.0606 / 0.1177 / 0.2296`,
  NDCG@5/10/20
  `0.03662857786324662 / 0.054867449700296195 / 0.08290060869107069`, and MRR
  `0.05933326491258513`; row counts passed for `scores.csv` (`1,010,001`
  lines), predictions (`10,000` lines), and `tables/ranking_eval_records.csv`
  (`10,001` lines). The local lightweight evidence package is
  `outputs/baselines/official_adapters/home_large10000_100neg_proex_profile_official_qwen3base_same_candidate/`.
  The completed intermediate adapter was removed after final/server/local gates
  passed and a 27-file sha256 cleanup manifest was written:
  `outputs/summary/home_proex_completed_adapter_cleanup_manifest_20260602.sha256`.
  Disk recovered from about `8.2G` to `16G` free without touching final
  evidence. Home `promax_profile` then launched at 2026-06-02 22:14 CST as the
  second home official row with runner PID `3026043`, adapter PID `3026052`,
  and log `baselines_new_domains_home_promax_20260602_2215.log`. It completed
  at 2026-06-03 02:53 CST as `implementation_status=official_completed` with
  `blockers=[]`, exact `score_coverage_rate=1.0`, server-final audit PASS,
  lightweight sync PASS, and local-light audit PASS. Full metrics over 10,000
  users and 101 candidates are HR@5/10/20 `0.0514 / 0.1019 / 0.2076`,
  NDCG@5/10/20
  `0.030788292596664168 / 0.04691808776215203 / 0.07326077825489297`, and MRR
  `0.053474908740382465`; row counts passed for `scores.csv` (`1,010,001`
  lines), predictions (`10,000` lines), and `tables/ranking_eval_records.csv`
  (`10,001` lines). The local lightweight evidence package is
  `outputs/baselines/official_adapters/home_large10000_100neg_promax_profile_official_qwen3base_same_candidate/`.
  The completed intermediate adapter was removed after final/server/local gates
  passed and a 22-file sha256 cleanup manifest was written:
  `outputs/summary/home_promax_completed_adapter_cleanup_manifest_20260602.sha256`.
  Disk recovered from about `7.5G` to `15G` free without touching final
  evidence. Home `elmrec_graph` then launched at 2026-06-03 03:02 CST as the
  third home official row with runner PID `3061705`, adapter PID `3061714`,
  and log `baselines_new_domains_home_elmrec_20260603_0302.log`. It completed
  at 2026-06-03 05:47 CST as `implementation_status=official_completed` with
  `blockers=[]`, exact `score_coverage_rate=1.0`, server-final audit PASS,
  lightweight sync PASS, and local-light audit PASS. Full metrics over 10,000
  users and 101 candidates are HR@5/10/20 `0.0509 / 0.1021 / 0.2018`,
  NDCG@5/10/20
  `0.029717257242599254 / 0.0460440741915887 / 0.0708856096588022`, and MRR
  `0.05195852255617441`; row counts passed for `scores.csv` (`1,010,001`
  lines), predictions (`10,000` lines), and `tables/ranking_eval_records.csv`
  (`10,001` lines). The local lightweight evidence package is
  `outputs/baselines/official_adapters/home_large10000_100neg_elmrec_graph_official_qwen3base_same_candidate/`.
  At the 2026-06-03 05:48 CST checkpoint no Pony/C-CRP/baseline Python process
  was active, GPU was idle, and disk was tight at about `6.5G` free; the next
  home row needed a fresh preflight and cleanup/storage decision for the
  completed ElmRec intermediate adapter. The completed intermediate adapter was
  then removed after exact realpath checks and a 16-file sha256 cleanup
  manifest
  `outputs/summary/home_elmrec_completed_adapter_cleanup_manifest_20260603.sha256`;
  a post-cleanup server-final audit remained `ok=true`, and final scores,
  provenance, audits, predictions, imported tables, model, and local
  lightweight evidence were preserved. Disk recovered from about `6.5G` to
  `14G` free. After a clean process/GPU/disk and no-existing-artifact preflight,
  home `llmemb` launched at 2026-06-03 06:08 CST as the fourth home row with
  `FAST_METHODS_OVERRIDE=llmemb`, active adapter PID `3085786`, PID file
  `baselines_new_domains_home_llmemb_adapter.pid`, and log
  `baselines_new_domains_home_llmemb_20260603_0608.log`. At first stable check
  it was in Qwen3 `hf_mean_pool` embedding at about `2808/385364`, GPU was
  `96%` with `16067 MiB / 49140 MiB`, disk was about `13G` free, and no final
  scores/provenance existed yet.
- Monitoring cadence updated 2026-06-01: no separate monitor automation is
  required while the active thread goal is running. Each continuation performs
  bounded read-only status checks, records material evidence changes, and must
  not start duplicate experiments.
- Monitoring checkpoint 2026-05-31 21:42 CST: runner PID `2794722` and child
  PID `2794731` are active; `llmemb` is encoding Qwen3 item/user text at about
  `28048/233470`; no baseline score/audit/import files are expected yet
  because the first baseline has not completed
- Monitoring checkpoint 2026-05-31 21:46 CST: runner PID `2794722` and child
  PID `2794731` are still active; `llmemb` is encoding at about
  `48056/233470`; GPU is about `100%` with `16285 MiB / 49140 MiB`; disk is
  still about `44G` free; no sports baseline has completed `scores.csv`,
  audit, import table, or final fairness provenance yet
- Monitoring checkpoint 2026-05-31 21:52 CST: runner PID `2794722` and child
  PID `2794731` are still active; `llmemb` is encoding at about
  `78888/233470`; GPU is about `100%` with `16285 MiB / 49140 MiB`; disk is
  still about `44G` free; all 8 sports official baseline rows still have no
  completed `scores.csv`, score-audit log, imported summary table, or final
  fairness provenance because the first baseline has not finished
- Monitoring checkpoint 2026-05-31 21:56 CST: runner PID `2794722` and child
  PID `2794731` are still active; `llmemb` is encoding at about
  `94488/233470`; elapsed time is about `24m`; GPU is about `96%` with
  `16285 MiB / 49140 MiB`; disk is still about `44G` free; recent log scan
  shows no error/blocker keywords; no sports row has completed
  `scores.csv`/audit/import/final provenance yet
- Monitoring checkpoint 2026-05-31 22:01 CST: runner PID `2794722` and child
  PID `2794731` are still active; `llmemb` is encoding at about
  `114984/233470`; elapsed time is about `29m`; GPU is about `95%` with
  `16285 MiB / 49140 MiB`; disk is still about `44G` free; recent log scan
  shows no error/blocker keywords; sports artifact matrix remains incomplete
  for all 8 official methods because `llmemb` has not finished scoring
- Monitoring checkpoint 2026-05-31 22:06 CST: runner PID `2794722` and child
  PID `2794731` are still active; `llmemb` is encoding at about
  `134848/233470`; elapsed time is about `33m`; GPU is about `95%` with
  `16285 MiB / 49140 MiB`; disk is still about `44G` free; recent log scan
  shows no error/blocker keywords; sports artifact matrix remains incomplete
  for all 8 official methods and only `llmemb` inspect provenance is present
- Monitoring checkpoint 2026-05-31 22:17 CST: active process is
  `main_run_llmemb_official_same_candidate_adapter.py` for sports `llmemb`
  (PID `2794731`); log progress is about `185136/233470`; GPU is about `96%`
  with `16285 MiB / 49140 MiB`; disk remains about `44G` free. The sports
  official artifact matrix is still incomplete for all eight canonical
  methods (`llmemb`, `llm2rec_sasrec`, `irllrec_intent`, `rlmrec_graphcl`,
  `proex_profile`, `promax_profile`, `llmesr_sasrec`, `elmrec_graph`):
  no completed `scores.csv`, score-audit JSON, imported table, or final
  `fairness_provenance.json` is present yet. This is expected while the first
  baseline is still embedding/scoring.
- Monitoring checkpoint 2026-05-31 22:32 CST: sports `llmemb` has completed
  the `hf_mean_pool` embedding pass (`233470/233470`) and advanced into
  `llmemb-sasrec` training, with log lines through epoch 35. PID `2794731`
  remains active under runner `2794722`; GPU is about `33%` with
  `16301 MiB / 49140 MiB`; disk has dropped to about `36G` free (`81%` used).
  No `scores.csv`, score-audit JSON, imported table, or final
  `fairness_provenance.json` exists yet for any of the eight sports official
  methods, so no baseline row is complete or table-eligible.
- Monitoring checkpoint 2026-05-31 22:52 CST: sports `llmemb` is still active
  under runner PID `2794722` and child PID `2794731`; `llmemb-sasrec` completed
  200 epochs and the `llmemb` training phase has reached epoch 175. GPU is
  about `83%`, `16301 MiB / 49140 MiB`; disk remains about `36G` free (`81%`
  used). Recent log and artifact scans show no `ERROR`, `WARN`, `Traceback`,
  `Killed`, `OOM`, `CUDA out`, or `FAILED`, and no sports official method has
  completed `scores.csv`, score-audit JSON, imported table, `report.json`, or
  final `fairness_provenance.json` yet.
- Completion checkpoint 2026-05-31 22:56 CST: sports `llmemb` reached
  `implementation_status=official_completed` with `blockers=[]`,
  `score_coverage_rate=1.0`, and exact same-candidate audit
  `audit_ok=True`. Full metrics are HR@5/10/20=`0.2124/0.3384/0.4900`,
  NDCG@5/10/20=`0.1388527216/0.1795004215/0.2176868359`, and
  MRR=`0.1538831336` over 10,000 users and 1,010,000 candidate scores.
  `scores.csv` has 1,010,001 lines including header; `rank_predictions.jsonl`
  has 10,000 rows. Lightweight evidence is backed up locally under
  `outputs/baselines/official_adapters/sports_large10000_100neg_llmemb_official_qwen3base_same_candidate/`.
  Large server-only artifacts are left on the server:
  `scores.csv`, `predictions/rank_predictions.jsonl`, and
  `llmemb_official_model.pt`. The runner has advanced to `proex_profile` on
  sports (active child PID `2805588`). Disk is now about `31G` free (`84%`
  used), so storage is a watch item for the next monitor cycle.
- Monitoring checkpoint 2026-05-31 23:29 CST: runner PID `2794722` is active
  and sports `proex_profile` child PID `2805588` is encoding with
  `hf_mean_pool` progress about `135560/233470`. GPU is about `95%` with
  `16285 MiB / 49140 MiB`; disk is about `31G` free (`84%` used). Recent log
  scan shows no `ERROR`, `WARN`, `Traceback`, `Killed`, `OOM`, `CUDA out`, or
  `FAILED` markers. Sports matrix remains: `llmemb` official-completed; the
  other seven canonical official baselines not yet complete.
- Completion checkpoint 2026-06-01 00:25 CST: sports `proex_profile`
  completed as `implementation_status=official_completed` with `blockers=[]`,
  `score_coverage_rate=1.0`, and exact same-candidate audit `audit_ok=True`.
  Full metrics are HR@5/10/20=`0.0821/0.1527/0.2777`,
  NDCG@5/10/20=`0.0516826556/0.0741722663/0.1054064715`, and
  MRR=`0.0742689715` over 10,000 users and 1,010,000 candidate scores.
  `scores.csv` has 1,010,001 lines including header; `rank_predictions.jsonl`
  has 10,000 rows. Lightweight evidence is backed up locally under
  `outputs/baselines/official_adapters/sports_large10000_100neg_proex_profile_official_qwen3base_same_candidate/`.
  Large server-only artifacts are left on the server:
  `scores.csv` and `predictions/rank_predictions.jsonl`. The runner advanced
  to `promax_profile` on sports (child PID `2816461` at the 00:26 CST
  checkpoint). Disk is now about `26G` free (`87%` used), so storage pressure
  remains a watch item.
- Completion checkpoint 2026-06-01 03:04 CST: sports `promax_profile`
  completed as `implementation_status=official_completed` with `blockers=[]`,
  `score_coverage_rate=1.0`, and exact same-candidate audit `audit_ok=True`.
  Full metrics are HR@5/10/20=`0.0825/0.1387/0.2370`,
  NDCG@5/10/20=`0.0541847954/0.0721533411/0.0967593591`, and
  MRR=`0.0741052747` over 10,000 users and 1,010,000 candidate scores.
  `scores.csv` has 1,010,001 lines including header; `rank_predictions.jsonl`
  has 10,000 rows. Lightweight evidence is backed up locally under
  `outputs/baselines/official_adapters/sports_large10000_100neg_promax_profile_official_qwen3base_same_candidate/`.
  Large server-only artifacts are left on the server:
  `scores.csv`, `predictions/rank_predictions.jsonl`, and
  `promax_official_model.pt`. The runner advanced to `elmrec_graph` on sports
  (child PID `2828395` at the 03:05 CST checkpoint). Disk is now about `20G`
  free (`90%` used), so storage pressure is a close watch item.
- Completion checkpoint 2026-06-01 04:37 CST: sports `elmrec_graph`
  completed as `implementation_status=official_completed` with `blockers=[]`,
  `score_coverage_rate=1.0`, and exact same-candidate audit `audit_ok=True`.
  Full metrics are HR@5/10/20=`0.0532/0.1054/0.2013`,
  NDCG@5/10/20=`0.0317045493/0.0483716358/0.0723504733`, and
  MRR=`0.0537009851` over 10,000 users and 1,010,000 candidate scores.
  `scores.csv` has 1,010,001 lines including header; `rank_predictions.jsonl`
  has 10,000 rows. Lightweight evidence is backed up locally under
  `outputs/baselines/official_adapters/sports_large10000_100neg_elmrec_graph_official_qwen3base_same_candidate/`.
  Large server-only artifacts are left on the server:
  `scores.csv`, `predictions/rank_predictions.jsonl`,
  `elmrec_official_model.pt`, and Qwen3 item embedding intermediates recorded
  by provenance. The runner advanced to `irllrec_intent` on sports (child PID
  `2835275` at the 04:38 CST checkpoint). Disk is now about `15G` free (`93%`
  used), so storage pressure is a close watch item but no `No space left`,
  OOM, or CUDA failure has been observed.
- Evidence packaging and cleanup checkpoint 2026-06-01 05:31 CST: local
  sports evidence for the four completed official rows
  (`llmemb`, `proex_profile`, `promax_profile`, `elmrec_graph`) now includes
  inspect provenance, final provenance, JSON/TXT score audits, run summaries,
  metric/coverage/exposure/summary tables, and per-event
  `tables/ranking_eval_records.csv` for later paired/statistical checks.
  Local/server file-size and line-count checks matched for the copied evidence;
  each `ranking_eval_records.csv` has 10,001 lines including header. After the
  local check, server-side completed-method working directories under
  `outputs/baselines/paper_adapters/` were removed for those four methods only.
  Final server outputs (`scores.csv`, `fairness_provenance.json`, score audits,
  imported tables, predictions, and compact checkpoints) remain in place. Disk
  recovered from about `15G` free (`93%` used) to about `33G` free (`83%`
  used). The active sports `irllrec_intent` adapter directory remains present.
- Monitoring/tooling checkpoint 2026-06-01 06:36 CST: runner PID `2794722`
  and sports `irllrec_intent` child PID `2835275` are still active. The log
  has reached epoch `1190` of the default `3000` IRLLRec official-adapter
  epochs, with latest train loss `0.625393`; no `Traceback`, `Killed`, OOM,
  CUDA, no-space, or fatal markers were found. GPU is about `98%` with
  `16295 MiB / 49140 MiB`; disk is about `29G` free (`85%` used). Based on
  observed epoch rate, IRLLRec training likely has roughly 3 more hours before
  scoring/import/audit overhead, but the next status source remains the log
  and final provenance, not the estimate. Four sports official rows remain
  complete (`llmemb`, `proex_profile`, `promax_profile`, `elmrec_graph`);
  `irllrec_intent`, `rlmrec_graphcl`, `llm2rec_sasrec`, and `llmesr_sasrec`
  are not complete yet. A new read-only package gate,
  `scripts/audit/main_audit_official_evidence_package.py`, was added and
  passed on all four local lightweight sports evidence packages. It checks
  final provenance, blockers, score coverage, full `@5/@10/@20 + MRR`
  metrics, row counts, score audits, run summaries, and per-event evaluation
  records before a copied package is treated as safely backed up.
- Server-final evidence audit 2026-06-01 06:44 CST: the same package gate was
  copied to `/tmp/pony_audit_official_evidence_package.py` on `pony-rec-gpu`
  and run in `server_final` mode against the four completed sports official
  output directories. All four passed, including server-side `scores.csv`
  line count `1,010,001`, `predictions/rank_predictions.jsonl` line count
  `10,000`, final provenance, score audits, full metrics, coverage/exposure
  tables, and per-event evaluation records. The active `irllrec_intent`
  process was not stopped, restarted, or modified.
- Monitoring checkpoint 2026-06-01 06:52 CST: sports `irllrec_intent` is still
  active under runner PID `2794722` and child PID `2835275`. The log has
  reached epoch `1520/3000`, latest train loss `0.624800`; GPU is about
  `99%`, `16295 MiB / 49140 MiB`, and disk remains about `29G` free (`85%`
  used). Fatal scan remains clean: no `Traceback`, `Killed`, OOM, CUDA,
  no-space, disk quota, exception, or runtime-error markers. No new sports
  official baseline row is final yet beyond the four already audited rows.
- Monitoring checkpoint 2026-06-01 07:18 CST: sports `irllrec_intent` remains
  active under runner PID `2794722` and child PID `2835275`. The log has
  reached epoch `2040/3000`, latest train loss `0.624872`; GPU is about
  `75%`, `16295 MiB / 49140 MiB`, and disk remains about `29G` free (`85%`
  used). Fatal scan remains clean: no `Traceback`, `Killed`, OOM, CUDA,
  no-space, disk quota, exception, or runtime-error markers. The sports
  official matrix is unchanged: four rows are final/audited, while
  `irllrec_intent`, `rlmrec_graphcl`, `llm2rec_sasrec`, and `llmesr_sasrec`
  are not final yet.
- Preflight audit 2026-06-01 07:21 CST: while IRLLRec continued running,
  the local and server unified official-runner dispatch were checked for the
  remaining sports methods. Server `adapters.py` contains real `run` branches
  for `rlmrec`, `llm2rec`, and `llmesr`, and each corresponding sports
  inspect provenance is `official_inspection_ready` with `blockers=[]`, a
  pinned official repo commit, and aligned train/valid/test/candidate task
  sources. The server repo is an experiment workspace with active dirty state,
  so no server pull, reset, cleanup, or process action was performed during
  the live runner.
- Evidence-sync tooling checkpoint 2026-06-01 07:26 CST:
  `scripts/audit/main_sync_official_evidence_package.py` was added as a
  local-side allowlist sync and server/local checksum verifier for official
  evidence packages. It excludes `scores.csv`, `predictions/`, checkpoints,
  embeddings, and other large binary artifacts by default, while copying and
  verifying final provenance, inspect provenance, score audits, run summaries,
  imported tables, and compact manifests. `python -m py_compile` passed, and a
  no-copy verification against the completed sports `llmemb` package matched
  10 lightweight files by size and sha256 while excluding the server-only
  score file, predictions file, and large checkpoint.
- Monitoring checkpoint 2026-06-01 07:32 CST: sports `irllrec_intent` remains
  active under runner PID `2794722` and child PID `2835275`. The log has
  reached epoch `2320/3000`, latest train loss `0.625049`; GPU is about
  `69%`, `16295 MiB / 49140 MiB`, and disk remains about `29G` free (`85%`
  used). Fatal scan remains clean: no `Traceback`, `Killed`, OOM, CUDA,
  no-space, disk quota, exception, or runtime-error markers. No final
  `fairness_provenance.json`, `scores.csv`, score audit, imported tables, or
  predictions exist yet for `irllrec_intent`, so no fifth sports official row
  has been recorded.
- Completion checkpoint 2026-06-01 08:10 CST: sports `irllrec_intent`
  completed as `implementation_status=official_completed` with `blockers=[]`,
  `score_coverage_rate=1.0`, exact same-candidate score coverage, and full
  imported metrics. Metrics: HR@5/10/20=`0.1573/0.2215/0.4016`,
  NDCG@5/10/20=`0.10642150916142634/0.12691703149297534/0.17128490034441315`,
  MRR=`0.12444202662842994`, `sample_count=10000`,
  `avg_candidates=101.0`, `score_rows=1010000`, and
  `candidate_rows=1010000`. Server row counts passed: `scores.csv` has
  `1,010,001` lines including header, `predictions/rank_predictions.jsonl`
  has `10,000` lines, and `tables/ranking_eval_records.csv` has `10,001`
  lines. Server-final audit, lightweight sync, and local-light audit all
  passed. Local sync copied 11 lightweight files with matching size/sha256,
  including server-final and local-light evidence audit JSONs; `scores.csv`,
  predictions, and `irllrec_official_model.pt` remain server-only. After the
  local backup and audits, the server intermediate adapter directory
  `outputs/baselines/paper_adapters/sports_large10000_100neg_irllrec_official_adapter`
  was removed, recovering disk from about `28G` free (`86%` used) to `32G`
  free (`83%` used). Final IRLLRec outputs were preserved. The runner advanced
  to sports `rlmrec_graphcl` child PID `2851207`; at 2026-06-01 08:18 CST it
  was generating Qwen embeddings at `35496/233470`, with no fatal markers.
- Monitoring/tooling checkpoint 2026-06-01 08:24 CST: sports `rlmrec_graphcl`
  remains active under runner PID `2794722` and child PID `2851207`. The log
  has reached Qwen embedding progress `57992/233470`; GPU is about `95%` with
  `16285 MiB / 49140 MiB`, disk is `32G` free (`83%` used), and fatal/OOM/CUDA/
  no-space scans remain clean. A read-only server scan found one empty malformed
  output directory,
  `outputs/sports_large10000_100neg_TRAIN_METHODS_OVERRIDE=_official_qwen3base_same_candidate/`;
  it has no files and is not evidence. The local runner now validates method
  tokens before creating output directories, preventing misquoted override
  tokens from creating malformed method directories in future launches. No
  server pull or cleanup was performed while the live runner is active.
- Monitoring checkpoint 2026-06-01 08:34 CST: sports `rlmrec_graphcl` remains
  active under runner PID `2794722` and child PID `2851207`. The log has reached
  Qwen embedding progress `105376/233470`; GPU is about `94%` with
  `16285 MiB / 49140 MiB`, disk is `32G` free (`83%` used), and fatal/OOM/CUDA/
  no-space scans remain clean. The RLMRec output directory still contains only
  `inspect_fairness_provenance.json`, so the row is not complete and must not
  be audited, synced, or recorded as a result yet.
- Monitoring checkpoint 2026-06-01 09:05 CST: sports `rlmrec_graphcl` reached
  Qwen embedding progress `233470/233470`. The child process `2851207` remains
  active after embedding completion with high CPU/memory and momentary GPU
  utilization `0%`, consistent with post-embedding graph or training
  preparation. Disk remains `32G` free (`83%` used), fatal/OOM/CUDA/no-space
  scans remain clean, and the RLMRec output directory still contains only
  `inspect_fairness_provenance.json`; no final scores/provenance/table package
  exists yet, so the row remains ineligible for audit, sync, or table import.
- Monitoring checkpoint 2026-06-01 09:08 CST: sports `rlmrec_graphcl` entered
  official training after embedding completion. The latest logged training
  line is `[rlmrec-official] epoch=10 train_loss=1.675038`; default RLMRec
  training is `3000` epochs with log interval `10`. GPU utilization is `100%`
  with `19943 MiB / 49140 MiB`; disk is `28G` free (`85%` used). Graph
  normalization emitted a non-fatal zero-degree inverse-degree warning already
  seen in earlier completed graph baselines and handled by setting `inf`
  inverse degrees to `0.0`. No final RLMRec score/provenance/table package
  exists yet.
- Monitoring checkpoint 2026-06-01 09:19 CST: sports `rlmrec_graphcl` is
  still training normally under child PID `2851207`. The latest logged line is
  `[rlmrec-official] epoch=140 train_loss=1.490221`; loss continues to decline,
  GPU is `100%` with `19943 MiB / 49140 MiB`, disk remains `28G` free (`85%`
  used), and fatal/OOM/CUDA/no-space scans remain clean. The output directory
  still contains only `inspect_fairness_provenance.json`, so no server-final
  audit, local sync, result table, or sixth sports official row is available.
- Monitoring checkpoint 2026-06-01 09:53 CST: sports `rlmrec_graphcl` passed
  the 500-epoch training checkpoint and reached
  `[rlmrec-official] epoch=510 train_loss=1.482085`. The epoch-500 loss was
  `1.480699`. The process remains active under child PID `2851207`; GPU was
  `36%` at sample time with `19943 MiB / 49140 MiB`, disk remained `28G` free
  (`85%` used), and fatal/OOM/CUDA/no-space scans remained clean. The output
  directory still contains only `inspect_fairness_provenance.json`; no final
  scores/provenance/audit/table package exists, so sports official evidence
  remains five completed rows.
- Monitoring checkpoint 2026-06-01 10:41 CST: sports `rlmrec_graphcl` passed
  the 1000-epoch training checkpoint and reached
  `[rlmrec-official] epoch=1030 train_loss=1.478778`. The epoch-1000 loss was
  `1.477797`. The process remains active under child PID `2851207`; GPU was
  `6%` at sample time with `19943 MiB / 49140 MiB`, disk remained `28G` free
  (`85%` used), and fatal/OOM/CUDA/no-space scans remained clean. The output
  directory still contains only `inspect_fairness_provenance.json`; no final
  scores/provenance/audit/table package exists, so sports official evidence
  remains five completed rows.
- Monitoring checkpoint 2026-06-01 12:10 CST: sports `rlmrec_graphcl` passed
  the 2000-epoch training checkpoint with
  `[rlmrec-official] epoch=2000 train_loss=1.476514`. The process remains
  active under child PID `2851207`; GPU was `70%` at sample time with
  `19943 MiB / 49140 MiB`, disk remained `28G` free (`85%` used), and
  fatal/OOM/CUDA/no-space scans remained clean. The output directory still
  contains only `inspect_fairness_provenance.json`; no final
  scores/provenance/audit/table package exists, so sports official evidence
  remains five completed rows.
- Completion/recovery checkpoint 2026-06-01 13:50 CST: sports
  `rlmrec_graphcl` completed as `implementation_status=official_completed`
  with `blockers=[]`, `score_coverage_rate=1.0`, server-final audit PASS,
  lightweight sync PASS, and local-light audit PASS. Full metrics:
  HR@5/10/20=`0.1212/0.1879/0.3009`,
  NDCG@5/10/20=`0.078580389191345/0.10001773336299705/0.12818232277286493`,
  MRR=`0.09720456858848743`, `sample_count=10000`, `avg_candidates=101.0`,
  and `score_rows=1010000`. Server row counts passed: `scores.csv` has
  `1,010,001` lines, predictions have `10,000` lines, and
  `tables/ranking_eval_records.csv` has `10,001` lines. The runner then
  advanced to sports `llm2rec_sasrec` and stopped during adapter export with an
  empty validation-history error for a valid-only user. Local fix and targeted
  unit test are in place; no baseline process is currently active, GPU is idle,
  and disk is about `28G` free.
- LLM2Rec recovery checkpoint 2026-06-01 14:03 CST: commit `657929e` was
  pushed locally, the fixed LLM2Rec exporter was copied to the dirty server
  worktree without resetting unrelated experiment changes, and sports
  `llm2rec_sasrec` was resumed as a single-row job rather than rerunning the
  full sports batch. Active PID is `2870575`, log path is
  `baselines_new_domains_sports_llm2rec_resume.log`, and the real sports run
  passed the previous adapter-export blocker. It is now in Qwen3
  `hf_mean_pool` embedding generation at about `3432/283760`; GPU is `100%`,
  memory is `16115 MiB / 49140 MiB`, and disk is about `27G` free. This is not
  a completed row yet.
- Storage/progress checkpoint 2026-06-01 14:09 CST: the sports LLM2Rec adapter
  package passed the server adapter audit as
  `ready_for_llm2rec_upstream_wrapper` with `valid_history_source` equal to
  `valid_task_train_interactions`, `10000` validation events, `10000` test
  events, `1010000` candidate rows, and zero missing mapped candidates.
  LLM2Rec embedding progress reached about `28736/283760`; GPU remained about
  `95%` with `16213 MiB / 49140 MiB`. After verifying RLMRec
  `server_final_evidence_audit.json` had `ok=true` and the absolute path was
  inside the project paper-adapter directory, the completed RLMRec intermediate
  adapter directory was removed, recovering about `4.5G`; final RLMRec evidence
  outputs and local lightweight package were preserved.
- LLM2Rec training-launch blocker checkpoint 2026-06-01 15:14 CST: sports
  `llm2rec_sasrec` completed Qwen3 item embedding (`283760/283760`) and wrote
  matching 4,649,140,352-byte embedding files at
  `outputs/baselines/paper_adapters/sports_large10000_100neg_llm2rec_official_adapter/llm2rec_item_embeddings.npy`
  and
  `/home/ajifang/projects/LLM2Rec/item_info/SportsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`.
  It then stopped before official SASRec training with
  `FileNotFoundError: [Errno 2] No such file or directory: 'python'` from
  `_train_with_official_entrypoint`. Local fix changes the official-entrypoint
  subprocess command to start with `sys.executable`, preserving the
  `evaluate_with_seqrec.py` entrypoint and SASRec arguments. Targeted local
  tests passed: `tests/test_llm2rec_upstream_adapter.py` (`5 passed`) and
  `tests/test_llm2rec_same_candidate_export.py` (`3 passed`). Next action:
  copy the fixed runner to the dirty server worktree, run `py_compile`, and
  resume only LLM2Rec with the existing upstream embedding path.
- LLM2Rec resume checkpoint 2026-06-01 15:51 CST: the fixed runner was present
  on the server and passed `py_compile`. The first direct single-row launch
  timed out locally but did start sports `llm2rec_sasrec`; a follow-up safety
  launcher detected the active process and refused to duplicate it. Active
  processes are adapter PID `2875446` and upstream official
  `evaluate_with_seqrec.py` PID `2875559`. The upstream official command uses
  the existing embedding path
  `/home/ajifang/projects/LLM2Rec/item_info/SportsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`.
  `llm2rec_official_training.log` reached epoch 15 validation and saved epoch
  5/10 checkpoints. GPU sample was `7%`, `9363 MiB / 49140 MiB`, and disk had
  `22G` free. No final scores/provenance/audit/imported metrics exist yet, so
  LLM2Rec remains running and not table-eligible.
- LLM2Rec completion checkpoint 2026-06-01 15:56 CST: sports
  `llm2rec_sasrec` completed as the seventh sports official row with
  `implementation_status=official_completed`, `blockers=[]`,
  `score_coverage_rate=1.0`, server-final audit PASS, lightweight sync PASS,
  and local-light audit PASS. Official training early-stopped at epoch 45 and
  loaded the best epoch 25 checkpoint. Full metrics over 10,000 users and 101
  candidates: HR@5/10/20=`0.1105/0.206/0.3657`,
  NDCG@5/10/20=`0.06514778914391295/0.09566791850988236/0.13561659669926907`,
  MRR=`0.08828933028385053`. Row counts passed: `scores.csv` has
  `1,010,001` lines, predictions have `10,000` lines, and
  `tables/ranking_eval_records.csv` has `10,001` lines. Disk is now `17G`
  free (`91%` used), so preflight cleanup/disk review is needed before
  launching sports `llmesr_sasrec`.
- Storage/LLM-ESR launch checkpoint 2026-06-01 16:28 CST: the completed
  LLM2Rec intermediate adapter directory
  `outputs/baselines/paper_adapters/sports_large10000_100neg_llm2rec_official_adapter`
  was removed after LLM2Rec server-final and local-light audits passed and
  final evidence paths were rechecked. This recovered about `5.3G`, raising
  free disk to about `23G`, and did not touch final LLM2Rec scores,
  provenance, audits, imported tables, predictions, checkpoints, or the
  upstream embedding under `/home/ajifang/projects/LLM2Rec/item_info/`.
  Sports `llmesr_sasrec` was then launched as the final sports official row
  with runner PID `2877443` and adapter PID `2877452`. It is currently in Qwen3
  `hf_mean_pool` embedding at about `51472/233470`; GPU sample was `95%`,
  `16285 MiB / 49140 MiB`, and disk had `22G` free. LLM-ESR is running and
  not table-eligible until final score/provenance/audit/import gates pass.
- Monitoring/gate checkpoint 2026-06-01 16:50 CST: sports `llmesr_sasrec`
  remains active under runner PID `2877443` and adapter PID `2877452`, with
  Qwen3 `hf_mean_pool` embedding progress about `141696/233470`. GPU sample
  was `95%`, `16285 MiB / 49140 MiB`, disk remained about `22G` free (`89%`
  used), and no final LLM-ESR `scores.csv`, final provenance, score audit,
  imported table, or predictions exist yet. A complete-metrics gate also
  rechecked the seven completed sports rows: each has HR@5/@10/@20,
  NDCG@5/@10/@20, MRR, `sample_count=10000`, `avg_candidates=101.0`, exact
  1,010,000/1,010,000 score coverage, final provenance, score audit, and
  imported `ranking_eval_records.csv`. The four earliest completed rows were
  missing only the newly standardized `server_final_evidence_audit.json`; that
  JSON was backfilled on the server, copied to local lightweight packages, and
  local-light audits passed for all four without changing any scores.
- Training checkpoint 2026-06-01 17:15 CST: sports `llmesr_sasrec` completed
  Qwen3 `hf_mean_pool` embedding (`233470/233470`) and entered official
  LLM-ESR training. Logged losses are epoch 1 `1.374167` and epoch 5
  `0.361412`. The same runner/adapter PIDs remain active (`2877443`/`2877452`);
  GPU sample was `100%` with `21215 MiB / 49140 MiB`, and disk was `15G` free
  (`93%` used). No final LLM-ESR `scores.csv`, final provenance, score audit,
  predictions, imported tables, or local evidence package exists yet, so the
  eighth sports row remains not table-eligible. A read-only storage review
  found the active LLM-ESR adapter at about `4.5G`; no cleanup was performed
  because meaningful large candidates were either active intermediates or
  protected final evidence from completed rows.
- LLM-ESR completion/package checkpoint 2026-06-01 18:42 CST: sports
  `llmesr_sasrec` completed as the eighth sports official row. The run reached
  epoch 200 with final train loss `0.011395`, saved final provenance, exported
  exact same-candidate scores, and imported metrics. Server-final audit passed
  with `implementation_status=official_completed`, `blockers=[]`,
  `score_coverage_rate=1.0`, `sample_count=10000`, and
  `avg_candidates=101.0`. Full metrics are HR@5/10/20=`0.0916/0.1564/0.2650`,
  NDCG@5/10/20=`0.054919833257876506/0.0758115528438973/0.10310478593304104`,
  and MRR=`0.0751149958885503`. Row counts passed: `scores.csv` has
  `1,010,001` lines, predictions have `10,000` lines, and
  `tables/ranking_eval_records.csv` has `10,001` lines. Lightweight sync and
  local-light audit passed; the local package is under
  `outputs/baselines/official_adapters/sports_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/`.
  After verifying no active LLM-ESR process and protected final outputs, the
  completed intermediate adapter directory was removed, recovering disk from
  `9.4G` to `14G` free. Sports official baselines are now 8/8 complete and
  should move to comparison-table construction and paired tests before any
  sports SOTA wording.
- Sports domain gate checkpoint 2026-06-01 19:08 CST: a read-only server gate
  using `scripts/audit/main_audit_domain_official_gate.py` verified sports has
  all eight official baselines plus `ccrp_v3_qwen3base_pointwise` with full
  HR@5/@10/@20, NDCG@5/@10/@20, MRR, `sample_count=10000`,
  `avg_candidates=101.0`, `score_coverage_rate=1.0`, expected row counts, and
  no gate failures. The generated summaries are
  `outputs/summary/sports_official_ccrp_gate_20260601.json` and `.csv` on both
  server and local. A stale non-experiment bash diagnostic process and the
  confirmed empty malformed `TRAIN_METHODS_OVERRIDE=` official-like directory
  were cleaned without touching final scores, provenance, or imported tables.
  This is a result-completeness gate only; sports still needs the full
  comparison table and paired tests before any SOTA wording.
- Sports comparison/statistical checkpoint 2026-06-01 19:20 CST: the read-only
  `scripts/experiments/main_build_domain_official_comparison.py` script built
  the sports C-CRP-vs-official comparison table and paired tests from existing
  server artifacts. Outputs were synced locally:
  `outputs/summary/sports_official_ccrp_20260601_comparison.csv`,
  `outputs/summary/sports_official_ccrp_20260601_comparison.md`,
  `outputs/summary/sports_official_ccrp_20260601_paired_tests.csv`, and
  `outputs/summary/sports_official_ccrp_20260601_paired_summary.json`. C-CRP
  is observed-best on all seven full metrics and rank 1 by NDCG@10. The
  paired-test family covers 8 official baselines x 7 metrics = 56 tests, all
  with `n_paired_events=10000`, positive deltas, 95% paired-bootstrap CIs above
  zero, and Holm-significant p-values. The closest official baseline is
  `llmemb` for all seven metrics; the smallest margin is HR@20 delta `0.0272`,
  CI `[0.0164, 0.0386]`, Holm p `1.219129314796352e-06`. This is a
  sports-domain pass, not a paper-wide SOTA claim.
- Storage/toys launch checkpoint 2026-06-01 19:48 CST: before launching the
  next domain, a server preflight confirmed no active Pony/baseline process,
  all four new-domain C-CRP reports present, sports official baselines 8/8,
  and toys/home/tools official baselines 0/8. Disk was only `14G` free, so a
  read-only storage audit was performed and disposable user caches under
  `/home/ajifang/.cache` were removed after path verification:
  `vllm`, `torch`, `google-chrome`, `mozilla`, and `JetBrains`. Disk recovered
  to about `19G` free. No project outputs, final scores, provenance, imported
  tables, predictions, checkpoints, or external task packages were deleted.
  Because disk remains tight for the storage-heavy baselines, toys was started
  as a single-row production loop rather than an all-method batch:
  `proex_profile` launched at 2026-06-01 19:44 CST with runner PID `2893793`,
  adapter PID `2893803`, PID file `baselines_new_domains_toys_proex.pid`, and
  log `baselines_new_domains_toys_proex_20260601_194414.log`. At the 19:48
  check it was in Qwen3 `hf_mean_pool` embedding at about `7088/215034`, GPU
  `95%`, and disk about `18G` free. The row is running and not table-eligible.
- Toys ProEx completion/package checkpoint 2026-06-01 21:13 CST:
  `proex_profile` completed as the first toys official row with
  `implementation_status=official_completed`, `blockers=[]`, and
  `score_coverage_rate=1.0`. Server-final audit passed with full metrics:
  HR@5/10/20=`0.0895/0.1615/0.3017`,
  NDCG@5/10/20=`0.058141214365017416/0.0810170703641553/0.11607709818340411`,
  and MRR=`0.08121671352544663`; `sample_count=10000` and
  `avg_candidates=101.0`. Row counts passed: `scores.csv` has `1,010,001`
  lines, predictions have `10,000` lines, and
  `tables/ranking_eval_records.csv` has `10,001` lines. The lightweight local
  package is under
  `outputs/baselines/official_adapters/toys_large10000_100neg_proex_profile_official_qwen3base_same_candidate/`;
  local sync and local-light audit both passed, with 11 allowed lightweight
  files and 4 server-only large files. A server-side sha256 manifest now records
  final `scores.csv`, predictions, and `proex_official_model.pt` without
  copying those large files locally. After verifying protected final evidence,
  the completed intermediate adapter directory was removed, recovering disk
  from about `14G` to `18G` free.
- Toys ProMax launch checkpoint 2026-06-01 21:31 CST: after the ProEx gate and
  cleanup, toys `promax_profile` was launched as the next single-row official
  baseline. The intended adapter is active as PID `2899998` under runner PID
  `2899989`; log path is
  `baselines_new_domains_toys_promax_20260601_212808.log`. At the first
  post-launch check it was in Qwen3 `hf_mean_pool` embedding at about
  `1312/215034`, GPU used about `14843 MiB / 49140 MiB`, disk was about `17G`
  free, and the error scan showed only the known model-loading note. No ProMax
  final scores/provenance/audit/import package exists yet, so the row is not
  table-eligible.
- Toys ProMax completion/package checkpoint 2026-06-02 00:02 CST:
  `promax_profile` completed as the second toys official row with
  `implementation_status=official_completed`, `blockers=[]`, and
  `score_coverage_rate=1.0`. Server-final audit passed with full metrics:
  HR@5/10/20=`0.0920/0.1435/0.2416`,
  NDCG@5/10/20=`0.06289618254810064/0.07937554863319267/0.10387644003990415`,
  and MRR=`0.08184625622431366`; `sample_count=10000` and
  `avg_candidates=101.0`. Row counts passed: `scores.csv` has `1,010,001`
  lines, predictions have `10,000` lines, and
  `tables/ranking_eval_records.csv` has `10,001` lines. The lightweight local
  package is under
  `outputs/baselines/official_adapters/toys_large10000_100neg_promax_profile_official_qwen3base_same_candidate/`;
  local sync and local-light audit both passed, with 11 allowed lightweight
  files and 4 server-only large files. A server-side sha256 manifest records
  final `scores.csv`, predictions, and `promax_official_model.pt` without
  copying those large files locally. After verifying protected final evidence,
  the completed intermediate adapter directory was removed, recovering disk
  from about `13G` to `17G` free.
- Toys ElmRec launch checkpoint 2026-06-02 00:10 CST: after the ProMax gate and
  cleanup, toys `elmrec_graph` was launched as the next single-row official
  baseline. The intended adapter is active as PID `2906455` under runner PID
  `2906447`; log path is
  `baselines_new_domains_toys_elmrec_20260602_000729.log`. At the first
  post-launch check it was in Qwen3 `hf_mean_pool` embedding at about
  `3624/215034`; at the 2026-06-02 00:15 CST monitor check it had advanced to
  about `21872/215034`, GPU used about `16213 MiB / 49140 MiB`, disk was about
  `16G` free, and the error scan showed only the known model-loading note. No
  ElmRec final scores/provenance/audit/import package exists yet, so the row is
  not table-eligible.
- Toys ElmRec completion/package checkpoint 2026-06-02 01:36 CST:
  `elmrec_graph` completed as the third toys official row with
  `implementation_status=official_completed`, `blockers=[]`, and
  `score_coverage_rate=1.0`. Server-final audit PASS, lightweight sync PASS,
  and local-light audit PASS. Full metrics over 10,000 users and 101
  candidates:
  HR@5/10/20 `0.0545 / 0.1043 / 0.2013`,
  NDCG@5/10/20
  `0.03259298673054038 / 0.04856005753116525 / 0.07278039157879498`,
  MRR `0.05431081812612059`. Row counts passed: `scores.csv` `1,010,001`
  lines, predictions `10,000` lines, and `tables/ranking_eval_records.csv`
  `10,001` lines. The local lightweight package is under
  `outputs/baselines/official_adapters/toys_large10000_100neg_elmrec_graph_official_qwen3base_same_candidate/`;
  final `scores.csv`, predictions, and `elmrec_official_model.pt` remain
  server-only and are covered by `server_large_artifact_manifest.sha256`.
  After verifying protected final evidence and no active ElmRec process, the
  completed intermediate adapter directory was removed, recovering disk from
  about `12G` to `16G` free.
- Toys LLMEmb completion checkpoint 2026-06-02 03:04 CST: toys `llmemb`
  completed as the fourth toys official row with
  `implementation_status=official_completed`, `blockers=[]`, and
  `score_coverage_rate=1.0`. Server-final and local-light audits passed with
  full metrics over 10,000 users and 101 candidates:
  HR@5/10/20 `0.2499 / 0.3505 / 0.4866`, NDCG@5/10/20
  `0.17252113274887534 / 0.20485045979333913 / 0.23905481091819092`, and MRR
  `0.1813804118284203`. Row counts passed for `scores.csv` (`1,010,001`
  lines), predictions (`10,000` lines), and
  `tables/ranking_eval_records.csv` (`10,001` lines). The local lightweight
  package is
  `outputs/baselines/official_adapters/toys_large10000_100neg_llmemb_official_qwen3base_same_candidate/`.
  A server-side sha256 manifest records `scores.csv`,
  `predictions/rank_predictions.jsonl`, and `llmemb_official_model.pt`; those
  large files remain server-only. After verifying protected final evidence and
  no active LLMEmb Python process, the completed intermediate adapter directory
  was removed, recovering disk from about `4.0G` to `8.3G` free without
  touching final evidence. Toys official baselines are now 4/8 complete.
- Toys IRLLRec launch checkpoint 2026-06-02 03:19 CST: after the LLMEmb gate
  and cleanup, toys `irllrec_intent` was launched as the next single-row
  official baseline. The intended adapter is active as PID `2923437` under
  runner PID `2923429`; log path is
  `baselines_new_domains_toys_irllrec_20260602_031623.log`. At the first
  stable check it was in Qwen3 `hf_mean_pool` embedding at about `1400/215034`,
  GPU used about `15945 MiB / 49140 MiB`, and disk was about `7.3G` free. No
  IRLLRec final scores/provenance/audit/import package exists yet, so the row
  is not table-eligible.
- IRLLRec storage checkpoint 2026-06-02 04:12 CST: after embeddings completed
  and official training reached epoch 30, disk had fallen to about `4.0G` free.
  Only disposable pip cache/temp paths were removed after scope checks
  (`/home/ajifang/.cache/pip` and `/tmp/pip-unpack-920865s3`), recovering disk
  to about `4.4G` free. No project evidence, final outputs, active adapters,
  checkpoints, scores, predictions, or other project directories were deleted.
- IRLLRec monitoring/cleanup checkpoint 2026-06-02 04:35 CST: toys
  `irllrec_intent` remains active under runner PID `2923429` and adapter PID
  `2923437`. The log has reached epoch `500/3000`, GPU is active, disk is
  still about `4.4G` free, and the final evidence directory still has no
  `scores.csv`, provenance, score audit, imported tables, or predictions. A
  read-only storage audit found no additional clearly disposable project
  artifact: the old `books_large10000_100neg_llmesr_adapter` directory is
  about `1.3G`, but it appears to contain historical adapter mapping/text data
  while the corresponding final books directory is table-only and no local
  lightweight package was found. It was not deleted. The next action remains
  to let IRLLRec finish, then run server-final audit, lightweight sync,
  local-light audit, and only then remove the completed toys IRLLRec
  intermediate adapter after path/final-output checks.
- IRLLRec cache-cleanup checkpoint 2026-06-02 04:47 CST: toys
  `irllrec_intent` remains active and reached epoch `760/3000`; no final
  score/provenance/audit/import package exists yet. To reduce no-space risk
  without touching project evidence or dependencies, only three user-level
  cache directories were removed after realpath allowlist checks:
  `.vscode-server/data/CachedExtensionVSIXs`, Chrome `component_crx_cache`,
  and Code `CachedData`. This recovered disk from about `4.4G` to `4.6G`
  free. Project outputs, active adapters, models, Python site-packages, and
  other projects were left untouched.
- IRLLRec IDE-cache cleanup checkpoint 2026-06-02 04:56 CST: toys
  `irllrec_intent` remains active and reached epoch `940/3000`; no final
  evidence files exist yet and error scans are clean. After confirming there
  were no VSCode server processes, five inactive
  `.vscode-server/cli/servers/Stable-*` cache directories were removed with
  realpath prefix checks, recovering disk from about `4.6G` to `6.4G` free.
  Project outputs, final evidence, active adapters, models, conda/Python
  environments, and other projects were not touched. Next action is still to
  wait for IRLLRec completion, run full evidence gates, sync the lightweight
  package, and then remove only the completed IRLLRec intermediate adapter.
- Toys IRLLRec completion checkpoint 2026-06-02 06:35 CST: toys
  `irllrec_intent` completed as the fifth toys official row with
  `implementation_status=official_completed`, `blockers=[]`, and
  `score_coverage_rate=1.0`. Server-final audit and local-light audit passed
  with full metrics over 10,000 users and 101 candidates: HR@5/10/20
  `0.1565 / 0.2293 / 0.4098`, NDCG@5/10/20
  `0.11049209461545026 / 0.13380144693674725 / 0.1785851471792316`, and MRR
  `0.1311986744710446`. Row counts passed for `scores.csv` (`1,010,001`
  lines), predictions (`10,000` lines), and
  `tables/ranking_eval_records.csv` (`10,001` lines). The local lightweight
  package is
  `outputs/baselines/official_adapters/toys_large10000_100neg_irllrec_intent_official_qwen3base_same_candidate/`.
  Server-only `scores.csv`, predictions, and `irllrec_official_model.pt` are
  covered by `server_large_artifact_manifest.sha256`. After audits and local
  sync passed, the completed intermediate adapter directory was removed,
  recovering disk from about `5.4G` to `9.7G` free without touching final
  evidence. Toys official baselines are now 5/8 complete.
- Toys RLMRec launch checkpoint 2026-06-02 06:47 CST: after the IRLLRec
  gate/package/cleanup preflight, toys `rlmrec_graphcl` was launched as the
  next single-row official baseline. The remote job continued after the local
  SSH command timed out, so PID files were corrected after process inspection:
  runner PID `2937284`, adapter PID `2937292`, log
  `baselines_new_domains_toys_rlmrec_20260602_064443.log`. At the first stable
  check it was active in Qwen3 `hf_mean_pool` embedding at about
  `1664/215034`, GPU was `99%` with `15945 MiB / 49140 MiB`, and disk was
  about `8.8G` free. Do not start another toys baseline until this row
  finishes or fails and has been audited.
- Toys RLMRec monitoring/gate checkpoint 2026-06-02 07:18 CST: active runner
  PID `2937284` and adapter PID `2937292` remain live. Qwen3 embedding reached
  about `133152/215034`; GPU was `98%` with `16285 MiB / 49140 MiB` at 75C,
  disk remained about `8.8G` free, and the recent error scan was clean. A
  read-only toys domain gate found five official rows passing all compact
  server checks (`llmemb`, `proex_profile`, `promax_profile`, `elmrec_graph`,
  `irllrec_intent`), while `rlmrec_graphcl`, `llm2rec_sasrec`, and
  `llmesr_sasrec` are incomplete as expected. Toys C-CRP core metrics and row
  counts are present under `outputs/toys_large10000_100neg_ccrp_v3`, but
  imported prediction/metric/coverage tables still needed reconciliation before
  a toys comparison table was paper-facing. That C-CRP import/gate issue was
  resolved at 2026-06-02 17:09 CST; the remaining toys gate blocker is the
  active `llmesr_sasrec` official row.
- Toys RLMRec training checkpoint 2026-06-02 07:48 CST: the Qwen3 embedding
  pass completed (`215034/215034`) and toys `rlmrec_graphcl` entered official
  training. The latest logged line was `[rlmrec-official] epoch=90
  train_loss=1.496428`; the familiar non-fatal graph normalization warnings
  appeared before training. No final score/provenance/audit/import package
  exists yet. Disk fell to about `5.4G` free (`98%` used) as the active RLMRec
  adapter grew to about `4.3G`. A read-only cleanup audit did not find a safe
  large deletion target: the active adapter must stay, the old books LLM-ESR
  adapter is not yet verified as disposable, and `.vscode-server` has live
  Code-related processes. Continue monitoring disk through final export.
- GPU: RTX 4090, active when official-baseline rows are running
- Disk: 44 GB free at launch check (2026-05-31)
- All experiments use: Qwen3-8B, vLLM, 10k users, 101 candidates (1+100neg)
