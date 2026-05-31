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
LLM recommendation confidence is informative but unreliable.
Task-grounded calibrated uncertainty can turn it into a decision-useful signal
under a controlled same-candidate recommendation protocol.
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

## Current Working Position (updated 2026-05-31)

The repository is now in M5 (multi-domain SOTA validation):

- C-CRP v3 completed on all 8 domains
- Official external baselines completed on original 4 domains (8 methods each)
- New domains (sports/toys/home/tools) official baselines are in Phase 2;
  sports launched on 2026-05-31 with the reconciled canonical 8-method runner,
  which imports full `@5/@10/@20 + MRR` metrics after each successful score
  audit
- Strategy: achieve SOTA only after the new-domain official baselines pass
  same-candidate score/provenance/import gates

### C-CRP v3 Results (all domains)

| Domain | Users | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR | Status |
|--------|-------|------|-------|-------|--------|---------|---------|-----|--------|
| beauty | 973 | 0.157 | 0.229 | 0.369 | 0.111 | 0.134 | 0.169 | 0.128 | #2 (ProEx=0.253) |
| books | 10000 | 0.374 | **0.476** | 0.592 | 0.300 | **0.333** | 0.362 | 0.306 | **SOTA** |
| electronics | 10000 | 0.218 | **0.299** | 0.418 | 0.157 | **0.183** | 0.213 | 0.168 | **SOTA** |
| movies | 10000 | 0.145 | 0.208 | 0.331 | 0.108 | 0.128 | 0.159 | 0.127 | #5 |
| sports | 10000 | 0.275 | 0.382 | 0.517 | 0.198 | 0.233 | 0.267 | 0.208 | baselines pending |
| toys | 10000 | 0.317 | 0.396 | 0.506 | 0.245 | 0.271 | 0.298 | 0.250 | baselines pending |
| home | 10000 | 0.156 | 0.226 | 0.351 | 0.110 | 0.132 | 0.164 | 0.126 | baselines pending |
| tools | 10000 | 0.194 | 0.270 | 0.393 | 0.142 | 0.166 | 0.197 | 0.156 | baselines pending |

Original-domain C-CRP v3 formal reports are under
`outputs/ccrp_v3_formal/<domain>/report.json`; the four-domain comparison with
the canonical official baseline block is
`outputs/ccrp_v3_formal/main_comparison_table.csv`. New-domain artifact
completeness: each of sports/toys/home/tools has `report.json`, `scores.csv`
with 1,010,000 candidate-score rows plus header, and `user_ranks.jsonl` with
10,000 user-rank rows.

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
2. 8 official baselines on 4 new domains (Phase 2) — sports running with
   `llmemb`, `proex_profile`, `promax_profile`, and `elmrec_graph` complete;
   continue the single-domain
   production loop after each domain passes provenance, exact-score, import,
   and storage checks
3. Full comparison table + statistical tests (Phase 3)
4. Paper writing with ARIS skill (Phase 4)
5. GPT-5.5/Codex review cycle until 8/10 (Phase 5)

### Server State

- Batch script complete: `run_ccrp_v3_all_new_domains.sh` (sports/toys/home/tools)
- Phase 2 sports official-baseline run started 2026-05-31:
  `baselines_new_domains_sports.log`, runner PID `2794722`. Sports completed
  `llmemb`, `proex_profile`, `promax_profile`, and `elmrec_graph`; the current
  active row is `irllrec_intent`.
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
- GPU: RTX 4090, active for the sports official-baseline run
- Disk: 44 GB free at launch check (2026-05-31)
- All experiments use: Qwen3-8B, vLLM, 10k users, 101 candidates (1+100neg)
