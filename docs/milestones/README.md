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

## Current Working Position (updated 2026-06-02)

The repository is now in M5 (multi-domain SOTA validation):

- C-CRP v3 completed on all 8 domains
- Official external baselines completed on original 4 domains (8 methods each)
- New domains (sports/toys/home/tools) official baselines are in Phase 2.
  Sports has all eight official rows plus C-CRP imported evidence through the
  domain gate. Toys has seven audited official rows complete and the eighth
  row, `llmesr_sasrec`, is running; every completed row imports full
  `@5/@10/@20 + MRR` metrics after score audit.
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

New-domain imported-table note (2026-06-02): toys C-CRP v3 is metric-complete
under `outputs/toys_large10000_100neg_ccrp_v3` (`report.json`, `scores.csv`,
and `user_ranks.jsonl` are present with aligned 10k-user/101-candidate counts),
but the current unified domain gate looks for imported C-CRP artifacts under
`outputs/toys_large10000_100neg_ccrp_v3_qwen3base_pointwise`. Until the import
step or gate mapping is reconciled, do not treat toys as having passed the
same comparison-table gate even though the core C-CRP metrics exist.

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
2. 8 official baselines on 4 new domains (Phase 2) — sports has all eight
   audited official rows complete; toys has seven audited official rows complete
   with `llmesr_sasrec` currently running; home/tools remain pending.
3. Full comparison table + statistical tests (Phase 3)
4. Paper writing with ARIS skill (Phase 4)
5. GPT-5.5/Codex review cycle until 8/10 (Phase 5)

### Server State

- Batch script complete: `run_ccrp_v3_all_new_domains.sh` (sports/toys/home/tools)
- Phase 2 sports official-baseline run started 2026-05-31 and is now complete:
  all eight sports official rows passed final provenance/coverage/import gates.
  The last separate LLM-ESR runner (`2877443`/`2877452`) finished at
  2026-06-01 18:31 CST.
- Phase 2 toys official-baseline run is in progress. As of 2026-06-02
  16:35 CST, seven toys rows are audited complete
  (`proex_profile`, `promax_profile`, `elmrec_graph`, `llmemb`,
  `irllrec_intent`, `rlmrec_graphcl`, and `llm2rec_sasrec`). The latest
  completed row, `llm2rec_sasrec`, passed server-final audit, lightweight sync,
  local-light audit, full metrics, exact coverage, and row-count gates after
  the disk-full recovery described below. The previous completed row,
  `rlmrec_graphcl`, also passed all gates and its completed intermediate
  adapter was safely removed after final evidence and local backup were
  verified. Toys `llm2rec_sasrec` first launched at
  2026-06-02 14:26 CST completed Qwen3 embedding (`254815/254815`) but failed
  when disk reached `0` free; the adapter-side embedding was verified complete
  with shape `(254816, 4096)` and the upstream LLM2Rec item-info file was an
  incomplete disk-full copy. Recovery removed the incomplete upstream copy,
  temporary files, and the old non-active books LLM-ESR intermediate adapter,
  then symlinked the upstream Toys item-info path to the complete adapter
  embedding. The recovery job launched at 2026-06-02 16:04 CST with runner PID
  `2965472`, adapter PID `2965476`, official training child PID `2965700`, and
  log `baselines_new_domains_toys_llm2rec_recovery_20260602_1603.log`. It
  finished at 16:18 CST as `official_completed` with `blockers=[]`,
  `score_coverage_rate=1.0`, full metrics HR@5/10/20
  `0.2202 / 0.3172 / 0.4652`, NDCG@5/10/20
  `0.1475691807818137 / 0.17887285724512209 / 0.21609262826220665`, MRR
  `0.15921596430464027`, and aligned row counts (`scores.csv` `1,010,001`,
  predictions `10,000`, `tables/ranking_eval_records.csv` `10,001`). The local
  lightweight package is
  `outputs/baselines/official_adapters/toys_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate/`.
  Completed adapter intermediate CSVs were compressed, and the old sports
  LLM2Rec upstream item-info cache was removed after recording sha256
  `41e968bc31de1454eb3deab08eff6e06e1d68308d7ed2b25137f0b377f6b9a2c`; final
  scores, provenance, audits, predictions, imported tables, checkpoints, and
  the toys embedding were not deleted. Toys `llmesr_sasrec` is now active
  under wrapper PID `2970036`, runner PID `2970047`, adapter PID `2970055`,
  with log `baselines_new_domains_toys_llmesr_20260602_1635.log`.
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
  imported prediction/metric/coverage tables still need reconciliation before
  a toys comparison table is paper-facing.
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
