# Pony-rec / Uncertainty Active TODO

Last updated: 2026-06-01 14:03 CST

This is the cumulative execution TODO for the active Pony-rec / Uncertainty
goal. It is a handoff artifact, not a claim of paper readiness. Update it after
each completed official row, blocker, cleanup decision, comparison-table build,
or review cycle.

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

## Current Server State

- Server: `pony-rec-gpu`
- Server repo: `~/projects/pony-rec-rescue-shadow-v6`
- Active runner: resumed single-row sports `llm2rec_sasrec`, PID `2870575`,
  launched with `DOMAINS_OVERRIDE=sports`,
  `FAST_METHODS_OVERRIDE=`, and `TRAIN_METHODS_OVERRIDE=llm2rec_sasrec`; do
  not restart the whole eight-method script because six sports rows are already
  completed and audited
- Latest checked state: 2026-06-01 14:02 CST, LLM2Rec passed the previous
  adapter-export blocker and entered Qwen3 embedding generation at about
  `3432/283760`; GPU was `100%`, `16115 MiB / 49140 MiB`, and disk had `27G`
  free (`86%` used)
- Latest completed row: sports `rlmrec_graphcl`, completed 2026-06-01
  13:43 CST with `implementation_status=official_completed`, `blockers=[]`,
  `score_coverage_rate=1.0`, server-final audit PASS, lightweight sync PASS,
  and local-light audit PASS
- Current blocker/recovery: sports `llm2rec_sasrec` failed during adapter
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
  server after commit `657929e`; server lacks `pytest`, but
  `py_compile` passed and the real sports export passed far enough to start
  full Qwen3 embedding generation.
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
| `llm2rec_sasrec` | running | valid-history fix deployed; export passed; Qwen3 embedding active under PID `2870575` |
| `llmesr_sasrec` | pending | inspect-only placeholder |

Completed sports rows have server-side `scores.csv` line count `1,010,001`,
`predictions/rank_predictions.jsonl` line count `10,000`, final provenance,
score audits, full metric tables, coverage/exposure tables, and
`tables/ranking_eval_records.csv`.
RLMRec is now a completed row. LLM2Rec and LLM-ESR are still missing final
score/provenance/table packages and must not enter a comparison table yet.

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

## Required Next Actions

1. Monitor active sports `llm2rec_sasrec` PID `2870575` without stopping it.
   It is currently embedding Qwen3 item text (`hf_mean_pool`) after passing the
   repaired adapter export path.
2. If LLM2Rec completes, run server-final audit, lightweight sync,
   local-light audit, and full metric/row-count recording.
3. Repeat the evidence loop for sports `llmesr_sasrec`.
4. After all eight sports official rows complete, build the sports comparison
   table and paired/statistical tests. Do not claim sports SOTA until the
   complete same-candidate table and paired tests pass.
5. Continue the same official-baseline protocol for toys, home, and tools.
6. Only after the declared experiments, comparisons, ablations, provenance,
    statistical tests, and figure checks are complete, move to ARIS paper
    writing and GPT-5.5/Codex xhigh review. The review loop must reach at
    least 8/10 before submission-level readiness is claimed.

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

```bash
/home/ajifang/miniconda3/bin/python /tmp/pony_audit_official_evidence_package.py \
  --evidence_dir outputs/<EXP>_<METHOD>_official_qwen3base_same_candidate \
  --mode=server_final \
  --quiet
```

Passing these gates is necessary but not sufficient for paper readiness; final
paper claims still require full cross-baseline comparison tables, paired tests,
claim audit, citation audit, and GPT-5.5/Codex review.
