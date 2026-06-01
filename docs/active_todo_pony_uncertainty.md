# Pony-rec / Uncertainty Active TODO

Last updated: 2026-06-01 08:34 CST

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
- Active runner: `baselines_new_domains_sports.log`, runner PID `2794722`
- Active row: sports `rlmrec_graphcl`, child PID `2851207`
- Latest checked progress: 2026-06-01 08:34 CST, RLMRec Qwen embedding
  generation `105376/233470`; no RLMRec training epochs yet
- GPU/disk at latest check: GPU `94%`, `16285 MiB / 49140 MiB`, disk `32G`
  free (`83%` used)
- Latest fatal scan: no `Traceback`, `Killed`, OOM, CUDA, no-space, disk quota,
  exception, or runtime-error markers
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
| `rlmrec_graphcl` | running | Qwen embedding generation active; wait for final scores/provenance |
| `llm2rec_sasrec` | pending | inspect-only placeholder |
| `llmesr_sasrec` | pending | inspect-only placeholder |

Completed sports rows have server-side `scores.csv` line count `1,010,001`,
`predictions/rank_predictions.jsonl` line count `10,000`, final provenance,
score audits, full metric tables, coverage/exposure tables, and
`tables/ranking_eval_records.csv`.
RLMRec is not yet a completed row: as of 2026-06-01 08:34 CST its output
directory contains only `inspect_fairness_provenance.json`; there is no final
`scores.csv`, score audit, imported table, run summary, or final
`fairness_provenance.json` to audit or sync.

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

## Required Next Actions

1. Monitor the active sports `rlmrec_graphcl` row until it produces final
   `scores.csv`, `fairness_provenance.json`, score audits, imported tables,
   predictions, and run summary.
2. If RLMRec completes, run the server-side package audit:

   ```bash
   cd ~/projects/pony-rec-rescue-shadow-v6
   /home/ajifang/miniconda3/bin/python /tmp/pony_audit_official_evidence_package.py \
     --evidence_dir outputs/sports_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate \
     --mode=server_final \
     --quiet
   ```

3. Copy the lightweight RLMRec evidence package to local
   `outputs/baselines/official_adapters/sports_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate/`.
   Include final provenance, inspect provenance, JSON/TXT score audits, run
   summary, imported `tables/`, and compact manifests if present. Do not copy
   huge score/prediction/checkpoint files unless a recovery decision requires
   it.
   Preferred command:

   ```powershell
   python scripts\audit\main_sync_official_evidence_package.py `
     --remote_evidence_dir outputs/sports_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate `
     --local_evidence_dir outputs\baselines\official_adapters\sports_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate `
     --copy `
     --quiet
   ```

4. Run the local package audit:

   ```powershell
   python scripts\audit\main_audit_official_evidence_package.py `
     --evidence_dir outputs\baselines\official_adapters\sports_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate `
     --mode local_light `
     --quiet
   ```

5. Update shared memory, `docs/milestones/README.md`,
   `docs/paper_claims_and_status.md`, and this TODO with the complete metrics,
   row counts, provenance status, copied files, and any cleanup decision.
6. Commit and push only the related docs/manifests/code changes from the local
   repo.
7. Let the runner continue to `llm2rec_sasrec`, then repeat the same evidence
   loop for `llm2rec_sasrec` and `llmesr_sasrec`.
8. After all eight sports official rows complete, build the sports comparison
   table and paired/statistical tests. Do not claim sports SOTA until the
   complete same-candidate table and paired tests pass.
9. Continue the same official-baseline protocol for toys, home, and tools.
10. Only after the declared experiments, comparisons, ablations, provenance,
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
