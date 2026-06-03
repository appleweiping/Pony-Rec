# Pony-rec / Uncertainty Active TODO

Last updated: 2026-06-03 09:55 CST

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
- Disk policy update: after a completed official row has passed server-final
  audit and local-light sync, the huge server-side
  `predictions/rank_predictions.jsonl` may be deleted with sha256 manifest and
  docs/memory note if disk pressure threatens active progress. Domain gate and
  comparison scripts now accept the row only if `server_final_evidence_audit.json`
  certifies the missing prediction file's original line count. This exception
  does not cover `scores.csv`, provenance, score audits, run summaries,
  imported `tables/`, models, checkpoints, or local evidence packages.
- Active runner: home `irllrec_intent` official row, launched 2026-06-03
  13:55 CST after a clean process/GPU/disk preflight and LLMEmb cleanup.
  Runner PID `3147646`, adapter PID `3147655`, PID file
  `baselines_new_domains_home_irllrec_20260603_1355.pid`, log
  `baselines_new_domains_home_irllrec_20260603_1355.log`. At the 13:57 CST
  checkpoint it was CPU-side active, GPU idle (`0%`, `15 MiB / 49140 MiB`),
  adapter directory about `1.1G`, final output directory empty, and disk about
  `6.1G` free. No scores/provenance/imported tables exist yet, so this is not
  table-eligible.
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
  local-light package were preserved; disk recovered to about `3.1G`.
- Latest completed home row: `llmemb`, completed 2026-06-03 09:55 CST after a
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
  keeping those files server-only. Home now has 4/8 completed official
  baseline rows. After local evidence was refreshed, the completed
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
| `irllrec_intent` | active | launched 2026-06-03 13:55 CST; runner PID `3147646`, adapter PID `3147655`, log `baselines_new_domains_home_irllrec_20260603_1355.log`; no final scores/provenance yet |
| `rlmrec_graphcl` | pending | do not launch until the previous row is packaged/audited and disk has enough margin |
| `llm2rec_sasrec` | pending | do not launch until the previous row is packaged/audited and disk has enough margin |
| `llmesr_sasrec` | pending | do not launch until the previous row is packaged/audited and disk has enough margin |

Home official baselines are now 4/8 complete (`proex_profile`,
`promax_profile`, `elmrec_graph`, `llmemb`). All completed rows passed final
provenance, exact score coverage, server-final package audit, lightweight
local sync, local-light audit, full metrics, and row-count gates. Home is not
domain-gate eligible until all eight official rows and imported C-CRP evidence
pass the same checks.

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

## Required Next Actions

1. Monitor the active home `irllrec_intent` row. Do not launch another
   baseline while runner PID `3147646` / adapter PID `3147655` are active.
   Watch disk closely: it fell to about `30M` free before emergency cleanup
   recovered it to about `2.0G` at the 2026-06-03 17:27 CST checkpoint, then a
   further audited Toys ProEx prediction cleanup recovered it to about `2.5G`
   at the 17:45 CST checkpoint, and an audited Sports ProMax prediction cleanup
   recovered it to about `3.1G` at the 18:08 CST checkpoint. The
   final row is not table-eligible until the full
   score/provenance/import/audit/local-light gate passes.
2. After each completed home/tools row, verify full HR@5/@10/@20,
   NDCG@5/@10/@20, MRR, `n_users=10000`, `avg_candidates=101`,
   score/candidate row counts, exact same-candidate coverage, provenance,
   score audit, imported tables, server-final audit, lightweight sync, and
   local-light audit before recording it as official evidence.
3. Build domain comparison + paired-test gates only after all eight official
   rows for that domain and C-CRP imported evidence pass the domain gate.
4. Only after the declared experiments, comparisons, ablations, provenance,
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
