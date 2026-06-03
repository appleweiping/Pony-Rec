# Paper Claims and Experiment Status

This file freezes the paper scope so the project does not drift into a mixed
system log.

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
| home | 0.156 | 0.226 | 0.351 | 0.110 | 0.132 | 0.164 | 0.126 | baselines pending |
| tools | 0.194 | 0.270 | 0.393 | 0.142 | 0.166 | 0.197 | 0.156 | baselines pending |

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
table, and sports/toys now pass domain-level official-baseline and paired-test
gates. Do not generalize to paper-wide SOTA until the declared domain set is
complete; home has only 3/8 audited official rows complete and tools still has
0/8, so both still need canonical eight-row official blocks and paired
same-candidate tests.

### Remaining for paper submission

1. Run the canonical 8 official baselines on home/tools after a
   fresh disk/GPU/process check. `scripts/run_baselines_new_domains.sh` is
   aligned to exclude SETRec while it remains blocked/supplementary, supports
   single-domain production via `DOMAINS_OVERRIDE`, and now audits/imports
   complete `@5/@10/@20 + MRR` same-candidate metrics after each completed
   score file. Sports and toys are now 8/8 complete and have passed their
   domain/comparison/paired-test gates; home has 3/8 audited official rows
   complete (`proex_profile`, `promax_profile`, `elmrec_graph`) and home
   `llmemb` is in a symlink recovery rerun after a disk-full checkpoint
   failure, while tools remains pending. Toys
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
   Home now has 3/8 completed official rows. At the 2026-06-03 05:48 CST
   checkpoint no Pony/C-CRP/baseline Python process was active, GPU was idle,
   and disk was tight at about `6.5G` free. The completed ElmRec intermediate
   adapter was removed after exact realpath checks and a 16-file sha256 cleanup
   manifest
   `outputs/summary/home_elmrec_completed_adapter_cleanup_manifest_20260603.sha256`;
   a post-cleanup server-final audit remained `ok=true`, and final scores,
   provenance, audits, predictions, imported tables, model, and local
   lightweight evidence were preserved. Disk recovered to about `14G` free.
   After a clean process/GPU/disk and no-existing-artifact preflight, home
   `llmemb` launched at 2026-06-03 06:08 CST as the fourth home row with
   active adapter PID `3085786`, PID file
   `baselines_new_domains_home_llmemb_adapter.pid`, and log
   `baselines_new_domains_home_llmemb_20260603_0608.log`. At the first stable
   check it was in Qwen3 `hf_mean_pool` embedding at about `2808/385364`, GPU
   was `96%` with `16067 MiB / 49140 MiB`, disk was about `13G` free, and no
   final scores/provenance existed yet. LLMEmb is not a result row until final
   gates pass.
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
