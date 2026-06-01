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
| sports | 0.275 | 0.382 | 0.517 | 0.198 | 0.233 | 0.267 | 0.208 | baselines pending |
| toys | 0.317 | 0.396 | 0.506 | 0.245 | 0.271 | 0.298 | 0.250 | baselines pending |
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

C-CRP v3 achieves SOTA on books and electronics. For sports, toys, home, and
tools, do not claim SOTA until the canonical 8 official baselines finish and
paired same-candidate tests pass. Current values are candidate results awaiting
external-baseline comparison.

### Remaining for paper submission

1. Run the canonical 8 official baselines on sports/toys/home/tools after a
   fresh disk/GPU/process check. `scripts/run_baselines_new_domains.sh` is
   aligned to exclude SETRec while it remains blocked/supplementary, supports
   single-domain production via `DOMAINS_OVERRIDE`, and now audits/imports
   complete `@5/@10/@20 + MRR` same-candidate metrics after each completed
   score file. Sports is currently running from
   `baselines_new_domains_sports.log` with runner PID `2794722`; the active
   child at the 2026-05-31 22:32 CST checkpoint is sports `llmemb` PID
   `2794731`. It has completed the `hf_mean_pool` embedding pass
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
   official evidence count remains five completed rows.
2. Import and audit each remaining new-domain baseline row with exact score coverage,
   full @5/@10/@20 metrics, provenance, and row-count checks.
3. Full @5/@10/@20 comparison table across all domains
4. Statistical significance tests (paired t-test, 20+ seeds or bootstrap)
5. Paper writing
6. GPT-5.5/Codex review cycle (target: 8/10)
