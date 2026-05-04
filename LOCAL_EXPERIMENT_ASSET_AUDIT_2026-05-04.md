# Local Experiment Asset Audit - 2026-05-04

This audit records what experiment assets are present locally on the
pre-Cursor rescue branch workspace. It is meant to prevent losing track of
which results are local and which likely live only on the server.

## Local Result Locations Checked

- `outputs/`
- `outputs/summary/`
- `outputs_backup_old/`
- `Paper/pony2/results/`
- `Paper/pony2/root_docs_archive/`
- `docs/`
- `data/processed/`
- `server_sync/`

## Key Finding

The local workspace contains useful summary artifacts and some real prediction
files, but it does not contain the full server-sync directory referenced by the
week7.7 report.

`server_sync/` is not present locally.

Therefore, server-side raw run folders may still need to be recovered from the
server before any final paper-result claim or rerun comparison.

## Present Locally

### Paper-level server-synced conclusions

`Paper/pony2/results/` contains week7.7 server-synced summary artifacts:

- `week7_7_four_domain_srpd_final_matrix.md`
- `week7_7_four_domain_srpd_final_matrix.csv`
- `week7_7_four_domain_srpd_full_metric_matrix.md`
- `week7_7_four_domain_srpd_full_metric_matrix.csv`
- `week7_7_teacher_reliability_summary.md`
- `week7_7_teacher_reliability_summary.csv`
- `week7_7_calibration_coverage_robustness_integration.md`

Important recorded conclusions:

- Beauty full973 winner: `SRPD-v2`, NDCG@10 `0.635366`, MRR `0.518448`.
- Books small500 winner: `SRPD-v2`, NDCG@10 `0.705707`, MRR `0.611767`.
- Electronics small500 winner: `SRPD-v5`, NDCG@10 `0.662100`, MRR `0.552833`.
- Movies small500 winner: `structured_risk`, NDCG@10 `0.573183`, MRR `0.439100`; `SRPD-v4` is the best SRPD repair row.

Teacher reliability summary records:

- Beauty: AUROC `0.618473`, ECE `0.218777`.
- Books: AUROC `0.667293`, ECE `0.044753`.
- Electronics: AUROC `0.610643`, ECE `0.187660`.
- Movies: AUROC `0.433191`, ECE `0.261853`, warning that teacher reliability is weak.

These are the strongest local paper-direction assets found.

### Output summaries

`outputs/summary/` contains 93 CSV files, 13 Markdown files, and related local
summary assets. Notable files:

- `candidate_protocol_audit.csv`
- `baseline_reliability_proxy_audit.csv`
- `generative_title_bridge_status.csv`
- `week7_9_shadow_small_prior_summary.csv`
- `week7_9_shadow_small_prior_summary.md`
- SRPD data summaries for books/electronics/movies small500 and Beauty full973.

Important local caveat:

- `week7_9_shadow_small_prior_summary.csv` currently records the shadow small
  prior row as `missing` for pointwise, calibration, rerank, noisy pointwise,
  and noisy rerank. This means local shadow small-prior results are not present
  as completed local evidence.

### Raw-ish local prediction outputs

`outputs/` contains non-placeholder files:

- 93 `.csv`
- 39 `.jsonl`
- 18 `.png`
- 13 `.md`
- 9 `.json`

Large local prediction/result folders include:

- `electronics_deepseek_pointwise_full3000/`
- `books_deepseek_pointwise_full3000/`
- `movies_deepseek_pointwise_full3000/`
- `beauty_deepseek_rank_full973_structured_risk/`
- `books_deepseek_rank_full500/`
- `electronics_deepseek_rank_full500/`
- `movies_deepseek_rank_full500/`
- pairwise coverage folders for Beauty, Books, Electronics, and Movies.

These are useful for archaeology and possibly rerun validation, but they must
be checked against protocol docs before being treated as paper evidence.

### Local v6 bridge smoke

`outputs/_tmp_shadow_v6_bridge_smoke/` is present locally and contains:

- `reranked/shadow_v6_bridge_rows.jsonl`
- `reranked/shadow_v6_decision_reranked.jsonl`
- `tables/rerank_results.csv`
- `tables/shadow_v6_bridge_rows.csv`
- `tables/shadow_v6_bridge_summary.csv`

Recorded smoke result:

- Direct candidate ranking: NDCG@10 `0.752726`, MRR `0.673533`.
- `shadow_v6_decision_bridge`: NDCG@10 `0.706906`, MRR `0.612567`.
- Changed ranking fraction: `0.392`.
- Matched signal rate: `0.5`.
- Fallback rate: `0.5`.

Interpretation: this is a smoke / diagnostic bridge check only, not evidence
that v6 improves ranking.

### Processed data

`data/processed/` contains many local processed domains and variants:

- `amazon_beauty`
- `amazon_books`
- `amazon_electronics`
- `amazon_movies`
- small / medium / noisy variants
- `processed_4domains`
- `movielens_1m`

These local processed files are useful for reproducing or inspecting old runs,
but server parity should be checked before launching new server jobs.

## Not Present Locally

- `server_sync/` is absent.
- `output-repaired/summary/` is absent.
- Completed local shadow v1-v5 small-prior result folders are not visible under
  `outputs/` by directory name.

## Server Follow-Up

Before claiming final status or running the next large job, recover or verify
on the server:

1. The source folder referenced by `Paper/pony2/results`:
   `server_sync/week7_20260421/week7_7_four_domain_final`.
2. Any completed shadow v1-v5 pointwise/calibration/rerank outputs.
3. Any full replay outputs for Beauty full973 and Books/Electronics/Movies
   small500.
4. The exact command logs and environment used for week7.7 and week7.9.

## Working Rule

Use local summaries as navigation and hypothesis evidence. Use server raw
artifacts plus manifests/logs as the final authority before paper-result
promotion.
