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

## Server Cross-Check - 2026-05-04

The user checked the remote server workspace under `~/projects` and preserved
the command outputs in a server-side audit directory:

- `~/projects/server_audit_20260504_225229`

Top-level server storage shows the likely authoritative experiment directories:

- `~/projects/uncertainty-llm4rec/` - `13G`
- `~/projects/uncertainty-llm4rec-codex-apr12-preserve-local/` - `1.1G`
- `~/projects/uncertainty-llm4rec_backup_before_week7_5/` - `544M`
- `~/projects/fresh/` - `18G`

### Week7.7 SRPD server export

The week7.7 four-domain export is present on the server at:

- `~/projects/uncertainty-llm4rec/export/week7_7_four_domain_final`

This directory contains direct ranking, structured-risk rerank, SRPD prediction
folders, summary files, and logs. The local paper summaries therefore do have a
server-side raw/export source, even though `server_sync/` is absent locally.

Server summary files include teacher-data alignment CSVs for:

- Beauty full973: SRPD-v1, v2, v3, v4, v5.
- Books small500: SRPD-v2, v4, v5.
- Electronics small500: SRPD-v2, v4, v5.
- Movies small500: SRPD-v2, v4, v5.

All checked SRPD teacher-data summaries report:

- `status`: `srpd_teacher_data_ready`
- `teacher_match_rate`: `1.0`
- `missing_base_rows`: `0`
- Beauty: `base_rows=973`, `matched_rows=973`, `train_rows=778`,
  `valid_rows=195`
- Books/Electronics/Movies: `base_rows=500`, `matched_rows=500`,
  `train_rows=400`, `valid_rows=100`

Core week7.7 prediction counts are complete:

- Beauty direct rank and SRPD-v1..v5 prediction files each have `973` rows.
- Books direct rank and SRPD-v2/v4/v5 prediction files each have `500` rows.
- Electronics direct rank and SRPD-v2/v4/v5 prediction files each have `500`
  rows.
- Movies direct rank and SRPD-v2/v4/v5 prediction files each have `500` rows.

Checked week7.7 metrics from server tables:

- Beauty full973: best SRPD row is `SRPD-v2`, NDCG@10 `0.6353658183`,
  MRR `0.5184480987`. Structured-risk rerank is NDCG@10 `0.6140777971`,
  MRR `0.4900479616`.
- Books small500: best SRPD row is `SRPD-v2`, NDCG@10 `0.7057074243`,
  MRR `0.6117666667`. Structured-risk rerank is NDCG@10 `0.6395142413`,
  MRR `0.5235`.
- Electronics small500: best SRPD row is `SRPD-v5`, NDCG@10 `0.6621001801`,
  MRR `0.5528333333`. Structured-risk rerank is NDCG@10 `0.6583007652`,
  MRR `0.5471666667`.
- Movies small500: structured-risk rerank remains the best checked method,
  NDCG@10 `0.5731826886`, MRR `0.4391000000`. Best SRPD repair row is
  `SRPD-v4`, NDCG@10 `0.5463885554`, MRR `0.4043666667`.

These server values match the local `Paper/pony2/results` conclusions.

### Week7.9 shadow server backup

The completed shadow summaries are present on the server at:

- `~/projects/uncertainty-llm4rec-codex-apr12-preserve-local/outputs/summary`

The server has completed `week7_9_shadow_small_prior_summary.csv` and
`week7_9_shadow_full_replay_summary.csv` files, unlike the incomplete local
summary visible in this rescue workspace.

Checked small-prior shadow rows:

- Beauty has ready `shadow_v1`, `shadow_v2`, and `shadow_v5` rows with
  pointwise, calibration, rerank, noisy pointwise, and noisy rerank all marked
  `ready`.
- Books has ready `shadow_v1`, `shadow_v2`, and `shadow_v5` rows with the same
  ready statuses.
- Beauty small-prior rerank NDCG@10 is `0.6394303493` for v1/v2/v5; noisy
  rerank NDCG@10 is `0.6277602928`.
- Books small-prior best checked rerank row is `shadow_v5`, NDCG@10
  `0.6516282885`, MRR `0.5400555556`; noisy rerank NDCG@10 is
  `0.6567818881`.

Checked full-replay shadow rows:

- Full replay has ready `shadow_v1` rows for Beauty, Books, Electronics, and
  Movies, including noisy variants.
- Beauty `shadow_v1` full replay: pointwise AUROC `0.5723036899`, rerank
  NDCG@10 `0.6353973143`, MRR `0.5184823570`.
- Books `shadow_v1` full replay: pointwise AUROC `0.7133176`, rerank NDCG@10
  `0.6362591123`, MRR `0.5191`.
- Electronics `shadow_v1` full replay: pointwise AUROC `0.6230092`, rerank
  NDCG@10 `0.6574286221`, MRR `0.5460333333`.
- Movies `shadow_v1` full replay: pointwise AUROC `0.5223908`, rerank NDCG@10
  `0.5725075524`, MRR `0.4381666667`.

Practical implication:

- Server results support `shadow_v1` as the current full-replay winner signal
  source for a conservative v6 bridge.
- `shadow_v2` and `shadow_v5` are small-prior alternatives for Beauty/Books,
  but they are not four-domain full-replay winners in the checked summaries.
- The next server step should not rerun everything blindly. First copy or
  reference the server audit files and use the server raw/export folders as the
  authoritative source for any v6 bridge command construction.

## Server Follow-Up

Before claiming final status or running the next large job, recover or preserve
from the server:

1. The server audit directory:
   `~/projects/server_audit_20260504_225229`.
2. The week7.7 source export:
   `~/projects/uncertainty-llm4rec/export/week7_7_four_domain_final`.
3. The week7.9 shadow summaries and raw output folders:
   `~/projects/uncertainty-llm4rec-codex-apr12-preserve-local/outputs`.
4. The exact command logs and environment used for week7.7 and week7.9.

## Working Rule

Use local summaries as navigation and hypothesis evidence. Use server raw
artifacts plus manifests/logs as the final authority before paper-result
promotion.
