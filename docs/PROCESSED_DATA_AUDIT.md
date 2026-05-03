# Processed Data Audit

## Scope

Workdir:

```text
/home/ajifang/projects/fresh/uncertainty-llm4rec
```

Original audit source:

The old project source was `/home/ajifang/projects/uncertainty-llm4rec/data/processed`. It was used only to create clean symlinks and must not be modified by the fresh project.

Important source/entry distinction:

- `data/processed` in the fresh project is now the only allowed processed-data entrypoint for current experiments.
- It is not the old full symlink. It is a clean local directory containing only four official domain directories and four symlinked CSV base tables per domain: `interactions.csv`, `items.csv`, `popularity_stats.csv`, and `users.csv`.
- The old full processed tree may contain `*_small`, `*_noisy`, `srpd/`, split JSONL, ranking/pairwise, prompt, prediction, LoRA, and repaired artifacts; those are explicitly excluded from the fresh project entrypoint.
- Current official dataset configs point to `data/processed/...`, where that path means the clean fresh-project entrypoint.

Only these four full-domain processed directories were audited:

- `data/processed/amazon_beauty`
- `data/processed/amazon_books`
- `data/processed/amazon_electronics`
- `data/processed/amazon_movies`

Excluded by design: raw data, `tests/fixtures`, `srpd/`, `small/`, `*_small`, `*_noisy`, `outputs`, repaired predictions, old LoRA predictions.

## Executive Conclusion

ACCEPT_AS_IS:

- The four domain CSV bundles can be accepted as processed source tables for a new reprocessing step. Each domain has `interactions.csv`, `items.csv`, `users.csv`, and `popularity_stats.csv`; all are UTF-8 CSV-like files with headers and the expected semantic columns.
- `interactions.csv` has `user_id`, `item_id`, `rating`, and `timestamp`; all audited ratings are already `>= 4`.
- `items.csv` has unique `item_id`, title text, `candidate_text`, and domain text fields usable for LLM prompts.
- `users.csv` has unique `user_id`, and all interaction users are covered.

ACCEPT_AFTER_REPROCESS:

- Use these CSVs only after rebuilding per-user temporal leave-one-out splits, train-only popularity counts/buckets, deterministic candidates, and manifests in the fresh repo.
- Existing full-domain `popularity_stats.csv` matches full interactions exactly, but that means it may include validation/test targets. Treat it as a descriptive full-corpus statistic, not as a leakage-safe feature for evaluation.
- Existing old split/sample files are incomplete across domains and should not be reused as experiment evidence.

REJECT_FOR_PAPER_USE:

- Existing `train.jsonl`, `valid.jsonl`, `test.jsonl`, `pairwise_*`, `ranking_*`, `pairwise_coverage_*`, old predictions, repaired predictions, and LoRA artifacts are rejected for paper use unless regenerated under the fresh repo protocol with manifests.
- The current CSV state is not sufficient for final paper results because split/candidate generation and train-only popularity must be regenerated and logged.

Next allowed step, if proceeding, should be a small pilot from reprocessed samples. Do not jump directly to a full experiment.

## Per-Domain Audit

### `amazon_beauty`

Exact path: `data/processed/amazon_beauty`

Files:

- `interactions.csv`: 6,583 rows; columns `user_id`, `item_id`, `rating`, `timestamp`; UTF-8; comma delimiter; header present.
- `items.csv`: 1,184 rows; columns `item_id`, `title`, `categories`, `description`, `candidate_text`, `popularity_group`; UTF-8; comma delimiter; header present.
- `users.csv`: 992 rows; columns `user_id`; UTF-8; header present.
- `popularity_stats.csv`: 1,184 rows; columns `item_id`, `interaction_count`, `popularity_group`; UTF-8; comma delimiter; header present.

Schema findings:

- Interactions are not already sorted by `user_id/timestamp/item_id`; re-sort before splitting.
- Duplicate `user-item-time`: 0.
- Users: 992; items: 1,184; interactions: 6,583.
- User interaction count distribution: min 3, median 4, mean 6.64, p90 12, p95 20, p99 36.09, max 107.
- Item interaction count distribution: min 3, median 5, mean 5.56, p90 9, p95 11, p99 15, max 39.
- Core status: satisfies 3-core; does not satisfy 5-core or 10-core.
- Rating filter: already rating `>= 4`; rating values are 2,024 rows with `4.0` and 4,559 rows with `5.0`.
- After item de-duplication by user, 973 users have at least 3 unique interactions and 436 users have at least 5.

Item/user/popularity consistency:

- `item_id` unique in `items.csv`; no interaction item is missing from items; no extra item without interaction.
- `user_id` unique in `users.csv`; no interaction user is missing from users; no extra user without interaction.
- `popularity_stats.csv` item IDs are unique and counts match full `interactions.csv` exactly.
- Buckets exist: head 236, mid 474, tail 474.
- Leakage risk: popularity is full-interaction popularity, not confirmed train-only popularity.

Split/candidate status:

- Existing old files: `train.jsonl`, `valid.jsonl`, `test.jsonl`, `pairwise_valid.jsonl`, `pairwise_test.jsonl`, `ranking_valid.jsonl`, `ranking_test.jsonl`, `pairwise_coverage_valid.jsonl`, `pairwise_coverage_test.jsonl`.
- These should not be trusted for new evidence; regenerate splits/candidates.
- Candidate generation from CSV is feasible for sizes 19, 99, and 199.

### `amazon_books`

Exact path: `data/processed/amazon_books`

Files:

- `interactions.csv`: 12,442,971 rows; columns `user_id`, `item_id`, `rating`, `timestamp`; UTF-8; comma delimiter; header present.
- `items.csv`: 963,824 rows; columns `item_id`, `title`, `categories`, `description`, `candidate_text`, `popularity_group`; UTF-8; comma delimiter; header present.
- `users.csv`: 1,711,729 rows; columns `user_id`; UTF-8; header present.
- `popularity_stats.csv`: 963,824 rows; columns `item_id`, `interaction_count`, `popularity_group`; UTF-8; comma delimiter; header present.

Schema findings:

- Interactions are not already sorted by `user_id/timestamp/item_id`; re-sort before splitting.
- Duplicate `user-item-time`: 0.
- Users: 1,711,729; items: 963,824; interactions: 12,442,971.
- User interaction count distribution: min 3, median 4, mean 7.27, p90 12, p95 19, p99 51, max 3,037.
- Item interaction count distribution: min 3, median 5, mean 12.91, p90 23, p95 39, p99 118, max 14,503.
- Core status: satisfies 3-core; does not satisfy 5-core or 10-core.
- Rating filter: already rating `>= 4`; 2,651,385 rows with rating 4 and 9,791,586 with rating 5.
- After item de-duplication by user, 1,711,590 users have at least 3 unique interactions and 779,508 have at least 5.

Item/user/popularity consistency:

- `item_id` unique in `items.csv`; all interaction items are covered; no extra item without interaction.
- `user_id` unique in `users.csv`; all interaction users are covered; no extra user without interaction.
- Text readiness sample: title non-empty 100%; category non-empty about 96.78%; candidate text/text field non-empty 100%.
- `popularity_stats.csv` counts match full `interactions.csv` exactly.
- Buckets exist: head 192,764, mid 385,530, tail 385,530.
- Leakage risk: full-interaction popularity must not be used as validation/test feature without recomputing from train.

Split/candidate status:

- No full-domain `train/valid/test` split files found in this directory.
- Candidate generation from CSV is feasible for sizes 19, 99, and 199.

### `amazon_electronics`

Exact path: `data/processed/amazon_electronics`

Files:

- `interactions.csv`: 17,515,103 rows; columns `user_id`, `item_id`, `rating`, `timestamp`; UTF-8; comma delimiter; header present.
- `items.csv`: 533,532 rows; columns `item_id`, `title`, `categories`, `description`, `candidate_text`, `popularity_group`; UTF-8; comma delimiter; header present.
- `users.csv`: 3,010,273 rows; columns `user_id`; UTF-8; header present.
- `popularity_stats.csv`: 533,532 rows; columns `item_id`, `interaction_count`, `popularity_group`; UTF-8; comma delimiter; header present.

Schema findings:

- Interactions are not already sorted by `user_id/timestamp/item_id`; re-sort before splitting.
- Duplicate `user-item-time`: 0.
- Users: 3,010,273; items: 533,532; interactions: 17,515,103.
- User interaction count distribution: min 3, median 4, mean 5.82, p90 10, p95 14, p99 29, max 866.
- Item interaction count distribution: min 3, median 8, mean 32.83, p90 52, p95 104, p99 411, max 63,607.
- Core status: satisfies 3-core; does not satisfy 5-core or 10-core.
- Rating filter: already rating `>= 4`; 2,928,948 rows with rating 4 and 14,586,155 with rating 5.
- After item de-duplication by user, 3,007,066 users have at least 3 unique interactions and 1,249,031 have at least 5.

Item/user/popularity consistency:

- `item_id` unique in `items.csv`; all interaction items are covered; no extra item without interaction.
- `user_id` unique in `users.csv`; all interaction users are covered; no extra user without interaction.
- Text readiness sample: title non-empty 100%; category non-empty about 95.83%; candidate text/text field non-empty 100%.
- `popularity_stats.csv` counts match full `interactions.csv` exactly.
- Buckets exist: head 106,706, mid 213,413, tail 213,413.
- Leakage risk: full-interaction popularity must be replaced by train-only popularity for experimental features and head/mid/tail analysis.

Split/candidate status:

- No full-domain `train/valid/test` split files found in this directory.
- Candidate generation from CSV is feasible for sizes 19, 99, and 199.

### `amazon_movies`

Exact path: `data/processed/amazon_movies`

Files:

- `interactions.csv`: 7,895,387 rows; columns `user_id`, `item_id`, `rating`, `timestamp`; UTF-8; comma delimiter; header present.
- `items.csv`: 258,113 rows; columns `item_id`, `title`, `categories`, `description`, `candidate_text`, `popularity_group`; UTF-8; comma delimiter; header present.
- `users.csv`: 1,174,881 rows; columns `user_id`; UTF-8; header present.
- `popularity_stats.csv`: 258,113 rows; columns `item_id`, `interaction_count`, `popularity_group`; UTF-8; comma delimiter; header present.

Schema findings:

- Interactions are not already sorted by `user_id/timestamp/item_id`; re-sort before splitting.
- Duplicate `user-item-time`: 0.
- Users: 1,174,881; items: 258,113; interactions: 7,895,387.
- User interaction count distribution: min 3, median 4, mean 6.72, p90 12, p95 17, p99 42, max 2,098.
- Item interaction count distribution: min 3, median 8, mean 30.59, p90 59, p95 107, p99 336.88, max 27,705.
- Core status: satisfies 3-core; does not satisfy 5-core or 10-core.
- Rating filter: already rating `>= 4`; 1,608,475 rows with rating 4 and 6,286,912 with rating 5.
- After item de-duplication by user, 1,174,876 users have at least 3 unique interactions and 527,704 have at least 5.

Item/user/popularity consistency:

- `item_id` unique in `items.csv`; all interaction items are covered; no extra item without interaction.
- `user_id` unique in `users.csv`; all interaction users are covered; no extra user without interaction.
- Text readiness sample: title non-empty 100%; category non-empty about 99.45%; candidate text/text field non-empty 100%.
- `popularity_stats.csv` counts match full `interactions.csv` exactly.
- Buckets exist: head 51,622, mid 103,245, tail 103,246.
- Leakage risk: full-interaction popularity must be replaced by train-only popularity for leakage-safe evaluation.

Split/candidate status:

- Existing `train.jsonl` has 56,690 rows, but `valid.jsonl` and `test.jsonl` are absent.
- Existing split state is incomplete and should not be used.
- Candidate generation from CSV is feasible for sizes 19, 99, and 199.

## Split Protocol Assessment

No domain should use the old split/sample artifacts as-is.

Recommended split:

- Sort each user history by `timestamp`, tie-breaking by `item_id`.
- De-duplicate repeated user-item interactions by keeping the first occurrence if the task is next new-item recommendation.
- Use per-user temporal leave-one-out:
  - `train`: each user's history except last two interactions.
  - `valid`: second-last interaction.
  - `test`: last interaction.
- Require at least 3 unique interactions per user for the 3-split protocol. If a stricter experiment needs 5-core behavior, apply that explicitly and report the smaller user set.

Leakage note:

- Per-user temporal leave-one-out is appropriate for sequential recommendation and prevents each user's future target from appearing in that user's own history.
- It is not a global time split. If the paper claim requires strict global chronology, this must be separately implemented and may reduce usable data.
- Literature on offline recommendation evaluation warns that ignoring temporal order can leak future behavior into training/evaluation. This audit therefore rejects random or unknown old splits.

## Candidate Generation Readiness

All four domains can support target-plus-negative candidate sets with 19, 99, and 199 negatives.

Required rules:

- Negative items must not contain the target.
- Negative items must not contain any item in the user's history.
- Negative items must not contain any item in the user's full positive sequence.
- Sampling must use a fixed seed and deterministic per-user/per-target RNG.
- Candidate outputs must include candidate IDs and prompt-ready candidate text.

## Current Code Compatibility

Existing code:

- `src/data/protocol.py` supports raw-to-processed preprocessing and can build candidates from existing `train/valid/test.jsonl`.
- `src/cli/infer.py` expects `*_candidates.jsonl`, not raw processed CSVs.
- `src/baselines/recbole_adapter.py` can export `interactions.csv` to RecBole atomic format if the schema is present.
- Existing code did not provide a clean loader whose contract is “load four processed CSVs, reject old artifacts, recompute popularity, rebuild splits/candidates.”

Implemented minimal compatibility layer:

- Added `src/data/processed_loader.py`.
- Supports only the four full-domain processed directories by default.
- Requires the dataset source to live under the fresh project's clean `data/processed`.
- Reads only `interactions.csv`, `items.csv`, `users.csv`, and `popularity_stats.csv`.
- Rejects `srpd`, `small`, `noisy`, `outputs`, `predictions`, and `repaired` source paths.
- Rejects any extra file or directory inside a domain source, including old `train.jsonl`, `valid.jsonl`, `test.jsonl`, ranking/pairwise files, prompts, predictions, LoRA files, and repaired artifacts.
- Normalizes semantic columns to `user_id`, `item_id`, `rating`, `timestamp`.
- Validates user/item/popularity coverage.
- Does not trust old popularity by default; exposes recomputation and comparison.
- Provides per-user temporal leave-one-out split generation.
- Provides deterministic target-plus-negative candidate construction.

Tests added:

- `tests/test_processed_loader.py`
- Covered processed file loading, required column detection, rejection of `srpd`, `*_small`, and `*_noisy` sources, rejection of old split/prediction artifacts, rejection of non-clean roots, item/user consistency, temporal split no leakage, negative candidate no target/history/full-positive leakage, popularity recomputation mismatch reporting, and official configs pointing at the clean fresh-project `data/processed` entry.
- Test command run:

```bash
.venv_lora/bin/python3.11 -m pytest tests/test_processed_loader.py
```

Result: 7 passed.

## Suitability By Use Case

DeepSeek API pilot:

- Status: ACCEPT_AFTER_REPROCESS.
- Use a small, regenerated pilot split/candidate file from the CSVs. Do not use old split/jsonl artifacts.

Baseline comparison:

- Status: ACCEPT_AFTER_REPROCESS.
- Rebuild temporal splits and export RecBole data from the regenerated protocol. Existing interactions schema is suitable, but evaluation protocol must be regenerated and documented.

LoRA debug:

- Status: ACCEPT_AFTER_REPROCESS.
- Use regenerated small ranking/candidate samples. The CSVs have enough data and item text, but old LoRA predictions/adapters are not evidence.

Final paper results:

- Status: REJECT_FOR_PAPER_USE until reprocessed.
- These CSVs may serve as the processed source tables, but final evidence requires fresh deterministic splits, train-only popularity, candidates, manifests, and rerun predictions/evaluations under the fresh repository.

## Minimum Repair Plan

1. Use `ProcessedDatasetLoader` to load each of the four full-domain CSV directories through the `data/processed` symlink.
2. Re-sort interactions by `user_id`, `timestamp`, `item_id`.
3. Rebuild per-user temporal leave-one-out splits from interactions.
4. Recompute popularity using training interactions only; treat old `popularity_stats.csv` only as an audit reference.
5. Build deterministic candidates with sizes needed for pilot/debug: 19, 99, or 199 negatives.
6. Write fresh manifests that record source symlink, config hash, split method, seed, candidate size, and popularity source.
7. Start with a small pilot only.
