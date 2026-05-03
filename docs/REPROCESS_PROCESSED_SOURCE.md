# Reprocess Processed Source

## Purpose

`data/processed` is the clean official processed source entry for the fresh project. It contains only four audited Amazon domains and only the four base CSV tables per domain. It must not read or depend on old `train.jsonl`, `valid.jsonl`, `test.jsonl`, `srpd`, prediction, repair, LoRA, or output artifacts from the old project.

The old project source remains external at `/home/ajifang/projects/uncertainty-llm4rec/data/processed` and must not be modified.

## Command

Tiny smoke command run:

```bash
.venv_lora/bin/python3.11 -m src.cli.reprocess_processed_source \
  --source_root data/processed \
  --output_dir outputs/reprocessed_processed_source \
  --max_users_per_domain 20 \
  --candidate_size 19 \
  --seed 42
```

This is a smoke run only. It intentionally limits each domain to 20 users.

## Source Tables

Each domain reads exactly:

- `data/processed/<domain>/interactions.csv`
- `data/processed/<domain>/items.csv`
- `data/processed/<domain>/popularity_stats.csv`
- `data/processed/<domain>/users.csv`

The clean source layout was verified to contain 16 files total:

- `amazon_beauty`: 6,583 interactions; 1,184 items; 992 users; 1,184 full-corpus popularity rows.
- `amazon_books`: 12,442,971 interactions; 963,824 items; 1,711,729 users; 963,824 full-corpus popularity rows.
- `amazon_electronics`: 17,515,103 interactions; 533,532 items; 3,010,273 users; 533,532 full-corpus popularity rows.
- `amazon_movies`: 7,895,387 interactions; 258,113 items; 1,174,881 users; 258,113 full-corpus popularity rows.

The source `popularity_stats.csv` files are treated as audit references only because they match full interactions and may include validation/test targets.

## Output Directory

Outputs are written under `outputs/reprocessed_processed_source/`.

For each domain, the CLI writes:

- `train.jsonl`
- `valid.jsonl`
- `test.jsonl`
- `train_candidates.jsonl`
- `valid_candidates.jsonl`
- `test_candidates.jsonl`
- `train_popularity.csv`
- `manifest.json`
- `split_statistics.json`
- `leakage_report.json`

The smoke run also writes `outputs/reprocessed_processed_source/summary_manifest.json`.

## Split Protocol

The split protocol is per-user temporal leave-one-out:

- Sort interactions by `user_id`, `timestamp`, and `item_id`.
- Drop duplicate `user_id` and `item_id` pairs, keeping the earliest event.
- Keep users with at least three interactions.
- Train uses the user history before the final two events.
- Validation target is the second-last event.
- Test target is the last event.

This protocol prevents per-user future target leakage, but it is not a global chronological split.

## Candidate Protocol

Candidates are generated as target plus seeded uniform negatives:

- `candidate_size=19` in the smoke run, meaning one target and 18 negatives.
- Negatives exclude the target item.
- Negatives exclude every item in the user's full positive history.
- Sampling is deterministic from `seed`, split name, user id, target item id, and candidate size.
- Candidate rows include item ids, item text fields, train-only popularity counts, and train-only head/mid/tail buckets.

## Leakage Checks

The CLI writes a leakage report per domain. The smoke run passed for all four domains:

- `amazon_beauty`: 20 train, 20 valid, 20 test rows; candidate size 19; leakage violations 0.
- `amazon_books`: 20 train, 20 valid, 20 test rows; candidate size 19; leakage violations 0.
- `amazon_electronics`: 20 train, 20 valid, 20 test rows; candidate size 19; leakage violations 0.
- `amazon_movies`: 20 train, 20 valid, 20 test rows; candidate size 19; leakage violations 0.

The leakage report checks that targets are not in history, negatives do not include the target, and negatives do not include any user history or full-positive item.

## Eligibility

- Smoke: eligible. The command above is the verified tiny run.
- Pilot: not yet eligible from this smoke alone. Use a larger but still bounded reprocess run and inspect manifests before API or model pilots.
- Paper result: not eligible from this smoke. Paper outputs require a full reprocess run with documented settings, stable manifests, train-only popularity, leakage reports, and downstream experiment provenance.
