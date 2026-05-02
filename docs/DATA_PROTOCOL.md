# Data Protocol

Raw files are the source of truth. The new path supports Amazon JSONL/GZip review/meta files and Movie CSV-style files.

Default protocol:

- Normalize to `user_id`, `item_id`, `rating`, `timestamp`, `title`, `categories`, `description`, `candidate_text`.
- Keep positive interactions with `rating >= 4`.
- Apply iterative user-item k-core filtering, default `k_core: 5`.
- Build chronological user sequences.
- Use temporal leave-one-out: train history is all except last two interactions, validation target is second last, test target is last.
- Compute popularity buckets from training histories only.
- Build deterministic target-plus-negative candidates under seed.
- Exclude the target item, user history, and all user positives from candidate negatives.

Commands:

```bash
python -m src.cli.preprocess --config configs/datasets/amazon_beauty.yaml
python -m src.cli.build_candidates --config configs/datasets/amazon_beauty.yaml --negative_count 99 --strategy popularity_stratified
```

The smoke dataset under `tests/fixtures/` is only for tests and smoke validation.

Raw-data readiness check:

```bash
python -m src.cli.validate_raw_data --dataset_config configs/datasets/amazon_beauty.yaml
python -m src.cli.validate_raw_data --dataset_config configs/datasets/amazon_electronics.yaml
python -m src.cli.validate_raw_data --dataset_config configs/datasets/amazon_books.yaml
python -m src.cli.validate_raw_data --dataset_config configs/datasets/movie.yaml
```

If files are absent, the validator fails with the exact expected path and field format.
