# 100-User Reprocess Protocol (c19, seed42)

This document records the controlled 100-user pilot reprocess inputs/outputs.

- `run_type`: pilot
- `is_paper_result`: false
- `seed`: 42
- `candidate_size`: 19

## Command

```bash
cd /home/ajifang/projects/fresh/uncertainty-llm4rec
.venv_lora/bin/python3.11 -m src.cli.reprocess_processed_source \
  --source_root data/processed \
  --output_dir outputs/reprocessed_processed_source_100u_c19_seed42 \
  --max_users_per_domain 100 \
  --candidate_size 19 \
  --seed 42
```

## Source / Output

- Source root: `data/processed`
- Output root: `outputs/reprocessed_processed_source_100u_c19_seed42`

## Per-domain Row Counts

All four domains produced 100 rows for each split:

- `amazon_beauty`: train=100, valid=100, test=100
- `amazon_books`: train=100, valid=100, test=100
- `amazon_electronics`: train=100, valid=100, test=100
- `amazon_movies`: train=100, valid=100, test=100

Each split also produced matching candidate files with `candidate_size=19`.

## Leakage / Popularity Verification

- Leakage validation from reprocess command: passed for all four domains (`leakage_passed=true`).
- Leakage report files written per domain:
  - `outputs/reprocessed_processed_source_100u_c19_seed42/<domain>/leakage_report.json`
- Train-only popularity recomputation confirmed in manifest:
  - `popularity_source: "train_only"`
- Per-domain manifests written:
  - `outputs/reprocessed_processed_source_100u_c19_seed42/<domain>/manifest.json`

## Path Hygiene

No `srpd`, old split, or old prediction path references were found under:

- `outputs/reprocessed_processed_source_100u_c19_seed42`

## Pilot Scope Statement

This 100-user reprocess is a pilot-stage protocol artifact and is **not** a paper result.
