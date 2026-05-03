# Repository Cleanup Audit

Date: 2026-05-03  
Scope: `/home/ajifang/projects/fresh/uncertainty-llm4rec` only. Old project and symlink targets outside this repo were not touched.

## Summary

Root-level `main_*.py` scripts (50 files) were **moved** to `legacy/root_main/` so the repo root no longer advertises deprecated entrypoints. References were updated in `src/utils/exp_launcher.py`, selected `src/` imports, `configs/batch/*.yaml`, `scripts/run_*.sh`, `README.md`, and `configs/lora/qwen3_rank_beauty_framework_v1.yaml`. Historical markdown reports at repo root were moved to `legacy/docs/`.

**Current primary interfaces:** `python -m src.cli.*` and `python -m src.cli.reprocess_processed_source` (see `docs/REPROCESS_PROCESSED_SOURCE.md`).

## Classification Table

| path | type | reason | action | replacement | risk |
| --- | --- | --- | --- | --- | --- |
| `legacy/root_main/main_*.py` (~50 files) | Python | Superseded by `src.cli` for new work; retained for week7/part5 reproduction and internal imports | MOVE_TO_LEGACY | `python -m src.cli.infer`, `preprocess`, `evaluate`, `rerank`, etc. (`docs/LEGACY_MIGRATION.md`) | Low if all references updated; some external notes may still mention old paths |
| `legacy/docs/week7_9_shadow_observation_report.md` | Markdown | Week7 narrative; not needed for fresh pilot surface | MOVE_TO_LEGACY | `src/shadow`, `docs/METHOD.md` as needed | Low |
| `legacy/docs/part5_artifact_map.md` | Markdown | Part5 artifact map; historical | MOVE_TO_LEGACY | `src/cli/aggregate.py`, `export_paper_tables` | Low |
| `src/cli/*.py` | Python | Supported entrypoints | KEEP | — | — |
| `src/data/processed_loader.py`, `src/cli/reprocess_processed_source.py` | Python | Clean processed pipeline | KEEP | — | — |
| `tests/` | Python | Regression | KEEP | — | — |
| `configs/` (non-batch) | YAML | Experiment definitions | KEEP | — | Large srpd/lora set is legacy-oriented but not removed |
| `configs/batch/*.yaml` | YAML | Week7 batch commands referenced `main_*.py` | KEEP (paths patched) | Commands now use `python legacy/root_main/main_*.py` where embedded | Medium: batches still encode srpd/week7 workflows |
| `scripts/test_deepseek_live.sh` | Shell | Required tiny API check | KEEP | — | — |
| `scripts/run_*.sh` (non-destructive) | Shell | Staged runs | KEEP (paths patched) | Prefer `src.cli` where possible | Low |
| `Paper/**` | `.gitkeep` only | Placeholder tree | KEEP | — | None |
| `data/processed/` | Data (ignored) | Clean local entrypoint; symlinks only in workspace | IGNORE_LOCAL | `reprocess_processed_source` | Do not delete targets |
| `outputs/**` | Generated (ignored) | Experiment outputs | IGNORE_LOCAL | — | None |
| `.venv*`, `.conda_pkgs/`, `.env` | Local env | Not in git | IGNORE_LOCAL | — | None |
| `prompts/` | Text | Active templates for `src` | KEEP | `src/prompts` for structured builders | Low |
| `docs/PROCESSED_DATA_AUDIT.md`, `docs/REPROCESS_PROCESSED_SOURCE.md`, `docs/SERVER_CONTEXT_RECOVERY.md`, `docs/BLOCKER_RESOLUTION.md` | Markdown | Required project docs | KEEP | — | — |
| `README.md`, `AGENTS.md` | Markdown | Project entry | KEEP | — | — |
| `src/utils/exp_launcher.py` | Python | Batch launcher built `python main_*.py` | KEEP (updated paths) | Uses `legacy/root_main/*.py` when no explicit command | Low |
| `src/methods/baseline_ranker_multitask.py` | Python | Imported helpers from `main_rank_rerank` | KEEP (import from `legacy.root_main`) | Long term: extract helpers into `src/` | Low |
| `src/analysis/aggregate_fused_alpha_ablation.py` | Python | Imported `evaluate_estimator` from legacy compare | KEEP (import from `legacy.root_main`) | Long term: move function into `src/` | Low |

## Not Done (Conservative)

- **Deleting** `configs/srpd/*`, `configs/exp/*` subsets, or large analysis scripts: still referenced by historical workflows; removing would exceed “conservative” cleanup without a dedicated config retirement pass.
- **Deduplicating** all docs under `docs/` beyond `LEGACY_MIGRATION` updates: deferred.
- **Removing** `scripts/generate_noisy_data.py` and other one-offs: still potentially useful; not referenced by core tests.

## Verification Commands (post-cleanup)

See Step 4–6 in the cleanup task: `git grep` for `main_`, `processed_clean`, `srpd`, `data/processed`; `pytest`; `reprocess_processed_source` smoke.
