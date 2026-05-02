# Legacy Migration

The supported Phase-1 interface is `python -m src.cli.*`. Legacy `main_*.py` scripts remain only as historical implementation references until their remaining useful logic is migrated and tested.

## Superseded Entry Points

| Legacy script | Replacement |
| --- | --- |
| `main_preprocess.py` | `python -m src.cli.preprocess` |
| `main_build_samples.py` | `python -m src.cli.build_candidates` after `src.cli.preprocess` |
| `main_infer.py` | `python -m src.cli.infer` |
| `main_calibrate.py`, `main_calibrate_shadow.py` | `python -m src.cli.calibrate` |
| `main_rerank.py`, `main_rank_rerank.py` | `python -m src.cli.rerank` |
| `main_eval.py`, `main_eval_rank.py`, `main_eval_pairwise.py`, `main_eval_shadow.py` | `python -m src.cli.evaluate` |
| `main_aggregate_all.py`, `main_part5_artifacts.py` | `python -m src.cli.aggregate` and `python -m src.cli.export_paper_tables` |
| `main_lora_train_rank.py`, `main_eval_lora_rank.py` | `python -m src.cli.train_lora` and `python -m src.cli.eval_lora` |
| `main_run_literature_baselines.py`, `main_literature_baseline.py`, `main_compare_baselines.py` | `python -m src.cli.baselines`, `python -m src.cli.export_recbole_data`, `python -m src.cli.run_recbole_baseline` |

## Safe To Archive After Verification

- Week-specific comparison scripts: `main_week7_*`, `main_compare_week7_medium_scale.py`, `main_compare_teacher_requested_line.py`.
- Old aggregation/report scripts whose outputs are replaced by `src.cli.aggregate`.
- Noisy-data one-offs once uncertainty-pruned/weighted LoRA data builders fully replace them.

## Still Useful Logic

- `src/data/raw_loaders.py`: raw Amazon field normalization ideas.
- `src/llm/parser.py`: legacy parser edge cases worth preserving as regression tests.
- `src/training/lora_rank_trainer.py`: server LoRA implementation details.
- `src/analysis/*`: plotting and aggregation utilities that can be called by the clean CLI.
- `src/shadow/*`: structured-risk experimental logic, kept as a research branch but not the main paper interface.

## Dangerous Or Deprecated

- Legacy scripts that read/write implicit paths under `outputs/` without manifest metadata.
- Any script that calibrates or tunes on `test` predictions.
- Week-specific scripts with hardcoded experiment names, sample caps, or backend assumptions.
- Any script producing mock/smoke artifacts without `run_type`, `backend_type`, and `is_paper_result`.

New users should not call `main_*.py`. If a legacy script is needed for archaeology, run it only after checking this document and recording why the clean CLI was insufficient.
