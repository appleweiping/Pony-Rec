# RecBole 100-User Pilot (Atomic Export + Baselines)

Run scope: pilot-only, `backend_type=baseline`, `is_paper_result=false`.

## Command

```bash
cd /home/ajifang/projects/fresh/uncertainty-llm4rec
.venv_lora/bin/python3.11 -m src.cli.run_recbole_smoke_reprocessed \
  --reprocess_root outputs/reprocessed_processed_source_100u_c19_seed42 \
  --processed_root data/processed \
  --output_root outputs/pilots/recbole_processed_100u_c19_seed42 \
  --domains amazon_beauty amazon_books amazon_electronics amazon_movies \
  --seed 42
```

## Models run

- Pop
- BPR
- LightGCN
- SASRec
- BERT4Rec

## Output root

- `outputs/pilots/recbole_processed_100u_c19_seed42/`

Includes:

- `atomic/<domain>/<domain>.inter` / `<domain>.item`
- `runs/<domain>/<model>/result.json`
- `smoke_run_summary.json`

## Protocol gap (important)

RecBole here consumes **atomic files**, not LLM candidate JSONL ranking outputs directly.

- LLM/CARE pipelines are listwise-candidate JSONL based.
- RecBole pipelines are interaction-matrix / sequence baseline training based.
- Therefore this pilot is a baseline-track reference, not a drop-in substitute for LLM candidate rerank evaluation.

## Pilot notes

- 100-user cohort alignment comes from reprocessed 100u user selection.
- Baseline outputs are intended for controlled pilot comparison and sanity checks, not paper-scale claims.
