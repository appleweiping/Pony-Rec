# RecBole smoke baseline on reprocessed processed-source cohort

End-to-end **pilot** check (`run_type=pilot`, `backend_type=baseline`, `is_paper_result=false`) of RecBole on the **same 20-user-per-domain** cohort as `outputs/reprocessed_processed_source/` (seed 42). This is **not** tuned, **not** full data, and **not** a paper comparison.

## Explicit scope (read before interpreting numbers)

1. **RecBole uses exported atomic files** — training/evaluation read RecBole tab-separated **`*.inter`** and **`*.item`** under `outputs/pilots/recbole_smoke_processed_20u_seed42/atomic/<domain>/` (plus generated side files RecBole writes beside them). Those atomic files are produced from **`data/processed/<domain>/interactions.csv`** (and **`items.csv`**) restricted to users listed in the reprocess split JSONL.
2. **RecBole does not consume LLM candidate JSONL** — `train_candidates.jsonl`, `valid_candidates.jsonl`, and `test_candidates.jsonl` are **not** inputs to this RecBole smoke. They remain part of the **LLM listwise** protocol only.
3. **Smoke baseline, not a paper result** — goal is integration confidence (export → RecBole → metrics) on a **tiny** cohort, not leaderboard-quality scores. Do **not** treat these metrics as publication baselines.
4. **Hyperparameters are tiny smoke settings, not tuned** — e.g. Pop uses **1** epoch; other models **3** epochs with reduced `train_batch_size` / `eval_batch_size` and smaller embeddings/hidden sizes than a real study would use.
5. **Outputs are labeled pilot baseline** — `smoke_run_summary.json` and each `runs/<domain>/<model>/result.json` → `meta` record **`run_type=pilot`**, **`backend_type=baseline`**, **`is_paper_result=false`**.

## What runs

1. **Export** RecBole atomic `.inter` / `.item` per domain: all interaction rows from `data/processed/<domain>/interactions.csv` whose `user_id` appears in that domain’s `train.jsonl`, `valid.jsonl`, or `test.jsonl` under `outputs/reprocessed_processed_source/<domain>/`.
2. **Train/eval** via RecBole’s built-in temporal leave-one-out style split (`LS: valid_and_test`, `order: TO`), **not** the LLM `*_candidates.jsonl` ranking task. Candidate JSONL remains the protocol reference for LLM work; RecBole uses the full filtered interaction log.

Models (RecBole class names): **Pop**, **BPR** (BPRMF-style pairwise MF), **LightGCN**, **SASRec**, **BERT4Rec**.

## Commands

From repo root (`/home/ajifang/projects/fresh/uncertainty-llm4rec`):

```bash
.venv_lora/bin/python3.11 -m pip install -r configs/baselines/requirements-recbole-smoke.txt
.venv_lora/bin/python3.11 -m src.cli.run_recbole_smoke_reprocessed
```

Defaults:

- `--reprocess_root outputs/reprocessed_processed_source`
- `--processed_root data/processed`
- `--output_root outputs/pilots/recbole_smoke_processed_20u_seed42`
- Domains: `amazon_beauty`, `amazon_books`, `amazon_electronics`, `amazon_movies`
- `--seed 42`

## Outputs

- **Atomic datasets:** `outputs/pilots/recbole_smoke_processed_20u_seed42/atomic/<domain>/`
- **Per run:** `outputs/pilots/recbole_smoke_processed_20u_seed42/runs/<domain>/<model>/`
  - `baseline_smoke.yaml` — merged smoke hyperparameters
  - `*_recbole.yaml` — RecBole-resolved config
  - `result.json` — `test_result`, `best_valid_result`, `meta` (`run_type`, `backend_type`, `is_paper_result`, `config_hash`)
  - `saved/` — checkpoints (pilot only; safe to delete)
- **Aggregate:** `outputs/pilots/recbole_smoke_processed_20u_seed42/smoke_run_summary.json`

## Dependency failures encountered (resolved for this smoke)

| Issue | Mitigation |
|--------|------------|
| `ModuleNotFoundError: pkg_resources` (Ray) | `setuptools<81` |
| `numpy.bool8` removed (Ray + NumPy 2) | `numpy<2` |
| SASRec / BERT4Rec: `train_neg_sample_args` must be `None` for CE loss | Set in smoke merge (`src/cli/run_recbole_smoke_reprocessed.py`) |
| LightGCN: `dok_matrix` has no `_update` | `scipy>=1.10,<1.11` (e.g. `1.10.1`) |

Install line for a clean venv:

```text
pip install -r configs/baselines/requirements-recbole-smoke.txt
```

## Smoke status (from `smoke_run_summary.json`)

**Per-run `ok`:** all **20** cells are `true` (4 domains × 5 models: Pop, BPR, LightGCN, SASRec, BERT4Rec).

| domain | Pop | BPR | LightGCN | SASRec | BERT4Rec |
|--------|-----|-----|------------|--------|----------|
| amazon_beauty | ok | ok | ok | ok | ok |
| amazon_books | ok | ok | ok | ok | ok |
| amazon_electronics | ok | ok | ok | ok | ok |
| amazon_movies | ok | ok | ok | ok | ok |

All runs used the dependency pins below.

## Test metrics snapshot (RecBole `test_result`, NDCG@10)

Smoke epochs: Pop **1**; others **3** with small batch sizes and reduced embedding/hidden sizes. Values are **diagnostic only** (tiny data, no tuning).

| domain | Pop | BPR | LightGCN | SASRec | BERT4Rec |
|--------|-----|-----|------------|--------|----------|
| amazon_beauty | 0.0178 | 0.0366 | 0.0393 | 0.0687 | 0.0 |
| amazon_books | 0.0178 | 0.0167 | 0.0 | 0.0145 | 0.0 |
| amazon_electronics | 0.0486 | 0.0 | 0.0 | 0.0 | 0.0 |
| amazon_movies | 0.0494 | 0.05 | 0.0 | 0.0 | 0.0 |

Many zeros are expected on **20 users** and 3 epochs (especially sequential models). BERT4Rec often stays at zero in this regime.

## Next blockers before serious baselines

1. **Dependencies:** Pin `numpy`, `setuptools`, and `scipy` for RecBole 1.2.x or upgrade RecBole when upstream fixes Ray / sparse / pandas CoW issues.
2. **Scale:** Re-export atomic data for larger user samples; tune `epochs`, `eval_batch_size`, and model capacity separately from this smoke.
3. **Alignment with LLM eval:** RecBole ranking metrics are on its internal split of the **interaction matrix**, not on the fixed 19-item candidate lists used for LLM listwise ranking. A fair comparison needs a dedicated scoring step (e.g. score candidate sets from `*_candidates.jsonl`).
4. **Artifacts:** Per-run `saved/` checkpoints grow; add retention policy or `saved=False` in a future wrapper if disk matters.

## Code entrypoints

- `src.cli.run_recbole_smoke_reprocessed` — smoke driver
- `src.baselines.recbole_adapter.export_recbole_atomic_for_reprocess_users` — cohort-filtered export
- `src.cli.run_recbole_baseline` — single-model runner (unchanged contract)
