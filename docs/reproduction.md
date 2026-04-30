# Reproduction

This repository supports two reproduction levels:

- smoke reproduction: parser, metric, protocol, and audit sanity checks
- main-table reproduction: aggregation and paper-facing table generation from
  existing experiment outputs

## Environment

Python 3.12 is recommended.

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Smoke test

Linux/macOS:

```bash
bash scripts/reproduce_smoke_test.sh
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe main_baseline_reliability_audit.py
.\.venv\Scripts\python.exe main_audit_candidate_protocol.py --domain beauty --data_dir data/processed/amazon_beauty
.\.venv\Scripts\python.exe main_generative_title_bridge_status.py
```

Expected smoke outputs:

- `outputs/summary/baseline_reliability_proxy_audit.csv`
- `outputs/summary/candidate_protocol_audit.csv`
- `outputs/summary/generative_title_bridge_status.csv`

## Main table reproduction

Linux/macOS:

```bash
bash scripts/reproduce_main_tables.sh
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe main_aggregate_all.py --output_root outputs
.\.venv\Scripts\python.exe main_baseline_reliability_audit.py
.\.venv\Scripts\python.exe main_audit_candidate_protocol.py --domain beauty --data_dir data/processed/amazon_beauty
.\.venv\Scripts\python.exe main_generative_title_bridge_status.py
```

Expected summary outputs:

- `outputs/summary/final_results.csv`
- `outputs/summary/weekly_summary.csv`
- `outputs/summary/model_results.csv`
- `outputs/summary/baseline_reliability_proxy_audit.csv`
- `outputs/summary/candidate_protocol_audit.csv`
- `outputs/summary/main_table_with_ci.csv` when paired method records are
  provided to `main_stat_tests.py`

## Statistical tests

Use paired event records with stable event ids:

```bash
python main_stat_tests.py \
  --input_paths direct=outputs/direct/tables/ranking_eval_records.csv ccrp=outputs/ccrp/tables/ranking_eval_records.csv \
  --output_dir outputs/summary
```

The script writes:

- `outputs/summary/significance_tests.csv`
- `outputs/summary/main_table_with_ci.csv`

Rows whose confidence interval crosses zero or whose corrected p-value is not
significant are labeled `observed_best`, not `winner`.
