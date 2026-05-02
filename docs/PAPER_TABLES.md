# Paper Tables

Generated artifacts:

- `main_results.md`: ranking utility table.
- `calibration.md`: ECE, MCE, Brier.
- `exposure.md`: head/tail exposure and long-tail coverage.
- `risk_coverage.csv`: selective prediction curve points.
- `echo_chamber.json`: popularity-confidence and exposure concentration diagnostics.

Command:

```bash
python -m src.cli.aggregate --metrics_glob "outputs/**/eval/metrics.json" --output_path outputs/paper/aggregate.csv
python -m src.cli.export_paper_tables --aggregate_csv outputs/paper/aggregate.csv --output_dir outputs/paper/tables
```

Tables from smoke or mock runs are formatting checks only.
