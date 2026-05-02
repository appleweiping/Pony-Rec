# Reproducibility

Every output row produced by the new inference path includes dataset, domain, split, seed, method, backend/model, prompt template ID, config hash, git commit when available, timestamp, raw response, usage, latency, retry count, and cache-hit status.

Artifact classes:

- `smoke`: tiny mock/local runs proving pipeline mechanics.
- `pilot`: small real-backend runs used to estimate cost and failure modes.
- `diagnostic`: analysis artifacts used to debug methods and data.
- `paper_result`: full protocol outputs generated from raw data, approved configs, and non-mock backends/baselines.

Only `paper_result` artifacts may be cited as evidence.

Each CLI run writes `manifest.json` where practical. The manifest records git commit, config hash, dataset, raw and processed paths, method, backend, model, prompt template, seed, candidate size, calibration source, command, environment, API-key use, mock-data use, `run_type`, `backend_type`, and `is_paper_result`.

`python -m src.cli.export_paper_tables` refuses smoke/mock/missing-metadata rows unless `--allow_smoke` is passed.
