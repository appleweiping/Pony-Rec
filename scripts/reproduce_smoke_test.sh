#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"

"${PYTHON_CMD[@]}" -m unittest discover -s tests
"${PYTHON_CMD[@]}" main_baseline_reliability_audit.py
"${PYTHON_CMD[@]}" main_audit_candidate_protocol.py \
  --domain beauty \
  --data_dir data/processed/amazon_beauty \
  --negative_sampling_strategy sampled_candidate_one_positive \
  --output_path outputs/summary/candidate_protocol_audit.csv
"${PYTHON_CMD[@]}" main_generative_title_bridge_status.py
