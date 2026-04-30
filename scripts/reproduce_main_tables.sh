#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

"${PYTHON_CMD[@]}" main_aggregate_all.py --output_root "${OUTPUT_ROOT}"
"${PYTHON_CMD[@]}" main_baseline_reliability_audit.py \
  --output_path "${OUTPUT_ROOT}/summary/baseline_reliability_proxy_audit.csv"
"${PYTHON_CMD[@]}" main_audit_candidate_protocol.py \
  --domain beauty \
  --data_dir data/processed/amazon_beauty \
  --negative_sampling_strategy sampled_candidate_one_positive \
  --output_path "${OUTPUT_ROOT}/summary/candidate_protocol_audit.csv"
"${PYTHON_CMD[@]}" main_generative_title_bridge_status.py \
  --output_path "${OUTPUT_ROOT}/summary/generative_title_bridge_status.csv"
