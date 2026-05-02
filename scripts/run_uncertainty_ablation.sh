#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python}"

INPUT="${1:-outputs/smoke_mock/calibrated/test_calibrated.jsonl}"
OUT_ROOT="${2:-outputs/smoke_mock/uncertainty_ablation}"
mkdir -p "$OUT_ROOT"

for lambda in 0.0 0.25 0.5 1.0
do
  $PYTHON_BIN -m src.cli.rerank \
    --input_path "$INPUT" \
    --output_path "$OUT_ROOT/rerank_lambda_${lambda}.jsonl" \
    --lambda_penalty "$lambda"
  $PYTHON_BIN -m src.cli.evaluate \
    --predictions_path "$OUT_ROOT/rerank_lambda_${lambda}.jsonl" \
    --output_dir "$OUT_ROOT/eval_lambda_${lambda}"
done
