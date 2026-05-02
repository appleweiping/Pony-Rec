#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python}"

$PYTHON_BIN -m src.cli.preprocess --config configs/datasets/smoke_amazon_tiny.yaml
$PYTHON_BIN -m src.cli.build_candidates --config configs/datasets/smoke_amazon_tiny.yaml
$PYTHON_BIN -m src.cli.infer --config configs/experiments/smoke_mock.yaml --split valid
$PYTHON_BIN -m src.cli.infer --config configs/experiments/smoke_mock.yaml --split test
$PYTHON_BIN -m src.cli.calibrate \
  --valid_path outputs/smoke_mock/predictions/valid_raw.jsonl \
  --test_path outputs/smoke_mock/predictions/test_raw.jsonl \
  --output_path outputs/smoke_mock/calibrated/test_calibrated.jsonl \
  --method isotonic
$PYTHON_BIN -m src.cli.rerank \
  --input_path outputs/smoke_mock/calibrated/test_calibrated.jsonl \
  --output_path outputs/smoke_mock/reranked/test_reranked.jsonl \
  --lambda_penalty 0.5 \
  --popularity_penalty 0.05 \
  --exploration_bonus 0.02
$PYTHON_BIN -m src.cli.evaluate \
  --predictions_path outputs/smoke_mock/reranked/test_reranked.jsonl \
  --output_dir outputs/smoke_mock/eval
$PYTHON_BIN -m src.cli.echo_chamber_analysis \
  --predictions_path outputs/smoke_mock/reranked/test_reranked.jsonl \
  --output_path outputs/smoke_mock/eval/echo_chamber.json
$PYTHON_BIN -m src.cli.train_lora --config configs/lora/smoke_mock_lora.yaml --smoke_check
$PYTHON_BIN -m src.cli.aggregate \
  --metrics_glob "outputs/smoke_mock/eval/metrics.json" \
  --output_path outputs/smoke_mock/tables/aggregate.csv
$PYTHON_BIN -m src.cli.export_paper_tables \
  --aggregate_csv outputs/smoke_mock/tables/aggregate.csv \
  --output_dir outputs/smoke_mock/paper_tables
