#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python}"

scripts/run_api_multidomain_deepseek.sh
scripts/run_baselines_recbole.sh configs/datasets/amazon_beauty.yaml
scripts/run_uncertainty_ablation.sh outputs/deepseek_multidomain/amazon_beauty/calibrated/test_calibrated.jsonl outputs/deepseek_multidomain/amazon_beauty/uncertainty_ablation
scripts/run_echo_chamber_analysis.sh outputs/deepseek_multidomain/amazon_beauty/reranked/test_reranked.jsonl outputs/deepseek_multidomain/amazon_beauty/eval/echo_chamber.json
scripts/run_lora_ablation.sh configs/lora/server_qwen_rank.yaml
$PYTHON_BIN -m src.cli.aggregate --metrics_glob "outputs/**/eval/metrics.json" --output_path outputs/paper/aggregate.csv
$PYTHON_BIN -m src.cli.export_paper_tables --aggregate_csv outputs/paper/aggregate.csv --output_dir outputs/paper/tables
