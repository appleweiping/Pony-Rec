#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python}"

$PYTHON_BIN -m src.cli.train_lora --config "${1:-configs/lora/qwen_server_rank.yaml}"
