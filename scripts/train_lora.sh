#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python}"

CONFIG="${1:-configs/lora/server_qwen_rank.yaml}"
$PYTHON_BIN -m src.cli.train_lora --config "$CONFIG"
