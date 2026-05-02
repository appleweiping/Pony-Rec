#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python}"

CONFIG="${1:-configs/lora/server_qwen_rank.yaml}"
echo "[lora_weighting] Use a config whose train_path contains uncertainty weights."
$PYTHON_BIN -m src.cli.train_lora --config "$CONFIG"
