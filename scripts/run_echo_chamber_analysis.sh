#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python}"

PREDICTIONS="${1:-outputs/smoke_mock/reranked/test_reranked.jsonl}"
OUTPUT="${2:-outputs/smoke_mock/eval/echo_chamber.json}"
$PYTHON_BIN -m src.cli.echo_chamber_analysis --predictions_path "$PREDICTIONS" --output_path "$OUTPUT"
