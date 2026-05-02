#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python}"

DATASET_CFG="${1:-configs/datasets/amazon_beauty.yaml}"
$PYTHON_BIN -m src.cli.preprocess --config "$DATASET_CFG"
$PYTHON_BIN -m src.cli.build_candidates --config "$DATASET_CFG" --negative_count 99 --strategy popularity_stratified

PROCESSED_DIR="$($PYTHON_BIN -c "import yaml;print(yaml.safe_load(open('$DATASET_CFG'))['processed_dir'])")"
OUT_ROOT="outputs/baselines/$(basename "$PROCESSED_DIR")"
mkdir -p "$OUT_ROOT"
$PYTHON_BIN -m src.cli.export_recbole_data --dataset_config "$DATASET_CFG" --output_dir "$OUT_ROOT/recbole_atomic"

for method in random popularity bm25
do
  $PYTHON_BIN -m src.cli.baselines \
    --input_path "$PROCESSED_DIR/test_candidates.jsonl" \
    --output_path "$OUT_ROOT/${method}_test.jsonl" \
    --method "$method"
  $PYTHON_BIN -m src.cli.evaluate \
    --predictions_path "$OUT_ROOT/${method}_test.jsonl" \
    --output_dir "$OUT_ROOT/${method}_eval"
done

for cfg in configs/baselines/pop.yaml configs/baselines/bprmf.yaml configs/baselines/lightgcn.yaml configs/baselines/gru4rec.yaml configs/baselines/sasrec.yaml configs/baselines/bert4rec.yaml
do
  $PYTHON_BIN -m src.cli.run_recbole_baseline \
    --baseline_config "$cfg" \
    --dataset_dir "$OUT_ROOT/recbole_atomic" \
    --output_dir "$OUT_ROOT/recbole_runs"
done
