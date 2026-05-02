#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python}"

DATASET_CFG="${1:-configs/datasets/amazon_beauty.yaml}"
$PYTHON_BIN -m src.cli.preprocess --config "$DATASET_CFG"
$PYTHON_BIN -m src.cli.build_candidates --config "$DATASET_CFG" --negative_count 99 --strategy popularity_stratified

PROCESSED_DIR="$($PYTHON_BIN -c "import yaml;print(yaml.safe_load(open('$DATASET_CFG'))['processed_dir'])")"
OUT_ROOT="outputs/baselines/$(basename "$PROCESSED_DIR")"
mkdir -p "$OUT_ROOT"

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

echo "[recbole] Install recbole and map $PROCESSED_DIR to RecBole atomic format for BPR/LightGCN/GRU4Rec/SASRec/BERT4Rec/FMLPRec."
echo "[recbole] Official integration status and modifications must be recorded in docs/BASELINES.md before paper use."
