#!/usr/bin/env bash
set -e

echo "======================================"
echo " Running CLEAN experiment (MovieLens 1M) "
echo "======================================"

python legacy/root_main/main_infer.py \
  --exp_name clean \
  --input_path data/processed/movielens_1m/test.jsonl \
  --overwrite

python legacy/root_main/main_eval.py \
  --exp_name clean

python legacy/root_main/main_calibrate.py \
  --exp_name clean

python legacy/root_main/main_rerank.py \
  --exp_name clean \
  --lambda_penalty 0.5

echo "======================================"
echo " CLEAN experiment finished "
echo "======================================"