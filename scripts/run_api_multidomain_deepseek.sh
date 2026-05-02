#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python}"

: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY before running DeepSeek experiments.}"

for cfg in \
  configs/datasets/amazon_beauty.yaml \
  configs/datasets/amazon_electronics.yaml \
  configs/datasets/amazon_books.yaml \
  configs/datasets/movie.yaml
do
  $PYTHON_BIN -m src.cli.preprocess --config "$cfg"
  $PYTHON_BIN -m src.cli.build_candidates --config "$cfg" --negative_count 99 --strategy popularity_stratified
done

$PYTHON_BIN - <<'PY'
import yaml
from pathlib import Path

datasets = [
    ("amazon_beauty", "Beauty", "data/processed/amazon_beauty"),
    ("amazon_electronics", "Electronics", "data/processed/amazon_electronics"),
    ("amazon_books", "Books", "data/processed/amazon_books"),
    ("movie", "Movie", "data/processed/movie"),
]
backend = yaml.safe_load(Path("configs/backends/deepseek_v4_flash.yaml").read_text())
for dataset, domain, processed_dir in datasets:
    cfg = {
        "seed": 42,
        "method": "deepseek_zero_shot_listwise",
        "output_dir": f"outputs/deepseek_multidomain/{dataset}",
        "dataset": {"dataset": dataset, "domain": domain, "processed_dir": processed_dir},
        "backend": backend,
        "inference": {"prompt_id": "listwise_ranking_v1", "topk": 10},
    }
    out = Path(f"outputs/deepseek_multidomain/{dataset}/experiment.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

for dataset in amazon_beauty amazon_electronics amazon_books movie
do
  exp_cfg="outputs/deepseek_multidomain/${dataset}/experiment.yaml"
  $PYTHON_BIN -m src.cli.infer --config "$exp_cfg" --split valid --resume
  $PYTHON_BIN -m src.cli.infer --config "$exp_cfg" --split test --resume
  $PYTHON_BIN -m src.cli.calibrate \
    --valid_path "outputs/deepseek_multidomain/${dataset}/predictions/valid_raw.jsonl" \
    --test_path "outputs/deepseek_multidomain/${dataset}/predictions/test_raw.jsonl" \
    --output_path "outputs/deepseek_multidomain/${dataset}/calibrated/test_calibrated.jsonl" \
    --method isotonic
  $PYTHON_BIN -m src.cli.rerank \
    --input_path "outputs/deepseek_multidomain/${dataset}/calibrated/test_calibrated.jsonl" \
    --output_path "outputs/deepseek_multidomain/${dataset}/reranked/test_reranked.jsonl" \
    --lambda_penalty 0.5 \
    --popularity_penalty 0.05 \
    --exploration_bonus 0.02
  $PYTHON_BIN -m src.cli.evaluate \
    --predictions_path "outputs/deepseek_multidomain/${dataset}/reranked/test_reranked.jsonl" \
    --output_dir "outputs/deepseek_multidomain/${dataset}/eval"
done
