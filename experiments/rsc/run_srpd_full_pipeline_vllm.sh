#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vllm
cd ~/projects/pony-rec-rescue-shadow-v6
export PYTHONPATH="$HOME/projects/pony-rec-rescue-shadow-v6:${PYTHONPATH:-}"

MODEL=/home/ajifang/models/Qwen/Qwen3-8B
DATA_ROOT=outputs/baselines/external_tasks
VLLM_CFG=configs/model/qwen3_8b_vllm_rank_safe.yaml
HF_CFG=configs/model/qwen3_8b_local_rank.yaml

echo "=============================================="
echo "SRPD FULL PIPELINE (vLLM) — $(date)"
echo "=============================================="

# ============================================================
# STEP 1: Anchor rank inference (vLLM, ~2h/domain-split)
# Beauty already done. Run books/electronics/movies.
# ============================================================
echo ""
echo "=== STEP 1: Anchor Rank Inference (vLLM) ==="

for DOMAIN in books electronics movies; do
  PREFIX="${DOMAIN}_large10000_100neg"

  VALID_INPUT="${DATA_ROOT}/${PREFIX}_valid_same_candidate/ranking_valid.jsonl"
  VALID_OUTPUT="outputs/${PREFIX}_srpd_anchor_rank_valid"

  if [ ! -f "${VALID_OUTPUT}/predictions/rank_predictions.jsonl" ]; then
    rm -rf "${VALID_OUTPUT}" 2>/dev/null || true
    echo "  [$DOMAIN] Anchor rank VALID... ($(date))"
    python scripts/pipeline/main_rank.py \
      --exp_name "${PREFIX}_srpd_anchor_rank_valid" \
      --input_path "$VALID_INPUT" \
      --model_config "$VLLM_CFG" \
      --prompt_path prompts/candidate_ranking.txt \
      --output_root outputs \
      --topk 10 \
      --max_new_tokens 256 \
      --resume_partial \
      --checkpoint_every_batches 5 \
      --seed 20260525
    echo "  [$DOMAIN] Anchor rank VALID done. ($(date))"
  else
    echo "  [$DOMAIN] Anchor rank VALID exists, skipping."
  fi

  TEST_INPUT="${DATA_ROOT}/${PREFIX}_test_same_candidate/ranking_test.jsonl"
  TEST_OUTPUT="outputs/${PREFIX}_srpd_anchor_rank_test"

  if [ ! -f "${TEST_OUTPUT}/predictions/rank_predictions.jsonl" ]; then
    rm -rf "${TEST_OUTPUT}" 2>/dev/null || true
    echo "  [$DOMAIN] Anchor rank TEST... ($(date))"
    python scripts/pipeline/main_rank.py \
      --exp_name "${PREFIX}_srpd_anchor_rank_test" \
      --input_path "$TEST_INPUT" \
      --model_config "$VLLM_CFG" \
      --prompt_path prompts/candidate_ranking.txt \
      --output_root outputs \
      --topk 10 \
      --max_new_tokens 256 \
      --resume_partial \
      --checkpoint_every_batches 5 \
      --seed 20260525
    echo "  [$DOMAIN] Anchor rank TEST done. ($(date))"
  else
    echo "  [$DOMAIN] Anchor rank TEST exists, skipping."
  fi
done

echo ""
echo "=== STEP 1 COMPLETE ($(date)) ==="

# ============================================================
# STEP 2: Teacher reranking (CPU, fast)
# ============================================================
echo ""
echo "=== STEP 2: Teacher Reranking ==="

for DOMAIN in beauty books electronics movies; do
  if [ "$DOMAIN" = "beauty" ]; then
    PREFIX="beauty_supplementary_smallerN_100neg"
  else
    PREFIX="${DOMAIN}_large10000_100neg"
  fi

  ANCHOR_DIR="outputs/${PREFIX}_srpd_anchor_rank_valid"
  TEACHER_OUTPUT="outputs/${PREFIX}_srpd_teacher"

  if [ ! -f "${TEACHER_OUTPUT}/teacher_rankings.jsonl" ]; then
    echo "  [$DOMAIN] Teacher reranking... ($(date))"
    python scripts/pipeline/teacher_rerank.py \
      --anchor_predictions "${ANCHOR_DIR}/predictions/rank_predictions.jsonl" \
      --output_dir "$TEACHER_OUTPUT" \
      --topk 10 \
      --risk_mode structured
    echo "  [$DOMAIN] Teacher done. ($(date))"
  else
    echo "  [$DOMAIN] Teacher exists, skipping."
  fi
done

echo ""
echo "=== STEP 2 COMPLETE ($(date)) ==="

# ============================================================
# STEP 3: Build SRPD training data (CPU, fast)
# ============================================================
echo ""
echo "=== STEP 3: Build Training Data ==="

for DOMAIN in beauty books electronics movies; do
  if [ "$DOMAIN" = "beauty" ]; then
    PREFIX="beauty_supplementary_smallerN_100neg"
  else
    PREFIX="${DOMAIN}_large10000_100neg"
  fi

  TEACHER_DIR="outputs/${PREFIX}_srpd_teacher"
  TRAIN_OUTPUT="outputs/${PREFIX}_srpd_train_data"

  if [ ! -f "${TRAIN_OUTPUT}/train.jsonl" ]; then
    echo "  [$DOMAIN] Building training data... ($(date))"
    python scripts/pipeline/build_srpd_data.py \
      --teacher_rankings "${TEACHER_DIR}/teacher_rankings.jsonl" \
      --anchor_predictions "outputs/${PREFIX}_srpd_anchor_rank_valid/predictions/rank_predictions.jsonl" \
      --output_dir "$TRAIN_OUTPUT" \
      --mode dpo
    echo "  [$DOMAIN] Training data built. ($(date))"
  else
    echo "  [$DOMAIN] Training data exists, skipping."
  fi
done

echo ""
echo "=== STEP 3 COMPLETE ($(date)) ==="

# ============================================================
# STEP 4: LoRA Training (GPU, ~4-6h/domain)
# ============================================================
echo ""
echo "=== STEP 4: LoRA Training ==="

for DOMAIN in beauty books electronics movies; do
  if [ "$DOMAIN" = "beauty" ]; then
    PREFIX="beauty_supplementary_smallerN_100neg"
  else
    PREFIX="${DOMAIN}_large10000_100neg"
  fi

  TRAIN_DATA="outputs/${PREFIX}_srpd_train_data/train.jsonl"
  LORA_OUTPUT="outputs/${PREFIX}_srpd_lora"

  if [ ! -d "${LORA_OUTPUT}/checkpoint-final" ] && [ ! -d "${LORA_OUTPUT}/adapter_model" ]; then
    echo "  [$DOMAIN] LoRA training... ($(date))"
    python scripts/pipeline/train_lora.py \
      --train_data "$TRAIN_DATA" \
      --model_name_or_path "$MODEL" \
      --output_dir "$LORA_OUTPUT" \
      --lora_r 16 \
      --lora_alpha 32 \
      --num_epochs 2 \
      --batch_size 2 \
      --gradient_accumulation_steps 8 \
      --learning_rate 2e-5 \
      --max_length 2048 \
      --seed 20260525
    echo "  [$DOMAIN] LoRA training done. ($(date))"
  else
    echo "  [$DOMAIN] LoRA checkpoint exists, skipping."
  fi
done

echo ""
echo "=== STEP 4 COMPLETE ($(date)) ==="
