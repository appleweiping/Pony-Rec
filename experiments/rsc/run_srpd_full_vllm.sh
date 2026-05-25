#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vllm
cd ~/projects/pony-rec-rescue-shadow-v6
export PYTHONPATH="$HOME/projects/pony-rec-rescue-shadow-v6:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "=============================================="
echo "SRPD FULL PIPELINE (vLLM anchor) — $(date)"
echo "=============================================="

# ============================================================
# STEP 1: Anchor rank via vLLM (title-only, ~2.5K tokens/prompt)
# Full scale: 10K users × 101 candidates per domain
# ============================================================
echo ""
echo "=== STEP 1: Anchor Rank (vLLM) ==="

for DOMAIN in beauty books electronics movies; do
  for SPLIT in valid test; do
    echo "  [$DOMAIN/$SPLIT] Starting... ($(date))"
    python experiments/rsc/run_srpd_anchor_rank_vllm.py \
      --domain "$DOMAIN" \
      --split "$SPLIT" \
      --topk 10 \
      --max_model_len 4096 \
      --batch_size 256 \
      --gpu_memory_utilization 0.88
    echo "  [$DOMAIN/$SPLIT] Done. ($(date))"
  done
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

MODEL=/home/ajifang/models/Qwen/Qwen3-8B

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

# ============================================================
# STEP 5: Test inference with LoRA (HF backend, batch_size=1)
# This is the slowest step (~16s/sample × 10K = ~46h/domain)
# ============================================================
echo ""
echo "=== STEP 5: Test Inference (LoRA + HF) ==="

for DOMAIN in beauty books electronics movies; do
  if [ "$DOMAIN" = "beauty" ]; then
    PREFIX="beauty_supplementary_smallerN_100neg"
  else
    PREFIX="${DOMAIN}_large10000_100neg"
  fi

  LORA_DIR="outputs/${PREFIX}_srpd_lora"
  TEST_OUTPUT="outputs/${PREFIX}_srpd_final_test"

  ADAPTER_PATH=""
  if [ -d "${LORA_DIR}/checkpoint-final" ]; then
    ADAPTER_PATH="${LORA_DIR}/checkpoint-final"
  elif [ -d "${LORA_DIR}/adapter_model" ]; then
    ADAPTER_PATH="${LORA_DIR}/adapter_model"
  else
    ADAPTER_PATH=$(ls -d ${LORA_DIR}/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
  fi

  if [ -z "$ADAPTER_PATH" ]; then
    echo "  [$DOMAIN] ERROR: No LoRA checkpoint found, skipping."
    continue
  fi

  if [ ! -f "${TEST_OUTPUT}/predictions/rank_predictions.jsonl" ]; then
    echo "  [$DOMAIN] Test inference with LoRA... ($(date))"
    python scripts/pipeline/main_rank.py \
      --exp_name "${PREFIX}_srpd_final_test" \
      --input_path "outputs/baselines/external_tasks/${PREFIX}_test_same_candidate/ranking_test.jsonl" \
      --model_config configs/model/qwen3_8b_local_rank.yaml \
      --prompt_path prompts/candidate_ranking.txt \
      --output_root outputs \
      --topk 10 \
      --max_new_tokens 256 \
      --resume_partial \
      --checkpoint_every_batches 1 \
      --seed 20260525
    echo "  [$DOMAIN] Test inference done. ($(date))"
  else
    echo "  [$DOMAIN] Test predictions exist, skipping."
  fi
done

echo ""
echo "=== STEP 5 COMPLETE ($(date)) ==="

# ============================================================
# STEP 6: Evaluate
# ============================================================
echo ""
echo "=== STEP 6: Evaluation ==="

for DOMAIN in beauty books electronics movies; do
  if [ "$DOMAIN" = "beauty" ]; then
    PREFIX="beauty_supplementary_smallerN_100neg"
  else
    PREFIX="${DOMAIN}_large10000_100neg"
  fi

  TEST_OUTPUT="outputs/${PREFIX}_srpd_final_test"
  EVAL_OUTPUT="outputs/${PREFIX}_srpd_eval"

  if [ ! -f "${EVAL_OUTPUT}/report.json" ]; then
    echo "  [$DOMAIN] Evaluating... ($(date))"
    python scripts/pipeline/evaluate_ranking.py \
      --predictions "${TEST_OUTPUT}/predictions/rank_predictions.jsonl" \
      --output_dir "$EVAL_OUTPUT" \
      --metrics hr@5,hr@10,hr@20,ndcg@5,ndcg@10,ndcg@20,mrr
    echo "  [$DOMAIN] Evaluation done. ($(date))"
  else
    echo "  [$DOMAIN] Evaluation exists, skipping."
  fi
done

echo ""
echo "=============================================="
echo "SRPD FULL PIPELINE COMPLETE! ($(date))"
echo "=============================================="
