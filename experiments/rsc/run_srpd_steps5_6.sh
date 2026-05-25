#!/usr/bin/env bash
# Part 2 of SRPD pipeline: Steps 5-6 (test inference + evaluation)
# Run after run_srpd_full_pipeline_vllm.sh completes Steps 1-4
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vllm
cd ~/projects/pony-rec-rescue-shadow-v6
export PYTHONPATH="$HOME/projects/pony-rec-rescue-shadow-v6:${PYTHONPATH:-}"

MODEL=/home/ajifang/models/Qwen/Qwen3-8B
DATA_ROOT=outputs/baselines/external_tasks
VLLM_CFG=configs/model/qwen3_8b_vllm_rank_safe.yaml

# ============================================================
# STEP 5: Test Inference with LoRA (vLLM doesn't support LoRA
# adapters easily, so use HF backend with batch_size=1)
# ============================================================
echo ""
echo "=== STEP 5: Test Inference with LoRA ==="

for DOMAIN in beauty books electronics movies; do
  if [ "$DOMAIN" = "beauty" ]; then
    PREFIX="beauty_supplementary_smallerN_100neg"
  else
    PREFIX="${DOMAIN}_large10000_100neg"
  fi

  TEST_INPUT="${DATA_ROOT}/${PREFIX}_test_same_candidate/ranking_test.jsonl"
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
    TEMP_CFG="/tmp/qwen3_8b_lora_${DOMAIN}.yaml"
    python3 -c "
import yaml
cfg = {
    'backend_name': 'local_hf',
    'provider': 'local_hf',
    'model_name': 'qwen3-8b-srpd-${DOMAIN}',
    'model_name_or_path': '${MODEL}',
    'tokenizer_name_or_path': '${MODEL}',
    'runtime': {
        'device': 'cuda',
        'device_map': 'auto',
        'dtype': 'bfloat16',
        'batch_size': 1,
        'local_files_only': True,
        'trust_remote_code': True,
        'load_in_4bit': False,
        'load_in_8bit': False,
        'adapter_path': '${ADAPTER_PATH}',
    },
    'generation': {
        'temperature': 0.0,
        'top_p': 1.0,
        'max_new_tokens': 256,
        'use_chat_template': True,
        'enable_thinking': False,
    }
}
with open('${TEMP_CFG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"

    python scripts/pipeline/main_rank.py \
      --exp_name "${PREFIX}_srpd_final_test" \
      --input_path "$TEST_INPUT" \
      --model_config "$TEMP_CFG" \
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
# STEP 6: Evaluate + Export Scores
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
