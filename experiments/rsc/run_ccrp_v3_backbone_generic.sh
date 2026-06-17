#!/usr/bin/env bash
# GENERIC second/third-backbone C-CRP v3 runner. Re-runs the SAME pointwise
# relevance-posterior scoring (run_ccrp_v3_domain_seeded.py) with an arbitrary
# vLLM-supported backbone over a list of domains, for the backbone-robustness
# study. Same prompt/parse/metrics/output layout as the Qwen + Llama runs;
# --guided-json forces structured decoding (needed for non-Qwen backbones that
# don't reliably follow the free-form "Return ONLY JSON" instruction).
#
# Usage (env vars):
#   MODEL=/home/ajifang/models/Mistral-7B-Instruct-v0.3 \
#   BACKBONE=mistral-7b \
#   SUFFIX=ccrp_v3_mistral \
#   DOMAINS="beauty books electronics movies sports toys home tools" \
#   bash experiments/rsc/run_ccrp_v3_backbone_generic.sh
#
# DETACHED LAUNCH (exact):
#   ssh pony-rec-gpu 'cd ~/projects/pony-rec-rescue-shadow-v6 && MODEL=... BACKBONE=... SUFFIX=... DOMAINS="..." \
#     nohup bash experiments/rsc/run_ccrp_v3_backbone_generic.sh > logs/ccrp_v3_${BACKBONE}.log 2>&1 &'
#
# Skips any domain whose report.json already exists (crash/restart safe at
# domain granularity).
set -uo pipefail
cd /home/ajifang/projects/pony-rec-rescue-shadow-v6
export PYTHONPATH=/home/ajifang/projects/pony-rec-rescue-shadow-v6:
PYTHON=/home/ajifang/miniconda3/envs/qwen_vllm/bin/python
mkdir -p logs

: "${MODEL:?set MODEL}"; : "${BACKBONE:?set BACKBONE}"; : "${SUFFIX:?set SUFFIX}"
: "${DOMAINS:?set DOMAINS}"
GPU_MEM="${GPU_MEM:-0.85}"
NUSERS="${NUSERS:-10000}"

for domain in $DOMAINS; do
    # Beauty uses the smaller supplementary panel (973 users); the rest use the 10k panel.
    if [ "$domain" = "beauty" ]; then
        exp_prefix="beauty_supplementary_smallerN_100neg"
    else
        exp_prefix="${domain}_large10000_100neg"
    fi
    data_path="outputs/baselines/external_tasks/${exp_prefix}_test_same_candidate/ranking_test.jsonl"
    output_dir="outputs/${exp_prefix}_${SUFFIX}"

    echo ""
    echo "============================================"
    echo "C-CRP v3 [${BACKBONE}]: ${domain} ($(date))"
    echo "============================================"

    if [ -f "${output_dir}/report.json" ]; then
        echo "  SKIP: report.json already exists at ${output_dir}"; continue
    fi
    if [ ! -f "$data_path" ]; then
        echo "  SKIP (no task data): ${data_path} -- regenerate the panel first"; continue
    fi

    $PYTHON experiments/rsc/run_ccrp_v3_domain_seeded.py \
        --data "$data_path" \
        --output "$output_dir" \
        --model "$MODEL" \
        --backbone "$BACKBONE" \
        --n_users "$NUSERS" \
        --gpu_mem "$GPU_MEM" \
        --guided-json
    if [ $? -ne 0 ]; then echo "  FAILED: ${domain}"; exit 1; fi
    echo "  DONE: ${domain} -> ${output_dir}"
done

echo ""
echo "============================================"
echo "C-CRP v3 [${BACKBONE}] over [${DOMAINS}] finished ($(date))"
echo "============================================"
