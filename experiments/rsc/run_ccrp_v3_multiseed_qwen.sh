#!/usr/bin/env bash
# MULTI-SEED run-variance experiment (Opus gap-to-8 #3).
# Re-run the Qwen3-8B C-CRP v3 scoring with 3 generation seeds (2026/2027/2028)
# on the 4 gated domains -> mean+/-std NDCG@10 per domain. Converts the current
# paired-event bootstrap CI (sampling variance) into genuine generation/run
# variance at temp 0.1.
#
# The seed is threaded into vLLM SamplingParams.seed (+ LLM seed) by
# run_ccrp_v3_domain_seeded.py; the stock VLLMBackend does NOT set a seed, which
# is exactly why this variant script is required. Everything else (prompt, parse,
# metrics, output layout) is identical to the original Qwen runs.
#
# 12 runs total (3 seeds x 4 domains), each ~8.5h on RTX 4090 -> ~4.3 days serial.
# Each run writes its own report.json; the aggregator computes mean+/-std after.
#
# CONSTRAINT: only launch when the GPU is FREE (TGL LoRA PID 4090183 done).
# Recommended ordering: run the SECOND-BACKBONE (Llama) experiment FIRST (it is
# the #1 blocker and only 4 runs); start multi-seed after, or interleave per GPU
# availability. To save wall-time, seed 2026 of each domain can REUSE the existing
# Qwen run IF it had been seeded — but the originals were unseeded, so all 12 run.
#
# DETACHED LAUNCH (exact):
#   ssh pony-rec-gpu 'cd ~/projects/pony-rec-rescue-shadow-v6 && \
#     nohup bash experiments/rsc/run_ccrp_v3_multiseed_qwen.sh \
#       > logs/ccrp_v3_multiseed_qwen.log 2>&1 &'
set -uo pipefail
cd /home/ajifang/projects/pony-rec-rescue-shadow-v6
export PYTHONPATH=/home/ajifang/projects/pony-rec-rescue-shadow-v6:
PYTHON=/home/ajifang/miniconda3/envs/qwen_vllm/bin/python
mkdir -p logs

DOMAINS="sports toys home tools"
SEEDS="2026 2027 2028"
MODEL=/home/ajifang/models/Qwen/Qwen3-8B
BACKBONE=qwen3-8b

for domain in $DOMAINS; do
  for seed in $SEEDS; do
    exp_prefix="${domain}_large10000_100neg"
    data_path="outputs/baselines/external_tasks/${exp_prefix}_test_same_candidate/ranking_test.jsonl"
    output_dir="outputs/${exp_prefix}_ccrp_v3_seed${seed}"

    echo ""
    echo "============================================"
    echo "C-CRP v3 [QWEN seed=${seed}]: ${domain} ($(date))"
    echo "============================================"

    if [ -f "${output_dir}/report.json" ]; then
        echo "  SKIP: report.json already exists at ${output_dir}"
        continue
    fi

    $PYTHON experiments/rsc/run_ccrp_v3_domain_seeded.py \
        --data "$data_path" \
        --output "$output_dir" \
        --model "$MODEL" \
        --backbone "$BACKBONE" \
        --seed "$seed" \
        --n_users 10000 \
        --gpu_mem 0.85
    if [ $? -ne 0 ]; then echo "  FAILED: ${domain} seed=${seed}"; exit 1; fi
    echo "  DONE: ${domain} seed=${seed} -> ${output_dir}"
  done
done

echo ""
echo "============================================"
echo "ALL MULTI-SEED QWEN C-CRP v3 (3 seeds x 4 domains) COMPLETE"
echo "Aggregate with: $PYTHON experiments/rsc/aggregate_multiseed_ndcg.py"
echo "============================================"
