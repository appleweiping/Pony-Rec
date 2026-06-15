#!/usr/bin/env bash
# SECOND-BACKBONE experiment (Opus gap-to-8 #1 remaining blocker).
# Re-run the SAME C-CRP v3 pointwise scoring with Llama-3.1-8B-Instruct (instead
# of Qwen3-8B) on the 4 cleanest-gate domains, to show the 6/8 win pattern + the
# eta=0 negative result replicate on a NON-Qwen backbone.
#
# vs the Qwen runs, --model changes AND --guided-json is added (Llama-only). The
# prompt, parse_score regex, metrics and output layout are identical
# (run_ccrp_v3_domain_seeded.py). The chat template is applied generically via the
# tokenizer, so no Qwen tokens/format are hard-coded. No seed is set here (matches
# the original Qwen runs' default RNG); run-to-run variance is the separate
# multi-seed experiment.
#
# WHY --guided-json HERE (and NOT on Qwen/multiseed): the smoke showed Llama does
# not reliably follow the free-form "Return ONLY JSON" instruction (~13% nonzero,
# 87% degenerate 0.0), while Qwen3-8B followed it naturally. --guided-json forces
# vLLM structured decoding to the schema {"relevance_probability": number[0,1],
# "reason": string}, isolating the relevance signal from Llama's formatting quirk.
# This is a Llama-only fix; the Qwen and multiseed scripts stay free-form so the
# Qwen path is byte-identical to the published numbers. (A matched Qwen+guided-JSON
# control may be needed for strict decoding-parity in the paper — see PLAN.md.)
#
# CONSTRAINT: only launch when the GPU is FREE (TGL LoRA PID 4090183 done).
# Run the smoke test (smoke_test_llama_ccrp_v3.sh) FIRST and confirm PASS.
#
# DETACHED LAUNCH (exact):
#   ssh pony-rec-gpu 'cd ~/projects/pony-rec-rescue-shadow-v6 && \
#     nohup bash experiments/rsc/run_ccrp_v3_second_backbone_llama.sh \
#       > logs/ccrp_v3_llama_4domains.log 2>&1 &'
#   then:  ssh pony-rec-gpu 'tail -f ~/projects/pony-rec-rescue-shadow-v6/logs/ccrp_v3_llama_4domains.log'
set -uo pipefail
cd /home/ajifang/projects/pony-rec-rescue-shadow-v6
export PYTHONPATH=/home/ajifang/projects/pony-rec-rescue-shadow-v6:
PYTHON=/home/ajifang/miniconda3/envs/qwen_vllm/bin/python
mkdir -p logs

DOMAINS="sports toys home tools"
MODEL=/home/ajifang/models/Llama-3.1-8B-Instruct
BACKBONE=llama3.1-8b

for domain in $DOMAINS; do
    exp_prefix="${domain}_large10000_100neg"
    data_path="outputs/baselines/external_tasks/${exp_prefix}_test_same_candidate/ranking_test.jsonl"
    output_dir="outputs/${exp_prefix}_ccrp_v3_llama"

    echo ""
    echo "============================================"
    echo "C-CRP v3 [LLAMA]: ${domain} ($(date))"
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
        --n_users 10000 \
        --gpu_mem 0.85 \
        --guided-json
    if [ $? -ne 0 ]; then echo "  FAILED: ${domain}"; exit 1; fi
    echo "  DONE: ${domain} -> ${output_dir}"
done

echo ""
echo "============================================"
echo "ALL LLAMA C-CRP v3 (4 domains) COMPLETE"
echo "Reports: outputs/{sports,toys,home,tools}_large10000_100neg_ccrp_v3_llama/report.json"
echo "============================================"
