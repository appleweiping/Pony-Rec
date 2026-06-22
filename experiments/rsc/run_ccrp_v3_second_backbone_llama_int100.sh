#!/usr/bin/env bash
# SECOND-BACKBONE experiment (int100 score-scale variant).
# Re-run the SAME C-CRP v3 pointwise scoring with Llama-3.1-8B-Instruct on the 4
# cleanest-gate domains, BUT eliciting a 0-100 INTEGER relevance_score (then /100)
# instead of a verbalized [0,1] probability.
#
# WHY int100: Llama-3.1-8B's verbalized [0,1] relevance_probability FLOORS
# off-category candidates at exactly 0.0 (~87% of scores 0.0 = degenerate ranking;
# a genuine model judgment, not a format bug), whereas Qwen3-8B gives a GRADED
# posterior. The int100 prompt explicitly asks the model to distinguish even weak/
# partial relevance (give small positive scores like 5/10/15 to loosely-related
# items), testing whether a granular integer scale elicits a GRADED posterior.
#
# vs the Qwen runs: --model changes, --guided-json is added (schema-enforced
# integer JSON {"relevance_score": int 0..100, "reason": string}), and
# --score-scale int100 selects the int100 prompt + parser. The history/candidate
# framing, metrics, and output layout are otherwise identical
# (run_ccrp_v3_domain_seeded.py). The chat template is applied generically via the
# tokenizer, so no Qwen tokens/format are hard-coded. No seed is set here (matches
# the original Qwen runs' default RNG); run-to-run variance is a separate experiment.
#
# CONTROL NOTE (for the paper): a matched Qwen3-8B + same 0-100 elicitation control
# is needed for strict decoding/scale parity (Qwen published numbers use the
# verbalized [0,1] free-form path). See docs.
#
# CONSTRAINT: only launch when the GPU is FREE.
# Run smoke_test_llama_ccrp_v3_int100.sh FIRST and confirm PASS.
#
# DETACHED LAUNCH (exact, survives logout/reboot):
#   ssh pony-rec-gpu 'cd ~/projects/pony-rec-rescue-shadow-v6 && \
#     nohup bash experiments/rsc/run_ccrp_v3_second_backbone_llama_int100.sh \
#       > logs/ccrp_v3_llama_int100_4domains.log 2>&1 &'
#   then:  ssh pony-rec-gpu 'tail -f ~/projects/pony-rec-rescue-shadow-v6/logs/ccrp_v3_llama_int100_4domains.log'
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
    output_dir="outputs/${exp_prefix}_ccrp_v3_llama_int100"

    echo ""
    echo "============================================"
    echo "C-CRP v3 [LLAMA int100]: ${domain} ($(date))"
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
        --guided-json \
        --score-scale int100
    if [ $? -ne 0 ]; then echo "  FAILED: ${domain}"; exit 1; fi
    echo "  DONE: ${domain} -> ${output_dir}"
done

echo ""
echo "============================================"
echo "ALL LLAMA int100 C-CRP v3 (4 domains) COMPLETE"
echo "Reports: outputs/{sports,toys,home,tools}_large10000_100neg_ccrp_v3_llama_int100/report.json"
echo "============================================"
