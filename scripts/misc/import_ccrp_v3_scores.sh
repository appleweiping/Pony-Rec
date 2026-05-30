#!/usr/bin/env bash
set -euo pipefail

# Import C-CRP v3 scores into unified evaluation pipeline.
# Run from: /home/ajifang/projects/pony-rec-rescue-shadow-v6
# After: C-CRP v3 inference completes for all domains.

export PYTHONPATH=/home/ajifang/projects/pony-rec-rescue-shadow-v6:$PYTHONPATH
PYTHON=/home/ajifang/miniconda3/envs/qwen_vllm/bin/python

DOMAINS="sports toys home tools"

for domain in $DOMAINS; do
    exp_prefix="${domain}_large10000_100neg"
    task_dir="outputs/baselines/external_tasks/${exp_prefix}_test_same_candidate"
    scores_path="outputs/${exp_prefix}_ccrp_v3/scores.csv"
    exp_name="${exp_prefix}_ccrp_v3_qwen3base_pointwise_same_candidate"

    echo "=== Importing C-CRP v3 scores for ${domain} ==="

    if [ ! -f "$scores_path" ]; then
        echo "  SKIP: scores not found at $scores_path"
        continue
    fi

    $PYTHON scripts/misc/main_import_same_candidate_baseline_scores.py \
        --baseline_name ccrp_v3_qwen3base_pointwise \
        --exp_name "$exp_name" \
        --domain "$domain" \
        --ranking_input_path "${task_dir}/ranking_test.jsonl" \
        --scores_path "$scores_path" \
        --status_label same_schema_internal_method \
        --artifact_class completed_result \
        --ks "5,10,20" \
        --allow_partial_scores

    echo "  Done: $domain"
done

echo "=== All C-CRP v3 imports complete ==="
