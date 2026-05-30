#!/usr/bin/env bash
# Run C-CRP v3 on remaining new domains (home, tools).
# Execute from: ~/projects/pony-rec-rescue-shadow-v6/
# Prerequisites: GPU must be free (toys experiment finished).
set -euo pipefail

cd ~/projects/pony-rec-rescue-shadow-v6

MODEL=/home/ajifang/models/Qwen/Qwen3-8B

for domain in home tools; do
  DATA="outputs/baselines/external_tasks/${domain}_large10000_100neg_test_same_candidate/ranking_test.jsonl"
  OUTPUT="outputs/${domain}_large10000_100neg_ccrp_v3"

  if [ -f "${OUTPUT}/scores.csv" ]; then
    echo "[$(date)] SKIP $domain (scores.csv exists)"
    continue
  fi

  echo "[$(date)] START C-CRP v3 on $domain (10000 users)"
  python experiments/rsc/run_ccrp_v3_domain.py \
    --data "$DATA" \
    --output "$OUTPUT" \
    --model "$MODEL" \
    --n_users 10000 \
    --gpu_mem 0.85

  echo "[$(date)] DONE C-CRP v3 on $domain"
  echo "---"
done

echo "[$(date)] All C-CRP v3 new domain runs complete."
