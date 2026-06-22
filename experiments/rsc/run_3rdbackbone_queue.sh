#!/usr/bin/env bash
set -uo pipefail
cd /home/ajifang/projects/pony-rec-rescue-shadow-v6
echo "=== MISTRAL 8 domains start $(date) ==="
MODEL=/home/ajifang/models/Mistral-7B-Instruct-v0.3 BACKBONE=mistral-7b SUFFIX=ccrp_v3_mistral DOMAINS="beauty books electronics movies sports toys home tools" bash experiments/rsc/run_ccrp_v3_backbone_generic.sh
echo "=== LLAMA 4 missing domains start $(date) ==="
MODEL=/home/ajifang/models/Llama-3.1-8B-Instruct BACKBONE=llama3.1-8b SUFFIX=ccrp_v3_llama DOMAINS="beauty books electronics movies" bash experiments/rsc/run_ccrp_v3_backbone_generic.sh
echo "=== 3RD-BACKBONE QUEUE DONE $(date) ==="
