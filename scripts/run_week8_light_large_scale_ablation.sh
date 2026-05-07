#!/usr/bin/env bash
set -euo pipefail

cd "${PONY_REC_ROOT:-$HOME/projects/pony-rec-rescue-shadow-v6}"

CONFIG="${CONFIG:-configs/week8_large_scale_future_framework.yaml}"
DOMAINS="${DOMAINS:-all}"
MAX_EVENTS_ARG=()
if [[ -n "${MAX_EVENTS:-}" ]]; then
  MAX_EVENTS_ARG=(--max_events "$MAX_EVENTS")
fi

for d in books electronics movies; do
  for split in valid test; do
    path="outputs/baselines/external_tasks/${d}_large10000_100neg_${split}_same_candidate/ranking_${split}.jsonl"
    if [[ ! -f "$path" ]]; then
      echo "Missing large-scale task package: $path" >&2
      echo "Run scripts/run_week8_large_scale_10k_100neg.sh first and verify it completed." >&2
      exit 1
    fi
  done
done

mkdir -p outputs/summary/logs
command_script="outputs/summary/week8_large10000_100neg_light_commands.sh"
log="outputs/summary/logs/week8_light_large10000_100neg_$(date +%F_%H%M%S).log"

python main_make_week8_future_framework_commands.py \
  --config "$CONFIG" \
  --stage light \
  --domains "$DOMAINS" \
  --output_path "$command_script" \
  "${MAX_EVENTS_ARG[@]}"

bash "$command_script" 2>&1 | tee "$log"
