#!/usr/bin/env bash
set -euo pipefail

cd /home/ajifang/projects/pony-rec-rescue-shadow-v6

LOG="baselines_new_domains_toys_llmesr_20260602_1635.log"
RUNNER_PID_FILE="baselines_new_domains_toys_llmesr_runner.pid"

OUT_DIR="outputs/toys_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate"
SCORES="${OUT_DIR}/scores.csv"
SUMMARY="${OUT_DIR}/tables/same_candidate_external_baseline_summary.csv"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] PRECHECK toys llmesr_sasrec" >> "$LOG"

if ps aux | grep python | grep -v grep | grep -iE 'llm2rec|llmesr|sasrec|ccrp|baseline' >> "$LOG"; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ABORT active relevant python process detected" >> "$LOG"
  exit 7
fi

free_kb="$(df -Pk /home/ajifang | awk 'NR==2 {print $4}')"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] disk_free_kb=${free_kb}" >> "$LOG"
if [ "$free_kb" -lt 5000000 ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ABORT disk free below 5GB guard" >> "$LOG"
  exit 8
fi

if [ -f "$SCORES" ] && [ -f "$SUMMARY" ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP toys llmesr_sasrec already complete" >> "$LOG"
  exit 0
fi

echo "$$" > "$RUNNER_PID_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] START toys llmesr_sasrec single-row official loop" >> "$LOG"

DOMAINS_OVERRIDE="toys" \
FAST_METHODS_OVERRIDE="" \
TRAIN_METHODS_OVERRIDE="llmesr_sasrec" \
bash scripts/run_baselines_new_domains.sh >> "$LOG" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE toys llmesr_sasrec single-row official loop" >> "$LOG"
