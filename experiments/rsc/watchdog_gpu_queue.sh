#!/usr/bin/env bash
# Server-side GPU-queue WATCHDOG. Runs ON pony-rec-gpu (independent of the laptop), so it
# survives local-machine shutdown / network loss. Installed via cron (every 15 min) — and
# cron itself survives server reboots. Keeps the CURRENT heavy GPU experiment alive: if the
# run is dead AND its outputs are incomplete, it relaunches it (with a 20-min cooldown so a
# just-launched run is given time to start, preventing restart loops). Logs every check to
# logs/watchdog.log so progress/restarts are auditable.
#
# As the GPU queue advances (Llama 2nd-backbone -> TGL eval -> TRUCE rollout), update the
# "current GPU job spec" block below (JOB / PROC_PAT / LAUNCH / DONE_CHECK).
set -uo pipefail
ROOT=/home/ajifang/projects/pony-rec-rescue-shadow-v6
cd "$ROOT" || exit 1
LOG="$ROOT/logs/watchdog.log"; mkdir -p "$ROOT/logs"
TS() { date '+%F %T'; }

# ---- current GPU job spec ----
JOB="llama_2nd_backbone"
PROC_PAT='run_ccrp_v3_domain_seeded.py.*llama3.1-8b'
LAUNCH='nohup bash experiments/rsc/run_ccrp_v3_second_backbone_llama.sh >> logs/ccrp_v3_llama_4domains.log 2>&1 &'
DONE_CHECK() { for d in sports toys home tools; do [ -f "outputs/${d}_large10000_100neg_ccrp_v3_llama/report.json" ] || return 1; done; return 0; }
STAMP="$ROOT/logs/.watchdog_${JOB}_lastlaunch"

if DONE_CHECK; then
  echo "$(TS) [$JOB] ALL DONE -- no action" >> "$LOG"; exit 0
fi
n=$(pgrep -fc "$PROC_PAT" 2>/dev/null || echo 0)
if [ "${n:-0}" -ge 1 ]; then
  echo "$(TS) [$JOB] ALIVE (procs=$n)" >> "$LOG"; exit 0
fi
now=$(date +%s); last=0; [ -f "$STAMP" ] && last=$(cat "$STAMP" 2>/dev/null || echo 0)
if [ $((now - last)) -lt 1200 ]; then
  echo "$(TS) [$JOB] dead+incomplete but relaunched <20min ago -- waiting" >> "$LOG"; exit 0
fi
echo "$(TS) [$JOB] DEAD + incomplete -- RELAUNCHING" >> "$LOG"
echo "$now" > "$STAMP"
( cd "$ROOT" && eval "$LAUNCH" )
echo "$(TS) [$JOB] relaunch issued" >> "$LOG"
