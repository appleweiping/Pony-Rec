#!/usr/bin/env bash
# SMOKE TEST — run FIRST when the GPU frees, BEFORE any full 4-domain run.
# Validates, on Llama-3.1-8B-Instruct over 20 sports users (~2020 prompts, ~1-2 min):
#   (1) the Llama chat template is applied generically (no Qwen hard-coding),
#   (2) the model actually emits parseable {"relevance_probability": x} JSON,
#   (3) parse_score extracts a non-trivial spread of probabilities (NOT all 0.0).
# If probabilities are all 0.0 or degenerate, STOP and inspect raw_text before the
# full run (would indicate Llama ignores the JSON instruction / different format).
#
# CONSTRAINT: only run when nvidia-smi shows the GPU is FREE (LoRA PID 4090183 gone).
set -uo pipefail
cd /home/ajifang/projects/pony-rec-rescue-shadow-v6
export PYTHONPATH=/home/ajifang/projects/pony-rec-rescue-shadow-v6:
PYTHON=/home/ajifang/miniconda3/envs/qwen_vllm/bin/python

MODEL=/home/ajifang/models/Llama-3.1-8B-Instruct
DATA=outputs/baselines/external_tasks/sports_large10000_100neg_test_same_candidate/ranking_test.jsonl
OUT=outputs/_smoke/llama_sports_20u

echo "=== SMOKE: Llama-3.1-8B C-CRP v3, 20 sports users ($(date)) ==="
$PYTHON experiments/rsc/run_ccrp_v3_domain_seeded.py \
    --data "$DATA" \
    --output "$OUT" \
    --model "$MODEL" \
    --backbone llama3.1-8b \
    --n_users 20 \
    --gpu_mem 0.85
rc=$?
[ $rc -ne 0 ] && { echo "SMOKE FAILED rc=$rc"; exit $rc; }

echo "=== SMOKE VALIDATION: score distribution from scores.csv ==="
$PYTHON - <<'PY'
import csv, collections
rows = list(csv.DictReader(open("outputs/_smoke/llama_sports_20u/scores.csv")))
scores = [float(r["score"]) for r in rows]
nz = [s for s in scores if s > 0.0]
print(f"n_scores={len(scores)}  nonzero={len(nz)} ({100*len(nz)/max(len(scores),1):.0f}%)")
print(f"min={min(scores):.3f} max={max(scores):.3f} mean={sum(scores)/max(len(scores),1):.3f}")
print(f"distinct_values={len(set(scores))}")
import json
rep = json.load(open("outputs/_smoke/llama_sports_20u/report.json"))
print(f"report NDCG@10={rep['NDCG@10']:.4f} HR@10={rep['HR@10']:.4f} backbone={rep['backbone']}")
ok = len(nz) >= 0.3*len(scores) and len(set(scores)) >= 5
print("SMOKE VERDICT:", "PASS — proceed to full 4-domain Llama run" if ok
      else "FAIL — inspect raw_text; do NOT launch full run")
PY
echo "=== If PASS: launch run_ccrp_v3_second_backbone_llama.sh ==="
