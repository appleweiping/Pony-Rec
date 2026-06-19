#!/usr/bin/env bash
# SMOKE TEST (int100 score-scale) — run FIRST when the GPU is FREE, BEFORE any full
# 4-domain Llama int100 run.
#
# CONTEXT: Llama-3.1-8B-Instruct's verbalized [0,1] relevance_probability FLOORS
# off-category candidates at exactly 0.0 (guided-JSON works, parse works, but ~87%
# of scores are 0.0 = degenerate ranking; this is a genuine model judgment, not a
# format bug). Qwen3-8B gives a GRADED posterior. This smoke tests whether a
# granular 0-100 INTEGER scale (with an explicit instruction to give small positive
# scores to weak/partial relevance) elicits a GRADED posterior from Llama instead.
#
# Validates on Llama-3.1-8B-Instruct over 20 sports users (~2020 prompts, ~2-3 min):
#   (1) the Llama chat template is applied generically (no Qwen hard-coding),
#   (2) the model emits a parseable {"relevance_score": int 0..100} JSON,
#   (3) parse_score_int100 extracts a GRADED spread (NOT floored at 0).
#
# PASS criteria (from the int100 design decision):
#   nonzero >= 70%  AND  distinct_values >= 8  AND  NDCG@10 in a sane range
#   (Qwen sports NDCG@10 was 0.2287; sane ~= 0.10..0.45 for this 101-candidate task).
# If FAIL (still floored): do NOT launch the full run; inspect raw_text_samples.jsonl
# and the distribution — Llama's verbalized relevance posterior is genuinely coarse
# for this 101-candidate task (a documented scope finding for external-validity).
#
# CONSTRAINT: only run when nvidia-smi shows the GPU is FREE.
set -uo pipefail
cd /home/ajifang/projects/pony-rec-rescue-shadow-v6
export PYTHONPATH=/home/ajifang/projects/pony-rec-rescue-shadow-v6:
PYTHON=/home/ajifang/miniconda3/envs/qwen_vllm/bin/python

MODEL=/home/ajifang/models/Llama-3.1-8B-Instruct
DATA=outputs/baselines/external_tasks/sports_large10000_100neg_test_same_candidate/ranking_test.jsonl
OUT=outputs/_smoke/llama_sports_20u_int100

echo "=== SMOKE (int100): Llama-3.1-8B C-CRP v3, 20 sports users ($(date)) ==="
$PYTHON experiments/rsc/run_ccrp_v3_domain_seeded.py \
    --data "$DATA" \
    --output "$OUT" \
    --model "$MODEL" \
    --backbone llama3.1-8b \
    --n_users 20 \
    --gpu_mem 0.85 \
    --guided-json \
    --score-scale int100
rc=$?
[ $rc -ne 0 ] && { echo "SMOKE FAILED rc=$rc"; exit $rc; }

echo "=== SMOKE VALIDATION (int100): score distribution from scores.csv ==="
OUT="$OUT" $PYTHON - <<'PY'
import csv, collections, json, os
out = os.environ["OUT"]
rows = list(csv.DictReader(open(f"{out}/scores.csv")))
scores = [float(r["score"]) for r in rows]
nz = [s for s in scores if s > 0.0]
pct_nz = 100*len(nz)/max(len(scores),1)
distinct = len(set(scores))
print(f"n_scores={len(scores)}  nonzero={len(nz)} ({pct_nz:.0f}%)")
print(f"min={min(scores):.3f} max={max(scores):.3f} mean={sum(scores)/max(len(scores),1):.3f}")
print(f"distinct_values={distinct}")
hist = collections.Counter(round(s,2) for s in scores)
print("top score buckets:", sorted(hist.items(), key=lambda x:-x[1])[:12])
rep = json.load(open(f"{out}/report.json"))
ndcg10 = rep['NDCG@10']
print(f"report NDCG@10={ndcg10:.4f} HR@10={rep['HR@10']:.4f} "
      f"backbone={rep['backbone']} score_scale={rep.get('score_scale')}")
# PASS: nonzero>=70% AND distinct>=8 AND NDCG@10 in sane range (0.05..0.45).
ok = (pct_nz >= 70.0) and (distinct >= 8) and (0.05 <= ndcg10 <= 0.45)
print("SMOKE VERDICT:", "PASS - proceed to full 4-domain Llama int100 run" if ok
      else "FAIL - Llama posterior still coarse; do NOT launch; inspect raw_text_samples.jsonl")
PY
echo "=== raw_text samples (first 5) ==="
head -5 "$OUT/raw_text_samples.jsonl" 2>/dev/null
echo "=== If PASS: launch run_ccrp_v3_second_backbone_llama_int100.sh ==="
