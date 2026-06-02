#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/pony-rec-rescue-shadow-v6

PYTHON="${PYTHON:-/home/ajifang/miniconda3/bin/python}"
export PYTHONPATH="$PWD:$PWD/scripts/adapters:$PWD/scripts/audit:$PWD/scripts/build:$PWD/scripts/train${PYTHONPATH:+:$PYTHONPATH}"

BACKBONE=/home/ajifang/models/Qwen/Qwen3-8B
POLICY_ID="official_code_qwen3base_default_hparams_declared_adaptation_v1"
VARIANT="official_code_qwen3base_default_hparams_declared_adaptation"

domain=toys
method=llm2rec_sasrec
exp="${domain}_large10000_100neg"
task_dir="outputs/baselines/external_tasks/${exp}_test_same_candidate"
valid_dir="outputs/baselines/external_tasks/${exp}_valid_same_candidate"
out_dir="outputs/${exp}_${method}_official_qwen3base_same_candidate"
scores_path="${out_dir}/scores.csv"
prov_path="${out_dir}/fairness_provenance.json"
audit_log="${out_dir}/${method}_same_candidate_score_audit.txt"
embedding_path="/home/ajifang/projects/pony-rec-rescue-shadow-v6/outputs/baselines/paper_adapters/toys_large10000_100neg_llm2rec_official_adapter/llm2rec_item_embeddings.npy"

mkdir -p "$out_dir"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] RECOVERY START llm2rec_sasrec on toys using existing embedding: $embedding_path"

"$PYTHON" scripts/adapters/main_run_llm2rec_official_same_candidate_adapter.py \
  --stage run --domain "$domain" \
  --task_dir "$task_dir" --valid_task_dir "$valid_dir" \
  --output_scores_path "$scores_path" \
  --provenance_output_path "$prov_path" \
  --fairness_policy_id "$POLICY_ID" \
  --comparison_variant "$VARIANT" \
  --backbone_path "$BACKBONE" \
  --llm_adaptation_mode frozen_base_embedding \
  --embedding_backend hf_mean_pool \
  --embedding_max_length 128 --hf_device_map auto \
  --llm2rec_item_embedding_path "$embedding_path" \
  --llm2rec_link_mode symlink

if [ -f "$scores_path" ]; then
  "$PYTHON" scripts/audit/main_audit_same_candidate_score_file.py \
    --candidate_items_path "${task_dir}/candidate_items.csv" \
    --scores_path "$scores_path" > "$audit_log"
  "$PYTHON" scripts/misc/main_import_same_candidate_baseline_scores.py \
    --baseline_name "$method" \
    --exp_name "${exp}_${method}_official_qwen3base_same_candidate" \
    --domain "$domain" \
    --ranking_input_path "${task_dir}/ranking_test.jsonl" \
    --scores_path "$scores_path" \
    --output_root outputs \
    --ks 5,10,20 \
    --k 10 \
    --artifact_class completed_result \
    --status_label same_schema_external_baseline \
    --min_score_coverage 1.0 \
    --provenance_path "$prov_path" \
    --require_fairness_provenance
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] RECOVERY DONE llm2rec_sasrec on toys"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] RECOVERY ERROR llm2rec_sasrec on toys: no scores produced"
  exit 1
fi
