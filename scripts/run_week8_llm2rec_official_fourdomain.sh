#!/usr/bin/env bash
set -euo pipefail

for domain in beauty books electronics movies; do
  if [ "$domain" = "beauty" ]; then
    exp="beauty_supplementary_smallerN_100neg"
  else
    exp="${domain}_large10000_100neg"
  fi

  python main_run_llm2rec_official_same_candidate_adapter.py \
    --stage run \
    --domain "$domain" \
    --task_dir "outputs/baselines/external_tasks/${exp}_test_same_candidate" \
    --valid_task_dir "outputs/baselines/external_tasks/${exp}_valid_same_candidate" \
    --output_scores_path "outputs/baselines/official_adapters/${exp}_llm2rec_official/llm2rec_official_scores.csv" \
    --provenance_output_path "outputs/baselines/official_adapters/${exp}_llm2rec_official/fairness_provenance.json" \
    --fairness_policy_id official_code_qwen3base_default_hparams_declared_adaptation_v1 \
    --comparison_variant official_code_qwen3base_default_hparams_declared_adaptation \
    --backbone_path /home/ajifang/models/Qwen/Qwen3-8B \
    --llm_adaptation_mode frozen_base_embedding \
    --hparam_policy official_default_or_recommended \
    --embedding_backend hf_mean_pool \
    --embedding_max_length 128 \
    --hf_device_map auto

  python main_audit_same_candidate_score_file.py \
    --candidate_items_path "outputs/baselines/external_tasks/${exp}_test_same_candidate/candidate_items.csv" \
    --scores_path "outputs/baselines/official_adapters/${exp}_llm2rec_official/llm2rec_official_scores.csv"
done
