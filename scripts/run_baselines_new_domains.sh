#!/usr/bin/env bash
# Run all 8 official baselines on new Amazon 2023 domains (sports, toys, home, tools).
# Execute from: ~/projects/pony-rec-rescue-shadow-v6/scripts/adapters/
# Prerequisites: toys C-CRP v3 must be finished (GPU free).
#
# Execution order: fastest baselines first (embedding-only), then training-based.
# Each domain runs all baselines before moving to the next.
set -euo pipefail

cd ~/projects/pony-rec-rescue-shadow-v6

BACKBONE=/home/ajifang/models/Qwen/Qwen3-8B
POLICY_ID="official_code_qwen3base_default_hparams_declared_adaptation_v1"
VARIANT="official_code_qwen3base_default_hparams_declared_adaptation"

DOMAINS=(sports toys home tools)

log_progress() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_baseline_on_domain() {
  local method=$1
  local domain=$2
  local exp="${domain}_large10000_100neg"
  local task_dir="outputs/baselines/external_tasks/${exp}_test_same_candidate"
  local valid_dir="outputs/baselines/external_tasks/${exp}_valid_same_candidate"
  local out_dir="outputs/${exp}_${method}_official_qwen3base_same_candidate"
  local scores_path="${out_dir}/scores.csv"
  local prov_path="${out_dir}/fairness_provenance.json"

  if [ -f "$scores_path" ]; then
    log_progress "SKIP $method on $domain (scores exist)"
    return 0
  fi

  log_progress "START $method on $domain"
  mkdir -p "$out_dir"

  case "$method" in
    llmemb)
      python scripts/adapters/main_run_llmemb_official_same_candidate_adapter.py \
        --stage run --domain "$domain" \
        --task_dir "$task_dir" --valid_task_dir "$valid_dir" \
        --output_scores_path "$scores_path" \
        --provenance_output_path "$prov_path" \
        --fairness_policy_id "$POLICY_ID" \
        --comparison_variant "$VARIANT" \
        --backbone_path "$BACKBONE" \
        --llm_adaptation_mode frozen_base_embedding \
        --embedding_backend hf_mean_pool \
        --embedding_max_length 128 --hf_device_map auto
      ;;
    proex_profile)
      python scripts/adapters/main_run_proex_official_same_candidate_adapter.py \
        --stage run --domain "$domain" \
        --task_dir "$task_dir" --valid_task_dir "$valid_dir" \
        --output_scores_path "$scores_path" \
        --provenance_output_path "$prov_path" \
        --fairness_policy_id "$POLICY_ID" \
        --comparison_variant "$VARIANT" \
        --backbone_path "$BACKBONE" \
        --llm_adaptation_mode frozen_base_embedding \
        --embedding_backend hf_mean_pool \
        --embedding_max_length 128 --hf_device_map auto
      ;;
    promax_profile)
      python scripts/adapters/main_run_promax_official_same_candidate_adapter.py \
        --stage run --domain "$domain" \
        --task_dir "$task_dir" --valid_task_dir "$valid_dir" \
        --output_scores_path "$scores_path" \
        --provenance_output_path "$prov_path" \
        --fairness_policy_id "$POLICY_ID" \
        --comparison_variant "$VARIANT" \
        --backbone_path "$BACKBONE" \
        --llm_adaptation_mode frozen_base_embedding \
        --embedding_backend hf_mean_pool \
        --embedding_max_length 128 --hf_device_map auto
      ;;
    elmrec_graph)
      python scripts/adapters/main_run_elmrec_official_same_candidate_adapter.py \
        --stage run --domain "$domain" \
        --task_dir "$task_dir" --valid_task_dir "$valid_dir" \
        --output_scores_path "$scores_path" \
        --provenance_output_path "$prov_path" \
        --fairness_policy_id "$POLICY_ID" \
        --comparison_variant "$VARIANT" \
        --backbone_path "$BACKBONE" \
        --llm_adaptation_mode frozen_base_embedding \
        --embedding_backend hf_mean_pool \
        --embedding_max_length 128 --hf_device_map auto
      ;;
    irllrec_intent)
      python scripts/adapters/main_run_irllrec_official_same_candidate_adapter.py \
        --stage run --domain "$domain" \
        --task_dir "$task_dir" --valid_task_dir "$valid_dir" \
        --output_scores_path "$scores_path" \
        --provenance_output_path "$prov_path" \
        --fairness_policy_id "$POLICY_ID" \
        --comparison_variant "$VARIANT" \
        --backbone_path "$BACKBONE" \
        --llm_adaptation_mode frozen_base_embedding \
        --embedding_backend hf_mean_pool \
        --embedding_max_length 128 --hf_device_map auto
      ;;
    rlmrec_graphcl)
      python scripts/adapters/main_run_rlmrec_official_same_candidate_adapter.py \
        --stage run --domain "$domain" \
        --task_dir "$task_dir" --valid_task_dir "$valid_dir" \
        --output_scores_path "$scores_path" \
        --provenance_output_path "$prov_path" \
        --fairness_policy_id "$POLICY_ID" \
        --comparison_variant "$VARIANT" \
        --backbone_path "$BACKBONE" \
        --llm_adaptation_mode frozen_base_embedding \
        --embedding_backend hf_mean_pool \
        --embedding_max_length 128 --hf_device_map auto
      ;;
    llm2rec_sasrec)
      python scripts/adapters/main_run_llm2rec_official_same_candidate_adapter.py \
        --stage run --domain "$domain" \
        --task_dir "$task_dir" --valid_task_dir "$valid_dir" \
        --output_scores_path "$scores_path" \
        --provenance_output_path "$prov_path" \
        --fairness_policy_id "$POLICY_ID" \
        --comparison_variant "$VARIANT" \
        --backbone_path "$BACKBONE" \
        --llm_adaptation_mode frozen_base_embedding \
        --embedding_backend hf_mean_pool \
        --embedding_max_length 128 --hf_device_map auto
      ;;
    llmesr_sasrec)
      python scripts/adapters/main_run_llmesr_official_same_candidate_adapter.py \
        --stage run --domain "$domain" \
        --task_dir "$task_dir" --valid_task_dir "$valid_dir" \
        --output_scores_path "$scores_path" \
        --provenance_output_path "$prov_path" \
        --fairness_policy_id "$POLICY_ID" \
        --comparison_variant "$VARIANT" \
        --backbone_path "$BACKBONE" \
        --llm_adaptation_mode frozen_base_embedding \
        --embedding_backend hf_mean_pool \
        --embedding_max_length 128 --hf_device_map auto
      ;;
    setrec_identifier)
      python scripts/adapters/main_run_setrec_official_same_candidate_adapter.py \
        --stage run --domain "$domain" \
        --task_dir "$task_dir" --valid_task_dir "$valid_dir" \
        --output_scores_path "$scores_path" \
        --provenance_output_path "$prov_path" \
        --fairness_policy_id "$POLICY_ID" \
        --comparison_variant "$VARIANT" \
        --backbone_path "$BACKBONE" \
        --llm_adaptation_mode frozen_base_embedding \
        --embedding_backend hf_mean_pool \
        --embedding_max_length 128 --hf_device_map auto
      ;;
    *)
      log_progress "ERROR: Unknown method $method"
      return 1
      ;;
  esac

  if [ -f "$scores_path" ]; then
    python scripts/audit/main_audit_same_candidate_score_file.py \
      --candidate_items_path "${task_dir}/candidate_items.csv" \
      --scores_path "$scores_path" 2>&1 || true
    log_progress "DONE $method on $domain"
  else
    log_progress "WARN $method on $domain: no scores produced"
  fi
}

# Embedding-only baselines first (fastest), then training-based
FAST_METHODS=(llmemb proex_profile promax_profile)
TRAIN_METHODS=(elmrec_graph irllrec_intent rlmrec_graphcl llm2rec_sasrec llmesr_sasrec setrec_identifier)

log_progress "=== Starting baseline runs on new domains ==="
log_progress "Domains: ${DOMAINS[*]}"
log_progress "Fast methods: ${FAST_METHODS[*]}"
log_progress "Training methods: ${TRAIN_METHODS[*]}"

# Phase 1: Fast baselines on all domains
for domain in "${DOMAINS[@]}"; do
  for method in "${FAST_METHODS[@]}"; do
    run_baseline_on_domain "$method" "$domain"
  done
done

# Phase 2: Training-based baselines on all domains
for domain in "${DOMAINS[@]}"; do
  for method in "${TRAIN_METHODS[@]}"; do
    run_baseline_on_domain "$method" "$domain"
  done
done

log_progress "=== All baseline runs complete ==="
