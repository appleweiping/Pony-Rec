#!/usr/bin/env bash
# Run all 8 official baselines on new Amazon 2023 domains (sports, toys, home, tools).
# Execute from anywhere; the script cd's to ~/projects/pony-rec-rescue-shadow-v6.
# Prerequisites: C-CRP v3 must be finished on sports/toys/home/tools (GPU free).
#
# Execution order: fastest baselines first (embedding-only), then training-based.
# Each domain runs all baselines before moving to the next.
set -euo pipefail

cd ~/projects/pony-rec-rescue-shadow-v6

PYTHON="${PYTHON:-/home/ajifang/miniconda3/bin/python}"
export PYTHONPATH="$PWD:$PWD/scripts/adapters:$PWD/scripts/audit:$PWD/scripts/build:$PWD/scripts/train${PYTHONPATH:+:$PYTHONPATH}"
BACKBONE=/home/ajifang/models/Qwen/Qwen3-8B
POLICY_ID="official_code_qwen3base_default_hparams_declared_adaptation_v1"
VARIANT="official_code_qwen3base_default_hparams_declared_adaptation"

# Optional overrides support the documented single-domain production loop:
#   DOMAINS_OVERRIDE="sports" TRAIN_METHODS_OVERRIDE="llm2rec_sasrec" bash ...
if [ "${DOMAINS_OVERRIDE+x}" ]; then
  read -r -a DOMAINS <<< "$DOMAINS_OVERRIDE"
else
  read -r -a DOMAINS <<< "sports toys home tools"
fi

log_progress() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_baseline_on_domain() {
  local method=$1
  local domain=$2
  case "$method" in
    llmemb|proex_profile|promax_profile|elmrec_graph|irllrec_intent|rlmrec_graphcl|llm2rec_sasrec|llmesr_sasrec)
      ;;
    ""|*=*)
      log_progress "ERROR: Invalid method token '$method'. Check FAST_METHODS_OVERRIDE/TRAIN_METHODS_OVERRIDE quoting."
      return 1
      ;;
    *)
      log_progress "ERROR: Unknown method $method"
      return 1
      ;;
  esac
  local exp="${domain}_large10000_100neg"
  local task_dir="outputs/baselines/external_tasks/${exp}_test_same_candidate"
  local valid_dir="outputs/baselines/external_tasks/${exp}_valid_same_candidate"
  local out_dir="outputs/${exp}_${method}_official_qwen3base_same_candidate"
  local scores_path="${out_dir}/scores.csv"
  local prov_path="${out_dir}/fairness_provenance.json"
  local audit_log="${out_dir}/${method}_same_candidate_score_audit.txt"
  local import_summary="${out_dir}/tables/same_candidate_external_baseline_summary.csv"

  if [ -f "$scores_path" ] && [ -f "$import_summary" ]; then
    log_progress "SKIP $method on $domain (scores and imported metrics exist)"
    return 0
  fi

  mkdir -p "$out_dir"

  if [ ! -f "$scores_path" ]; then
    log_progress "START $method on $domain"

    case "$method" in
      llmemb)
        "$PYTHON" scripts/adapters/main_run_llmemb_official_same_candidate_adapter.py \
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
        "$PYTHON" scripts/adapters/main_run_proex_official_same_candidate_adapter.py \
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
        "$PYTHON" scripts/adapters/main_run_promax_official_same_candidate_adapter.py \
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
        "$PYTHON" scripts/adapters/main_run_elmrec_official_same_candidate_adapter.py \
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
        "$PYTHON" scripts/adapters/main_run_irllrec_official_same_candidate_adapter.py \
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
        "$PYTHON" scripts/adapters/main_run_rlmrec_official_same_candidate_adapter.py \
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
          --embedding_max_length 128 --hf_device_map auto
        ;;
      llmesr_sasrec)
        "$PYTHON" scripts/adapters/main_run_llmesr_official_same_candidate_adapter.py \
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
  else
    log_progress "IMPORT $method on $domain (scores exist, metrics missing)"
  fi

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
    log_progress "DONE $method on $domain"
  else
    log_progress "ERROR $method on $domain: no scores produced"
    return 1
  fi
}

# Embedding-only baselines first (fastest), then training-based
if [ "${FAST_METHODS_OVERRIDE+x}" ]; then
  read -r -a FAST_METHODS <<< "$FAST_METHODS_OVERRIDE"
else
  read -r -a FAST_METHODS <<< "llmemb proex_profile promax_profile"
fi
# Canonical main official block excludes SETRec while it remains blocked/supplementary.
if [ "${TRAIN_METHODS_OVERRIDE+x}" ]; then
  read -r -a TRAIN_METHODS <<< "$TRAIN_METHODS_OVERRIDE"
else
  read -r -a TRAIN_METHODS <<< "elmrec_graph irllrec_intent rlmrec_graphcl llm2rec_sasrec llmesr_sasrec"
fi

log_progress "=== Starting baseline runs on new domains ==="
log_progress "Python: $PYTHON"
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
