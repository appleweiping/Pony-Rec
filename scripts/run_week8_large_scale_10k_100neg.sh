#!/usr/bin/env bash
set -euo pipefail

cd "${PONY_REC_ROOT:-$HOME/projects/pony-rec-rescue-shadow-v6}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
USER_LIMIT="${USER_LIMIT:-10000}"
BEAUTY_USER_LIMIT="${BEAUTY_USER_LIMIT:-0}"
NUM_NEGATIVES="${NUM_NEGATIVES:-100}"
NEGATIVE_SAMPLING="${NEGATIVE_SAMPLING:-popularity}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GRAPH_BATCH_SIZE="${GRAPH_BATCH_SIZE:-512}"
DEVICE="${DEVICE:-auto}"
QWEN3_MODEL="${QWEN3_MODEL:-$HOME/models/Qwen/Qwen3-8B}"

RUN_LLMESR_STYLE="${RUN_LLMESR_STYLE:-1}"
LLMESR_REPO_DIR="${LLMESR_REPO_DIR:-$HOME/projects/LLM-ESR}"

RUN_LLM2REC_STYLE="${RUN_LLM2REC_STYLE:-1}"
LLM2REC_REPO_DIR="${LLM2REC_REPO_DIR:-$HOME/projects/LLM2Rec}"
LLM2REC_SAVE_INFO="${LLM2REC_SAVE_INFO:-pony_qwen3_8b}"
LLM2REC_EPOCHS="${LLM2REC_EPOCHS:-300}"
LLM2REC_TRAIN_BATCH_SIZE="${LLM2REC_TRAIN_BATCH_SIZE:-256}"
LLM2REC_EVAL_BATCH_SIZE="${LLM2REC_EVAL_BATCH_SIZE:-64}"
LLM2REC_PATIENCE="${LLM2REC_PATIENCE:-20}"
REUSE_LLMESR_EMB_FOR_LLM2REC="${REUSE_LLMESR_EMB_FOR_LLM2REC:-1}"

SUMMARY_TAG="${SUMMARY_TAG:-week8_fourdomain_100neg_full_external}"
CURRENT_SUMMARY_GLOB="$OUTPUT_ROOT/*/tables/same_candidate_external_baseline_summary.csv"
DOMAINS_CSV="beauty,books,electronics,movies"
METHODS_CSV="sasrec,gru4rec,bert4rec,lightgcn,llm2rec_style_qwen3_sasrec,llmesr_style_qwen3_sasrec,llmemb_style_qwen3_sasrec,rlmrec_style_qwen3_graphcl,irllrec_style_qwen3_intent,setrec_style_qwen3_identifier"

declare -A DATASET=(
  [beauty]=amazon_beauty
  [books]=amazon_books
  [electronics]=amazon_electronics
  [movies]=amazon_movies
)

declare -A EXP_PREFIX=(
  [beauty]=beauty_supplementary_smallerN_100neg
  [books]=books_large${USER_LIMIT}_${NUM_NEGATIVES}neg
  [electronics]=electronics_large${USER_LIMIT}_${NUM_NEGATIVES}neg
  [movies]=movies_large${USER_LIMIT}_${NUM_NEGATIVES}neg
)

declare -A USER_LIMIT_BY_DOMAIN=(
  [beauty]=$BEAUTY_USER_LIMIT
  [books]=$USER_LIMIT
  [electronics]=$USER_LIMIT
  [movies]=$USER_LIMIT
)

declare -A LLM2REC_ALIAS=(
  [beauty]=BeautySupplementary100Neg
  [books]=BooksLarge10k100Neg
  [electronics]=ElectronicsLarge10k100Neg
  [movies]=MoviesLarge10k100Neg
)

declare -A LLMESR_ALIAS=(
  [beauty]=beauty_supplementary_100neg
  [books]=books_large10k_100neg
  [electronics]=electronics_large10k_100neg
  [movies]=movies_large10k_100neg
)

domains=(beauty books electronics movies)

require_dir() {
  local label="$1"
  local path="$2"
  if [[ ! -d "$path" ]]; then
    echo "Missing required $label directory: $path" >&2
    exit 1
  fi
}

domain_env_value() {
  local prefix="$1"
  local domain="$2"
  local suffix="$3"
  local upper
  upper="$(echo "$domain" | tr '[:lower:]' '[:upper:]')"
  local var="${prefix}_${upper}_${suffix}"
  echo "${!var:-}"
}

latest_checkpoint_since() {
  local root="$1"
  local marker="$2"
  find "$root" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) -newer "$marker" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

run_llm2rec_style() {
  local d="$1"
  local exp="$2"
  local task="$3"
  local ranking="$4"
  local llmesr_adapter="$5"
  local llm2rec_adapter="$OUTPUT_ROOT/baselines/paper_adapters/${exp}_llm2rec_adapter"
  local dataset_alias="${LLM2REC_ALIAS[$d]}"
  local item_embedding_path="$LLM2REC_REPO_DIR/item_info/$dataset_alias/${LLM2REC_SAVE_INFO}_title_item_embs.npy"
  local scores_path="$llm2rec_adapter/llm2rec_same_candidate_scores.csv"

  echo "================ LLM2Rec-style Qwen3 + SASRec for $d ================"
  python main_export_llm2rec_same_candidate_task.py \
    --task_dir "$task" \
    --exp_name "${exp}_llm2rec_adapter" \
    --output_root "$OUTPUT_ROOT" \
    --dataset_alias "$dataset_alias"

  python main_prepare_llm2rec_upstream_adapter.py \
    --adapter_dir "$llm2rec_adapter" \
    --llm2rec_repo_dir "$LLM2REC_REPO_DIR" \
    --dataset_alias "$dataset_alias" \
    --link_mode copy

  if [[ "$REUSE_LLMESR_EMB_FOR_LLM2REC" == "1" ]]; then
    python main_reuse_llmesr_embeddings_for_llm2rec.py \
      --llmesr_adapter_dir "$llmesr_adapter" \
      --llm2rec_adapter_dir "$llm2rec_adapter" \
      --llm2rec_repo_dir "$LLM2REC_REPO_DIR" \
      --save_info "$LLM2REC_SAVE_INFO"
  else
    python main_generate_llm2rec_sentence_embeddings.py \
      --adapter_dir "$llm2rec_adapter" \
      --backend hf_mean_pool \
      --model_name "$QWEN3_MODEL" \
      --llm2rec_repo_dir "$LLM2REC_REPO_DIR" \
      --save_info "$LLM2REC_SAVE_INFO" \
      --batch_size 2 \
      --max_length 128 \
      --trust_remote_code \
      --torch_dtype bfloat16 \
      --hf_device_map auto
  fi

  local checkpoint
  checkpoint="$(domain_env_value LLM2REC "$d" CHECKPOINT)"
  if [[ -z "$checkpoint" ]]; then
    local marker
    marker="$(mktemp)"
    touch "$marker"
    (
      cd "$LLM2REC_REPO_DIR"
      python evaluate_with_seqrec.py \
        --model SASRec \
        --dataset "$dataset_alias" \
        --embedding "./item_info/$dataset_alias/${LLM2REC_SAVE_INFO}_title_item_embs.npy" \
        --exp_type srec \
        --lr=1.0e-3 \
        --weight_decay=1.0e-4 \
        --dropout=0.3 \
        --loss_type=ce \
        --run_id "LLM2Rec_style_qwen3_${d}_${NUM_NEGATIVES}neg" \
        --max_seq_length=10 \
        --train_batch_size="$LLM2REC_TRAIN_BATCH_SIZE" \
        --eval_batch_size="$LLM2REC_EVAL_BATCH_SIZE" \
        --epochs="$LLM2REC_EPOCHS" \
        --eval_interval=5 \
        --patience="$LLM2REC_PATIENCE"
    )
    checkpoint="$(latest_checkpoint_since "$LLM2REC_REPO_DIR" "$marker")"
    rm -f "$marker"
  fi
  if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
    echo "Could not resolve LLM2Rec checkpoint for $d." >&2
    echo "Set LLM2REC_$(echo "$d" | tr '[:lower:]' '[:upper:]')_CHECKPOINT=/path/to/checkpoint and rerun." >&2
    exit 1
  fi

  python main_score_llm2rec_same_candidate_adapter.py \
    --adapter_dir "$llm2rec_adapter" \
    --llm2rec_repo_dir "$LLM2REC_REPO_DIR" \
    --model SASRec \
    --item_embedding_path "$item_embedding_path" \
    --checkpoint_path "$checkpoint" \
    --output_scores_path "$scores_path" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"
  python main_import_same_candidate_baseline_scores.py \
    --baseline_name llm2rec_style_qwen3_sasrec \
    --exp_name "${exp}_llm2rec_style_qwen3_sasrec_same_candidate" \
    --domain "$d" \
    --ranking_input_path "$ranking" \
    --scores_path "$scores_path" \
    --status_label same_schema_external_baseline \
    --artifact_class completed_result
  python main_audit_same_candidate_score_file.py \
    --candidate_items_path "$task/candidate_items.csv" \
    --scores_path "$scores_path"
}

if [[ "$RUN_LLMESR_STYLE" == "1" ]]; then
  require_dir "LLM-ESR repo" "$LLMESR_REPO_DIR"
fi
if [[ "$RUN_LLM2REC_STYLE" == "1" ]]; then
  require_dir "LLM2Rec repo" "$LLM2REC_REPO_DIR"
fi

for d in "${domains[@]}"; do
  dataset="${DATASET[$d]}"
  exp="${EXP_PREFIX[$d]}"
  limit="${USER_LIMIT_BY_DOMAIN[$d]}"
  echo "================ Build $d same-candidate runtime ================"
  python main_build_large_scale_same_candidate_runtime.py \
    --processed_dir "$HOME/projects/uncertainty-llm4rec/data/processed/$dataset" \
    --domain "$d" \
    --dataset_name "$dataset" \
    --output_root "$OUTPUT_ROOT" \
    --exp_prefix "$exp" \
    --user_limit "$limit" \
    --num_negatives "$NUM_NEGATIVES" \
    --max_history_len 50 \
    --min_sequence_length 3 \
    --seed 20260506 \
    --shuffle_seed 42 \
    --splits valid,test \
    --selection_strategy random \
    --negative_sampling "$NEGATIVE_SAMPLING" \
    --test_history_mode train_plus_valid
done

for d in "${domains[@]}"; do
  exp="${EXP_PREFIX[$d]}"
  task="$OUTPUT_ROOT/baselines/external_tasks/${exp}_test_same_candidate"
  ranking="$task/ranking_test.jsonl"
  llmesr_adapter="$OUTPUT_ROOT/baselines/paper_adapters/${exp}_llmesr_adapter"
  emb="$llmesr_adapter/llm_esr/handled/itm_emb_np.pkl"
  item_map="$llmesr_adapter/item_id_map.csv"

  echo "================ Export LLM-ESR adapter + Qwen embeddings for $d ================"
  python main_export_llmesr_same_candidate_task.py \
    --task_dir "$task" \
    --exp_name "${exp}_llmesr_adapter" \
    --output_root "$OUTPUT_ROOT"

  python main_generate_llmesr_sentence_embeddings.py \
    --adapter_dir "$llmesr_adapter" \
    --backend hf_mean_pool \
    --model_name "$QWEN3_MODEL" \
    --batch_size 2 \
    --max_length 128 \
    --trust_remote_code \
    --torch_dtype bfloat16 \
    --hf_device_map auto

  if [[ "$RUN_LLM2REC_STYLE" == "1" ]]; then
    run_llm2rec_style "$d" "$exp" "$task" "$ranking" "$llmesr_adapter"
  fi

  echo "================ Classical baselines for $d ================"
  python main_train_sasrec_same_candidate.py \
    --task_dir "$task" \
    --output_scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_sasrec_adapter/sasrec_scores.csv" \
    --hidden_size 64 \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"
  python main_import_same_candidate_baseline_scores.py \
    --baseline_name sasrec \
    --exp_name "${exp}_sasrec_same_candidate" \
    --domain "$d" \
    --ranking_input_path "$ranking" \
    --scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_sasrec_adapter/sasrec_scores.csv" \
    --status_label same_schema_external_baseline \
    --artifact_class completed_result
  python main_audit_same_candidate_score_file.py \
    --candidate_items_path "$task/candidate_items.csv" \
    --scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_sasrec_adapter/sasrec_scores.csv"

  python main_train_gru4rec_same_candidate.py \
    --task_dir "$task" \
    --output_scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_gru4rec_adapter/gru4rec_scores.csv" \
    --hidden_size 64 \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"
  python main_import_same_candidate_baseline_scores.py \
    --baseline_name gru4rec \
    --exp_name "${exp}_gru4rec_same_candidate" \
    --domain "$d" \
    --ranking_input_path "$ranking" \
    --scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_gru4rec_adapter/gru4rec_scores.csv" \
    --status_label same_schema_external_baseline \
    --artifact_class completed_result
  python main_audit_same_candidate_score_file.py \
    --candidate_items_path "$task/candidate_items.csv" \
    --scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_gru4rec_adapter/gru4rec_scores.csv"

  python main_train_bert4rec_same_candidate.py \
    --task_dir "$task" \
    --output_scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_bert4rec_adapter/bert4rec_scores.csv" \
    --hidden_size 64 \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"
  python main_import_same_candidate_baseline_scores.py \
    --baseline_name bert4rec \
    --exp_name "${exp}_bert4rec_same_candidate" \
    --domain "$d" \
    --ranking_input_path "$ranking" \
    --scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_bert4rec_adapter/bert4rec_scores.csv" \
    --status_label same_schema_external_baseline \
    --artifact_class completed_result
  python main_audit_same_candidate_score_file.py \
    --candidate_items_path "$task/candidate_items.csv" \
    --scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_bert4rec_adapter/bert4rec_scores.csv"

  python main_train_lightgcn_same_candidate.py \
    --task_dir "$task" \
    --output_scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_lightgcn_adapter/lightgcn_scores.csv" \
    --embedding_size 64 \
    --epochs "$EPOCHS" \
    --batch_size "$GRAPH_BATCH_SIZE" \
    --device "$DEVICE"
  python main_import_same_candidate_baseline_scores.py \
    --baseline_name lightgcn \
    --exp_name "${exp}_lightgcn_same_candidate" \
    --domain "$d" \
    --ranking_input_path "$ranking" \
    --scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_lightgcn_adapter/lightgcn_scores.csv" \
    --status_label same_schema_external_baseline \
    --artifact_class completed_result
  python main_audit_same_candidate_score_file.py \
    --candidate_items_path "$task/candidate_items.csv" \
    --scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_lightgcn_adapter/lightgcn_scores.csv"

  echo "================ Qwen3 paper-project style baselines for $d ================"
  if [[ "$RUN_LLMESR_STYLE" == "1" ]]; then
    echo "================ LLM-ESR-style Qwen3 + LLMESR-SASRec for $d ================"
    python main_train_score_llmesr_upstream_adapter.py \
      --adapter_dir "$llmesr_adapter" \
      --llmesr_repo_dir "$LLMESR_REPO_DIR" \
      --dataset_alias "${LLMESR_ALIAS[$d]}" \
      --output_scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_llmesr_adapter/llmesr_upstream_scores.csv" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --device "$DEVICE"
    python main_import_same_candidate_baseline_scores.py \
      --baseline_name llmesr_style_qwen3_sasrec \
      --exp_name "${exp}_llmesr_style_qwen3_sasrec_same_candidate" \
      --domain "$d" \
      --ranking_input_path "$ranking" \
      --scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_llmesr_adapter/llmesr_upstream_scores.csv" \
      --status_label same_schema_external_baseline \
      --artifact_class completed_result
    python main_audit_same_candidate_score_file.py \
      --candidate_items_path "$task/candidate_items.csv" \
      --scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_llmesr_adapter/llmesr_upstream_scores.csv"
  fi

  python main_train_llmemb_style_same_candidate.py \
    --task_dir "$task" \
    --embedding_path "$emb" \
    --item_map_path "$item_map" \
    --output_scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_llmemb_style_adapter/llmemb_style_scores.csv" \
    --hidden_size 128 \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"
  python main_import_same_candidate_baseline_scores.py \
    --baseline_name llmemb_style_qwen3_sasrec \
    --exp_name "${exp}_llmemb_style_qwen3_sasrec_same_candidate" \
    --domain "$d" \
    --ranking_input_path "$ranking" \
    --scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_llmemb_style_adapter/llmemb_style_scores.csv" \
    --status_label same_schema_external_baseline \
    --artifact_class completed_result
  python main_audit_same_candidate_score_file.py \
    --candidate_items_path "$task/candidate_items.csv" \
    --scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_llmemb_style_adapter/llmemb_style_scores.csv"

  python main_train_rlmrec_style_same_candidate.py \
    --task_dir "$task" \
    --embedding_path "$emb" \
    --item_map_path "$item_map" \
    --output_scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_rlmrec_style_adapter/rlmrec_style_scores.csv" \
    --embedding_size 64 \
    --epochs "$EPOCHS" \
    --batch_size "$GRAPH_BATCH_SIZE" \
    --device "$DEVICE"
  python main_import_same_candidate_baseline_scores.py \
    --baseline_name rlmrec_style_qwen3_graphcl \
    --exp_name "${exp}_rlmrec_style_qwen3_graphcl_same_candidate" \
    --domain "$d" \
    --ranking_input_path "$ranking" \
    --scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_rlmrec_style_adapter/rlmrec_style_scores.csv" \
    --status_label same_schema_external_baseline \
    --artifact_class completed_result
  python main_audit_same_candidate_score_file.py \
    --candidate_items_path "$task/candidate_items.csv" \
    --scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_rlmrec_style_adapter/rlmrec_style_scores.csv"

  python main_train_irllrec_style_same_candidate.py \
    --task_dir "$task" \
    --embedding_path "$emb" \
    --item_map_path "$item_map" \
    --output_scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_irllrec_style_adapter/irllrec_style_scores.csv" \
    --hidden_size 128 \
    --num_intents 4 \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"
  python main_import_same_candidate_baseline_scores.py \
    --baseline_name irllrec_style_qwen3_intent \
    --exp_name "${exp}_irllrec_style_qwen3_intent_same_candidate" \
    --domain "$d" \
    --ranking_input_path "$ranking" \
    --scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_irllrec_style_adapter/irllrec_style_scores.csv" \
    --status_label same_schema_external_baseline \
    --artifact_class completed_result
  python main_audit_same_candidate_score_file.py \
    --candidate_items_path "$task/candidate_items.csv" \
    --scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_irllrec_style_adapter/irllrec_style_scores.csv"

  python main_train_setrec_style_same_candidate.py \
    --task_dir "$task" \
    --embedding_path "$emb" \
    --item_map_path "$item_map" \
    --output_scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_setrec_style_adapter/setrec_style_scores.csv" \
    --hidden_size 128 \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"
  python main_import_same_candidate_baseline_scores.py \
    --baseline_name setrec_style_qwen3_identifier \
    --exp_name "${exp}_setrec_style_qwen3_identifier_same_candidate" \
    --domain "$d" \
    --ranking_input_path "$ranking" \
    --scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_setrec_style_adapter/setrec_style_scores.csv" \
    --status_label same_schema_external_baseline \
    --artifact_class completed_result
  python main_audit_same_candidate_score_file.py \
    --candidate_items_path "$task/candidate_items.csv" \
    --scores_path "$OUTPUT_ROOT/baselines/paper_adapters/${exp}_setrec_style_adapter/setrec_style_scores.csv"
done

python main_build_external_only_baseline_comparison.py \
  --external_summary_glob "$CURRENT_SUMMARY_GLOB" \
  --domains "$DOMAINS_CSV" \
  --methods "$METHODS_CSV" \
  --output_root "$OUTPUT_ROOT/summary" \
  --output_name "external_only_baseline_comparison_${SUMMARY_TAG}"

python main_run_week8_external_only_phenomenon_diagnostics.py \
  --external_summary_glob "$CURRENT_SUMMARY_GLOB" \
  --domains "$DOMAINS_CSV" \
  --external_methods "$METHODS_CSV" \
  --base_reference best_single_external \
  --output_dir "$OUTPUT_ROOT/summary/${SUMMARY_TAG}_external_only_phenomenon"

python main_run_week8_external_paired_stat_tests.py \
  --external_summary_glob "$CURRENT_SUMMARY_GLOB" \
  --domains "$DOMAINS_CSV" \
  --output_dir "$OUTPUT_ROOT/summary/${SUMMARY_TAG}_external_stat_tests" \
  --baselines "$METHODS_CSV" \
  --bootstrap_iters 2000 \
  --permutation_iters 2000

echo "Four-domain 100neg full-external run complete."
echo "Main table: $OUTPUT_ROOT/summary/external_only_baseline_comparison_${SUMMARY_TAG}.md"
echo "External-only diagnostics: $OUTPUT_ROOT/summary/${SUMMARY_TAG}_external_only_phenomenon"
