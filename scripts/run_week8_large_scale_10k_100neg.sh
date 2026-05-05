#!/usr/bin/env bash
set -euo pipefail

cd "${PONY_REC_ROOT:-$HOME/projects/pony-rec-rescue-shadow-v6}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
USER_LIMIT="${USER_LIMIT:-10000}"
NUM_NEGATIVES="${NUM_NEGATIVES:-100}"
NEGATIVE_SAMPLING="${NEGATIVE_SAMPLING:-popularity}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GRAPH_BATCH_SIZE="${GRAPH_BATCH_SIZE:-512}"
DEVICE="${DEVICE:-auto}"
QWEN3_MODEL="${QWEN3_MODEL:-$HOME/models/Qwen/Qwen3-8B}"
LLMESR_REPO_DIR="${LLMESR_REPO_DIR:-$HOME/projects/LLM-ESR}"
RUN_LLMESR_STYLE="${RUN_LLMESR_STYLE:-0}"

declare -A DATASET=(
  [books]=amazon_books
  [electronics]=amazon_electronics
  [movies]=amazon_movies
)

declare -A LLM2REC_ALIAS=(
  [books]=BooksLarge10k100Neg
  [electronics]=ElectronicsLarge10k100Neg
  [movies]=MoviesLarge10k100Neg
)

declare -A LLMESR_ALIAS=(
  [books]=books_large10k_100neg
  [electronics]=electronics_large10k_100neg
  [movies]=movies_large10k_100neg
)

domains=(books electronics movies)

for d in "${domains[@]}"; do
  dataset="${DATASET[$d]}"
  exp="${d}_large${USER_LIMIT}_${NUM_NEGATIVES}neg"
  echo "================ Build $d large-scale runtime ================"
  python main_build_large_scale_same_candidate_runtime.py \
    --processed_dir "$HOME/projects/uncertainty-llm4rec/data/processed/$dataset" \
    --domain "$d" \
    --dataset_name "$dataset" \
    --output_root "$OUTPUT_ROOT" \
    --exp_prefix "$exp" \
    --user_limit "$USER_LIMIT" \
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
  exp="${d}_large${USER_LIMIT}_${NUM_NEGATIVES}neg"
  task="$OUTPUT_ROOT/baselines/external_tasks/${exp}_test_same_candidate"
  ranking="$task/ranking_test.jsonl"
  llmesr_adapter="$OUTPUT_ROOT/baselines/paper_adapters/${exp}_llmesr_adapter"
  emb="$llmesr_adapter/llm_esr/handled/itm_emb_np.pkl"
  item_map="$llmesr_adapter/item_id_map.csv"

  echo "================ Export adapters + embeddings for $d ================"
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

  python main_export_llm2rec_same_candidate_task.py \
    --task_dir "$task" \
    --exp_name "${exp}_llm2rec_adapter" \
    --output_root "$OUTPUT_ROOT" \
    --dataset_alias "${LLM2REC_ALIAS[$d]}"

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

  if [[ "$RUN_LLMESR_STYLE" == "1" ]]; then
    echo "================ Optional upstream LLM-ESR-style for $d ================"
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
done

python main_build_external_only_baseline_comparison.py \
  --external_summary_glob "$OUTPUT_ROOT/*large${USER_LIMIT}_${NUM_NEGATIVES}neg*/tables/same_candidate_external_baseline_summary.csv" \
  --domains books,electronics,movies \
  --output_root "$OUTPUT_ROOT/summary" \
  --output_name "external_only_baseline_comparison_week8_large${USER_LIMIT}_${NUM_NEGATIVES}neg"

python main_run_week8_external_only_phenomenon_diagnostics.py \
  --external_summary_glob "$OUTPUT_ROOT/*large${USER_LIMIT}_${NUM_NEGATIVES}neg*/tables/same_candidate_external_baseline_summary.csv" \
  --domains books,electronics,movies \
  --external_methods sasrec,gru4rec,bert4rec,lightgcn,llmemb_style_qwen3_sasrec,rlmrec_style_qwen3_graphcl,irllrec_style_qwen3_intent,setrec_style_qwen3_identifier,llmesr_style_qwen3_sasrec \
  --base_reference best_single_external \
  --output_dir "$OUTPUT_ROOT/summary/week8_large${USER_LIMIT}_${NUM_NEGATIVES}neg_external_only_phenomenon"

python main_run_week8_external_paired_stat_tests.py \
  --external_summary_glob "$OUTPUT_ROOT/*large${USER_LIMIT}_${NUM_NEGATIVES}neg*/tables/same_candidate_external_baseline_summary.csv" \
  --domains books,electronics,movies \
  --output_dir "$OUTPUT_ROOT/summary/week8_large${USER_LIMIT}_${NUM_NEGATIVES}neg_external_stat_tests" \
  --baselines sasrec,gru4rec,bert4rec,lightgcn,llmemb_style_qwen3_sasrec,rlmrec_style_qwen3_graphcl,irllrec_style_qwen3_intent,setrec_style_qwen3_identifier,llmesr_style_qwen3_sasrec \
  --bootstrap_iters 2000 \
  --permutation_iters 2000

echo "Large-scale 10k/100neg run complete."
echo "Main table: $OUTPUT_ROOT/summary/external_only_baseline_comparison_week8_large${USER_LIMIT}_${NUM_NEGATIVES}neg.md"
echo "External-only diagnostics: $OUTPUT_ROOT/summary/week8_large${USER_LIMIT}_${NUM_NEGATIVES}neg_external_only_phenomenon"
