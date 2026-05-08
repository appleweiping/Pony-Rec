# Two More Senior Baselines Plan - 2026-05-05

Superseded status:

```text
This four-paper expansion has been superseded by the six-paper result in
LLM_PROJECT_QWEN3_6PAPER_BASELINE_RESULT_2026-05-05.md.
```

This note records the next two senior-recommended paper-project baselines to
add after completing:

```text
LLM2Rec-style Qwen3-8B Emb. + SASRec
LLM-ESR-style Qwen3-8B Emb. + LLMESR-SASRec
```

The target is to move from:

```text
4 classical + 2 paper-project baselines
```

to:

```text
4 classical + 4 paper-project baselines
```

Only rows with full exact same-candidate score coverage should be imported as
`same_schema_external_baseline`.

## Added Entrypoints

Two new same-candidate score emitters are available:

```text
main_train_llmemb_style_same_candidate.py
main_train_rlmrec_style_same_candidate.py
```

They both emit:

```text
source_event_id,user_id,item_id,score
```

so they can be imported through:

```text
main_import_same_candidate_baseline_scores.py
```

## Baseline 3: LLMEmb-Style

Paper-source audit row:

```text
LLMEmb: Large Language Model Can Be a Good Embedding Generator for Sequential Recommendation
https://github.com/Applied-Machine-Learning-Lab/LLMEmb
```

Local baseline name:

```text
llmemb_style_qwen3_sasrec
```

Paper-ready display name:

```text
LLMEmb-style Qwen3-8B Emb. + SASRec
```

Protocol:

- Use the same train split and exact same-candidate rows.
- Use the Qwen3 item embedding matrix already produced for the adapter stack.
- Train a local LLMEmb-style item-embedding SASRec adapter.
- Score exact candidates and import only if coverage is `1.0`.

Safe wording:

```text
LLMEmb-style same-backbone adapter baseline
```

Unsafe wording:

```text
official LLMEmb reproduction
```

unless the full official SCFT and recommendation adaptation pipeline is run.

## Baseline 4: RLMRec-Style

Paper-source audit row:

```text
Representation Learning with Large Language Models for Recommendation
https://github.com/HKUDS/RLMRec
```

Local baseline name:

```text
rlmrec_style_qwen3_graphcl
```

Paper-ready display name:

```text
RLMRec-style Qwen3-8B GraphCL
```

Protocol:

- Use the same train split and exact same-candidate rows.
- Train a graph contrastive representation model over the same user-item train
  graph.
- Optionally inject the same Qwen3 item semantic embeddings through an item
  semantic adapter.
- Score exact candidates and import only if coverage is `1.0`.

Safe wording:

```text
RLMRec-style graph/representation same-candidate baseline
```

Unsafe wording:

```text
official RLMRec reproduction
```

unless the full upstream RLMRec preprocessing and training pipeline is run.

## Server Run Template

Assumes the four domain task packages and LLM-ESR adapter embeddings already
exist under `outputs/baselines/paper_adapters`.

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
git pull --ff-only

declare -A dataset=(
  [beauty]=amazon_beauty
  [books]=amazon_books_small
  [electronics]=amazon_electronics_small
  [movies]=amazon_movies_small
)

for d in beauty books electronics movies; do
  python main_export_same_candidate_baseline_task.py \
    --processed_dir ~/projects/uncertainty-llm4rec/data/processed/${dataset[$d]} \
    --ranking_input_path ~/projects/uncertainty-llm4rec/data/processed/${dataset[$d]}/ranking_test.jsonl \
    --exp_name ${d}_same_candidate \
    --dataset_name ${d}_same_candidate
done

for d in beauty books electronics movies; do
  task=outputs/baselines/external_tasks/${d}_same_candidate
  emb_adapter=outputs/baselines/paper_adapters/${d}_llmesr_same_candidate_adapter
  emb=$emb_adapter/llm_esr/handled/itm_emb_np.pkl
  item_map=$emb_adapter/item_id_map.csv

  python main_train_llmemb_style_same_candidate.py \
    --task_dir "$task" \
    --embedding_path "$emb" \
    --item_map_path "$item_map" \
    --output_scores_path outputs/baselines/paper_adapters/${d}_llmemb_style_same_candidate_adapter/llmemb_style_same_candidate_scores.csv \
    --hidden_size 128 \
    --epochs 80 \
    --batch_size 128 \
    --device auto

  python main_import_same_candidate_baseline_scores.py \
    --baseline_name llmemb_style_qwen3_sasrec \
    --exp_name ${d}_llmemb_style_qwen3_sasrec_same_candidate \
    --domain "$d" \
    --ranking_input_path ~/projects/uncertainty-llm4rec/data/processed/${dataset[$d]}/ranking_test.jsonl \
    --scores_path outputs/baselines/paper_adapters/${d}_llmemb_style_same_candidate_adapter/llmemb_style_same_candidate_scores.csv \
    --status_label same_schema_external_baseline \
    --artifact_class completed_result

  python main_audit_same_candidate_score_file.py \
    --candidate_items_path "$task/candidate_items.csv" \
    --scores_path outputs/baselines/paper_adapters/${d}_llmemb_style_same_candidate_adapter/llmemb_style_same_candidate_scores.csv

  python main_train_rlmrec_style_same_candidate.py \
    --task_dir "$task" \
    --embedding_path "$emb" \
    --item_map_path "$item_map" \
    --output_scores_path outputs/baselines/paper_adapters/${d}_rlmrec_style_same_candidate_adapter/rlmrec_style_same_candidate_scores.csv \
    --embedding_size 64 \
    --epochs 80 \
    --batch_size 512 \
    --device auto

  python main_import_same_candidate_baseline_scores.py \
    --baseline_name rlmrec_style_qwen3_graphcl \
    --exp_name ${d}_rlmrec_style_qwen3_graphcl_same_candidate \
    --domain "$d" \
    --ranking_input_path ~/projects/uncertainty-llm4rec/data/processed/${dataset[$d]}/ranking_test.jsonl \
    --scores_path outputs/baselines/paper_adapters/${d}_rlmrec_style_same_candidate_adapter/rlmrec_style_same_candidate_scores.csv \
    --status_label same_schema_external_baseline \
    --artifact_class completed_result

  python main_audit_same_candidate_score_file.py \
    --candidate_items_path "$task/candidate_items.csv" \
    --scores_path outputs/baselines/paper_adapters/${d}_rlmrec_style_same_candidate_adapter/rlmrec_style_same_candidate_scores.csv
done
```

After those imports, rebuild:

```bash
python main_build_unified_method_matrix.py \
  --week77_root ~/projects/uncertainty-llm4rec/export/week7_7_four_domain_final \
  --shadow_matrix_path outputs/summary/shadow_v1_to_v6_status_matrix.csv \
  --external_summary_glob "outputs/*/tables/same_candidate_external_baseline_summary.csv" \
  --output_root outputs/summary \
  --output_name unified_method_matrix_week77_shadow_external_qwen_4paper

python main_build_paper_ready_baseline_comparison.py \
  --unified_matrix_path outputs/summary/unified_method_matrix_week77_shadow_external_qwen_4paper.csv \
  --output_root outputs/summary \
  --output_name paper_ready_baseline_comparison_week77_qwen_4paper

python main_run_week8_llm2rec_paired_stat_tests.py \
  --week77_root ~/projects/uncertainty-llm4rec/export/week7_7_four_domain_final \
  --external_summary_glob "outputs/*/tables/same_candidate_external_baseline_summary.csv" \
  --output_dir outputs/summary/week8_llm_project_qwen3_4paper_stat_tests \
  --baselines direct,structured_risk,llm2rec_style_qwen3_sasrec,llmesr_style_qwen3_sasrec,llmemb_style_qwen3_sasrec,rlmrec_style_qwen3_graphcl
```

## Verification Greps

```bash
grep "llm2rec_style_qwen3_sasrec\|llmesr_style_qwen3_sasrec\|llmemb_style_qwen3_sasrec\|rlmrec_style_qwen3_graphcl" \
  outputs/summary/unified_method_matrix_week77_shadow_external_qwen_4paper.csv

grep "LLM2Rec-style\|LLM-ESR-style\|LLMEmb-style\|RLMRec-style" \
  outputs/summary/paper_ready_baseline_comparison_week77_qwen_4paper.md
```

## Completed Result

The decision boundary has been met:

- all four domains were imported for LLMEmb-style and RLMRec-style,
- score coverage is `1.0`,
- the paper-ready table has four paper-project rows per domain,
- paired statistical tests were rerun in
  `outputs/summary/week8_llm_project_qwen3_4paper_stat_tests`.

Final result note:

```text
LLM_PROJECT_QWEN3_4PAPER_BASELINE_RESULT_2026-05-05.md
```

Paper-safe final count:

```text
4 classical same-candidate baselines
4 senior-recommended LLM-rec paper-project same-schema baselines
```
