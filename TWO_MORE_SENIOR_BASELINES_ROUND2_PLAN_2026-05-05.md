# Two More Senior Baselines Round 2 Plan - 2026-05-05

This note records the next two senior-recommended external paper-project
baselines after the completed four-paper block:

```text
LLM2Rec-style
LLM-ESR-style
LLMEmb-style
RLMRec-style
```

The Round 2 targets are:

```text
IRLLRec-style Qwen3-8B IntentRep
SETRec-style Qwen3-8B Identifier
```

Target final count:

```text
4 classical + 6 senior-recommended LLM-rec paper-project baselines
```

## Added Entrypoints

```text
main_train_irllrec_style_same_candidate.py
main_train_setrec_style_same_candidate.py
```

Both emit:

```text
source_event_id,user_id,item_id,score
```

and should be imported through:

```text
main_import_same_candidate_baseline_scores.py
```

## Baseline 5: IRLLRec-Style

Paper-source audit row:

```text
Intent Representation Learning with Large Language Model for Recommendation
https://github.com/wangyu0627/IRLLRec
```

Local baseline name:

```text
irllrec_style_qwen3_intent
```

Paper-ready display name:

```text
IRLLRec-style Qwen3-8B IntentRep
```

Protocol:

- Use the same train split and exact same-candidate rows.
- Use Qwen3 item embeddings from the existing LLM-ESR adapter.
- Train a local multi-intent representation adapter.
- Score exact candidates and import only if coverage is `1.0`.

Safe wording:

```text
IRLLRec-style intent representation same-candidate baseline
```

Unsafe wording:

```text
official IRLLRec reproduction
```

unless the full official IRLLRec pipeline is reproduced.

## Baseline 6: SETRec-Style

Paper-source audit row:

```text
Order-agnostic Identifier for Large Language Model-based Generative Recommendation
https://github.com/Linxyhaha/SETRec
```

Local baseline name:

```text
setrec_style_qwen3_identifier
```

Paper-ready display name:

```text
SETRec-style Qwen3-8B Identifier
```

Protocol:

- Use the same train split and exact same-candidate rows.
- Use Qwen3 item embeddings from the existing LLM-ESR adapter.
- Train a local order-agnostic set/identifier adapter.
- Score exact candidates and import only if coverage is `1.0`.

Safe wording:

```text
SETRec-style order-agnostic identifier same-candidate baseline
```

Unsafe wording:

```text
official SETRec reproduction
```

unless the full official SETRec generative identifier pipeline is reproduced.

## Server Run Template

Assumes the external task packages already exist under:

```text
outputs/baselines/external_tasks/<domain>_same_candidate
```

If they do not exist, rerun `main_export_same_candidate_baseline_task.py` first.

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
  task=outputs/baselines/external_tasks/${d}_same_candidate
  emb_adapter=outputs/baselines/paper_adapters/${d}_llmesr_same_candidate_adapter
  emb=$emb_adapter/llm_esr/handled/itm_emb_np.pkl
  item_map=$emb_adapter/item_id_map.csv

  echo "================ $d IRLLRec-style ================"
  python main_train_irllrec_style_same_candidate.py \
    --task_dir "$task" \
    --embedding_path "$emb" \
    --item_map_path "$item_map" \
    --output_scores_path outputs/baselines/paper_adapters/${d}_irllrec_style_same_candidate_adapter/irllrec_style_same_candidate_scores.csv \
    --hidden_size 128 \
    --num_intents 4 \
    --epochs 80 \
    --batch_size 128 \
    --device auto

  python main_import_same_candidate_baseline_scores.py \
    --baseline_name irllrec_style_qwen3_intent \
    --exp_name ${d}_irllrec_style_qwen3_intent_same_candidate \
    --domain "$d" \
    --ranking_input_path ~/projects/uncertainty-llm4rec/data/processed/${dataset[$d]}/ranking_test.jsonl \
    --scores_path outputs/baselines/paper_adapters/${d}_irllrec_style_same_candidate_adapter/irllrec_style_same_candidate_scores.csv \
    --status_label same_schema_external_baseline \
    --artifact_class completed_result

  python main_audit_same_candidate_score_file.py \
    --candidate_items_path "$task/candidate_items.csv" \
    --scores_path outputs/baselines/paper_adapters/${d}_irllrec_style_same_candidate_adapter/irllrec_style_same_candidate_scores.csv

  echo "================ $d SETRec-style ================"
  python main_train_setrec_style_same_candidate.py \
    --task_dir "$task" \
    --embedding_path "$emb" \
    --item_map_path "$item_map" \
    --output_scores_path outputs/baselines/paper_adapters/${d}_setrec_style_same_candidate_adapter/setrec_style_same_candidate_scores.csv \
    --hidden_size 128 \
    --epochs 80 \
    --batch_size 128 \
    --device auto

  python main_import_same_candidate_baseline_scores.py \
    --baseline_name setrec_style_qwen3_identifier \
    --exp_name ${d}_setrec_style_qwen3_identifier_same_candidate \
    --domain "$d" \
    --ranking_input_path ~/projects/uncertainty-llm4rec/data/processed/${dataset[$d]}/ranking_test.jsonl \
    --scores_path outputs/baselines/paper_adapters/${d}_setrec_style_same_candidate_adapter/setrec_style_same_candidate_scores.csv \
    --status_label same_schema_external_baseline \
    --artifact_class completed_result

  python main_audit_same_candidate_score_file.py \
    --candidate_items_path "$task/candidate_items.csv" \
    --scores_path outputs/baselines/paper_adapters/${d}_setrec_style_same_candidate_adapter/setrec_style_same_candidate_scores.csv
done
```

## Rebuild After Import

```bash
python main_build_unified_method_matrix.py \
  --week77_root ~/projects/uncertainty-llm4rec/export/week7_7_four_domain_final \
  --shadow_matrix_path outputs/summary/shadow_v1_to_v6_status_matrix.csv \
  --external_summary_glob "outputs/*/tables/same_candidate_external_baseline_summary.csv" \
  --output_root outputs/summary \
  --output_name unified_method_matrix_week77_shadow_external_qwen_6paper

python main_build_paper_ready_baseline_comparison.py \
  --unified_matrix_path outputs/summary/unified_method_matrix_week77_shadow_external_qwen_6paper.csv \
  --output_root outputs/summary \
  --output_name paper_ready_baseline_comparison_week77_qwen_6paper

python main_run_week8_llm2rec_paired_stat_tests.py \
  --week77_root ~/projects/uncertainty-llm4rec/export/week7_7_four_domain_final \
  --external_summary_glob "outputs/*/tables/same_candidate_external_baseline_summary.csv" \
  --output_dir outputs/summary/week8_llm_project_qwen3_6paper_stat_tests \
  --baselines direct,structured_risk,llm2rec_style_qwen3_sasrec,llmesr_style_qwen3_sasrec,llmemb_style_qwen3_sasrec,rlmrec_style_qwen3_graphcl,irllrec_style_qwen3_intent,setrec_style_qwen3_identifier
```

## Verification

```bash
grep "llm2rec_style_qwen3_sasrec\|llmesr_style_qwen3_sasrec\|llmemb_style_qwen3_sasrec\|rlmrec_style_qwen3_graphcl\|irllrec_style_qwen3_intent\|setrec_style_qwen3_identifier" \
  outputs/summary/unified_method_matrix_week77_shadow_external_qwen_6paper.csv

grep "LLM2Rec-style\|LLM-ESR-style\|LLMEmb-style\|RLMRec-style\|IRLLRec-style\|SETRec-style" \
  outputs/summary/paper_ready_baseline_comparison_week77_qwen_6paper.md
```

Do not update the final completed count from `4` to `6` until all four domains
are imported for both new baselines, score coverage is `1.0`, the paper-ready
table has six paper-project rows per domain, and paired statistical tests have
been rerun.
