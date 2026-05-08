# LLM2Rec Adapter Audit - 2026-05-05

## Upstream Findings

Repo inspected:

```text
https://github.com/HappyPointer/LLM2Rec
```

Key upstream files:

- `extract_llm_embedding.py`
- `evaluate_with_seqrec.py`
- `seqrec/recdata.py`
- `seqrec/runner.py`
- `seqrec/models/SASRec/_model.py`
- `script_extract_and_evaluate.sh`

LLM2Rec's downstream sequential evaluation expects a directory like:

```text
data/<dataset_path>/item_titles.json
data/<dataset_path>/data.txt
data/<dataset_path>/train_data.txt
data/<dataset_path>/val_data.txt
data/<dataset_path>/test_data.txt
```

The sequence files contain whitespace-separated 1-based item ids. In
`seqrec/recdata.py`, each line is interpreted as a sequence where the final item
is the label and previous items are history. `item_titles.json` is a dictionary
whose keys start at `1`; item id `0` is reserved as padding.

Important friction:

- Dataset aliases and paths are hard-coded in upstream `source_dict` mappings.
- Native evaluation ranks against the full item pool, not our exact candidate
  rows.
- Completed-result status requires an upstream-compatible extraction/training
  and scoring wrapper that emits `source_event_id,user_id,item_id,score` with
  full same-candidate coverage.

## Implemented Adapter Package

New exporter:

```bash
python main_export_llm2rec_same_candidate_task.py \
  --task_dir outputs/baselines/external_tasks/beauty_week8_same_candidate_external \
  --exp_name beauty_llm2rec_same_candidate_adapter \
  --output_root outputs \
  --dataset_alias beauty_same_candidate
```

New audit:

```bash
python main_audit_llm2rec_adapter_package.py \
  --adapter_dir outputs/baselines/paper_adapters/beauty_llm2rec_same_candidate_adapter
```

The exporter writes:

```text
adapter_metadata.json
README.md
candidate_items_mapped.csv
same_candidate_events.csv
user_id_map.csv
item_id_map.csv
item_text_seed.csv
llm2rec/data/<dataset_alias>/downstream/item_titles.json
llm2rec/data/<dataset_alias>/downstream/data.txt
llm2rec/data/<dataset_alias>/downstream/train_data.txt
llm2rec/data/<dataset_alias>/downstream/val_data.txt
llm2rec/data/<dataset_alias>/downstream/test_data.txt
```

Status remains:

```text
adapter_package_only
```

The package does not claim a completed baseline result.

## Same-Candidate Scoring Handoff

The adapter preserves exact candidate rows in:

```text
candidate_items_mapped.csv
same_candidate_events.csv
```

Expected completed-result score schema remains:

```text
source_event_id,user_id,item_id,score
```

Possible wrapper paths:

1. Patch upstream `seqrec/recdata.py` to register the adapter's `dataset_alias`
   and `llm2rec/data/<dataset_alias>/downstream` directory.
2. Train/evaluate upstream SASRec/GRU4Rec with LLM2Rec item embeddings.
3. Patch model scoring so it returns scores for `candidate_items_mapped.csv`
   instead of only full-pool top-k metrics.
4. Import full-coverage score CSV through:

```bash
python main_import_same_candidate_baseline_scores.py \
  --baseline_name llm2rec \
  --exp_name beauty_llm2rec_same_candidate \
  --domain beauty \
  --ranking_input_path ~/projects/uncertainty-llm4rec/data/processed/amazon_beauty/ranking_test.jsonl \
  --scores_path /path/to/llm2rec_scores.csv \
  --status_label same_schema_external_baseline
```

Only step 4 after a genuine upstream-compatible scoring run can enter the main
external baseline table as `completed_result`.

## Four-Domain Export Commands

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
git pull --ff-only

for d in beauty books electronics movies; do
  python main_export_llm2rec_same_candidate_task.py \
    --task_dir outputs/baselines/external_tasks/${d}_week8_same_candidate_external \
    --exp_name ${d}_llm2rec_same_candidate_adapter \
    --output_root outputs \
    --dataset_alias ${d}_same_candidate

  python main_audit_llm2rec_adapter_package.py \
    --adapter_dir outputs/baselines/paper_adapters/${d}_llm2rec_same_candidate_adapter
done
```

Expected audit diagnosis:

```text
diagnosis=ready_for_llm2rec_upstream_wrapper
```

## Current Recommendation

Use this adapter package as the Week8.4 LLM2Rec handoff. The next engineering
step is not to import a scaffold metric, but to patch/run the upstream LLM2Rec
seqrec scorer so it can score our exact candidate rows.

For the paper-project baseline, the safest practical choice is now:

```text
LLM2Rec-style Qwen3-8B Emb. + SASRec
```

That means:

- Qwen3-8B is used as the item embedding backbone.
- We generate padded LLM2Rec-compatible `.npy` embeddings locally.
- The upstream LLM2Rec `SASRec` downstream runner trains on those embeddings.
- We do **not** claim an official LLM2Rec CSFT/IEM reproduction unless the
  upstream LLM2Rec extraction/training path is actually used.

## Upstream Wrapper Step Added

Two local entry points now cover the next handoff:

```bash
python main_prepare_llm2rec_upstream_adapter.py \
  --adapter_dir outputs/baselines/paper_adapters/beauty_llm2rec_same_candidate_adapter \
  --llm2rec_repo_dir ~/projects/LLM2Rec \
  --link_mode copy
```

This installs:

```text
~/projects/LLM2Rec/data/beauty_same_candidate/downstream/
```

and patches upstream:

```text
seqrec/recdata.py
extract_llm_embedding.py
```

so `beauty_same_candidate` is recognized by native LLM2Rec extraction and
seqrec evaluation scripts. Repeat for `books_same_candidate`,
`electronics_same_candidate`, and `movies_same_candidate`.

After native LLM2Rec embedding extraction and seqrec training produce an item
embedding `.npy` and a seqrec checkpoint, emit exact same-candidate scores:

```bash
python main_score_llm2rec_same_candidate_adapter.py \
  --adapter_dir outputs/baselines/paper_adapters/beauty_llm2rec_same_candidate_adapter \
  --llm2rec_repo_dir ~/projects/LLM2Rec \
  --model SASRec \
  --item_embedding_path ~/projects/LLM2Rec/item_info/beauty_same_candidate/<save_info>_title_item_embs.npy \
  --checkpoint_path ~/projects/LLM2Rec/seqrec/ckpt/<best_checkpoint>.pth \
  --output_scores_path outputs/baselines/paper_adapters/beauty_llm2rec_same_candidate_adapter/llm2rec_same_candidate_scores.csv
```

Then audit and import only if coverage is complete:

```bash
python main_audit_same_candidate_score_file.py \
  --candidate_items_path outputs/baselines/external_tasks/beauty_week8_same_candidate_external/candidate_items.csv \
  --scores_path outputs/baselines/paper_adapters/beauty_llm2rec_same_candidate_adapter/llm2rec_same_candidate_scores.csv

python main_import_same_candidate_baseline_scores.py \
  --baseline_name llm2rec \
  --exp_name beauty_llm2rec_same_candidate \
  --domain beauty \
  --ranking_input_path data/processed/amazon_beauty/ranking_test.jsonl \
  --scores_path outputs/baselines/paper_adapters/beauty_llm2rec_same_candidate_adapter/llm2rec_same_candidate_scores.csv \
  --status_label same_schema_external_baseline \
  --artifact_class completed_result
```

This keeps the protocol boundary explicit: upstream native full-pool metrics
are still not imported; only exact same-candidate score CSVs enter the local
baseline matrix.

## Small-Model Path

If the server environment lacks `llm2vec` or `flash-attn`, use the local
Qwen3-8B route instead of blocking the baseline:

```bash
python main_generate_llm2rec_sentence_embeddings.py \
  --adapter_dir outputs/baselines/paper_adapters/beauty_llm2rec_same_candidate_adapter \
  --backend hf_mean_pool \
  --model_name /path/to/local/Qwen3-8B \
  --llm2rec_repo_dir ~/projects/LLM2Rec \
  --save_info pony_qwen3_8b
```

This writes:

```text
outputs/baselines/paper_adapters/beauty_llm2rec_same_candidate_adapter/llm2rec_item_embeddings.npy
~/projects/LLM2Rec/item_info/beauty_same_candidate/pony_qwen3_8b_title_item_embs.npy
```

The generated matrix is padded with a zero row at index 0 so the upstream
LLM2Rec `SASRec`/`GRU4Rec` loaders can consume it directly.
