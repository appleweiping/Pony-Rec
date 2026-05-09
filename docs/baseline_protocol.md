# Baseline Protocol

The baseline layer separates same-schema evidence from proxy positioning.

## Baseline groups

Main paper comparisons should include these groups when available under the
same schema:

- Non-LLM recommenders: SASRec, BERT4Rec, GRU4Rec, or LightGCN.
- Simple recommendation priors: popularity, recency, history overlap, BM25 or
  title embedding.
- LLM direct ranking: same prompt and candidate set, no uncertainty signal.
- Uncertainty baselines: raw confidence, Platt or isotonic calibrated
  confidence, self-consistency, entropy or logprob when available.

Internal SRPD/shadow variants are ablations, not substitutes for external
baselines.

## Standardized external-baseline comparison policy

Different papers use different comparison conventions. We keep the policy
explicit instead of claiming a single absolute notion of fairness.

Future agents must also follow the official-baseline guardrails in
`AGENTS.md`: wrapper names, plan rows, and `official` filenames are not
evidence unless provenance, exact score coverage, import, and paired-test gates
pass.

Common acceptable modes:

1. Run each official implementation with its original backbone and default
   settings, changing only the data interface.
2. Reuse a prior work's dataset/protocol and compare against its reported
   baseline block.
3. Use official implementations but standardize the LLM/text backbone to one
   model, then run each method's official/default training recipe.
4. Validation-tune every baseline and our method under an equal tuning budget.

The primary paper policy is mode 3:

```text
official or official-code-level source implementation
+ our same-candidate dataset/protocol
+ shared Qwen3-8B base model for all LLM/text components
+ frozen Qwen3-8B base except method-declared adapter or representation module
+ baseline official default or recommended hyperparameters
+ our hyperparameters selected on validation only or fixed before test
+ shared score schema: source_event_id,user_id,item_id,score
```

This is the standard controlled academic comparison used for the main external
baseline table. Full fine-tuning of Qwen3-8B, original-backbone runs, and fully
retuned-baseline sensitivity studies can appear only as explicitly labeled
supplementary tables.

The canonical primary comparison variant is:

```text
official_code_qwen3base_default_hparams_declared_adaptation
```

For official main-table rows, the score file must exactly match the candidate
rows: unique `source_event_id,user_id,item_id` keys, no missing candidate
scores, no extra score keys, and finite numeric `score` values. Scores may use
any method-native scale as long as they are orderable within each event; ties
are broken by stable candidate order after descending score. Candidate order
cannot be used as an implicit score.

Beauty rows are supplementary smaller-N same-candidate 100neg rows unless the
eligible user count reaches the main 10k-domain target. Beauty pilot or adapter
checks do not enter the main four-domain table.

Implementation status is table-gating metadata:

```text
style_adapter_only -> supplementary only
partial_official_adapter_exists -> not main-table eligible
official_completed -> eligible only if provenance, score coverage, and paired-test inputs pass
```

For storage-heavy official LLM-rec baselines, execute and archive one
method-domain row at a time. The completed LLM2Rec large-domain evidence
packages are the template: keep the score CSV, fairness provenance, score
audit, run summary, training/server log, imported prediction/table outputs,
comparison summaries, and checkpoint/embedding sha256 manifests in a
lightweight archive. Full checkpoints are optional separate preservation
artifacts; they should not block the next domain unless the user explicitly
asks for full model backup. Server-side intermediates can be deleted only after
the corresponding evidence archive has been copied off the server and confirmed
locally.

## Reliability proxy audit

The old "baseline confidence formulation audit" is renamed:

```text
baseline reliability proxy audit
```

Run:

```bash
python main_baseline_reliability_audit.py \
  --config configs/baseline_reliability/week7_9_manifest.yaml \
  --output_path outputs/summary/baseline_reliability_proxy_audit.csv
```

The schema includes:

- `baseline_name`
- `baseline_family`
- `confidence_semantics`
- `calibration_target`
- `is_relevance_calibratable`
- `can_compute_ece`
- `can_run_selective_analysis`
- `risk_of_unfair_comparison`
- `protocol_gap`
- `status_label`

## ECE boundary

ECE and Brier are valid only for relevance-calibratable signals:

- `self_reported_confidence`
- `calibrated_relevance_probability`

Signals such as `exposure_policy_certainty`, candidate order, or pure
popularity are not relevance probabilities. They may appear in exposure or
policy audit tables, but not in the relevance calibration table.

## Related-work reported numbers

Reported numbers from different protocols must not enter the same main ranking
table. They can appear only in a proxy table with `protocol_gap`, for example:

- `different_candidate_space`
- `full_catalog_vs_sampled`
- `different_backbone`
- `no_confidence_output`
- `not_relevance_calibratable`

## Week8 Baseline Gates

The rescue branch distinguishes four baseline layers:

1. Runnable task-aligned baselines:
   candidate order, popularity prior, long-tail prior, history/title overlap,
   and pairwise analogues. These are protocol/sanity baselines, not enough for
   a final external-baseline claim.
2. Classical recommender baselines:
   SASRec, BERT4Rec, GRU4Rec, LightGCN, or equivalent RecBole-style methods.
   These must score the exact candidate set used by the ranking task.
3. Senior-recommended paper baselines:
   methods selected from `Paper/BASELINE/NH` and `Paper/BASELINE/NR`. Each
   paper/project needs a protocol-gap audit before entering a result table.
4. Our trainable framework line:
   SRPD is the current self-trained ranking framework evidence; shadow Signal
   LoRA and Decision LoRA are future stages until run under the same protocol.

SRPD/shadow variants are not substitutes for external baselines. They answer
method-ablation and trainable-framework questions. External baselines answer
whether the framework is competitive with prior recommendation systems under a
matched protocol.

## Senior-Recommended Paper Audit

Audit `Paper/BASELINE/NH` and `Paper/BASELINE/NR` before selecting paper-specific
external baselines:

```bash
python main_audit_baseline_papers.py \
  --baseline_root Paper/BASELINE \
  --collections NH,NR \
  --output_root outputs/summary \
  --output_name baseline_paper_audit_matrix \
  --include_archives
```

The audit matrix separates:

- `B_adapter_candidate`: code/title suggests a possible same-candidate adapter,
  but no result claim exists yet.
- `C_proxy_only`: useful for related work or motivation, not ready for main
  result comparison.
- `D_related_only`: keep outside result tables unless a runnable implementation
  is found.

The first external result layer should still be SASRec, BERT4Rec, GRU4Rec, and
LightGCN-style same-candidate baselines. Paper-specific adapters such as OpenP5,
SLMRec, LLM-ESR, LLMEmb, LLM2Rec, RLMRec, IRLLRec, SETRec, PAD, ED2, or LRD are
second-layer candidates after the classical baselines are stable.

The selected Week8 manifest is:

```text
configs/baseline/week8_external_same_candidate_manifest.yaml
```

Every external baseline must emit the same ranking prediction schema and carry
`status_label=same_schema_external_baseline` before it can enter the unified
method matrix.

## Same-Candidate Adapter

Use the export/import pair for SASRec, BERT4Rec, GRU4Rec, LightGCN, or any
paper-specific implementation:

```bash
python main_export_same_candidate_baseline_task.py \
  --processed_dir data/processed/amazon_beauty \
  --ranking_input_path data/processed/amazon_beauty/ranking_test.jsonl \
  --exp_name beauty_week8_same_candidate_external
```

The external model should train on the exported `train_interactions.csv` or
RecBole-style `.inter` file, then score every row in `candidate_items.csv`.

For SASRec, the repository now includes a lightweight PyTorch trainer that does
not require RecBole:

```bash
python main_train_sasrec_same_candidate.py \
  --task_dir outputs/baselines/external_tasks/beauty_week8_same_candidate_external \
  --epochs 80 \
  --hidden_size 64 \
  --num_layers 2 \
  --num_heads 2 \
  --batch_size 128 \
  --device auto
```

It writes:

```text
outputs/baselines/external_tasks/beauty_week8_same_candidate_external/sasrec_scores.csv
```

For GRU4Rec, use the analogous lightweight PyTorch trainer:

```bash
python main_train_gru4rec_same_candidate.py \
  --task_dir outputs/baselines/external_tasks/beauty_week8_same_candidate_external \
  --epochs 60 \
  --hidden_size 64 \
  --num_layers 1 \
  --batch_size 128 \
  --device auto
```

It writes:

```text
outputs/baselines/external_tasks/beauty_week8_same_candidate_external/gru4rec_scores.csv
```

For BERT4Rec, use the masked-sequence trainer:

```bash
python main_train_bert4rec_same_candidate.py \
  --task_dir outputs/baselines/external_tasks/beauty_week8_same_candidate_external \
  --epochs 60 \
  --hidden_size 64 \
  --num_layers 2 \
  --num_heads 2 \
  --batch_size 128 \
  --device auto
```

It writes:

```text
outputs/baselines/external_tasks/beauty_week8_same_candidate_external/bert4rec_scores.csv
```

For LightGCN, use the graph collaborative-filtering trainer:

```bash
python main_train_lightgcn_same_candidate.py \
  --task_dir outputs/baselines/external_tasks/beauty_week8_same_candidate_external \
  --epochs 80 \
  --embedding_size 64 \
  --num_layers 2 \
  --batch_size 512 \
  --device auto
```

It writes:

```text
outputs/baselines/external_tasks/beauty_week8_same_candidate_external/lightgcn_scores.csv
```

Expected score file schema:

```text
source_event_id,user_id,item_id,score
```

Import and evaluate:

```bash
python main_import_same_candidate_baseline_scores.py \
  --baseline_name sasrec \
  --exp_name beauty_sasrec_same_candidate \
  --domain beauty \
  --ranking_input_path data/processed/amazon_beauty/ranking_test.jsonl \
  --scores_path outputs/baselines/external_tasks/beauty_week8_same_candidate_external/sasrec_scores.csv \
  --artifact_class completed_result \
  --status_label same_schema_external_baseline
```

The import step writes `predictions/rank_predictions.jsonl` and the same
ranking metrics used by direct/SRPD/shadow rows. For
`status_label=same_schema_external_baseline`, the import step requires full
score coverage by default.

If coverage fails, audit the score file before importing:

```bash
python main_audit_same_candidate_score_file.py \
  --candidate_items_path outputs/baselines/external_tasks/beauty_week8_same_candidate_external/candidate_items.csv \
  --scores_path outputs/baselines/external_tasks/beauty_week8_same_candidate_external/sasrec_scores.csv
```

The audit must show `invalid_scores=0` and full exact-key coverage. A score
file full of `nan` values is invalid even if the row count and keys match.

After importing real external baseline scores, include those rows in the unified
method matrix with:

```bash
python main_build_unified_method_matrix.py \
  --week77_root ~/projects/uncertainty-llm4rec/export/week7_7_four_domain_final \
  --shadow_matrix_path outputs/summary/shadow_v1_to_v6_status_matrix.csv \
  --external_summary_glob "outputs/*/tables/same_candidate_external_baseline_summary.csv" \
  --output_root outputs/summary \
  --output_name unified_method_matrix_week77_shadow_external
```

## Paper-Project Adapters

The first paper-project target after the completed classical suite is LLM-ESR.
Export an adapter package from an existing same-candidate task:

```bash
python main_export_llmesr_same_candidate_task.py \
  --task_dir outputs/baselines/external_tasks/beauty_week8_same_candidate_external \
  --exp_name beauty_llmesr_same_candidate_adapter \
  --output_root outputs
```

This writes an `adapter_package_only` bundle under:

```text
outputs/baselines/paper_adapters/beauty_llmesr_same_candidate_adapter/
```

The package contains mapped 1-based user/item ids, LLM-ESR-style `inter.txt`,
candidate rows with mapped ids, item text seeds, and similar-user fallback
files. It is not a completed result. A paper-project row can enter the unified
matrix only after the adapted repo emits the same score schema with full
candidate coverage:

```text
source_event_id,user_id,item_id,score
```

Audit the package before attempting embedding generation or scorer wrapping:

```bash
python main_audit_llmesr_adapter_package.py \
  --adapter_dir outputs/baselines/paper_adapters/beauty_llmesr_same_candidate_adapter
```

The expected pre-embedding diagnosis is
`adapter_core_ready_embeddings_missing_or_invalid`. The package is not ready
for scoring until both `llm_esr/handled/itm_emb_np.pkl` and
`llm_esr/handled/pca64_itm_emb_np.pkl` are present with one row per mapped item.
The `pca64` file must have 64 columns because the upstream LLM-ESR model adds
it directly to a 64-dimensional position embedding.

Note that LLM-ESR item ids remain 1-based, but `sim_user_100.pkl` stores
0-based dataset row indices because the upstream dataset code indexes
`self.data[user]`.

For adapter/scorer smoke tests, generate deterministic text-hash scaffold
embeddings:

```bash
python main_generate_llmesr_text_embeddings.py \
  --adapter_dir outputs/baselines/paper_adapters/beauty_llmesr_same_candidate_adapter
```

These scaffold embeddings are not a completed paper-project baseline. Replace
them with true LLM item embeddings before marking an LLM-ESR row as
`completed_result`.

To smoke-test the exact-candidate scoring and import path, score the adapter
package:

```bash
python main_score_llmesr_same_candidate_adapter.py \
  --adapter_dir outputs/baselines/paper_adapters/beauty_llmesr_same_candidate_adapter
```

This writes `llmesr_scaffold_scores.csv`. Import it only with a non-main status
label such as `llmesr_adapter_scaffold_score`; it is a protocol check, not a
paper-result row. The importer maps scaffold status labels to
`artifact_class=adapter_scaffold_score` to prevent accidental
`completed_result` labeling.

To run this scaffold protocol across all four domains:

```bash
python main_run_llmesr_scaffold_four_domain.py \
  --processed_root ~/projects/uncertainty-llm4rec/data/processed \
  --raw_metadata_root ~/projects/uncertainty-llm4rec/data/raw \
  --output_root outputs
```

When `--raw_metadata_root` is provided, the runner fails fast if a domain's
expected raw metadata file is missing. Use `--allow_missing_raw_metadata` only
for intentional smoke tests without raw catalog enrichment.

The generated four-domain summary is also a scaffold/protocol artifact, not a
main paper baseline table.

Before using a real text-embedding backend, enrich `item_text_seed.csv` from the
processed catalog and raw Amazon metadata:

```bash
python main_enrich_llmesr_item_text_seed.py \
  --adapter_dir outputs/baselines/paper_adapters/movies_llmesr_same_candidate_adapter \
  --processed_dir ~/projects/uncertainty-llm4rec/data/processed/amazon_movies_small \
  --raw_metadata_path ~/projects/uncertainty-llm4rec/data/raw/amazon_movies/meta_Movies_and_TV.jsonl.gz
```

Then replace the deterministic hash pickle files with true text embeddings:

```bash
python main_generate_llmesr_sentence_embeddings.py \
  --adapter_dir outputs/baselines/paper_adapters/movies_llmesr_same_candidate_adapter \
  --backend sentence_transformers \
  --model_name sentence-transformers/all-MiniLM-L6-v2
```

Local Hugging Face models such as Qwen3 8B can be used as a hidden-state
mean-pooling backend when no dedicated embedding model is available:

```bash
python main_generate_llmesr_sentence_embeddings.py \
  --adapter_dir outputs/baselines/paper_adapters/movies_llmesr_same_candidate_adapter \
  --backend hf_mean_pool \
  --model_name /path/to/local/qwen3-8b \
  --batch_size 2 \
  --max_length 256 \
  --torch_dtype bfloat16 \
  --hf_device_map auto \
  --trust_remote_code
```

For Movies, raw metadata may not contain titles for every Prime Video item, but
it can still fill useful `embedding_text` from main category and details. Track
both `title_seed_coverage` and `non_id_embedding_text_coverage`; the latter is
the cleaner readiness signal for text embeddings.

Even with true text item embeddings from sentence-transformers or HF mean
pooling, the current centroid scorer remains `adapter_scaffold_score`. It
verifies the same-candidate protocol and embedding files, but it is not an
upstream LLM-ESR completed result.
