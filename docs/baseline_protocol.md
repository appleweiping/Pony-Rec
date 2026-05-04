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
