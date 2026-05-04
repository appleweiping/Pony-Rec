# Baseline Paper Audit - 2026-05-04

This note records the first Week8 audit pass over the senior-recommended
baseline paper folders:

- `Paper/BASELINE/NH`
- `Paper/BASELINE/NR`

The generated matrix is intentionally a protocol audit, not a result table.
Rows from these papers cannot enter the main comparison until they produce
same-split, same-candidate, same-metric predictions.

## Rebuild Command

```bash
python main_audit_baseline_papers.py \
  --baseline_root Paper/BASELINE \
  --collections NH,NR \
  --output_root outputs/summary \
  --output_name baseline_paper_audit_matrix \
  --include_archives
```

For full PDF text extraction, run in an environment with `pypdf`. Without
`pypdf`, the script still produces a filename-level audit but cannot detect
titles and code links reliably.

Generated local outputs:

- `outputs/summary/baseline_paper_audit_matrix.csv`
- `outputs/summary/baseline_paper_audit_matrix.md`

These files stay under `outputs/summary` and are not committed because the
repository treats generated outputs as local artifacts.

## Matrix Summary

The local audit found 36 rows:

- 34 PDF rows from `NH` and `NR`.
- 1 duplicate/bundle PDF row across both folders.
- 1 archive inventory row for `recprefer.zip`.

Priority counts from the extracted local matrix:

- `B_adapter_candidate`: 12
- `C_proxy_only`: 22
- `D_related_only`: 2

No paper is marked as `completed_result`. Every candidate still has protocol
gaps until its code is adapted to our exact candidate-ranking task.

## Runnable Shortlist

The first runnable baseline layer should still be classical same-candidate
recommenders:

| method | priority | why first |
| --- | --- | --- |
| SASRec | A | Strong sequential baseline; easiest to train on our history split and score exact candidates. |
| BERT4Rec | A | Standard bidirectional sequential baseline; complements SASRec. |
| GRU4Rec | A | Lightweight recurrent sequential baseline; useful sanity check. |
| LightGCN | A/B | Strong collaborative-filtering baseline if the user-item graph is dense enough after preprocessing. |

The paper-derived adapter shortlist is second-layer:

| method/paper | collection | role |
| --- | --- | --- |
| OpenP5 | NH | Open LLM-rec platform; useful adapter/reference if it can score exact candidates. |
| SLMRec | NH | Sequential LLM distillation baseline; code detected. |
| LLM-ESR | NH | Long-tail sequential recommendation; especially relevant to exposure/long-tail claims. |
| LLMEmb | NH | LLM embedding generator for sequential recommendation; candidate-score adapter may be feasible. |
| LLM2Rec | NR | LLM embedding model for sequential recommendation; candidate-score adapter may be feasible. |
| RLMRec | NR | Representation-learning baseline with code detected. |
| IRLLRec | NR | Intent representation learning baseline with code detected. |
| SETRec | NR | Generative recommendation identifier baseline; code detected. |
| PAD / ED2 / LRD | NH | Sequential LLM-rec methods; inspect after the smaller shortlist. |
| Uncertainty Quantification and Decomposition for LLM-based Recommendation | NR | Closest uncertainty-related diagnostic reference; not a ranking baseline until output schema is matched. |

## Proxy-Only For Now

Keep these out of the main results table until a same-candidate adapter exists:

- Controllable recommendation and user-agent papers.
- Explanation/evaluation papers such as RecExplainer and LLM-as-recommender
  evaluation work.
- Multimodal or graph/knowledge-heavy papers whose data requirements do not
  match the current Amazon candidate task.
- Any paper with only reported numbers and no reproducible same-schema output.

They are still useful in related work and motivation, but not as leaderboard
rows.

## Same-Schema Contract

Any external baseline row must satisfy this contract before entering the unified
main matrix:

1. Train only on the training interactions for the same domain.
2. Score the exact `candidate_item_ids` in our ranking JSONL.
3. Emit `rank_predictions.jsonl` with `pred_ranked_item_ids`, `topk_item_ids`,
   `candidate_scores`, `parse_success`, and the original candidate metadata.
4. Evaluate with the same HR@10, NDCG@10, MRR, coverage@10,
   head-exposure@10, and long-tail-coverage@10 code path.
5. Attach `status_label=same_schema_external_baseline`.
6. Run candidate protocol audit and paired significance tests before any
   winner claim.

## Immediate Implementation Path

Week8.1 is now a reproducible paper audit:

- `main_audit_baseline_papers.py`
- `baseline_paper_audit_matrix.csv/md` generated locally

Week8.2 should implement the classical same-candidate adapter before cloning
paper-specific repositories:

- Build a small external-baseline manifest for SASRec/BERT4Rec/GRU4Rec/LightGCN.
- Add a common prediction writer that converts model scores into the existing
  rank prediction schema.
- Run one smoke domain first, then four-domain evaluation.

Only after those rows are stable should the paper-derived adapters be attempted.

## Week8.2 Adapter Update

The same-candidate adapter layer now has two entry points:

- `main_export_same_candidate_baseline_task.py`
- `main_import_same_candidate_baseline_scores.py`

The export step writes:

- `train_interactions.csv`
- `candidate_items.csv`
- `recbole/<dataset>.inter`
- `metadata.json`

The import step expects external model scores with:

```text
source_event_id,user_id,item_id,score
```

and writes:

- `predictions/rank_predictions.jsonl`
- `tables/ranking_metrics.csv`
- `tables/ranking_exposure_distribution.csv`
- `tables/same_candidate_external_baseline_summary.csv`

This is an adapter, not a completed SASRec/BERT4Rec/GRU4Rec/LightGCN result.
Those methods still need real model training before any baseline claim.
