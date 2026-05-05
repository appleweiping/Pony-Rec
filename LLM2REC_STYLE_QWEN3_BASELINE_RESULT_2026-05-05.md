# LLM2Rec-Style Qwen3 Baseline Result - 2026-05-05

This note records the first completed same-schema paper-project baseline using
the LLM2Rec adapter path:

```text
LLM2Rec-style Qwen3-8B Emb. + SASRec
```

It is a completed same-candidate baseline row, but it is **not** an official
LLM2Rec CSFT/IEM reproduction.

## Status

- Baseline name: `llm2rec_style_qwen3_sasrec`
- Display name: `LLM2Rec-style Qwen3-8B Emb. + SASRec`
- Status label: `same_schema_external_baseline`
- Artifact class: `completed_result`
- Table group: `paper_project_same_backbone_baseline`
- Backbone: local `Qwen3-8B` item embeddings
- Downstream model: upstream LLM2Rec `SASRec`
- Protocol: exact same-candidate scoring through
  `main_score_llm2rec_same_candidate_adapter.py` and
  `main_import_same_candidate_baseline_scores.py`
- Coverage: `score_coverage_rate=1.0` on all four checked domains

Paper-safe naming:

```text
LLM2Rec-style same-backbone adapter baseline
```

Do not call this row:

```text
official LLM2Rec CSFT/IEM reproduction
```

unless the upstream LLM2Rec CSFT/IEM checkpoint and native extraction path are
actually used.

## Generated Tables

Server-generated source tables:

```text
outputs/summary/unified_method_matrix_week77_shadow_external_qwen_llmesr_llm2rec.csv
outputs/summary/unified_method_matrix_week77_shadow_external_qwen_llmesr_llm2rec.md
outputs/summary/paper_ready_baseline_comparison_week77_qwen_llmesr_llm2rec.csv
outputs/summary/paper_ready_baseline_comparison_week77_qwen_llmesr_llm2rec.md
```

The paper-ready table correctly places this row under:

```text
paper_project_same_backbone_baseline
```

rather than the classical recommender baseline block.

## Four-Domain Result

Paper-ready row excerpt:

| domain | method_family | method | sample_count | NDCG@10 | MRR | delta vs direct | delta vs structured-risk | artifact_class | status_label |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| beauty | paper_project_same_backbone_baseline | LLM2Rec-style Qwen3-8B Emb. + SASRec | 973 | 0.551090 | 0.408565 | -0.062941 | -0.062988 | completed_result | same_schema_external_baseline |
| books | paper_project_same_backbone_baseline | LLM2Rec-style Qwen3-8B Emb. + SASRec | 500 | 0.555974 | 0.415167 | -0.083853 | -0.083540 | completed_result | same_schema_external_baseline |
| electronics | paper_project_same_backbone_baseline | LLM2Rec-style Qwen3-8B Emb. + SASRec | 500 | 0.558289 | 0.417733 | -0.100012 | -0.100012 | completed_result | same_schema_external_baseline |
| movies | paper_project_same_backbone_baseline | LLM2Rec-style Qwen3-8B Emb. + SASRec | 500 | 0.572491 | 0.435633 | -0.000569 | -0.000692 | completed_result | same_schema_external_baseline |

Compact numeric view:

| domain | sample_count | NDCG@10 | MRR | delta vs direct | delta vs structured-risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| beauty | 973 | 0.551090 | 0.408565 | -0.062941 | -0.062988 |
| books | 500 | 0.555974 | 0.415167 | -0.083853 | -0.083540 |
| electronics | 500 | 0.558289 | 0.417733 | -0.100012 | -0.100012 |
| movies | 500 | 0.572491 | 0.435633 | -0.000569 | -0.000692 |

Readout:

- The row is fully covered and importable as a completed same-schema baseline.
- It is close to the structured-risk reference on Movies.
- It remains clearly below the Qwen3 direct/structured-risk line on Beauty,
  Books, and Electronics.

## Classical Baseline Context

The classical same-candidate suite is complete and is not Qwen3-based:

| method | beauty NDCG@10 | books NDCG@10 | electronics NDCG@10 | movies NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| SASRec | 0.519685 | 0.431989 | 0.453392 | 0.460451 |
| GRU4Rec | 0.538966 | 0.540520 | 0.546086 | 0.536479 |
| BERT4Rec | 0.538195 | 0.458560 | 0.475590 | 0.478776 |
| LightGCN | 0.627705 | 0.515724 | 0.516162 | 0.527559 |
| LLM2Rec-style Qwen3-8B Emb. + SASRec | 0.551090 | 0.555974 | 0.558289 | 0.572491 |

Interpretation:

- The LLM2Rec-style same-backbone row is stronger than the classical sequential
  baselines on Books, Electronics, and Movies in this checked matrix.
- LightGCN remains stronger on Beauty among the external baselines.
- None of these external baseline rows should be compared to reported
  full-catalog numbers from other papers; all rows here are exact
  same-candidate results.

## SRPD and Shadow Context

Current method-side readout from the unified method interpretation:

| domain | strongest current method-side row | NDCG@10 | MRR | role |
| --- | --- | ---: | ---: | --- |
| beauty | `shadow_v6` / `SRPD-v2` near tie | 0.635397 | 0.518482 | v6 diagnostic bridge, SRPD framework evidence |
| books | `SRPD-v2` | 0.705707 | 0.611767 | strongest self-trained framework evidence |
| electronics | `shadow_v6` diagnostic | 0.663129 | 0.553467 | promising bridge, needs aligned/statistical gate |
| movies | `shadow_v6` diagnostic / structured-risk | 0.573384 | 0.439333 | near tie with structured-risk and LLM2Rec-style baseline |

Important interpretation:

- SRPD remains the strongest self-trained framework line on Beauty, Books, and
  Electronics.
- Shadow-v6 remains a diagnostic bridge, not yet the final promoted method.
- The LLM2Rec-style same-backbone baseline is a useful external method-design
  check: it uses the same Qwen3-8B backbone for embeddings, but does not use
  uncertainty, SRPD, shadow signals, or risk-aware decision logic.

## Paper-Safe Claim

Safe wording:

> Under the same Qwen3-8B backbone, an LLM2Rec-style embedding-plus-SASRec
> baseline is competitive and nearly matches the structured-risk reference on
> Movies, but it does not close the gap to the uncertainty-aware decision line
> on Beauty, Books, or Electronics.

Also safe:

> The external baseline block now includes four classical recommenders and one
> same-backbone paper-project adapter baseline, all evaluated under the same
> split, same candidate set, and same metric implementation.

Unsafe wording:

> We fully reproduce official LLM2Rec.

Why unsafe:

- The current row uses Qwen3-8B mean-pooled item embeddings through the local
  adapter path.
- It does not run LLM2Rec CSFT/IEM.
- It does not use native LLM2Rec full-pool metrics.

## Commands Used

The row was built with this path:

```bash
python main_generate_llm2rec_sentence_embeddings.py \
  --adapter_dir outputs/baselines/paper_adapters/<domain>_llm2rec_same_candidate_adapter \
  --backend hf_mean_pool \
  --model_name /home/ajifang/models/Qwen/Qwen3-8B \
  --llm2rec_repo_dir ~/projects/LLM2Rec \
  --save_info pony_qwen3_8b \
  --batch_size 2 \
  --max_length 128 \
  --trust_remote_code \
  --torch_dtype bfloat16 \
  --hf_device_map auto

python evaluate_with_seqrec.py \
  --model SASRec \
  --dataset <domain>_same_candidate \
  --embedding ./item_info/<domain>_same_candidate/pony_qwen3_8b_title_item_embs.npy \
  --exp_type srec \
  --lr=1.0e-3 \
  --weight_decay=1.0e-4 \
  --dropout=0.3 \
  --loss_type=ce \
  --run_id=LLM2Rec_style_qwen3_<domain> \
  --max_seq_length=10 \
  --train_batch_size=256 \
  --eval_batch_size=64 \
  --epochs=300 \
  --eval_interval=5 \
  --patience=20

python main_score_llm2rec_same_candidate_adapter.py \
  --adapter_dir outputs/baselines/paper_adapters/<domain>_llm2rec_same_candidate_adapter \
  --llm2rec_repo_dir ~/projects/LLM2Rec \
  --model SASRec \
  --item_embedding_path ~/projects/LLM2Rec/item_info/<domain>_same_candidate/pony_qwen3_8b_title_item_embs.npy \
  --checkpoint_path <domain_checkpoint>.pth \
  --output_scores_path outputs/baselines/paper_adapters/<domain>_llm2rec_same_candidate_adapter/llm2rec_same_candidate_scores.csv

python main_import_same_candidate_baseline_scores.py \
  --baseline_name llm2rec_style_qwen3_sasrec \
  --exp_name <domain>_llm2rec_style_qwen3_sasrec_same_candidate \
  --domain <domain> \
  --ranking_input_path /home/ajifang/projects/uncertainty-llm4rec/data/processed/<dataset>/ranking_test.jsonl \
  --scores_path outputs/baselines/paper_adapters/<domain>_llm2rec_same_candidate_adapter/llm2rec_same_candidate_scores.csv \
  --status_label same_schema_external_baseline \
  --artifact_class completed_result
```

## Remaining Gates

- Run paired statistical tests before using winner wording.
- Keep the row label as LLM2Rec-style unless official LLM2Rec CSFT/IEM is run.
- If official LLM2Rec is later run, add it as a separate row rather than
  overwriting this same-backbone adapter baseline.
