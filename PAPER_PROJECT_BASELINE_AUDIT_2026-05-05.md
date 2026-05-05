# Paper-Project Baseline Adapter Audit - 2026-05-05

## Current Report Table

The compact comparison table is generated from the unified matrix with:

```bash
python main_build_paper_ready_baseline_comparison.py \
  --unified_matrix_path outputs/summary/unified_method_matrix_week77_shadow_external_qwen_llmesr.csv \
  --output_root outputs/summary \
  --output_name paper_ready_baseline_comparison_week77_qwen_llmesr
```

Outputs:

```text
outputs/summary/paper_ready_baseline_comparison_week77_qwen_llmesr.csv
outputs/summary/paper_ready_baseline_comparison_week77_qwen_llmesr.md
```

This table is the preferred report-facing table for now. It contains:

- Week7.7 direct ranking.
- Week7.7 structured-risk rerank.
- Best SRPD LoRA variant per domain.
- Completed same-candidate classical baselines: SASRec, GRU4Rec, BERT4Rec, LightGCN.
- Qwen3-8B LLM-ESR adapter scaffold diagnostic.

The Qwen3/LLM-ESR row must remain:

```text
artifact_class=adapter_scaffold_score
paper_role_hint=adapter_scaffold_diagnostic_not_completed_result
```

It is a real local text-embedding backend and exact same-candidate protocol
check, but not a completed upstream LLM-ESR paper result.

## Adapter Audit Summary

| Candidate | Source | Fit To Same-Candidate Adapter | Main Friction | Recommendation |
| --- | --- | --- | --- | --- |
| LLM2Rec | `https://github.com/HappyPointer/LLM2Rec` | High for embedding-based diagnostic; medium for completed result. The repo exposes embedding extraction and downstream sequential evaluation scripts. | Heavy dependency chain: `torch >= 2.6.0`, `transformers >= 4.44.2`, `llm2vec == 0.2.3`, `flash-attn >= 2.7.4`; two-stage training if we want a completed result. | Next audit target after LLM-ESR. First build an adapter-package exporter/scorer around item or user/item embeddings; then decide whether to run CSFT/IEM. |
| SLMRec | `https://github.com/WujiangXu/SLMRec` | Medium. It is sequential recommendation and has train/eval scripts, but it is a distillation pipeline rather than a simple score-file producer. | Requires SR pretraining, embedding extraction, teacher model fine-tuning, then student distillation. README also notes GPU-dependent hyperparameter adjustment. | Keep as second target if we want a small-model/distillation baseline story. Do not start before LLM2Rec unless the paper narrative needs SLM distillation specifically. |
| OpenP5 | `https://github.com/agiresearch/OpenP5` | Low-to-medium for our immediate same-candidate protocol. It is broad and mature, but platform-style and prompt/generation oriented. | Needs OpenP5 data generation, checkpoint layout, and task-command adaptation. It supports many datasets/backbones, but exact candidate scoring will likely require more glue than embedding methods. | Useful reference/framework baseline, not the next implementation target. Revisit after one embedding-style paper baseline is completed. |

## Evidence From Upstream Repos

LLM2Rec:

- Repo title: KDD 2025 "LLM2Rec: Large Language Models Are Powerful Embedding Models for Sequential Recommendation".
- File layout includes `extract_llm_embedding.py`, `evaluate_with_seqrec.py`, `Baseline_inference.py`, and scripts for extraction/evaluation.
- README describes a two-stage training pipeline: Collaborative Supervised Fine-Tuning (CSFT) followed by Item-level Embedding Modeling (IEM).
- README lists dependencies including torch, transformers, llm2vec, and flash-attn.

SLMRec:

- Repo title: ICLR 2025 "SLMRec: Empowering Small Language Models for Sequential Recommendation".
- File layout includes `train_sr_trad.py`, `extract_emb.py`, `finetune.py`, `distill.py`, `run_finetune.sh`, and `run_distill.sh`.
- README describes first training an SR model and saving an embedding layer, then fine-tuning a teacher and distilling a student.
- README notes that learning rate and batch size may need adjustment for different GPUs.

OpenP5:

- Repo title: "OpenP5: An Open-Source Platform for Developing, Training, and Evaluating LLM-based Recommender Systems".
- File layout includes `command`, `data`, `preprocessing`, `src`, `test_command`, and `generate_dataset.sh`.
- README says OpenP5 supports T5 and LLaMA backbones, multiple datasets, and multiple item ID indexing methods.
- Usage expects OpenP5 dataset generation and command/test-command scripts.

## Recommended Next Implementation Step

Choose LLM2Rec as the next paper-project audit target, because it is closest to
our current adapter shape:

```text
processed same-candidate task
-> mapped user/item ids and histories
-> item or sequence embeddings
-> candidate score CSV
-> main_import_same_candidate_baseline_scores.py
```

Proposed Week8.4 tasks:

1. Clone or inspect `HappyPointer/LLM2Rec` in a separate external repo directory.
2. Identify the minimum input files expected by `extract_llm_embedding.py` and `evaluate_with_seqrec.py`.
3. Build `main_export_llm2rec_same_candidate_task.py` that writes mapped histories and candidate rows.
4. Add `main_audit_llm2rec_adapter_package.py` to verify id maps, histories, item text, and candidate coverage.
5. If the upstream extraction path can emit embeddings without full CSFT/IEM, create a scaffold diagnostic first.
6. Only promote to completed result after an upstream-compatible LLM2Rec run emits full-coverage same-candidate scores.

If LLM2Rec blocks on flash-attn or two-stage training, fall back to SLMRec audit
only if we can reuse its trained representations or sequence scorer without
running the full teacher/student chain.
