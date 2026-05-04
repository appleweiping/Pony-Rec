# Paper-Project Adapter Shortlist - 2026-05-04

This note records the first inspection pass over senior-recommended paper
repositories after the classical same-candidate baseline suite was completed.

The goal is to choose paper-project baselines that can eventually emit the same
score schema used by the unified matrix:

```text
source_event_id,user_id,item_id,score
```

Rows are not paper results until they pass `main_import_same_candidate_baseline_scores.py`
with full candidate coverage.

## First Target: LLM-ESR

- Paper: `LLM-ESR: Large Language Models Enhancement for Long-tailed Sequential Recommendation`
- Repo: `https://github.com/Applied-Machine-Learning-Lab/LLM-ESR`
- Role: first paper-project adapter target
- Why first:
  - Sequential recommendation fit is high.
  - Long-tail focus directly supports the exposure/long-tail story.
  - The repo exposes model-level `predict(seq, item_indices, positions)` methods
    for SASRec/BERT4Rec/GRU4Rec-style backbones, so exact-candidate scoring is
    structurally feasible.
- Current adapter status:
  - `main_export_llmesr_same_candidate_task.py` exports mapped ids, `inter.txt`,
    candidate rows, item text seeds, and similar-user fallback files.
  - Status is `adapter_package_only`, not `completed_result`.

Remaining blockers:

- Generate LLM-ESR-compatible item embeddings:
  - `itm_emb_np.pkl`
  - `pca64_itm_emb_np.pkl`
- Patch or wrap native inference so it scores `candidate_items_mapped.csv`
  instead of using sampled negative evaluation.
- Import the resulting score CSV through the same-candidate adapter before any
  result claim.

## Backup Target: LLM2Rec

- Paper: `LLM2Rec: Large Language Models Are Powerful Embedding Models for Sequential Recommendation`
- Repo: `https://github.com/HappyPointer/LLM2Rec`
- Role: backup paper-project adapter target
- Why second:
  - Embedding-model design can score exact candidates if item and sequence
    embeddings are available.
  - The repo has a downstream `seqrec` runner and model APIs, but the dependency
    chain is heavier (`llm2vec`, `flash-attn`, embedding extraction).

Recommended adaptation:

- First try precomputed item/sequence embeddings if available.
- Otherwise keep it as an appendix adapter candidate until LLM-ESR is finished.

## Deferred Candidates

| method | repo | reason for deferral |
| --- | --- | --- |
| OpenP5 | `https://github.com/agiresearch/OpenP5` | Platform-style pipeline with dataset generation and checkpoint assumptions; useful reference, not first adapter. |
| SLMRec | `https://github.com/WujiangXu/SLMRec` | Teacher/student distillation chain and LLM checkpoint dependencies make it heavier than LLM-ESR. |
| RLMRec | `https://github.com/HKUDS/RLMRec` | Graph/text-embedding framework; feasible after the sequential paper adapters. |
| IRLLRec | `https://github.com/wangyu0627/IRLLRec` | RLMRec-style graph/text-embedding pipeline with intent embeddings; useful second wave. |

## Server Start

After syncing `main`, export an LLM-ESR adapter package from an existing
same-candidate task:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
git pull --ff-only

python main_export_llmesr_same_candidate_task.py \
  --task_dir outputs/baselines/external_tasks/beauty_week8_same_candidate_external \
  --exp_name beauty_llmesr_same_candidate_adapter \
  --output_root outputs
```

Expected output:

```text
outputs/baselines/paper_adapters/beauty_llmesr_same_candidate_adapter/
```

This package is the handoff point for the LLM-ESR repo wrapper. It is not a
result row and should not be included in the unified method matrix until it
emits full-coverage same-candidate scores.
