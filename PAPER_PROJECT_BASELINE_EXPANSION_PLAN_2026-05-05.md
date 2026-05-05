# Paper-Project Baseline Expansion Plan - 2026-05-05

This note records the baseline coverage decision after completing the
LLM2Rec-style Qwen3 result and paired statistical tests.

## Current Count

Completed external baseline rows:

- Classical same-candidate baselines:
  `SASRec`, `GRU4Rec`, `BERT4Rec`, `LightGCN`.
- Senior-recommended paper-project baselines:
  `LLM2Rec-style Qwen3-8B Emb. + SASRec`.

Current paper-project completed count:

```text
1
```

This is enough for an internal checkpoint, but thin for the final paper if the
claim includes comparison against recent LLM-enhanced recommendation methods.

## Target Before Final Paper Wording

Minimum target:

```text
2 completed senior-recommended paper-project baselines
```

Preferred target:

```text
3 completed or clearly separated paper-project rows:
1. LLM2Rec-style completed result
2. LLM-ESR-style or LLMEmb-style completed result
3. one additional graph/representation or generative-adapter diagnostic row
```

Do not count adapter scaffolds as completed results.

## Recommended Next Candidate

### 1. LLM-ESR-Style Qwen3 Native Wrapper

Priority: highest.

Why:

- Already has a local same-candidate adapter package.
- It is sequential and long-tail focused, so it supports the exposure/long-tail
  story better than another generic sequential baseline.
- Upstream LLM-ESR expects exactly the artifact family our adapter already
  started to produce:
  `inter.txt`, `itm_emb_np.pkl`, `usr_emb_np.pkl`, `pca64_itm_emb_np.pkl`,
  and `sim_user_100.pkl`.

Current blocker:

- The current local LLM-ESR scorer is a centroid protocol scaffold.
- It must not be imported as `same_schema_external_baseline`.

Promotion condition:

- Generate Qwen3 item and user embedding pickles.
- Run or wrap the upstream LLM-ESR recommendation model/training path.
- Score exact same-candidate rows with the trained model/checkpoint.
- Import only after score coverage is `1.0`.

Paper-safe label if completed:

```text
LLM-ESR-style Qwen3-8B Emb. + native ESR SR model
```

Avoid:

```text
official LLM-ESR reproduction
```

unless the upstream preprocessing, embedding notebooks, and experiment scripts
are run exactly as designed.

### 2. LLMEmb-Style Qwen3 Adapter

Priority: second.

Why:

- Same lab/style as LLM-ESR and very close to the embedding-generator narrative.
- Strong complement to LLM2Rec because both test whether LLM item embeddings
  alone can close the gap.

Main friction:

- Upstream LLMEmb has two stages: supervised contrastive fine-tuning and
  recommendation adaptation training.
- If we skip SCFT and use local Qwen3 mean-pooled embeddings, the row must be
  labeled `LLMEmb-style`, not official LLMEmb reproduction.

Promotion condition:

- Use native LLMEmb recommendation adaptation training or a faithful wrapper
  around its trained adapter/model.
- Emit exact same-candidate score CSV with full coverage.

### 3. RLMRec-Style Representation Baseline

Priority: third.

Why:

- Different family from the sequential LLM embedding baselines.
- Graph/representation-learning baseline helps avoid a baseline block that is
  only SASRec-like models with different embeddings.

Main friction:

- Requires graph sparse dependencies and a user-item matrix adapter.
- Candidate scoring must be wrapped from learned user/item representations.

Promotion condition:

- Train native RLMRec or an explicitly labeled RLMRec-style representation
  model on the same train split.
- Score exact same-candidate rows with full coverage.

## Decision

The next implementation target should be LLM-ESR-style, not another classical
model. If LLM-ESR blocks on upstream training, switch to LLMEmb-style before
trying heavier graph/generative projects.

The paper-facing baseline statement should remain:

```text
We evaluate four classical recommenders and at least two same-schema
senior-recommended LLM-rec paper-project baselines under the same split,
candidate set, and metric implementation.
```

until the second paper-project row is completed.
