# Week8 Large-Scale 10k/100Neg Same-Candidate Plan - 2026-05-06

This is the next robustness gate after the six-paper baseline block and the
fusion/external-only diagnostics.

## Why Expand

Expanding Books/Electronics/Movies from 500 events to about 10k users is useful,
but not because a larger sample will magically make our standalone framework
beat every SOTA-style row.

The real value is:

- tighter estimates and less noisy domain conclusions,
- a harder candidate set with 101 candidates instead of 6,
- a more standard sampled-ranking protocol for sequential recommendation,
- a clearer test of whether the complementarity/uncertainty phenomenon survives
  outside the small Week7.7 replay set.

## Recommended Protocol

Primary protocol:

```text
Books/Electronics/Movies
10,000 users per domain when enough eligible users exist
leave-one-out temporal split
validation target = penultimate interaction
test target = last interaction
test history = all interactions before the last target
100 sampled negatives + 1 positive = 101 candidates
same candidate rows for every baseline
HR@10, NDCG@10, MRR, exposure metrics
paired tests only within this large-scale protocol
```

Negative sampling:

```text
Default: popularity sampled 100 negatives from items the user never interacted with.
Sensitivity option: uniform sampled 100 negatives by setting NEGATIVE_SAMPLING=uniform.
```

Rationale:

- RecBole supports `uni100` and `pop100` sampled evaluation modes; `uni100`
  uniformly samples 100 negatives for each positive test item, while `pop100`
  samples 100 negatives based on item popularity.
- BERT4Rec-style sampled metrics rank one positive target with 100 negative
  items under leave-one-out evaluation.
- Recent sampled-metric studies warn that model ranking can depend on the
  sampler, so the main run uses popularity sampling and keeps uniform as a
  sensitivity check.

References used for this setting:

- RecBole evaluation settings:
  `https://www.recbole.io/docs/user_guide/config/evaluation_settings.html`
- BERT4Rec-style leave-last-out and sampled-metric discussion:
  `https://www.sciencedirect.com/science/article/pii/S002002552200768X`
- BERT4Rec original paper:
  `https://arxiv.org/pdf/1904.06690`
- Sampling-strategy sensitivity:
  `https://arxiv.org/abs/2107.13045`

Do not mix this table with Week7.7 6-candidate direct/SRPD rows as direct
row-level comparisons. Candidate set size and sampled negatives differ.

## Implemented Artifacts

New scripts:

```text
main_build_large_scale_same_candidate_runtime.py
main_build_external_only_baseline_comparison.py
main_run_week8_external_paired_stat_tests.py
scripts/run_week8_large_scale_10k_100neg.sh
```

Adapter improvements:

```text
main_export_llmesr_same_candidate_task.py
main_export_llm2rec_same_candidate_task.py
```

The task builder writes `item_metadata.csv`, and the LLM2Rec/LLM-ESR adapters
now use it so train-only items get real title/text embeddings instead of falling
back to raw item ids.

## Main Server Command

Run this first:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
git pull --ff-only

bash scripts/run_week8_large_scale_10k_100neg.sh
```

This command builds Books/Electronics/Movies task packages, generates Qwen3 item
embeddings, trains/imports completed same-candidate rows, audits score coverage,
and writes summary tables.

Default methods:

```text
SASRec
GRU4Rec
BERT4Rec
LightGCN
LLMEmb-style Qwen3-8B Emb. + SASRec
RLMRec-style Qwen3-8B GraphCL
IRLLRec-style Qwen3-8B IntentRep
SETRec-style Qwen3-8B Identifier
```

Optional upstream LLM-ESR-style row:

```bash
RUN_LLMESR_STYLE=1 bash scripts/run_week8_large_scale_10k_100neg.sh
```

Useful overrides:

```bash
USER_LIMIT=10000 NUM_NEGATIVES=100 EPOCHS=80 DEVICE=auto bash scripts/run_week8_large_scale_10k_100neg.sh
NEGATIVE_SAMPLING=uniform bash scripts/run_week8_large_scale_10k_100neg.sh
QWEN3_MODEL=/home/ajifang/models/Qwen/Qwen3-8B bash scripts/run_week8_large_scale_10k_100neg.sh
```

## Output Paths

Main large-scale external-only table:

```text
outputs/summary/external_only_baseline_comparison_week8_large10000_100neg.csv
outputs/summary/external_only_baseline_comparison_week8_large10000_100neg.md
```

External-only phenomenon diagnostics:

```text
outputs/summary/week8_large10000_100neg_external_only_phenomenon/external_only_oracle_summary.csv
outputs/summary/week8_large10000_100neg_external_only_phenomenon/external_only_base_rank_bins.csv
outputs/summary/week8_large10000_100neg_external_only_phenomenon/external_only_disagreement_bins.csv
outputs/summary/week8_large10000_100neg_external_only_phenomenon/external_only_popularity_bins.csv
```

External-only paired tests:

```text
outputs/summary/week8_large10000_100neg_external_stat_tests/all_domains_significance_tests.csv
outputs/summary/week8_large10000_100neg_external_stat_tests/all_domains_main_table_with_ci.csv
outputs/summary/week8_large10000_100neg_external_stat_tests/input_manifest.csv
```

Per-domain runtime summaries:

```text
outputs/summary/books_large10000_100neg_runtime_summary.json
outputs/summary/electronics_large10000_100neg_runtime_summary.json
outputs/summary/movies_large10000_100neg_runtime_summary.json
```

Per-domain task packages:

```text
outputs/baselines/external_tasks/books_large10000_100neg_test_same_candidate
outputs/baselines/external_tasks/electronics_large10000_100neg_test_same_candidate
outputs/baselines/external_tasks/movies_large10000_100neg_test_same_candidate
```

## What To Look For

For main performance:

- Compare external rows only inside
  `external_only_baseline_comparison_week8_large10000_100neg.md`.
- Look for whether the strongest row remains IRLLRec/RLMRec/LLMEmb, or whether
  classical baselines become stronger under the harder 101-candidate protocol.
- Use paired stats from
  `week8_large10000_100neg_external_stat_tests/all_domains_significance_tests.csv`.

For phenomenon evidence:

- `oracle_gain_vs_best_single_NDCG@10 > 0` means external baselines retain
  complementary event-level signal.
- Larger oracle gain in weaker base-rank bins means the earlier hard-event
  phenomenon persists at larger scale.
- Larger oracle gain in high-disagreement bins means external-method
  disagreement remains a usable uncertainty proxy.

## If Runtime Is Too High

Fast first pass:

```bash
USER_LIMIT=3000 EPOCHS=40 bash scripts/run_week8_large_scale_10k_100neg.sh
```

Then rerun the full command after checking that all output files and coverage
audits look clean.

Optional sensitivity run:

```bash
NEGATIVE_SAMPLING=uniform USER_LIMIT=10000 bash scripts/run_week8_large_scale_10k_100neg.sh
```

This will overwrite the same output names unless `OUTPUT_ROOT` or `USER_LIMIT`
is changed. For a side-by-side sensitivity run, use:

```bash
NEGATIVE_SAMPLING=uniform OUTPUT_ROOT=outputs_uniform bash scripts/run_week8_large_scale_10k_100neg.sh
```

## Paper-Safe Wording

Safe:

```text
We add a large-scale sampled-ranking validation on Books, Electronics, and
Movies with 10k users per domain and 100 sampled negatives plus one positive
per test event. All baselines score the exact same candidate rows, and paired
tests are run only within this aligned 101-candidate protocol.
```

Also safe:

```text
The large-scale run is used as a robustness and generalization gate for the
small same-candidate replay findings, not as a direct row-level replacement for
the Week7.7 six-candidate LLM replay table.
```

Unsafe:

```text
The 10k/100neg table proves our Week7.7 direct/SRPD method beats the external
baselines.
```

Why unsafe:

- The large-scale external-only table and Week7.7 replay table use different
  candidate sets and different event files.
- Direct/SRPD large-scale 101-candidate LLM inference is a separate expensive
  optional tier, not included in the default script.
