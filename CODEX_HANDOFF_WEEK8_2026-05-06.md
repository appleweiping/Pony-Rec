# Codex Handoff - Week8 Large-Scale and External Baselines - 2026-05-06

This file is the current working handoff for the next Codex chat. Read this
first before changing scripts or rerunning experiments.

## Canonical First-Read Files

Before using this handoff, read:

```text
docs/milestones/README.md
docs/server_runbook.md
docs/top_conference_review_gate.md
```

Current milestone position:

```text
M4 baseline system -> M5 four-domain same-candidate validation.
M6 complete recommendation system remains a roadmap until official baselines,
Shadow large-scale diagnostics, LoRA modules, and generated-title verification
are complete under the shared protocol.
```

## Current High-Level State

The small/medium same-candidate proof block is complete:

- Four classical same-candidate baselines are available.
- Six senior-recommended LLM-rec paper-project style baselines are available.
- The official external-baseline upgrade contract is now explicit: final
  official rows must use pinned upstream implementations, the unified Qwen3-8B
  base model, each baseline's official LoRA/adapter or representation-training
  algorithm, and the unchanged same-candidate score schema.
- Ours + external rank-fusion diagnostic is available.
- External-only phenomenon diagnostic is available.

The main conclusion has changed from "ours simply beats all baselines" to:

```text
Our framework contributes a complementary risk/decision signal that can improve
or explain strong external LLM-rec baselines under the same-candidate protocol.
```

Do not claim a universal standalone SOTA win. The external baselines are strong,
especially IRLLRec-style and RLMRec-style.

Do not relabel the current `*_style_*` rows as official reproductions. They are
paper-style same-candidate adaptations until the official upgrade checklist in
`OFFICIAL_EXTERNAL_BASELINE_UPGRADE_PLAN_2026-05-07.md` is complete.

Official rows must keep this schema unchanged:

```text
source_event_id,user_id,item_id,score
```

The candidate protocol, metric schema, importer, coverage audit, and paired-test
path stay the same; only the implementation standard rises to official or
official-code-level adapters.

## Completed Six-Paper External Baseline Block

Completed same-candidate external methods:

```text
llm2rec_style_qwen3_sasrec
llmesr_style_qwen3_sasrec
llmemb_style_qwen3_sasrec
rlmrec_style_qwen3_graphcl
irllrec_style_qwen3_intent
setrec_style_qwen3_identifier
```

Strongest observed paper-project rows in the Week7.7 six-candidate protocol:

| domain | strongest observed external row | NDCG@10 |
| --- | --- | ---: |
| beauty | IRLLRec-style Qwen3-8B IntentRep | 0.662061 |
| books | IRLLRec-style Qwen3-8B IntentRep | 0.716744 |
| electronics | RLMRec-style Qwen3-8B GraphCL | 0.628100 |
| movies | IRLLRec-style Qwen3-8B IntentRep | 0.707149 |

Important output files:

```text
outputs/summary/unified_method_matrix_week77_shadow_external_qwen_6paper.csv
outputs/summary/paper_ready_baseline_comparison_week77_qwen_6paper.md
outputs/summary/week8_llm_project_qwen3_6paper_stat_tests/
```

## Completed Ours + External Fusion Diagnostic

Best observed diagnostic fusion rows:

| domain | fusion pair | ours weight | fused NDCG@10 | gain over best constituent |
| --- | --- | ---: | ---: | ---: |
| beauty | SRPD-best + IRLLRec-style | 0.3 | 0.663963 | +0.001902 |
| books | SRPD-best + IRLLRec-style | 0.3 | 0.730990 | +0.014247 |
| electronics | structured-risk + RLMRec-style | 0.7 | 0.676842 | +0.018542 |
| movies | structured-risk + IRLLRec-style | 0.3 | 0.708930 | +0.001781 |

Output path:

```text
outputs/summary/week8_ours_external_rank_fusion/fusion_best_by_domain.csv
```

Paper safety:

- This is a diagnostic upper bound because the best weight was selected on test.
- Safe wording: diagnostic rank-fusion / complementary signal.
- Do not call it a new main SOTA method unless the weight is fixed in advance
  or selected on validation.

## Completed External-Only Phenomenon Diagnostic

When structured-risk and SRPD are removed from the candidate method set, the
external paper-project baselines still show event-level complementarity.

External-only oracle gains:

| domain | best single external | best single NDCG@10 | external oracle NDCG@10 | oracle gain |
| --- | --- | ---: | ---: | ---: |
| beauty | IRLLRec-style | 0.662061 | 0.886395 | +0.224334 |
| books | IRLLRec-style | 0.716744 | 0.890521 | +0.173777 |
| electronics | RLMRec-style | 0.628100 | 0.828486 | +0.200385 |
| movies | IRLLRec-style | 0.707149 | 0.890900 | +0.183751 |

Output paths:

```text
outputs/summary/week8_external_only_phenomenon/external_only_oracle_summary.csv
outputs/summary/week8_external_only_phenomenon/external_only_base_rank_bins.csv
outputs/summary/week8_external_only_phenomenon/external_only_disagreement_bins.csv
outputs/summary/week8_external_only_phenomenon/external_only_popularity_bins.csv
```

Interpretation:

- Oracle gain is zero in rank-1 bins because the best single method already
  ranks the positive first.
- Oracle gain is large in rank-2/3 and rank-4/6 bins.
- High-disagreement bins generally show larger oracle gain than low-disagreement
  bins.
- Popularity-bin evidence is mixed, so do not overclaim long-tail effects.

## Current Large-Scale Experiment

The next robustness gate is the four-domain 100neg same-candidate protocol:

```text
Books/Electronics/Movies: 10,000 users per domain when enough eligible users exist
Beauty: supplementary smaller-N same-candidate 100neg run
leave-one-out temporal split
validation target = penultimate interaction
test target = last interaction
test history = all interactions before the last target
100 sampled negatives + 1 positive = 101 candidates
same candidate rows for every baseline
full external comparison within this 100neg protocol
```

Default baseline rows:

```text
SASRec
GRU4Rec
BERT4Rec
LightGCN
LLM2Rec-style Qwen3-8B Emb. + SASRec
LLM-ESR-style Qwen3-8B Emb. + LLMESR-SASRec
LLMEmb-style Qwen3-8B Emb. + SASRec
RLMRec-style Qwen3-8B GraphCL
IRLLRec-style Qwen3-8B IntentRep
SETRec-style Qwen3-8B Identifier
```

This aligns the large-scale paper-project block with the earlier six-paper
external baseline set while keeping four classical rows for context. Report
Beauty separately as supplementary smaller-N, not as a 10k domain.

Official-baseline interpretation:

```text
The large-scale `*_style_*` rows are still paper-style adapted baselines. The
next official external-baseline tier should reuse the same candidate rows and
output schema, but train/score through pinned official repositories with the
unified Qwen3-8B base model and each baseline's official adapter or
representation-learning algorithm retained.
```

Main script:

```text
scripts/run_week8_large_scale_10k_100neg.sh
```

Main plan:

```text
WEEK8_LARGE_SCALE_10K_100NEG_PLAN_2026-05-06.md
```

Expected final outputs:

```text
outputs/summary/external_only_baseline_comparison_week8_fourdomain_100neg_full_external.csv
outputs/summary/external_only_baseline_comparison_week8_fourdomain_100neg_full_external.md
outputs/summary/week8_fourdomain_100neg_full_external_external_only_phenomenon/
outputs/summary/week8_fourdomain_100neg_full_external_external_stat_tests/
```

Do not directly compare this 101-candidate table against Week7.7 six-candidate
direct/SRPD rows without explicitly stating the protocol difference.

## Official External Baseline Upgrade Status

Read this before editing external-baseline code:

```text
OFFICIAL_EXTERNAL_BASELINE_UPGRADE_PLAN_2026-05-07.md
configs/official_external_baselines.yaml
```

The YAML is the implementation contract and should not be edited casually. The
current target rows are:

| method | current row family | target official row |
| --- | --- | --- |
| LLM2Rec | `llm2rec_style_qwen3_sasrec` | `llm2rec_official_qwen3_lora_sasrec` |
| LLM-ESR | `llmesr_style_qwen3_sasrec` | `llmesr_official_qwen3_lora_sasrec` |
| LLMEmb | `llmemb_style_qwen3_sasrec` | `llmemb_official_qwen3_lora` |
| RLMRec | `rlmrec_style_qwen3_graphcl` | `rlmrec_official_qwen3_lora_graphcl` |
| IRLLRec | `irllrec_style_qwen3_intent` | `irllrec_official_qwen3_lora_intent` |
| SETRec | `setrec_style_qwen3_identifier` | `setrec_official_qwen3_lora_identifier` |

Official upgrade checklist:

```text
[ ] audit/pin official local repo checkout
[ ] identify official training/scoring entry points
[ ] preserve official algorithm, loss/objective, adapter or representation step,
    and scoring head
[ ] install the same-candidate task package in the official repo's native format
[ ] use unified Qwen3-8B base model for text/LLM representations
[ ] train or retain the official method-specific LoRA/adapter/representation
    artifact
[ ] select checkpoints on validation only
[ ] export exact source_event_id,user_id,item_id,score CSVs
[ ] import through main_import_same_candidate_baseline_scores.py
[ ] verify score coverage and paired-test inputs
[ ] write provenance with repo, commit, checkpoint, adapter, and score paths
```

Do LLM2Rec and LLM-ESR first because partial official adapter paths already
exist; then add LLMEmb, RLMRec, IRLLRec, and SETRec.

## Shadow Line Status And Future Large-Scale Tier

Do not forget why the project moved from the old `light` line to `shadow`.

The old `light` / verbalized-confidence line was not the same as LightGCN. It
was the earlier attempt to use directly verbalized LLM confidence/risk signals.
That line showed weak observations and collapse-like behavior: confidence was
not stable enough, calibration was not reliable enough, and the likely upside
looked limited. This motivated the move to the more task-grounded `shadow`
line.

The `shadow` line has not failed, but it is also not fully finished as a
paper-ready trained framework.

Completed or partially completed:

```text
shadow_v1 to shadow_v5: prompt-only task-grounded uncertainty signal candidates
small-prior screening: Beauty/Books
noisy robustness: top candidates on Beauty/Books
full-replay winner signal: shadow_v1 on Beauty/Books/Electronics/Movies
shadow_v6: diagnostic bridge from shadow_v1 signal to ranking decisions
```

Important previous shadow diagnostics:

| domain | direct anchor NDCG@10 | shadow_v6 NDCG@10 | delta |
| --- | ---: | ---: | ---: |
| beauty | 0.6353658183 | 0.6353973143 | +0.0000314960 |
| books | 0.6365719712 | 0.6568907475 | +0.0203187763 |
| electronics | 0.6574286221 | 0.6631285871 | +0.0056999649 |
| movies | 0.5723849700 | 0.5733843398 | +0.0009993699 |

What was still missing:

```text
validation-selected v6 gate / threshold sweep
accept-revise-fallback controller
generated-title verification path
chosen/rejected or pair-weight training data construction
Signal LoRA
Decision / Generative LoRA
paper-ready promotion against SRPD, structured-risk, and external baselines
```

Future decision:

```text
After the current external large-scale 10k/100neg protocol finishes and looks
healthy, consider a second large-scale shadow tier under the same protocol.
The goal is not to rerun shadow casually, but to make shadow comparable in the
new 101-candidate setting and test whether the v1/v6 phenomenon scales beyond
the old Week7.9 small/full-replay diagnostic setting.
```

Recommended future shadow large-scale design:

```text
1. Reuse the already built large10000_100neg task packages.
2. Run shadow_v1 signal inference on the same selected test events/candidates.
3. Calibrate on the corresponding valid split.
4. Apply a validation-selected shadow_v6 bridge/gate to test.
5. Report NDCG@10/MRR, calibration metrics, noisy or perturbation robustness if
   available, intervention rate, fallback rate, and paired tests.
6. Only then decide whether shadow becomes a paper-facing method or remains an
   explanatory/diagnostic tier.
```

Safe wording:

```text
Shadow is a planned large-scale extension after the external 10k/100neg
protocol is validated. The existing v6 result is a positive diagnostic bridge,
not yet a completed trained large-scale method.
```

## Live Server Run Status At Last User Check

The user started the full server command with logging:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
git pull --ff-only

mkdir -p outputs/summary/logs
log=outputs/summary/logs/week8_large10000_100neg_$(date +%F_%H%M%S).log

bash scripts/run_week8_large_scale_10k_100neg.sh 2>&1 | tee "$log"
```

Observed log file:

```text
outputs/summary/logs/week8_large10000_100neg_2026-05-06_165423.log
```

At the last check, the run was active:

```text
bash scripts/run_week8_large_scale_10k_100neg.sh
python main_build_large_scale_same_candidate_runtime.py \
  --processed_dir /home/ajifang/projects/uncertainty-llm4rec/data/processed/amazon_books \
  --domain books \
  --dataset_name amazon_books \
  --output_root outputs \
  --exp_prefix books_large10000_100neg \
  --user_limit 10000 \
  --num_negatives 100 \
  --max_history_len 50 \
  --min_sequence_length 3 \
  --seed 20260506 \
  --shuffle_seed 42 \
  --splits valid,test \
  --selection_strategy random \
  --negative_sampling popularity \
  --test_history_mode train_plus_valid
```

The process was using about 99 percent CPU and no GPU, which is normal for this
runtime-building stage.

New files observed within the active run:

```text
outputs/baselines/external_tasks/books_large10000_100neg_valid_same_candidate/
outputs/baselines/external_tasks/books_large10000_100neg_test_same_candidate/
outputs/summary/books_large10000_100neg_runtime_summary.json
```

This means the books runtime package had started writing successfully. At the
last recorded user check, the process still showed `--domain books`, so the next
Codex should first verify whether it has moved to electronics, embeddings, or
training.

## Immediate Monitoring Commands For Next Codex/User

Check whether the full script is still running:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
ps -ef | grep -E "run_week8_large_scale|main_build_large_scale|main_generate_llmesr|main_train_|main_import|main_run_week8" | grep -v grep
```

Check log tail:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
latest=$(ls -t outputs/summary/logs/week8_large10000_100neg_*.log | head -1)
echo "$latest"
tail -n 120 "$latest"
```

Follow live log:

```bash
tail -f "$latest"
```

Check recently modified files:

```bash
find outputs -type f -mmin -30 | sort | tail -80
```

Check GPU stage:

```bash
nvidia-smi
```

Expected stage order:

```text
Build books large-scale runtime
Build electronics large-scale runtime
Build movies large-scale runtime
Export adapters + embeddings for books
Classical baselines for books
Qwen3 paper-project style baselines for books
Repeat for electronics and movies
Build external-only large-scale comparison
Run external-only phenomenon diagnostics
Run external paired stat tests
Large-scale 10k/100neg run complete
```

The first build stages may not use GPU and may be quiet between banner lines.

## If The Run Finished

Check final artifacts:

```bash
ls -lh outputs/summary/external_only_baseline_comparison_week8_large10000_100neg.md
ls -lh outputs/summary/week8_large10000_100neg_external_only_phenomenon/
ls -lh outputs/summary/week8_large10000_100neg_external_stat_tests/
```

Print the main table and diagnostics:

```bash
cat outputs/summary/external_only_baseline_comparison_week8_large10000_100neg.md
cat outputs/summary/week8_large10000_100neg_external_only_phenomenon/external_only_oracle_summary.csv
cat outputs/summary/week8_large10000_100neg_external_only_phenomenon/external_only_base_rank_bins.csv
cat outputs/summary/week8_large10000_100neg_external_only_phenomenon/external_only_disagreement_bins.csv
cat outputs/summary/week8_large10000_100neg_external_stat_tests/all_domains_significance_tests.csv
```

Check completed per-baseline metrics:

```bash
find outputs -path "*large10000_100neg*/tables/ranking_metrics.csv" -print
```

## If The Run Died

First inspect the latest log:

```bash
latest=$(ls -t outputs/summary/logs/week8_large10000_100neg_*.log | head -1)
tail -n 200 "$latest"
```

Likely things to check:

- missing Qwen3 model path: `/home/ajifang/models/Qwen/Qwen3-8B`
- memory/OOM during embedding generation,
- a failed import/audit because score coverage is not 1.0,
- missing upstream LLM-ESR repo only if `RUN_LLMESR_STYLE=1` was enabled.

Do not restart blindly if partial outputs exist. First identify the failed
stage from the log, then rerun either the whole script or the specific command
for the failed domain/method.

## Paper-Safe Contribution Framing

Use:

```text
We evaluate all baselines through the same candidate rows and the same import,
audit, metric, and paired-test path. The Week7.7 six-candidate block shows broad
baseline coverage and event-level complementarity. The Week8 large-scale
10k/100neg protocol is a separate robustness gate using a harder 101-candidate
sampled-ranking setting on Books, Electronics, and Movies.
```

Avoid:

```text
The 10k/100neg table proves the Week7.7 direct/SRPD method beats every external
baseline.
```

Reason:

- Week7.7 direct/SRPD and Week8 large-scale external-only use different event
  files and candidate set sizes.
- Large-scale direct/SRPD 101-candidate LLM inference is not included in the
  default script.

## Suggested First Prompt In New Codex Chat

```text
Please first read CODEX_HANDOFF_WEEK8_2026-05-06.md and
WEEK8_LARGE_SCALE_10K_100NEG_PLAN_2026-05-06.md. Then check the server-side
Week8 large10000/100neg run progress. If it has finished, summarize the
conclusions. If it failed, use the latest log to locate the failed stage.
```

Also read these if continuing the future framework line:

```text
WEEK8_FUTURE_FRAMEWORK_ROADMAP_2026-05-06.md
PROJECT_LINEAGE_AND_FILE_INDEX_2026-05-06.md
```
