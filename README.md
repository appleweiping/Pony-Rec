# Task-Grounded Uncertainty for LLM-based Recommendation

This repository studies whether uncertainty from large language models can be
made reliable enough to support recommendation decisions.

The project is intentionally scoped as a controlled candidate-ranking and
reranking study. It asks a focused question:

> Can LLM recommendation confidence be upgraded from unreliable verbalized
> self-reporting into task-grounded calibrated uncertainty, and can that signal
> improve ranking reliability, coverage, and exposure behavior under a
> same-schema evaluation protocol?

The repository is not positioned as a full-catalog recommender SOTA claim, a
generative-title recommender system, or a LoRA distillation paper unless those
separate protocols are explicitly completed and labeled as such.

## Milestone Spine

The project is organized as a milestone-preserving research program rather than
a flat experiment log:

```text
Week1-4 / pony12 observation
-> Pony framework
-> Light series boundary test
-> Shadow series task-grounded uncertainty
-> same-candidate baseline system
-> small-domain to four-domain validation
-> complete recommendation-system roadmap
```

Canonical navigation starts here:

- [docs/milestones/README.md](docs/milestones/README.md) defines the M0-M6
  milestone map, evidence levels, and agent roles.
- [docs/top_conference_review_gate.md](docs/top_conference_review_gate.md)
  records the standing top-conference reviewer gate.
- [docs/server_runbook.md](docs/server_runbook.md) is the stable server-side
  execution entry point.

Legacy Week8 logs and plans remain in the tree for history, but they are not
the preferred starting point anymore. Use the canonical docs above first.

The current defended spine is:

```text
observation -> framework -> controlled same-candidate evidence
```

The complete recommendation-system layer is a staged roadmap until official
baselines, Shadow large-scale diagnostics, LoRA modules, and generated-title
verification are completed under the same protocol.

## Core Claim

Primary claim:

```text
Task-grounded calibrated uncertainty improves controlled candidate ranking /
reranking reliability under same-schema evaluation.
```

The main evidence chain is:

```text
Diagnosis -> Calibration -> Task-Grounded Uncertainty -> Decision-Time Reranking -> Protocol Audit
```

The paper-facing contribution is organized as:

1. Diagnosis: LLM verbalized confidence is informative in recommendation, but
   unreliable under miscalibration, confidence collapse, and domain-dependent
   failure.
2. Method: C-CRP, a calibrated candidate relevance posterior with uncertainty
   decomposed into boundary ambiguity, calibration gap, and evidence
   insufficiency.
3. Decision: risk-adjusted candidate ranking/reranking evaluated with utility,
   calibration, coverage, exposure, robustness, and paired statistical tests.

For the frozen claim and status rules, see
[docs/paper_claims_and_status.md](docs/paper_claims_and_status.md).

## What Is Implemented

The repository includes:

- Sequential recommendation data preprocessing for Amazon-style domains.
- Pointwise, pairwise, and candidate-ranking sample construction.
- LLM inference backends for API and local model experiments.
- Confidence parsing and normalization.
- Calibration diagnostics with ECE, Brier score, AUROC, reliability bins, and
  bootstrap confidence intervals.
- Strict validation-to-test calibration with split metadata, overlap checks,
  split hashes, and source file hashes.
- Candidate ranking/reranking evaluation with HR@K, NDCG@K, MRR, coverage,
  head exposure, long-tail coverage, parse success, and out-of-candidate rate.
- Candidate protocol audit for candidate set size, positive count, negative
  sampling, popularity distribution, title duplicates, split overlap, and
  HR/Recall equivalence.
- Baseline reliability proxy audit to prevent score-derived priors from being
  mislabeled as confidence.
- Paired bootstrap and permutation tests with Holm correction.
- C-CRP shadow method implementation and ablations.
- Generative-title bridge status tracking, explicitly outside the primary
  claim until fully completed.
- Official external-baseline upgrade contract for adapting pinned upstream
  LLM-rec baselines under the unchanged same-candidate protocol.
- Artifact documentation, smoke tests, and reproduction scripts.

## Project Tree

```text
.
|-- README.md
|-- requirements.txt
|-- environment.yml
|-- results_manifest.yaml
|-- OFFICIAL_EXTERNAL_BASELINE_UPGRADE_PLAN_2026-05-07.md
|-- configs/
|   |-- baseline/                     # literature baseline configs
|   |-- baseline_reliability/         # reliability proxy manifest
|   |-- batch/                        # staged experiment batches
|   |-- data/                         # domain preprocessing/sample configs
|   |-- exp/                          # experiment configs
|   |-- lora/                         # local LoRA training configs
|   |-- model/                        # API/local model configs
|   |-- official_external_baselines.yaml  # pinned official baseline contract
|   |-- shadow/                       # shadow/C-CRP runtime configs
|   |-- srpd/                         # SRPD training data configs
|   |-- task/                         # pointwise/pairwise/ranking task configs
|   `-- week8_large_scale_future_framework.yaml  # future shadow/light/LoRA scaffold config
|-- data/
|   `-- processed/                    # ignored by git; local processed data
|-- docs/
|   |-- paper_claims_and_status.md
|   |-- milestones/                   # canonical M0-M6 project map
|   |-- server_runbook.md             # server-side execution entry point
|   |-- top_conference_review_gate.md # reviewer/literature defense gate
|   |-- reproduction.md
|   |-- candidate_protocol.md
|   |-- baseline_protocol.md
|   |-- calibration_protocol.md
|   |-- shadow_method.md
|   |-- generative_title_bridge.md
|   |-- limitations.md
|   |-- experiments.md
|   `-- tables.md
|-- outputs/
|   |-- baselines/                    # ignored; external task packages/scores
|   `-- summary/                      # ignored except .gitkeep; generated tables
|-- prompts/                          # LLM prompt templates
|-- scripts/
|   |-- reproduce_smoke_test.sh
|   |-- reproduce_main_tables.sh
|   |-- run_week8_large_scale_10k_100neg.sh
|   |-- run_week8_shadow_large_scale_diagnostic.sh
|   |-- run_week8_light_large_scale_ablation.sh
|   |-- run_week8_generated_title_verification_scaffold.sh
|   `-- run_*.sh
|-- src/
|   |-- analysis/                     # aggregation, plotting, paper table export
|   |-- baselines/                    # literature/proxy baseline code
|   |-- data/                         # preprocessing and sample construction
|   |-- eval/                         # ranking, calibration, protocol, statistics
|   |-- llm/                          # model backends, prompts, parsers, inference
|   |-- methods/                      # rankers, rerankers, uncertainty aggregation
|   |-- shadow/                       # C-CRP and shadow signal code
|   |-- training/                     # LoRA/SRPD data and training utilities
|   |-- uncertainty/                  # confidence, calibration, reliability proxy
|   `-- utils/                        # IO, paths, registry, reproducibility
|-- tests/
|   |-- test_confidence_parser.py
|   |-- test_calibration_metrics.py
|   |-- test_ranking_metrics.py
|   |-- test_candidate_protocol.py
|   |-- test_shadow_parser.py
|   `-- test_statistical_tests.py
|-- main_preprocess.py
|-- main_build_samples.py
|-- main_infer.py
|-- main_eval.py
|-- main_calibrate.py
|-- main_rerank.py
|-- main_eval_rank.py
|-- main_audit_candidate_protocol.py
|-- main_audit_official_external_repos.py
|-- main_baseline_reliability_audit.py
|-- main_make_official_external_adapter_plan.py
|-- main_reuse_llmesr_embeddings_for_llm2rec.py
|-- main_make_official_external_adapter_plan.py
|-- main_export_llm2rec_same_candidate_task.py
|-- main_export_llmesr_same_candidate_task.py
|-- main_import_same_candidate_baseline_scores.py
|-- main_build_external_only_baseline_comparison.py
|-- main_stat_tests.py
|-- main_shadow_ccrp_eval.py
`-- main_generative_title_bridge_status.py
```

## Working File Map

The repository is intentionally organized around protocols rather than one
monolithic training entry point.

- `configs/official_external_baselines.yaml` is the machine-readable contract
  for pinned official external baselines. It defines the target baseline names,
  upstream repositories, pinned commits, shared Qwen3-8B base-model path,
  LoRA/adapter policy, and unchanged score schema.
- `main_audit_official_external_repos.py` and
  `main_make_official_external_adapter_plan.py` support the official-upgrade
  workflow before heavy training starts.
- `main_project_readiness_check.py` is the lightweight post-pull sanity check
  for the canonical milestone/reviewer/server-runbook layer.
- `main_export_*_same_candidate_task.py` scripts materialize unified
  same-candidate task packages in external-repository-friendly formats.
- `main_train_*_same_candidate.py` scripts train local classical or
  paper-style baselines under the shared candidate protocol.
- `main_import_same_candidate_baseline_scores.py` is the common ingestion
  gate for all baseline scores, including official external rows.
- `main_build_external_only_baseline_comparison.py`,
  `main_run_week8_external_only_phenomenon_diagnostics.py`, and
  `main_run_week8_external_paired_stat_tests.py` build paper-facing comparison,
  complementarity diagnostics, and paired tests from imported scores.
- `PROJECT_LINEAGE_AND_FILE_INDEX_2026-05-06.md` is the human navigation map
  for active, future, and historical files. Prefer updating the index before
  moving root scripts or markdown notes.
- `docs/milestones/README.md`, `docs/server_runbook.md`, and
  `docs/top_conference_review_gate.md` are the canonical first-read files for
  future Codex/server work.

## Method Overview

### 1. Verbalized confidence diagnosis

LLM outputs are parsed into recommendation decisions and confidence values.
The diagnostic layer asks whether confidence tracks correctness, whether it is
calibrated, and whether it interacts with popularity or exposure behavior.

Representative metrics:

- accuracy
- average confidence
- ECE and MCE
- Brier score
- AUROC
- wrong-high-confidence fraction
- reliability bins

### 2. Leakage-aware calibration

Calibration is fit on validation predictions and applied once to test
predictions.

The calibration entry point enforces strict checks by default:

- `--strict_split_check true`
- `--allow_user_overlap false`
- `--allow_item_overlap true`

User overlap between validation and test raises an error unless explicitly
allowed for a non-main diagnostic audit. The split metadata records sample
counts, user/item counts, overlap rates, split hash, source file SHA256 hashes,
calibration method, and bin counts.

See [docs/calibration_protocol.md](docs/calibration_protocol.md).

### 3. C-CRP: Calibrated Candidate Relevance Posterior

C-CRP is the main task-grounded uncertainty method. It uses a calibrated
candidate relevance posterior and defines uncertainty as:

```text
U = alpha * U_boundary
  + beta  * U_calibration_gap
  + gamma * U_evidence
```

where:

```text
U_boundary        = 4 * p_cal * (1 - p_cal)
U_calibration_gap = abs(p_raw - p_cal)
U_evidence        = 1 - clamp(evidence_support - counterevidence_strength, 0, 1)
```

The risk-adjusted score is:

```text
score = p_cal * (1 - U)^eta
```

Default fixed weights are `alpha=0.5`, `beta=0.3`, `gamma=0.2`. Any learned or
selected weights must be chosen on validation only.

See [docs/shadow_method.md](docs/shadow_method.md).

### 4. Decision-time reranking

Calibrated uncertainty is used at decision time to adjust candidate scores.
The evaluation reports both utility and distributional behavior:

- HR@K
- NDCG@K
- MRR
- coverage@K
- head exposure ratio
- long-tail coverage
- parse success
- out-of-candidate rate

Small differences are not called winners without paired statistical support.

## Baseline Policy

Baselines are divided into four groups:

1. Non-LLM recommenders: SASRec, BERT4Rec, GRU4Rec, LightGCN where available.
2. Simple recommendation priors: popularity, recency, history overlap, BM25 or
   title embedding.
3. LLM direct ranking: same candidate set and prompt schema, no uncertainty.
4. Uncertainty baselines: raw confidence, Platt/isotonic calibrated confidence,
   self-consistency, entropy or logprob when available.

Internal SRPD and shadow variants are ablations, not substitutes for external
baselines.

### Official external-baseline standard

The final external LLM-rec baseline standard is stricter than the current
paper-style adapted rows:

```text
pinned official or official-code-level implementation
+ official algorithm, loss, scoring head, and train/select procedure preserved
+ unified Qwen3-8B base model for LLM/text representations
+ LoRA/adapter trained and retained according to that baseline's official
  algorithm when the official method uses one
+ unchanged same-candidate train/valid/test rows
+ unchanged score schema: source_event_id,user_id,item_id,score
+ unified importer, metrics, coverage audit, and paired tests
```

Full-catalog metrics reported by an upstream repository are related-work or
sanity-check context only. Main same-candidate tables must use exact candidate
scores imported through `main_import_same_candidate_baseline_scores.py`.

The current `*_style_*` rows remain useful paper-style same-candidate
adaptations, but they should not be called official reproductions until the
pinned official baseline checklist is complete. See
[OFFICIAL_EXTERNAL_BASELINE_UPGRADE_PLAN_2026-05-07.md](OFFICIAL_EXTERNAL_BASELINE_UPGRADE_PLAN_2026-05-07.md).

Score-derived quantities are audited as reliability proxies, not automatically
treated as confidence. ECE and Brier are valid only for relevance-calibratable
signals such as self-reported confidence or calibrated relevance probability.

See [docs/baseline_protocol.md](docs/baseline_protocol.md).

## Candidate Ranking Protocol

The primary evaluation is controlled candidate ranking/reranking unless a
full-catalog audit says otherwise.

Every main candidate-ranking claim should be backed by:

```text
outputs/summary/candidate_protocol_audit.csv
```

The audit reports:

- candidate set size mean/min/max
- positives per event
- negative sampling strategy
- hard-negative ratio
- popularity-bin distribution
- duplicate-title rate
- valid/test user overlap
- train/test item overlap
- one-positive setting
- whether HR@K and Recall@K are numerically equivalent
- whether full-catalog evaluation is available

If `full_catalog_eval_available_flag=false`, the paper must not claim
full-catalog recommender SOTA.

See [docs/candidate_protocol.md](docs/candidate_protocol.md).

## Statistical Testing

Use paired tests before making winner claims.

Implemented in [main_stat_tests.py](main_stat_tests.py):

- paired bootstrap confidence intervals over users/events
- paired permutation tests
- delta vs direct
- delta vs structured risk
- Holm-Bonferroni correction

Rows whose confidence interval crosses zero or whose corrected p-value is not
significant are labeled `observed_best`, not `winner`.

## Environment

Python 3.12 is recommended.

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Conda users can start from:

```bash
conda env create -f environment.yml
conda activate uncertainty-rec
```

API-backed inference requires the relevant model key, for example:

```powershell
$env:DEEPSEEK_API_KEY="your_api_key"
```

Model configs read key names from `configs/model/*.yaml`.

## Quickstart

The commands below show the standard Beauty pipeline. Replace config files and
experiment names for other domains.

### 1. Preprocess

```bash
python main_preprocess.py --config configs/data/amazon_beauty.yaml
```

### 2. Build samples

```bash
python main_build_samples.py --config configs/data/amazon_beauty.yaml
```

### 3. Run inference

Validation split:

```bash
python main_infer.py \
  --config configs/exp/beauty_deepseek.yaml \
  --input_path data/processed/amazon_beauty/valid.jsonl \
  --output_path outputs/beauty_deepseek/predictions/valid_raw.jsonl \
  --split_name valid \
  --overwrite
```

Test split:

```bash
python main_infer.py \
  --config configs/exp/beauty_deepseek.yaml \
  --input_path data/processed/amazon_beauty/test.jsonl \
  --output_path outputs/beauty_deepseek/predictions/test_raw.jsonl \
  --split_name test \
  --overwrite
```

### 4. Evaluate predictions

```bash
python main_eval.py \
  --exp_name beauty_deepseek \
  --input_path outputs/beauty_deepseek/predictions/test_raw.jsonl
```

### 5. Calibrate

```bash
python main_calibrate.py \
  --exp_name beauty_deepseek \
  --valid_path outputs/beauty_deepseek/predictions/valid_raw.jsonl \
  --test_path outputs/beauty_deepseek/predictions/test_raw.jsonl \
  --method isotonic \
  --strict_split_check true \
  --allow_user_overlap false
```

If validation has too few rows for isotonic calibration, the script falls back
to Platt scaling and records both requested and effective methods.

### 6. Rerank

```bash
python main_rerank.py \
  --exp_name beauty_deepseek \
  --input_path outputs/beauty_deepseek/calibrated/test_calibrated.jsonl \
  --lambda_penalty 0.5
```

### 7. Audit protocols

Candidate protocol:

```bash
python main_audit_candidate_protocol.py \
  --domain beauty \
  --data_dir data/processed/amazon_beauty \
  --negative_sampling_strategy sampled_candidate_one_positive \
  --output_path outputs/summary/candidate_protocol_audit.csv
```

Baseline reliability proxy:

```bash
python main_baseline_reliability_audit.py \
  --config configs/baseline_reliability/week7_9_manifest.yaml \
  --output_path outputs/summary/baseline_reliability_proxy_audit.csv
```

Generative-title extension status template:

```bash
python main_generative_title_bridge_status.py
```

## Reproduction

Smoke test:

```bash
bash scripts/reproduce_smoke_test.sh
```

Main summary regeneration from existing outputs:

```bash
bash scripts/reproduce_main_tables.sh
```

On Windows without Bash, run the Python commands listed in
[docs/reproduction.md](docs/reproduction.md).

## Next-Step Checklist

Immediate repository priorities:

- Keep `configs/official_external_baselines.yaml` fixed unless the official
  baseline contract itself changes.
- Audit local official checkouts against pinned commits before adapting code.
- For each official baseline, write a provenance record containing upstream
  URL, pinned commit, local checkout path, preserved official modules, protocol
  changes, Qwen3-8B base-model/LoRA source, checkpoint or adapter path, and
  score coverage.
- Upgrade LLM2Rec and LLM-ESR first because partial official adapter paths
  already exist.
- Add official adapters for LLMEmb, RLMRec, IRLLRec, and SETRec by preserving
  each method's algorithm and only replacing the LLM/text representation source
  with the shared Qwen3-8B base plus the method-appropriate LoRA/adapter path.
- Emit only `source_event_id,user_id,item_id,score` candidate-score CSVs for
  the main protocol, then import them through the shared importer.
- Rebuild comparison tables using `*_official_qwen3_lora_*` rows for official
  baseline claims; keep `*_style_*` rows labeled as paper-style supplementary
  checks.
- Run coverage audits and paired tests before making winner claims.

## Key Outputs

Experiment-local outputs under `outputs/{exp_name}/`:

```text
predictions/valid_raw.jsonl
predictions/test_raw.jsonl
calibrated/valid_calibrated.jsonl
calibrated/test_calibrated.jsonl
tables/diagnostic_metrics.csv
tables/calibration_comparison.csv
tables/calibration_metric_ci.csv
tables/calibration_split_metadata.csv
tables/reliability_before_calibration.csv
tables/reliability_after_calibration.csv
tables/rerank_results.csv
figures/reliability_before_after_calibration.png
```

Summary outputs under `outputs/summary/`:

```text
final_results.csv
weekly_summary.csv
model_results.csv
domain_model_summary.csv
estimator_results.csv
robustness_results.csv
candidate_protocol_audit.csv
baseline_reliability_proxy_audit.csv
significance_tests.csv
main_table_with_ci.csv
generative_title_bridge_status.csv
```

Generated data, predictions, checkpoints, and large experiment outputs are
ignored by git. The reproducibility contract is documented through configs,
scripts, manifests, and protocol files.

## Paper-Facing Status Labels

Paper-facing tables use:

- `completed_result`: completed under the stated protocol.
- `runnable_not_complete`: code exists but the result is not complete enough
  for a main table.
- `design_only`: documented method or plan, not fully run.
- `proxy_only`: useful for related-work positioning or fairness audit, not
  same-schema evidence.
- `future_extension`: intentionally outside the main claim.

Only `completed_result` rows are eligible for main result tables, and only when
the protocol audits are present.

## Limitations

Current limitations are part of the artifact, not footnotes:

- Candidate-ranking results are not full-catalog SOTA unless audited as such.
- Proxy related-work numbers cannot be mixed into same-schema main tables.
- User overlap in valid/test splits invalidates main calibration claims unless
  the protocol explicitly defines a temporal same-user setting and labels it.
- Small NDCG/MRR differences require paired confidence intervals.
- Generative title recommendation is an extension bridge until catalog
  grounding, hallucination, unsupported confidence, and semantic audits are
  completed.

See [docs/limitations.md](docs/limitations.md).

## License

This project is released under the terms of the [LICENSE](LICENSE).
