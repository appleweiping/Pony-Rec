# Codex Task: Top-Conference Uncertainty-Aware LLM4Rec Framework, Experiments, and Baselines

## 0. Context and Non-Negotiable Goal

Repository:

```text
https://github.com/appleweiping/Pony-Rec.git
```

Current server project path:

```text
/home/ajifang/projects/fresh/uncertainty-llm4rec
```

Current known milestones:

```text
fc374a2  Add clean processed-source reprocessing pipeline
485cee9  Clean legacy project surface before pilot
aeb810d  Add RecBole processed-source smoke baselines
```

Current working assumptions:

- The clean project entrypoint for processed data is `data/processed` in the fresh repo.
- `data/processed` contains four official domains only:
  - `amazon_beauty`
  - `amazon_books`
  - `amazon_electronics`
  - `amazon_movies`
- Each domain must contain exactly four processed source tables:
  - `interactions.csv`
  - `items.csv`
  - `popularity_stats.csv`
  - `users.csv`
- The current raw-data limitation is accepted for now: old processed source tables are usable for pilot and reprocessing, but old `train/valid/test`, `srpd`, old LoRA artifacts, old prediction files, small/noisy dirs, and old batch configs must not be used as paper evidence.
- DeepSeek tiny live test passed.
- DeepSeek small pilot passed.
- RecBole smoke baseline passed.
- LoRA debug has not yet been completed.

The research idea is not simply “add uncertainty to recommendation”. The goal is to build a top-conference-level framework around a specific research claim:

> LLM-based recommenders should be evaluated and optimized not only by whether they produce correct recommendations, but by whether they know when they are likely to be correct, and whether their confidence systematically reinforces popular-item exposure and echo-chamber dynamics.

The method must not be a shallow combination of existing components such as `LLM + calibration + reranking`. It must produce a coherent framework with multiple connected contributions:

1. **Observation:** measure confidence-correctness reliability in LLM4Rec under identical recommendation settings.
2. **Diagnosis:** identify confidence-popularity coupling, high-confidence errors, low-confidence tail recommendations, and echo-chamber risk.
3. **Method:** use calibrated uncertainty and exposure-risk to improve ranking and train a local LoRA recommender.
4. **Training:** prune, reweight, or curriculum-order samples using uncertainty in a way that is explicitly recommendation-specific.
5. **Evaluation:** compare with traditional, sequential, LLM4Rec, uncertainty, and debiasing baselines under fair and clearly documented protocols.

Do not write the paper before experiments. Build the framework, experiments, metrics, and ablations first. Paper claims must be grounded in results.

---

## 1. Positioning: What This Project Should Contribute

### 1.1 Working Title

Suggested project name:

```text
CARE-Rec: Confidence-Aware Reliable Exposure Recommendation with LLMs
```

Alternative names:

```text
UCal-Rec  = Uncertainty-Calibrated Recommendation
TRUST-Rec = Trustworthy Recommendation via Uncertainty and Stable Training
RISK-Rec  = Recommendation with Item-level and Sequence-level Knowledge of Uncertainty
```

Use one name consistently in code/docs once chosen. For now use `CARE-Rec`.

### 1.2 Central Hypothesis

LLM recommendation confidence is not merely a calibration problem. In recommendation, confidence interacts with item popularity, user-history concentration, candidate ambiguity, and exposure feedback. Therefore, high confidence can be helpful when it predicts correctness, but harmful when it amplifies head items or repeated categories without sufficient evidence.

### 1.3 Main Contributions

The final paper should aim for the following contributions.

#### Contribution 1: Recommendation-Specific Confidence Reliability Benchmark

Build a benchmark that measures whether LLM recommendation confidence predicts correctness across:

- API LLMs such as DeepSeek.
- Local LoRA LLM recommender.
- Classical/sequential baselines where confidence can be formulated from scores, margins, entropy, or calibrated probabilities.
- Multiple domains: Beauty, Books, Electronics, Movies.
- Multiple data protocols: current processed leave-one-out protocol; optional user-filter/k-core protocol if raw data becomes available.

This directly addresses the reviewer concern: “You observed uncertainty on LLMs only; does the same setting exist for baselines?”

#### Contribution 2: Confidence-Popularity-Echo Diagnostic

Define metrics that answer:

- Are high-confidence predictions more likely to be head items?
- Are tail items systematically lower-confidence even when correct?
- Are high-confidence wrong predictions disproportionately popular items?
- Does confidence-aware reranking reduce or increase head exposure?
- Does the method increase repeated category exposure relative to user history?

This is not a generic fairness metric. It is a recommendation-specific uncertainty diagnostic.

#### Contribution 3: CARE Inference Policy

Develop a coherent inference policy derived from expected utility:

```text
Expected utility = relevance gain
                 - correctness-risk penalty
                 - exposure-harm penalty
                 + calibrated reliability reward
```

This avoids appearing as arbitrary “A+B+C” reranking. The method uses uncertainty only through a decision-theoretic lens: if the model is uncertain, the expected utility of recommending an item changes; if the model is overconfident on popular items, exposure harm increases.

#### Contribution 4: CARE-LoRA Training

Train the local small model using uncertainty-aware training rather than naive SFT:

- Standard SFT baseline.
- Confidence-only weighted SFT baseline.
- Drop high-uncertainty samples baseline.
- CARE-LoRA: use calibrated correctness risk, popularity-conditioned confidence gap, and tail-aware curriculum.

The novelty is not just pruning noisy data. The novelty is a recommendation-specific data policy:

- High-confidence correct samples are reliable anchors.
- High-confidence wrong head-item samples are anti-reliability signals and should be penalized or used as hard negatives.
- Low-confidence correct tail samples are not simply noise; they may be valuable long-tail learning signals and should be curriculum-scheduled rather than always removed.

#### Contribution 5: Protocol and Reproducibility

Use processed source tables now, but regenerate all experimental splits, candidates, train-only popularity, manifests, and leakage reports. Every output must state:

```text
run_type: smoke / pilot / full
backend_type: api / baseline / lora
is_paper_result: true / false
source_protocol: processed_source_reprocess / raw_rebuild
split_protocol: per_user_leave_one_out / sliding_prefix / global_time
candidate_protocol: sampled_19 / sampled_99 / full_sort / recbole_internal
```

---

## 2. Related Work and Why This Is Not Just Stitching

### 2.1 What Existing Work Covers

Use these works to position, not to copy.

#### Uncertainty Quantification for LLM-Based Recommendation

`Uncertainty Quantification and Decomposition for LLM-based Recommendation` studies predictive uncertainty in LLM recommendation and decomposes uncertainty sources. It is a direct related work and should be treated as the strongest uncertainty-related baseline/positioning reference.

Our difference:

- We focus on confidence-correctness-popularity coupling and exposure/echo risk.
- We test whether confidence phenomena exist not only in API LLMs but also in trained baselines and local LoRA models.
- We use uncertainty for both inference policy and training sample policy.
- We explicitly model high-confidence head-item errors and low-confidence tail correctness.

#### Decoding Matters

`Decoding Matters` identifies amplification bias and homogeneity issues in LLM recommendation decoding.

Our difference:

- We do not only change decoding.
- We measure whether confidence itself is part of amplification bias.
- We use confidence/exposure risk to identify when standard decoding produces overconfident head-heavy predictions.
- If local logits are available, we can implement a decoding baseline; if not, keep it as related work or local-model-only baseline.

#### SLMRec

SLMRec distills LLM recommendation ability into a smaller sequential model.

Our difference:

- Our local model is not just distilled from a large model.
- We use uncertainty-calibrated and popularity-aware training signals.
- We compare standard LoRA/SFT, naive distillation, and CARE-LoRA.

#### LLM-ESR

LLM-ESR enhances long-tail sequential recommendation using semantic embeddings from LLMs without adding LLM inference overhead.

Our difference:

- We study confidence disparity between head and tail items.
- We use uncertainty to decide which long-tail samples are useful versus noisy.
- We evaluate exposure and confidence gap, not only long-tail accuracy.

#### CoLLM / CLLM4Rec / LLaRA / LLM2Rec / LLMEmb / OpenP5

These works connect collaborative signals, sequential recommenders, or text-to-text LLM objectives with recommendation.

Our difference:

- We do not just align LLMs to recommendation.
- We ask whether the aligned model’s confidence is reliable and whether training improves or worsens confidence-popularity coupling.
- Our training objective is explicitly uncertainty- and exposure-aware.

### 2.2 Uploaded recprefer.zip Baseline Pool

The uploaded `recprefer.zip` contains many candidate papers. The extracted titles include, among others:

NH-style papers/metrics pool:

- SLMRec: Distilling Large Language Models into Small for Sequential Recommendation.
- Decoding Matters: Addressing Amplification Bias and Homogeneity Issue for LLM-based Recommendation.
- AGRec: Adapting Autoregressive Decoders with Graph Reasoning for LLM-based Sequential Recommendation.
- CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems.
- LLM4RSR: Large Language Models as Data Correctors for Robust Sequential Recommendation.
- OpenP5.
- RecExplainer.
- SPRec.
- LLM-ESR.
- Lost in Sequence.
- LLMEmb.

NR-style papers/metrics pool:

- CoLLM / Collaborative Large Language Model for Recommender Systems.
- Representation Learning with Large Language Models for Recommendation.
- Data-efficient Fine-tuning for LLM-based Recommendation.
- LLM2Rec.
- Denoising Recommendation.
- Uncertainty Quantification and Decomposition for LLM-based Recommendation.
- Order-agnostic Identifier for LLM-based Generative Recommendation.
- Review-driven Personalized Preference Reasoning.

Do not implement all of these immediately. Use them to choose minimal, high-value baselines and to avoid reviewer criticism.

---

## 3. Core Framework: CARE-Rec

### 3.1 Notation

For a domain `d`, each user `u` has a history:

```text
H_u = [i_1, i_2, ..., i_t]
```

A candidate set:

```text
C_u = {y_u} union N_u
```

where `y_u` is the ground-truth next item and `N_u` are negatives excluding user history and target.

An LLM or recommender returns:

```text
ranking R_u = [i_(1), i_(2), ..., i_(m)]
confidence c_u or item-level confidence c_{u,i}
raw response text
parsed output
validity flags
```

Correctness indicators:

```text
z_u@K = 1 if y_u appears in top-K
z_{u,i} = 1 if i == y_u
```

### 3.2 Uncertainty Features

For each user-candidate pair or user-ranking output, compute:

#### LLM/API uncertainty features

- `verbal_confidence`: model’s explicit confidence from the response.
- `rank_confidence`: confidence assigned to top item or list.
- `self_consistency_win_rate`: fraction of repeated samples selecting the same top item.
- `self_consistency_entropy`: entropy over top-item votes.
- `rank_variance`: Kendall/Spearman instability across repeated sampled rankings.
- `prompt_perturbation_variance`: output change under prompt paraphrase, candidate order shuffle, or confidence wording change.
- `invalid_output_flag`: invalid JSON, missing rank, duplicated item, hallucinated item, no confidence, etc.

#### Local model / baseline uncertainty features

For SASRec/BERT4Rec/LightGCN/BPR/LoRA:

- `score_margin`: top1 score minus top2 score.
- `softmax_entropy`: entropy over candidate scores.
- `calibrated_score`: validation-calibrated probability of correctness.
- `rank_margin`: score gap between target rank and competitors when target known for analysis.
- `dropout_variance`: optional MC dropout variance for local models.

#### Recommendation-specific context features

- `item_popularity_train`: popularity from training split only.
- `popularity_bucket`: head / mid / tail.
- `history_length_bucket`: short / medium / long.
- `history_concentration`: concentration of user history categories/items.
- `candidate_popularity_skew`: candidate set head-heavy ratio.
- `category_repeat_rate`: whether recommended item category repeats user’s dominant history category.
- `domain`: Beauty / Books / Electronics / Movies.

### 3.3 Calibration Model

Fit calibration only on validation predictions.

Allowed calibrators:

- Platt/logistic scaling.
- Isotonic regression.
- Temperature scaling if logits/probabilities are valid.
- Group-wise calibration by domain, popularity bucket, and history length.

Target:

```text
r_hat(u,i) = P(correct | features)
```

For listwise prediction, define:

```text
r_hat_top(u) = P(top prediction correct | output features)
```

Never fit calibrators on test.

### 3.4 Exposure Harm Risk

Define a recommendation-specific risk score:

```text
EHR(u,i) = max(0, raw_confidence(u,i) - calibrated_correctness(u,i))
           * head_bucket(i)
           * exposure_repeat_factor(u,i)
```

Interpretation:

- If the model is high-confidence but calibration says correctness is lower, risk increases.
- If the item is popular/head, risk increases because exposure amplification matters more.
- If the item repeats dominant user category/history, echo-chamber risk increases.

This is the core difference from generic calibration. We do not only ask “is the confidence right?” We ask “what harm does overconfidence create in a recommendation environment?”

### 3.5 CARE Inference Score

Use expected utility:

```text
CAREScore(u,i) = Rel(u,i)
               + alpha * CalibratedReliability(u,i)
               - beta  * Uncertainty(u,i)
               - gamma * ExposureHarmRisk(u,i)
               + delta * DiversityOrTailUtility(u,i)
```

Do not present this as arbitrary weighted sum. Present it as expected utility under uncertainty:

- `Rel` estimates relevance.
- `CalibratedReliability` estimates whether the model can trust its own prediction.
- `Uncertainty` penalizes low reliability.
- `ExposureHarmRisk` penalizes overconfident head/repeat errors.
- `DiversityOrTailUtility` is optional and must be ablated.

Required variants:

```text
BaseRel                   = relevance only
BaseRel+Calibration        = add calibrated reliability
BaseRel+Uncertainty        = add uncertainty penalty
BaseRel+PopularityDebias   = add popularity penalty only
CAREScore                 = reliability + uncertainty + exposure risk
CAREScore-no-tail-utility  = ablation
CAREScore-no-exposure-risk = ablation
CAREScore-no-calibration   = ablation
```

### 3.6 CARE-LoRA Training

Build training data from reprocessed source outputs. Use ground truth, teacher predictions, confidence, and popularity context.

#### Sample categorization

For each sample:

```text
correct_high_conf     = teacher correct and calibrated confidence high
correct_low_conf_tail = teacher correct, confidence low, item tail/mid
wrong_high_conf_head  = teacher wrong, confidence high, predicted item head
wrong_low_conf        = teacher wrong and confidence low
invalid_output        = teacher invalid/hallucinated/duplicate
```

#### Training policy

- `correct_high_conf`: high-weight anchor.
- `correct_low_conf_tail`: not noise; curriculum sample. Train later or moderate weight.
- `wrong_high_conf_head`: use as anti-overconfidence negative or contrastive hard negative.
- `wrong_low_conf`: low weight or excluded.
- `invalid_output`: exclude from teacher-distillation signal; can still use ground-truth SFT.

#### Loss design

Minimal implementation:

```text
L_total = L_rank_sft
        + lambda_pair * L_pairwise_preference
        + lambda_u    * L_uncertainty_weighted
        + lambda_pop  * L_popularity_confidence_regularizer
```

Where:

- `L_rank_sft`: standard listwise ranking generation SFT.
- `L_pairwise_preference`: target item preferred over negatives.
- `L_uncertainty_weighted`: sample weights from calibrated reliability.
- `L_popularity_confidence_regularizer`: penalizes high-confidence head errors or head-overconfidence gap.

Do not overcomplicate the first implementation. Implement a working version and ablate components.

---

## 4. Data Protocols

### 4.1 Current Protocol: Processed Source Reprocess

Use this immediately.

Input:

```text
data/processed/<domain>/interactions.csv
data/processed/<domain>/items.csv
data/processed/<domain>/popularity_stats.csv
data/processed/<domain>/users.csv
```

Domains:

```text
amazon_beauty
amazon_books
amazon_electronics
amazon_movies
```

Reprocess outputs:

```text
outputs/reprocessed_processed_source/<domain>/train.jsonl
outputs/reprocessed_processed_source/<domain>/valid.jsonl
outputs/reprocessed_processed_source/<domain>/test.jsonl
outputs/reprocessed_processed_source/<domain>/manifest.json
outputs/reprocessed_processed_source/<domain>/leakage_report.json
```

Required rules:

- Use interactions as source, not old jsonl splits.
- Recompute splits.
- Recompute train-only popularity.
- Generate deterministic candidates.
- Negatives must exclude target and user history.
- Save manifests and leakage reports.

### 4.2 Primary Split Protocol

Per-user temporal leave-one-out:

```text
train = all interactions except last two
valid = second last interaction
test  = last interaction
```

This follows common sequential recommendation practice.

### 4.3 Alternative Sliding Prefix Protocol

Implement as an ablation, not immediate default:

For each user with sequence length `n`, construct samples for:

```text
k in [4, n]
train history = items before k-2
valid = k-1
test = k
```

Or more generally:

```text
history = [i_1, ..., i_{k-2}]
validation target = i_{k-1}
test target = i_k
```

This produces multiple samples per user and can help data efficiency. It must be clearly marked as a different protocol.

### 4.4 k-core / User Filtering Protocol

If raw data becomes available, implement:

```text
Protocol A: user filter only, keep users with interactions > 3
Protocol B: iterative user-item k-core, k = 5 default
Protocol C: k-core sensitivity, k in {3,5,10}
```

For current processed source, audit whether it already satisfies 3/5/10-core; do not claim raw k-core unless raw rebuild exists.

### 4.5 Candidate Protocols

Use these levels:

```text
C19  = target + 18 negatives, pilot only
C99  = target + 98 negatives, main sampled setting
C199 = robustness
FullSort = if baseline supports full ranking
```

For LLM listwise ranking, `C99` may be expensive; use C19/C99 carefully.

---

## 5. Experimental Questions

### RQ1: Does LLM Confidence Predict Correctness?

Run on DeepSeek API and local LoRA.

Metrics:

- confidence-correctness correlation.
- AUROC/AUPRC for predicting correctness.
- ECE, MCE, Brier.
- confidence distributions for correct vs wrong predictions.
- invalid-output rate.

Report by:

- domain.
- valid/test split.
- popularity bucket.
- history length bucket.
- candidate size.

### RQ2: Is Confidence Popularity-Biased?

Metrics:

- Pearson/Spearman correlation between confidence and item popularity.
- high-confidence head error rate.
- tail correct but low-confidence rate.
- confidence gap: head vs mid vs tail.
- confidence gap under correct-only predictions.
- confidence gap under wrong-only predictions.

### RQ3: Does Confidence Cause Echo-Chamber Risk?

Define echo metrics:

- head exposure share.
- long-tail coverage.
- category repetition rate.
- user dominant-category reinforcement.
- HHI/Gini of exposure distribution.
- uplift in head exposure after confidence reranking.

Important comparison:

```text
base relevance ranking
confidence-only reranking
CARE reranking
popularity-only debiasing
```

### RQ4: Does CARE Improve Recommendation Quality and Reliability?

Compare:

- DeepSeek listwise baseline.
- DeepSeek + verbal confidence rerank.
- DeepSeek + calibration-only.
- DeepSeek + uncertainty-only rerank.
- DeepSeek + popularity debiasing.
- CARE rerank.

Metrics:

- HR@5/10/20.
- NDCG@5/10/20.
- Recall@5/10/20.
- MRR@10.
- ECE/Brier.
- risk-coverage curve.
- long-tail coverage.
- head exposure share.

### RQ5: Does CARE-LoRA Improve Local Model Training?

Compare:

- Qwen3-8B zero-shot / no adapter.
- Standard LoRA SFT.
- Teacher distillation LoRA.
- Confidence-weighted LoRA.
- Uncertainty-pruned LoRA.
- CARE-LoRA.

Measure:

- ranking quality.
- calibration reliability.
- invalid output rate.
- head/tail confidence gap.
- training efficiency.
- GPU memory.

### RQ6: Do Baselines Show Similar Confidence Phenomena?

For non-LLM baselines formulate confidence as:

```text
softmax over candidate scores
score margin
entropy
calibrated probability
```

Check if the same patterns hold:

- confidence-correctness correlation.
- high-confidence wrong head items.
- low-confidence tail correctness.
- popularity-confidence correlation.

This directly addresses the senior’s suggestion and reviewer risk.

---

## 6. Baseline Plan

### 6.1 Classical / Sequential Baselines

Use RecBole or existing smoke-compatible paths first.

Minimal baselines:

```text
Pop
BPR-MF
LightGCN
GRU4Rec
SASRec
BERT4Rec
```

Add if feasible:

```text
FMLP-Rec
CL4SRec / DuoRec / CoSeRec
```

For each baseline compute:

- ranking metrics.
- confidence features from scores.
- calibrated correctness.
- popularity-confidence diagnostics.

### 6.2 API LLM Baselines

```text
DeepSeek zero-shot listwise
DeepSeek few-shot listwise
DeepSeek pointwise yes/no scoring
DeepSeek pairwise comparison
DeepSeek self-consistency
DeepSeek confidence-prompting
DeepSeek calibration-only
DeepSeek uncertainty-only rerank
DeepSeek CARE rerank
```

Important:

- Use identical candidate sets.
- Use identical prompts except intended differences.
- Record invalid output flags.
- Save raw responses.
- Do not compare DeepSeek listwise candidate metrics directly to RecBole full-sort without clearly noting protocol differences.

### 6.3 Local LLM / LoRA Baselines

```text
Qwen3-8B no adapter
Qwen3-8B standard LoRA SFT
Qwen3-8B confidence-weighted LoRA
Qwen3-8B uncertainty-pruned LoRA
Qwen3-8B CARE-LoRA
```

Optional:

```text
Qwen3-8B DPO/ORPO preference optimization if stable
```

### 6.4 Literature Baselines from recprefer.zip

Implement only if feasible, label exact vs approximate.

Priority group:

```text
UNC_LLM_REC-style uncertainty decomposition
Decoding Matters-style decoding/logit baseline, local model only if logits available
SLMRec-style small-model distillation
LLM-ESR-style semantic long-tail enhancement
CoLLM/CLLM4Rec-style collaborative-token or embedding baseline
LLM2Rec/LLMEmb-style embedding generator baseline
OpenP5/P5 text-to-text baseline
LLM4RSR/Denoising-style data correction baseline
```

Rules:

- If official code is used, document commit and modifications.
- If approximated, name it `Approx-*` and state differences.
- Do not claim exact reproduction unless exact.

---

## 7. Metrics

### 7.1 Ranking Metrics

Report both NH and NR conventions:

NH-style:

```text
NDCG@5/10/20
HitRatio@5/10/20
```

NR-style:

```text
NDCG@5/10/20
Recall@5/10/20
```

Also report:

```text
MRR@10
InvalidRate
ValidItemRate
```

### 7.2 Calibration and Uncertainty Metrics

```text
ECE
MCE
Brier Score
NLL, if probabilities are valid
AUROC for correctness prediction
AUPRC for correctness prediction
Risk-Coverage curve
AURC
Confidence-Correctness correlation
Correct-vs-wrong confidence gap
```

### 7.3 Popularity / Exposure / Echo Metrics

```text
Head/Mid/Tail exposure share
Long-tail coverage
Average popularity of recommended items
Confidence-popularity correlation
High-confidence head error rate
Low-confidence tail-correct rate
Category repetition rate
Dominant-category reinforcement
Exposure Gini or HHI
Head-exposure uplift after reranking
```

### 7.4 Efficiency Metrics

```text
API latency
API tokens
API cost estimate
GPU memory
LoRA train time
LoRA inference latency
RecBole train/eval time
```

---

## 8. Implementation Plan by Files

### 8.1 New/Updated Docs

```text
docs/RESEARCH_FRAMEWORK_CARE_REC.md
docs/EXPERIMENT_MATRIX.md
docs/BASELINE_MATRIX.md
docs/UNCERTAINTY_METRICS.md
docs/ECHO_CHAMBER_METRICS.md
docs/CARE_LORA_TRAINING.md
docs/PAPER_CLAIM_GATES.md
```

### 8.2 Data and Reprocess

Existing:

```text
src/data/processed_loader.py
src/cli/reprocess_processed_source.py
```

Add/update:

```text
src/data/split_protocols.py
src/data/candidate_protocols.py
src/data/popularity.py
src/data/manifests.py
src/cli/run_protocol_ablation.py
```

### 8.3 Uncertainty Module

Add:

```text
src/uncertainty/features.py
src/uncertainty/estimators.py
src/uncertainty/calibration.py
src/uncertainty/group_calibration.py
src/uncertainty/risk_coverage.py
src/uncertainty/baseline_confidence.py
```

Expected functions/classes:

```python
class UncertaintyFeatures:
    verbal_confidence: float | None
    self_consistency_entropy: float | None
    score_margin: float | None
    popularity_bucket: str
    history_length_bucket: str
    invalid_flags: dict

class CalibrationModel:
    def fit_valid(self, rows): ...
    def predict_test(self, rows): ...

class BaselineConfidenceAdapter:
    def from_scores(self, candidate_scores): ...
```

### 8.4 Exposure/Echo Module

Add:

```text
src/exposure/popularity_bias.py
src/exposure/echo_metrics.py
src/exposure/exposure_harm.py
```

Expected functions:

```python
def exposure_harm_risk(raw_confidence, calibrated_correctness, popularity_bucket, repeat_factor): ...
def head_exposure_share(predictions): ...
def long_tail_coverage(predictions): ...
def category_reinforcement(user_history, recommended_items, item_categories): ...
```

### 8.5 CARE Methods

Add:

```text
src/methods/care_score.py
src/methods/care_rerank.py
src/methods/care_lora_data.py
src/methods/care_lora_loss.py
```

CLI:

```text
src/cli/run_uncertainty_probe.py
src/cli/run_calibrate_uncertainty.py
src/cli/run_care_rerank.py
src/cli/build_care_lora_data.py
src/cli/train_care_lora.py
src/cli/eval_care_lora.py
```

### 8.6 Baseline Integration

Existing:

```text
src/baselines/recbole_adapter.py
src/cli/run_recbole_smoke_reprocessed.py
```

Add/update:

```text
src/baselines/confidence_from_scores.py
src/cli/run_recbole_confidence_probe.py
src/cli/export_baseline_uncertainty_features.py
configs/baselines/*.yaml
```

### 8.7 Pilot/Experiment CLIs

Existing:

```text
src/cli/run_pilot_reprocessed_deepseek.py
```

Add:

```text
src/cli/run_deepseek_100u_pilot.py
src/cli/run_care_rerank_pilot.py
src/cli/run_lora_debug_reprocessed.py
src/cli/run_ablation_matrix.py
src/cli/export_reviewer_tables.py
```

### 8.8 Tests

Add tests:

```text
tests/test_uncertainty_features.py
tests/test_calibration_groupwise.py
tests/test_exposure_metrics.py
tests/test_care_score.py
tests/test_care_lora_data.py
tests/test_baseline_confidence.py
tests/test_protocol_ablation.py
```

Test requirements:

- No valid/test leakage.
- No candidate target/history leakage.
- Calibration fit only on valid.
- Test not used for threshold selection.
- High-confidence wrong head item gives higher exposure harm than low-confidence tail item.
- Baseline score-to-confidence outputs valid probabilities after calibration.
- CARE score ranks stable in deterministic fixtures.

---

## 9. Immediate Next Steps After Current State

Current status already achieved:

```text
DeepSeek small pilot: completed
RecBole smoke: completed
LoRA debug: next
```

### Step A: Finalize RecBole Smoke Commit

If not already committed, commit:

```text
Add RecBole processed-source smoke baselines
```

Do not commit outputs/data/logs.

### Step B: LoRA Debug

Run only Beauty, 20 users, candidate size 19, 5-20 optimizer steps.

Required output root:

```text
outputs/pilots/lora_qwen3_8b_processed_20u_c19_seed42_debug/
```

Use local model:

```text
/home/ajifang/models/Qwen/Qwen3-8B
```

Must prove:

- model loads.
- adapter saves.
- adapter reloads.
- predictions match API schema.
- metrics saved.
- invalid output rate measured.
- confidence unavailable marked explicitly if no confidence.
- GPU memory recorded.

### Step C: DeepSeek 100-user/domain Pilot

After LoRA debug, scale DeepSeek:

```text
max_users_per_domain = 100
candidate_size = 19 and 99
seed = 42
```

Do not full-scale until 100-user pilot passes.

### Step D: Baseline Confidence Probe

For RecBole models, compute candidate scores on the same candidate sets if possible. If RecBole full-sort protocol differs, clearly separate:

```text
RecBole internal evaluation
Candidate-set confidence probe
```

The confidence probe should use identical candidate sets to LLMs.

### Step E: CARE Rerank Pilot

Run CARE reranking on DeepSeek predictions:

```text
DeepSeek base
DeepSeek + verbal confidence
DeepSeek + calibrated confidence
DeepSeek + uncertainty penalty
DeepSeek + exposure risk
CARE full
```

### Step F: CARE-LoRA Pilot

Train:

```text
standard LoRA
uncertainty-pruned LoRA
confidence-weighted LoRA
CARE-LoRA
```

Start with one domain, then four-domain pilot.

---

## 10. Reviewer-Facing Experiment Matrix

### 10.1 Main Table

Rows:

```text
Pop
BPR-MF
LightGCN
GRU4Rec
SASRec
BERT4Rec
DeepSeek Listwise
DeepSeek + SelfConsistency
DeepSeek + Calibration
DeepSeek + Uncertainty Rerank
DeepSeek + CARE
Qwen3-8B LoRA-SFT
Qwen3-8B Uncertainty-Pruned LoRA
Qwen3-8B CARE-LoRA
```

Columns per domain:

```text
Recall@10
NDCG@10
Hit@10
MRR@10
ECE
Brier
HeadExposure
TailCoverage
InvalidRate
```

### 10.2 Confidence Reliability Table

Rows:

```text
DeepSeek
Qwen3-8B LoRA
SASRec
BERT4Rec
LightGCN
BPR
```

Columns:

```text
Confidence-Correctness Corr
AUROC correctness
ECE
Brier
Head confidence gap
Tail low-confidence correct rate
```

### 10.3 Echo Chamber Table

Rows:

```text
Base relevance
Confidence-only rerank
Popularity-only debias
CARE
```

Columns:

```text
Head exposure share
Head exposure uplift
Category repetition rate
Long-tail coverage
NDCG@10
Recall@10
```

### 10.4 LoRA Training Table

Rows:

```text
No adapter
Standard SFT
Teacher distillation
Drop high-uncertainty
Confidence weighted
CARE-LoRA
```

Columns:

```text
Recall@10
NDCG@10
ECE
High-conf wrong rate
Tail coverage
Invalid rate
Train time
GPU memory
```

---

## 11. Ablations

Required:

```text
CARE full
w/o calibration
w/o exposure risk
w/o popularity bucket
w/o self-consistency
w/o confidence prompt
w/o uncertainty training weight
w/o tail curriculum
only prune high uncertainty
only confidence weight
only popularity debias
```

Data protocol ablations:

```text
LOO split
sliding-prefix split
candidate_size 19 / 99 / 199
max_users 20 / 100 / 500 / full
history length 10 / 20 / 50
```

Model ablations:

```text
DeepSeek v4 flash vs pro if cost allows
Qwen LoRA rank 8 / 16 / 32
LoRA target modules q_proj/v_proj vs full attention projections
SFT vs pairwise vs listwise
```

---

## 12. Paper Claim Gates

Do not claim final paper result unless:

- `run_type=full`.
- `is_paper_result=true`.
- manifest exists.
- calibration fit only on valid.
- test never used for tuning.
- outputs not from mock.
- processed source protocol described.
- if raw unavailable, limitation stated.
- baseline protocol difference stated.
- at least three seeds or confidence intervals for final claims if feasible.

Pilot results can support engineering readiness, not final claims.

---

## 13. Codex Instructions for Next Implementation Round

Copy this section to Codex once ready.

```text
We are building a top-conference-level framework named CARE-Rec: Confidence-Aware Reliable Exposure Recommendation with LLMs.

Do not treat this as LLM + calibration + reranking. The coherent research claim is that LLM recommendation confidence must be calibrated and interpreted under recommendation-specific exposure risk, because high-confidence errors may amplify popular items and echo chambers, while low-confidence tail correctness may be useful training signal.

Current repo:
/home/ajifang/projects/fresh/uncertainty-llm4rec

Current accepted commits:
fc374a2 Add clean processed-source reprocessing pipeline
485cee9 Clean legacy project surface before pilot
aeb810d Add RecBole processed-source smoke baselines

Current data:
data/processed is clean and contains only four official domains x four CSV source tables.
Reprocessed output exists under outputs/reprocessed_processed_source.
DeepSeek small pilot and RecBole smoke are completed.

Next task:
Implement CARE-Rec framework modules and LoRA debug in staged order.

Do not run full experiments.
Do not use raw.
Do not use srpd.
Do not use old split/prediction/LoRA artifacts.
Do not use legacy/root_main or old batch configs.

Stage 1: LoRA debug
- implement/verify Qwen3-8B LoRA debug on amazon_beauty only
- 20 users, candidate_size 19, seed 42
- 5-20 optimizer steps only
- save/reload adapter
- output predictions in API-compatible schema
- save metrics and docs/PILOT_LORA_QWEN3_DEBUG.md
- run pytest
- commit

Stage 2: uncertainty feature extraction
- implement src/uncertainty/features.py and estimators.py
- read DeepSeek pilot outputs and RecBole smoke/candidate scores if available
- compute verbal confidence, invalid flags, self-consistency placeholders, score margins if available, popularity buckets, history length buckets
- write docs/UNCERTAINTY_FEATURES.md
- tests
- commit

Stage 3: calibration and diagnostics
- implement group calibration by domain/popularity/history length
- compute confidence-correctness, confidence-popularity, head/tail confidence gap, high-confidence wrong head rate, low-confidence correct tail rate
- write docs/UNCERTAINTY_DIAGNOSTIC_REPORT.md
- tests
- commit

Stage 4: CARE rerank pilot
- implement CARE expected utility scoring
- compare base DeepSeek, confidence-only, calibration-only, uncertainty-only, popularity-only, CARE full
- output metrics and exposure metrics
- write docs/PILOT_CARE_RERANK.md
- tests
- commit

Stage 5: CARE-LoRA pilot
- build uncertainty-aware LoRA data
- standard SFT vs confidence-weighted vs uncertainty-pruned vs CARE-LoRA
- start amazon_beauty only, then four-domain pilot if stable
- write docs/PILOT_CARE_LORA.md
- tests
- commit

All outputs must be pilot, backend_type correct, is_paper_result=false until explicitly promoted.
```

---

## 14. References to Keep in Docs

Use these in related work and experimental design docs:

- Uncertainty Quantification and Decomposition for LLM-based Recommendation: https://arxiv.org/abs/2501.17630
- Decoding Matters: https://aclanthology.org/2024.emnlp-main.589/
- SLMRec: https://openreview.net/forum?id=G4wARwjF8M
- LLM-ESR: https://proceedings.neurips.cc/paper_files/paper/2024/hash/2f0728449cb3150189d765fc87afc913-Abstract-Conference.html
- RecBole atomic files: https://recbole.io/docs/user_guide/data/atomic_files.html
- RecBole framework: https://github.com/RUCAIBox/RecBole
- CoLLM / CLLM4Rec: https://github.com/yaochenzhu/LLM4Rec
- OpenP5: https://arxiv.org/abs/2306.11134
- BERT4Rec: https://arxiv.org/abs/1904.06690
- FMLP-Rec: https://arxiv.org/abs/2202.13556
- PEFT LoRA docs: https://huggingface.co/docs/peft/developer_guides/lora

