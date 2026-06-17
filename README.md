# A Pointwise LLM Relevance Posterior Is a Strong Reranker

A controlled same-candidate study of uncertainty-adjusted LLM recommendation.

This repository builds a rigorous unified protocol for comparing LLM-based
recommendation scoring signals, and uses it to answer one focused question:

> Under a fixed candidate set, a shared backbone, and a shared evaluation
> pipeline, which LLM-derived scoring signal ranks the candidates most
> reliably -- and does a calibrated-uncertainty / risk-adjusted decomposition
> built on top of that signal help?

The headline answer is empirical: a **task-grounded pointwise LLM relevance
posterior** -- the model's own per-candidate relevance probability, queried
directly and used as the ranking score -- ranks **first in 6 of 8 Amazon
domains** against eight official-code-level baselines, improving NDCG@10 by
**+21.6% to +53.2%** over the strongest baseline. It is **competitive but not
first** in the other two: **rank 2 on Beauty (-11.0% NDCG@10 behind ProEx)** and
**rank 5 on Movies (-24.2% behind LLMEmb)**. We report those two losses
explicitly rather than restricting the table to the winning domains. The
uncertainty/risk machinery this project originally set out to validate turns out
**not** to improve same-candidate ranking; we report that as a characterized
**negative result** rather than hiding it.

This is the honest dual framing the paper makes throughout: the **pointwise
posterior is the working ranking signal** (positive, 6 of 8 domains), while the
**calibrated-uncertainty / risk decomposition is a characterized negative
result** (it does not improve same-candidate ranking over the posterior it is
built on).

The repository is explicitly **not** positioned as a full-catalog recommender
SOTA claim, a generative-title recommender, or an ECE/calibration-guarantee
result. The claim is scoped to same-candidate reranking.

> Note on naming: the method/artifact identifier `C-CRP` is retained throughout
> the code and result files because it is the canonical run name. After the
> reframe its expansion is **Candidate-Conditioned Relevance Posterior** -- a
> pointwise relevance posterior. It is **not** "Calibrated": the project makes
> no fitted-calibration-map, reliability-diagram, or expected-calibration-error
> claim.

## Two Contributions

```text
1. PROTOCOL   A unified same-candidate reranking benchmark
              (10k users x 101 candidates [Beauty 973], shared Qwen3-8B backbone,
               shared importer, paired Holm-corrected bootstrap, full provenance)
              + 8 official-code-level baselines over 8 Amazon domains.

2. FINDING    A task-grounded pointwise LLM relevance posterior ranks FIRST in
              6 of 8 domains (+21.6% to +53.2% NDCG@10 over the strongest
              baseline; on the 4 domains with full per-event signal rows, all
              56/56 per-domain paired tests are Holm-significant). It is
              competitive but NOT first in Beauty (#2, -11.0%) and Movies
              (#5, -24.2%), reported honestly.
              ... AND the calibrated-uncertainty/risk decomposition does NOT
              improve same-candidate ranking (a rigorous negative result).
```

### Contribution 1: a unified same-candidate reranking protocol

Comparisons across LLM-based recommenders are usually confounded by different
candidate sets, backbones, importers, and evaluation schemas, so it is rarely
clear whether a reported gain reflects the modeling idea or the setup. This
project holds all of that fixed:

- **Same candidates.** Each method ranks the same 101 candidates per event
  (1 ground-truth positive + 100 **popularity-sampled** negatives) for 10,000
  users per domain (1,010,000 candidate rows / domain); Beauty has a smaller
  catalog and yields 973 eligible test users (98,273 rows).
- **Same backbone.** Every method that requires an LLM or LLM-derived
  representation uses a single shared **Qwen3-8B** backbone.
- **Same importer + metrics.** All outputs are converted into the same
  `source_event_id,user_id,item_id,score` schema and run through one importer
  with exact coverage checks (coverage = 1.0, no missing/extra/duplicate keys,
  finite scores).
- **Same statistics.** Paired event-bootstrap confidence intervals (2,000
  samples, 95% percentile) with Holm correction at alpha=0.05.
- **Full provenance.** Each baseline row records upstream repo, pinned commit,
  adaptation mode, checkpoint/representation artifact, score path, score hash,
  and coverage audit.
- **8 official-code-level baselines:** ELMRec, IRLLRec, LLM2Rec, LLMEmb,
  LLM-ESR, ProEx, ProMax, RLMRec. (ProEx and ProMax are concurrent works from
  the **same official repository**, the ProRec line at
  [BlueGhostYi/ProRec](https://github.com/BlueGhostYi/ProRec) -- ProEx is the
  KDD 2026 entry and ProMax the SIGIR 2026 entry; we adapt both under the
  unified protocol.) A 9th baseline, **SETRec**, was run/attempted but is
  **excluded** from the main block: its official upstream tokenization runs out
  of CUDA memory on the large domains (blocker
  `setrec_upstream_tokenize_all_cuda_oom`), so it is held as
  `official_blocked` / supplementary and replaced by ELMRec/ProEx/ProMax.
- **8 Amazon domains:** Beauty, Books, Electronics, Movies (original four) +
  Sports, Toys, Home, Tools (new four).

The protocol deliberately isolates the **reranking** stage; it makes no
full-catalog retrieval claim.

### Contribution 2: the finding (one positive result + one negative result)

**Positive result.** The task-grounded pointwise LLM relevance posterior
(C-CRP, ranking on the raw per-candidate relevance probability with a seeded
tie-break) ranks **first in 6 of the 8 domains** (Books, Electronics, Sports,
Toys, Home, Tools), where it improves NDCG@10 over the strongest official
baseline by **+21.6% to +53.2%** and leads on essentially all seven metrics
(HR@5/@10/@20, NDCG@5/@10/@20, MRR; the lone exception is Books HR@20, where
LLMEmb's deeper recall edges it). In the remaining two domains it is competitive
but **not first**: **rank 2 on Beauty** (-11.0% NDCG@10 behind ProEx, 0.1341 vs.
0.1506) and **rank 5 on Movies** (-24.2% behind LLMEmb, 0.1281 vs. 0.1690). On
the four new domains with full per-event signal rows (Sports/Toys/Home/Tools),
**all 56 per-domain paired tests are positive and Holm-significant** (8 baselines
x 7 metrics) in every domain.

A random floor and a pure popularity ranking both sit near NDCG@10 ~ 0.045
(analytical expectation for 1 positive among 101 candidates). Because the
negatives are themselves popularity-sampled, the protocol neutralizes the
popularity shortcut, so the informative comparison is against the strongest
clearly-above-floor baseline -- **LLMEmb in seven of the eight domains**
(NDCG@10 0.094-0.274) and **ProEx in Beauty** (NDCG@10 0.151) -- not the weakest
rows.

**Negative result.** This project originally set out to study *actionable
uncertainty*. We instrument the posterior with a principled calibrated-
uncertainty decomposition (boundary ambiguity, calibration gap, evidence
support, counterevidence) and a one-parameter risk-adjusted ranking family
`r = p * (1 - U)^eta`. Under a leave-one-component-out ablation and a
raw-probability attribution, **this machinery does not improve same-candidate
ranking**:

- the zero-risk setting (`eta = 0`) is **test-best in all four domains**;
- a **confidence-only** variant (posterior, no uncertainty, no risk) is
  **non-worse than the full family in at least 3 of 4 domains**;
- ranking on the **raw `p-hat`** alone tracks the reported full-configuration
  NDCG@10 **within <= 0.008** across the three domains with local signal rows;
- the **boundary-uncertainty term is exactly inert (4/4 domains)**; removing
  counterevidence or the risk penalty is non-worse (4/4).

The working ranking signal is located **in the posterior itself**, not in the
uncertainty decomposition. As a minor, **caveated and partly-circular**
descriptive observation, the event-level uncertainty signal does stratify
ranking reliability (high-uncertainty bin worse on NDCG@10/MRR/HR@10 in all
four domains) -- but we do **not** claim it improves ranking.

We accordingly present the calibrated-uncertainty / risk family as an
**analyzed and characterized design space**, not as a validated headline
contribution. We make **no ECE/calibration-guarantee claim** and **no
full-catalog SOTA claim**.

For the frozen claim and status rules, see
[docs/paper_claims_and_status.md](docs/paper_claims_and_status.md).

## Main Results

Main same-candidate comparison across **all eight domains**: C-CRP (pointwise
posterior) vs. the strongest official baseline by NDCG@10, 10,000 users (Beauty
973) / 101 candidates per domain. Numbers match `Paper/tables/main_results.tex`,
`Paper/tables/improvement_over_strongest.tex`, and
`Paper/tables/full_official_ndcg10_ranking.tex`.

### Headline: rank and NDCG@10 improvement over the strongest baseline

C-CRP ranks **first in 6 of 8 domains**; competitive but not first in Beauty
(#2) and Movies (#5). The table below gives the per-domain NDCG@10 improvement of
C-CRP over the **strongest official baseline on NDCG@10**, **including the two
losses** (shown as negative deltas, reported honestly rather than dropped):

| Domain | C-CRP rank | Strongest baseline (NDCG@10) | C-CRP NDCG@10 | NDCG@10 improvement |
|--------|:---------:|------------------------------|---------------|:-------------------:|
| Books       | **1 of 9** | LLMEmb (0.2737) | **0.3328** | **+21.6%** |
| Electronics | **1 of 9** | LLMEmb (0.1196) | **0.1833** | **+53.2%** |
| Sports      | **1 of 9** | LLMEmb (0.1795) | **0.2329** | **+29.7%** |
| Toys        | **1 of 9** | LLMEmb (0.2049) | **0.2708** | **+32.2%** |
| Home        | **1 of 9** | LLMEmb (0.0939) | **0.1324** | **+41.0%** |
| Tools       | **1 of 9** | LLMEmb (0.1159) | **0.1661** | **+43.3%** |
| Beauty      | 2 of 9     | ProEx (0.1506)  | 0.1341     | **-11.0%** |
| Movies      | 5 of 9     | LLMEmb (0.1690) | 0.1281     | **-24.2%** |

So the honest headline is: **first in 6 of 8 domains (+21.6% to +53.2% NDCG@10
over the strongest baseline), competitive in Beauty (#2) and Movies (#5)** -- not
"first in every domain." The per-metric improvement table (including the
negative Beauty/Movies deltas across all seven metrics) is in
`Paper/tables/improvement_over_strongest.tex`.

### Backbone robustness: a second LLM (Llama-3.1-8B)

To test whether the headline is specific to the Qwen3-8B backbone, the identical
same-candidate pipeline was re-run with a different-family LLM
(Llama-3.1-8B-Instruct) on the four new winning domains, holding everything else
fixed (10k users, 101 candidates, same panels/schema/tie-break). The rank-first
NDCG@10 result **replicates in all four domains**, at a ~12-17% lower absolute
level that tracks backbone quality (a first cross-family test, not a broad
robustness guarantee -- the other four domains and other model families remain
future work):

| Domain | Qwen3-8B NDCG@10 | Llama-3.1-8B NDCG@10 | Strongest baseline (LLMEmb) | Llama vs. baseline |
|--------|:----------------:|:--------------------:|:---------------------------:|:------------------:|
| Sports | 0.2329 | **0.2054** | 0.1795 | **+14.4%** |
| Toys   | 0.2708 | **0.2274** | 0.2049 | **+11.0%** |
| Home   | 0.1324 | **0.1103** | 0.0939 | **+17.5%** |
| Tools  | 0.1661 | **0.1407** | 0.1159 | **+21.4%** |

The one consistent exception is deep recall (HR@20): the coarser Llama posterior
floors 84-90% of candidates at 0 (Qwen 19-34%) and the true positive in 53-78% of
events (Qwen 6-20%), which pushes floored positives past rank 20 while the top of
the list stays sharp. Evidence: `outputs/<domain>_large10000_100neg_ccrp_v3_llama/`,
`outputs/summary/paper_critical/second_backbone_llama_replication.md`, and the
posterior-degeneracy diagnostic. Paper: the "Backbone Robustness" subsection in
`Paper/sections/results.tex`.

### Full per-metric detail (the six winning domains)

In the six domains where C-CRP ranks first it leads on essentially all seven
metrics (the lone exception is Books HR@20, where LLMEmb's deeper recall edges
it). Absolute values, e.g.:

| Domain | Method | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR |
|--------|--------|------|-------|-------|--------|---------|---------|-----|
| Sports | LLMEmb (best official) | 0.2124 | 0.3384 | 0.4900 | 0.1389 | 0.1795 | 0.2177 | 0.1539 |
| Sports | **C-CRP** | **0.2745** | **0.3819** | **0.5172** | **0.1985** | **0.2329** | **0.2670** | **0.2076** |
| Toys   | LLMEmb (best official) | 0.2499 | 0.3505 | 0.4866 | 0.1725 | 0.2049 | 0.2391 | 0.1814 |
| Toys   | **C-CRP** | **0.3172** | **0.3964** | **0.5059** | **0.2452** | **0.2708** | **0.2983** | **0.2503** |
| Home   | LLMEmb (best official) | 0.1079 | 0.1856 | 0.3169 | 0.0690 | 0.0939 | 0.1267 | 0.0901 |
| Home   | **C-CRP** | **0.1561** | **0.2264** | **0.3505** | **0.1098** | **0.1324** | **0.1635** | **0.1259** |
| Tools  | LLMEmb (best official) | 0.1365 | 0.2257 | 0.3637 | 0.0875 | 0.1159 | 0.1505 | 0.1065 |
| Tools  | **C-CRP** | **0.1937** | **0.2696** | **0.3931** | **0.1419** | **0.1661** | **0.1970** | **0.1559** |

For Beauty and Movies the strongest baseline is ahead on NDCG@10 (Beauty: ProEx
0.1506 vs. C-CRP 0.1341; Movies: LLMEmb 0.1690 vs. C-CRP 0.1281). Random floor
and pure popularity both sit at NDCG@10 ~ 0.045 in the four new domains where
popularity was measured.

The full per-domain NDCG@10 ranking over all eight official baselines plus
C-CRP (with Random and Popularity reference rows) is in
`outputs/ccrp_v3_formal/` artifacts and mirrored in
`Paper/tables/full_official_ndcg10_ranking.tex`. The complete evidence ledger
contains **72 method-domain rows** (64 official baseline rows = 8 baselines x 8
domains, plus 8 C-CRP rows).

**Experiment protocol:** 10k test users (Beauty 973), 101 candidates (1 + 100
popularity-sampled negatives), shared Qwen3-8B via vLLM, same-candidate
evaluation, metrics @5/@10/@20 + MRR, paired Holm-corrected bootstrap.

## Honest Current Status

This is the load-bearing section: it states what is done for the validity of the
claim, including the bridge-fidelity experiment that was previously the only
pending validity item.

**Paper.** Reframed to the pointwise-posterior headline + uncertainty negative
result, now expanded to **all eight domains** with the honest 6/8 framing
(branch `paper/reframe-major-revision`, 8-domain commit `932d393`). Compiles
clean (`pdflatex` x3 + bibtex, 0 errors / 0 undefined refs) at **13 pages**.
Method full name is **Candidate-Conditioned Relevance Posterior** (a pointwise
relevance posterior -- **not** "Calibrated": no fitted-calibration-map,
reliability-diagram, or ECE claim).

**Done.**

- Unified same-candidate protocol + 8 official-code-level baselines over **all
  eight domains** (Beauty, Books, Electronics, Movies, Sports, Toys, Home,
  Tools), each 8/8 baselines complete with exact score coverage and full
  provenance (72-row ledger: 64 official + 8 C-CRP).
- C-CRP pointwise-posterior rows ranking **first in 6 of 8 domains** (+21.6% to
  +53.2% NDCG@10 over the strongest baseline); on the four new domains with full
  per-event signal rows, 56/56 per-domain paired tests positive and
  Holm-significant. Beauty (#2) and Movies (#5) reported as honest losses.
- **raw-`p-hat` attribution ablation** (the revision blocker that pins the win
  to the posterior, not the uncertainty machinery): commit `885f11c`.
- Leave-one-component-out ablations, `eta`/weight hyperparameter sensitivity,
  uncertainty-stratification observation module, framework-overview figure.
- **Bridge fidelity -- DONE.** All baselines use a shared Qwen3-8B
  representation **bridge** for same-candidate scoring (Limitation 5). To confirm
  the bridge is not responsible for low baseline scores, the strongest
  SASRec-class baseline **LLM2Rec was re-run in its native encoder** (an LLM2Vec
  model over a Qwen2-0.5B backbone) on the identical task, changing only the
  item-embedding source. The Qwen3-8B bridge **recovers ~90% of native NDCG@10**
  (Tools **110%**, Sports **78%**; HR@10 recovery 116% / 93%) and stays well
  above the random floor. Validated on the **two domains** whose native
  downstream task was prepared; a full-backbone reproduction of every baseline
  (e.g. LLMEmb) remains future work.

**Review.** Latest internal/ARIS review pass rates the paper **8.0/10
(conditional pass)**, scope-limited to the controlled same-candidate reranking
claim. Target venue: **SIGIR 2026** (full paper; ACM `acmart`/`sigconf`).

## What Is Implemented

The repository includes:

- Sequential recommendation data preprocessing for Amazon-style domains.
- Pointwise, pairwise, and candidate-ranking sample construction.
- LLM inference backends for API and local (vLLM Qwen3-8B) model experiments.
- Task-grounded relevance-posterior signal extraction, parsing, and
  normalization (the working ranking scorer).
- Calibration diagnostics with ECE, Brier score, AUROC, reliability bins, and
  bootstrap confidence intervals (used as *diagnostics*, not as a result claim).
- Strict validation-to-test split discipline with split metadata, overlap
  checks, split hashes, and source file hashes.
- Candidate ranking/reranking evaluation with HR@K, NDCG@K, MRR, coverage,
  head exposure, long-tail coverage, parse success, and out-of-candidate rate.
- Candidate protocol audit for candidate set size, positive count, negative
  sampling, popularity distribution, title duplicates, split overlap, and
  HR/Recall equivalence.
- Baseline reliability proxy audit to prevent score-derived priors from being
  mislabeled as confidence.
- Paired bootstrap and permutation tests with Holm correction.
- C-CRP pointwise-posterior method implementation and the analyzed
  uncertainty/risk design space, with leave-one-component-out and
  raw-probability attribution ablations.
- Formal C-CRP internal-method selector/exporter that selects mode and weights
  on validation only, emits exact same-candidate score CSVs, and imports
  through the shared score gate.
- SRPD trainable-framework tooling (ablation/supplementary line) with leakage
  audit hooks, sample-weighted LoRA loss support, and exact-score export.
- Generative-title bridge status tracking, explicitly outside the primary
  claim.
- Official external-baseline upgrade contract for adapting pinned upstream
  LLM-rec baselines under the unchanged same-candidate protocol.
- Artifact documentation, smoke tests, and reproduction scripts.

## Operations (server, env, runbook)

GPU experiments run on the server only; local is for version control and
writing.

- **Server:** `pony-rec-gpu` (`ssh pony-rec-gpu`, key-based auth;
  `125.71.97.70:15302`, user `ajifang`, RTX 4090 49GB VRAM).
- **Server project path:** `~/projects/pony-rec-rescue-shadow-v6`
- **Local project path:** `D:\Research\Uncertainty`
- **Python env:** `qwen_vllm`
  (`/home/ajifang/miniconda3/envs/qwen_vllm/bin/python`); do not rely on a
  bare `python` over non-interactive SSH.
- **HF mirror (used for the completed native-encoder bridge-fidelity run):**
  `HF_ENDPOINT=hf-mirror.com`.

Before doing anything, check whether the server already has a run in progress
(do not kill running processes or rerun completed experiments):

```bash
ssh pony-rec-gpu "ps aux | grep python | grep -v grep | grep -i 'pony-rec\|ccrp\|baseline\|uncertainty'"
ssh pony-rec-gpu "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
ssh pony-rec-gpu "tail -5 ~/projects/pony-rec-rescue-shadow-v6/ccrp_v3_all_domains.log 2>/dev/null"
```

See [docs/server_runbook.md](docs/server_runbook.md) for the full server-side
execution surface, and [AGENTS.md](AGENTS.md) for the operating contract,
non-toy experiment standards, and push hygiene.

## Artifact Rules

Keep locally (commit to GitHub): `report.json`, `user_ranks.jsonl`,
`main_comparison_table.csv`, `fairness_provenance.json`, `*_score_audit.json`,
`*_run_summary.json`, and the paper sources / scripts / configs / docs.

Server-only (do not download, do not commit): `scores.csv` (~87MB/domain,
regenerable), `predictions/` (600MB+), `embeddings/`, `checkpoints/`, raw data,
and model weights.

Never commit: API keys / credentials, `__pycache__/`, `*.pyc`, editor swap
files, or the large files above. Generated data, predictions, checkpoints, and
large experiment outputs are git-ignored. The reproducibility contract lives in
configs, scripts, manifests, and protocol docs.

## Canonical Navigation

- [AGENTS.md](AGENTS.md) -- authoritative operating contract: project direction,
  baseline standards, non-toy experiment rules, multi-agent collaboration,
  server paste-back protocol, GitHub push hygiene.
- [docs/milestones/README.md](docs/milestones/README.md) -- M0-M6 milestone map,
  evidence levels, agent roles.
- [docs/paper_claims_and_status.md](docs/paper_claims_and_status.md) -- frozen
  claims and status labels.
- [docs/top_conference_review_gate.md](docs/top_conference_review_gate.md) --
  standing reviewer gate.
- [docs/server_runbook.md](docs/server_runbook.md) -- server-side execution
  entry point.

Future agents should not rely on chat memory. If the working direction,
baseline status, command surface, or table eligibility changes, update the
canonical docs and push the change after smoke/readiness checks.

## Project Tree

```text
.
|-- README.md
|-- requirements.txt
|-- environment.yml
|-- results_manifest.yaml
|-- AGENTS.md
|-- CLAUDE.md
|-- Paper/                            # LaTeX sources + tables (do not edit from README work)
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
|   |-- ccrp_v3_formal/               # C-CRP v3 formal reports + comparison tables
|   |-- baselines/                    # ignored; external task packages/scores
|   `-- summary/                      # ignored except .gitkeep; generated tables
|-- prompts/                          # LLM prompt templates
|-- scripts/                          # reproduction + run scripts
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
`-- tests/

# entry points (under scripts/, not the repo root):
#   scripts/pipeline/   main_preprocess.py, main_infer.py, main_eval.py,
#                       main_eval_rank.py, main_calibrate.py, main_rerank.py
#   scripts/build/      main_build_samples.py, main_build_external_only_baseline_comparison.py
#   scripts/misc/       main_shadow_ccrp_eval.py, main_stat_tests.py,
#                       main_import_same_candidate_baseline_scores.py,
#                       main_generative_title_bridge_status.py
#   scripts/audit/      main_audit_candidate_protocol.py, main_baseline_reliability_audit.py
#   scripts/adapters/   main_run_<method>_official_same_candidate_adapter.py
#   scripts/analysis/   main_build_ccrp_component_ablation_summary.py,
#                       main_build_ccrp_hyperparameter_sweep.py,
#                       main_build_uncertainty_observation_study.py,
#                       main_build_framework_overview_figure.py, ...
```

## Method Overview

### Problem: same-candidate reranking

Given a user `u` with history `H_u` and a candidate set `C_u` of `n=101` items
(1 positive + 100 popularity-sampled negatives), order the candidates so the
held-out next item ranks as high as possible. Every method emits exactly one
finite score per `(u, c_i)` pair under the shared key schema.

### The working scorer: task-grounded pointwise relevance posterior

For each user-candidate pair, the LLM evaluates the candidate under the
**next-interaction task** (not generic similarity) and emits a pointwise
relevance probability `p_i = P(y_i = 1 | H_u, c_i)`. Ranking on this posterior
descending (with a deterministic seeded tie-break keyed by source event, user,
item) is the operative scorer and the source of the headline result.

The signal row also records a model-reported adjusted probability, evidence
support, counterevidence strength, and a short rationale. These feed the
*analyzed uncertainty design space* below; they do **not** drive the ranking.
Provenance records the Qwen3-8B vLLM path, temperature 0.1, 120 max new tokens,
5,000 users/chunk, 101 expected candidates/event, parse-failure count (zero in
paper packages, max allowed rate 0.005), row counts, and file hashes.

> `C-CRP` does **not** fit a post-hoc calibration map and makes no
> reliability-diagram / ECE guarantee. "Calibrated" in any field name refers to
> the self-reported probability target of the scoring prompt, not a measured
> calibration error.

### The analyzed (and inert) uncertainty / risk design space

The one-parameter ranking family is multiplicative:

```text
r_i = p_i * (1 - U_i)^eta
```

with `eta` fixed or selected on **validation only**; `eta = 0` recovers the
bare posterior `r_i = p_i`. Candidate uncertainty is

```text
U_i = clip_[0,1]( w_b * b_i + w_g * g_i + w_e * (1 - e_i) )
```

with boundary ambiguity `b_i = 4 * p_i * (1 - p_i)`, calibration gap
`g_i = |p_hat_i - p_i|`, and effective evidence `e_i = max(0, min(1, s_i - q_i))`
(support `s_i`, counterevidence `q_i`). Default preregistered weight triple
`(w_b, w_g, w_e) = (0.5, 0.3, 0.2)`.

Empirically, ablations show this family does **not** help ranking: `eta = 0` is
test-best in all four domains, boundary uncertainty is exactly inert, and the
confidence-only variant is non-worse than the full family in >= 3/4 domains.
The family is therefore reported as an analyzed, characterized design space.

### Diagnostics

- **Uncertainty-stratification observation:** bin events by event-level
  uncertainty into 5 quantiles; the high-uncertainty bin is worse on
  NDCG@10/MRR/HR@10 in all four domains. **Descriptive and partly mechanical**
  (events binned by the method's own uncertainty, which the score is a monotone
  function of) -- not a ranking-improvement claim.
- **Component ablation (leave-one-component-out):** removal-minus-full deltas on
  NDCG@10. Boundary uncertainty inert (mean 0, 4/4); removing counterevidence
  (+0.00168, non-worse 4/4) and risk penalty (+0.00082, non-worse 4/4)
  non-harmful; calibration gap mixed; evidence support directionally harmful but
  tiny.
- **Hyperparameter sensitivity:** `eta` and the uncertainty-weight grid;
  validation-selected vs. reporting-only test sweeps, stable within a 5%
  relative-drop tolerance in all four domains; test-best `eta` is zero
  everywhere.

See [docs/shadow_method.md](docs/shadow_method.md).

## Baseline Policy

The primary table uses **official-code-level** LLM-rec baselines under the
unchanged same-candidate protocol:

```text
pinned official or official-code-level implementation
+ official algorithm, loss, scoring head, and train/select procedure preserved
+ unified Qwen3-8B base model for LLM/text representations
+ frozen Qwen3-8B base except method-declared adapter, identifier,
  representation learner, graph/intent module, or downstream checkpoint
+ baseline official default/recommended hyperparameters in the primary table
+ our method hyperparameters selected on validation only or fixed before test
+ unchanged same-candidate train/valid/test rows
+ unchanged score schema: source_event_id,user_id,item_id,score
+ unified importer, metrics, coverage audit, and paired tests
```

Official main-table rows require `implementation_status=official_completed`,
exact score coverage, and fairness provenance. The score file is an exact key
contract: unique `source_event_id,user_id,item_id` rows, no missing/extra keys,
finite numeric scores; method-native score scales are allowed because the shared
importer applies one ranking/evaluation path.

**Full-catalog metrics reported by upstream repositories are related-work /
sanity context only** and are never mixed into the same-candidate main table.

**Bridge caveat (Limitation 5).** Baselines are run with a shared Qwen3-8B
representation bridge. This is a fair common-denominator reranking comparison,
**not** a reproduction of each method under its native full-catalog serving
stack. The **bridge-fidelity experiment is DONE**: LLM2Rec was re-run in its
native encoder (LLM2Vec over Qwen2-0.5B) on the identical task, and the Qwen3-8B
bridge recovers **~90% of native NDCG@10** (Tools 110%, Sports 78%), validated
on the two domains whose native downstream task was prepared. So the bridge
preserves most of a baseline's native same-candidate signal rather than
collapsing it; C-CRP's margin reflects a genuine reranking difference, not a
bridge artifact. A full-backbone reproduction of every baseline is left to
future work.

Score-derived quantities are audited as reliability proxies, not automatically
treated as confidence.

See [docs/baseline_protocol.md](docs/baseline_protocol.md).

## Candidate Ranking Protocol

The primary evaluation is controlled candidate reranking. Every main
candidate-ranking claim is backed by `outputs/summary/candidate_protocol_audit.csv`,
reporting candidate-set size, positives per event, negative-sampling strategy,
hard-negative ratio, popularity-bin distribution, duplicate-title rate,
valid/test user overlap, train/test item overlap, the one-positive setting,
HR@K/Recall@K equivalence, and full-catalog availability.

If `full_catalog_eval_available_flag=false`, the paper must not claim
full-catalog recommender SOTA. See
[docs/candidate_protocol.md](docs/candidate_protocol.md).

## Statistical Testing

Use paired tests before any winner claim. Implemented in
[scripts/misc/main_stat_tests.py](scripts/misc/main_stat_tests.py): paired event-bootstrap confidence
intervals, paired permutation tests, delta-vs-baseline comparisons, and
Holm-Bonferroni correction over the full 56-test family per domain. Rows whose
CI crosses zero or whose corrected p-value is not significant are labeled
`observed_best`, not `winner`.

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

API-backed inference requires the relevant model key, e.g.:

```powershell
$env:DEEPSEEK_API_KEY="your_api_key"
```

Model configs read key names from `configs/model/*.yaml`. GPU/vLLM Qwen3-8B
runs use the server `qwen_vllm` env (see Operations above), not the local
machine.

## Quickstart

The commands below show the standard Beauty diagnostic pipeline. Replace config
files and experiment names for other domains. The formal four-domain
(Sports/Toys/Home/Tools) same-candidate runs happen on the server.

### 1. Preprocess

```bash
python scripts/pipeline/main_preprocess.py --config configs/data/amazon_beauty.yaml
```

### 2. Build samples

```bash
python scripts/build/main_build_samples.py --config configs/data/amazon_beauty.yaml
```

### 3. Run inference

Validation split:

```bash
python scripts/pipeline/main_infer.py \
  --config configs/exp/beauty_deepseek.yaml \
  --input_path data/processed/amazon_beauty/valid.jsonl \
  --output_path outputs/beauty_deepseek/predictions/valid_raw.jsonl \
  --split_name valid \
  --overwrite
```

Test split:

```bash
python scripts/pipeline/main_infer.py \
  --config configs/exp/beauty_deepseek.yaml \
  --input_path data/processed/amazon_beauty/test.jsonl \
  --output_path outputs/beauty_deepseek/predictions/test_raw.jsonl \
  --split_name test \
  --overwrite
```

### 4. Evaluate predictions

```bash
python scripts/pipeline/main_eval.py \
  --exp_name beauty_deepseek \
  --input_path outputs/beauty_deepseek/predictions/test_raw.jsonl
```

### 5. Calibration diagnostics

```bash
python scripts/pipeline/main_calibrate.py \
  --exp_name beauty_deepseek \
  --valid_path outputs/beauty_deepseek/predictions/valid_raw.jsonl \
  --test_path outputs/beauty_deepseek/predictions/test_raw.jsonl \
  --method isotonic \
  --strict_split_check true \
  --allow_user_overlap false
```

Calibration is fit on validation and applied once to test, with strict split
checks (`--strict_split_check true`, `--allow_user_overlap false`). It is used
as a **diagnostic**; the paper makes no calibration-guarantee claim. If
validation has too few rows for isotonic calibration, the script falls back to
Platt scaling and records both requested and effective methods.

### 6. Rerank (analyzed risk family)

```bash
python scripts/pipeline/main_rerank.py \
  --exp_name beauty_deepseek \
  --input_path outputs/beauty_deepseek/calibrated/test_calibrated.jsonl \
  --lambda_penalty 0.5
```

### 7. C-CRP same-candidate scoring + eval

```bash
python scripts/misc/main_shadow_ccrp_eval.py --help
```

### 8. Audit protocols

Candidate protocol:

```bash
python scripts/audit/main_audit_candidate_protocol.py \
  --domain beauty \
  --data_dir data/processed/amazon_beauty \
  --negative_sampling_strategy sampled_candidate_one_positive \
  --output_path outputs/summary/candidate_protocol_audit.csv
```

Baseline reliability proxy:

```bash
python scripts/audit/main_baseline_reliability_audit.py \
  --config configs/baseline_reliability/week7_9_manifest.yaml \
  --output_path outputs/summary/baseline_reliability_proxy_audit.csv
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

## Key Outputs

Experiment-local outputs under `outputs/{exp_name}/`:

```text
predictions/valid_raw.jsonl
predictions/test_raw.jsonl
calibrated/valid_calibrated.jsonl
calibrated/test_calibrated.jsonl
tables/diagnostic_metrics.csv
tables/calibration_comparison.csv
tables/rerank_results.csv
figures/reliability_before_after_calibration.png
```

C-CRP v3 formal reports under `outputs/ccrp_v3_formal/<domain>/report.json`
(plus `main_comparison_table.csv`). New-domain same-candidate runs emit
`report.json`, `user_ranks.jsonl` (10,000 rows), and `scores.csv`
(1,010,000 candidate scores + header; server-only).

Summary outputs under `outputs/summary/` include `final_results.csv`,
`candidate_protocol_audit.csv`, `baseline_reliability_proxy_audit.csv`,
`significance_tests.csv`, and `main_table_with_ci.csv`. Large generated files
are git-ignored.

## Paper-Facing Status Labels

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

1. **No full-catalog claim.** Evaluation is controlled reranking over 101
   candidates per user; candidate generation / full-catalog retrieval is future
   work.
2. **Single LLM and domain family.** Unified Qwen3-8B backbone over eight Amazon
   domains; other LLMs / catalogs may behave differently.
3. **Descriptive uncertainty diagnostics.** The uncertainty-stratification
   observation is descriptive and partly mechanical, not a causal intervention.
4. **Uncertainty decomposition is inert for ranking.** Ablations + the
   raw-probability attribution show the calibrated-uncertainty / risk terms do
   not improve same-candidate ranking over the pointwise posterior; `eta = 0` is
   test-best in all four domains. Reported as a characterized negative result.
5. **External adapter / bridge policy.** Baselines use a shared Qwen3-8B
   representation bridge -- a fair controlled comparison, but not a reproduction
   under each method's native full-catalog serving stack. The bridge-fidelity
   experiment is **done** (native-encoder LLM2Rec recovers ~90% of native
   NDCG@10 under the bridge, validated on two domains); a full-backbone
   reproduction of every baseline remains future work.
6. **LLM cost and latency.** Not optimized for serving latency, token cost, or
   online batching.
7. **Prompt and parser dependence.** Signal rows are fail-closed and
   source-audited but still depend on the prompt, schema, and parser.
8. **Sampled negative candidate sets.** One positive + 100 popularity-sampled
   negatives per event; conclusions may differ under larger or
   retriever-generated candidate pools.

No ECE / calibration-guarantee claim is made. No full-catalog SOTA claim is
made. See [docs/limitations.md](docs/limitations.md).

## License

This project is released under the terms of the [LICENSE](LICENSE).
