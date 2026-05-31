# Paper Claims and Experiment Status

This file freezes the paper scope so the project does not drift into a mixed
system log.

## Primary claim

Task-grounded calibrated uncertainty improves controlled candidate ranking and
reranking reliability under same-schema evaluation.

The main paper studies whether uncertainty signals can be made decision-useful
for LLM-based recommendation. The contribution is not a new model backbone and
not a generic recommender system.

## Primary contribution statements

1. Diagnosis: verbalized LLM confidence is informative in recommendation, but
   unreliable under miscalibration, confidence collapse, and domain-dependent
   failure.
2. Method: C-CRP, a task-grounded calibrated candidate relevance posterior with
   boundary uncertainty, calibration gap, and evidence insufficiency.
3. Decision: risk-adjusted candidate ranking/reranking evaluated with utility,
   calibration, coverage, exposure, robustness, and paired statistical tests.

C-CRP is the main internal method line. SRPD is the trainable
framework/ablation line and becomes paper-facing only after leakage-clean
teacher generation, weighted-loss training when claimed, exact same-candidate
score export, and paired-test gates pass.

## Not primary claims unless completed

- Generative title recommendation.
- Full-catalog SOTA.
- Universal cross-domain winner.
- LoRA distillation as main novelty.
- Shadow v2-v6 as independent main methods.
- SRPD as a main-method substitute for external baselines unless its formal
  train/eval gates are completed.
- Proxy comparisons against reported numbers from incompatible protocols.

## Status labels

Every paper-facing summary table must include `status_label`.

- `completed_result`: runnable and completed under the stated protocol.
- `runnable_not_complete`: code path exists, but the result is not complete
  enough for a main table.
- `design_only`: method or plan is documented but not fully run under the
  frozen protocol.
- `proxy_only`: useful for related-work positioning or fairness audit, but not
  same-schema evidence.
- `future_extension`: intentionally outside the main paper claim.

Rows with `design_only`, `proxy_only`, or `future_extension` must not enter the
main result table. Rows from fallback internal validation splits are not main
table eligible.

Agent drift guard: every new experiment, baseline, model module, or generated
table must receive a status label and milestone eligibility before it is
described as evidence. If a future agent cannot prove eligibility from
provenance and audits, the row is diagnostic or supplementary.

## Milestone claim eligibility

The canonical milestone map lives in `docs/milestones/README.md`.

| milestone | paper role | main-table eligible |
| --- | --- | --- |
| M0 Week1-4 / pony12 | diagnosis and motivation | no |
| M1 Pony framework | method and protocol bridge | yes, if completed under frozen protocol |
| M2 Light series | precursor / negative-control ablation | supplementary unless explicitly completed under main protocol |
| M3 Shadow series | task-grounded signal family | diagnostic unless large-scale protocol and validation gates are complete |
| M4 Baseline system | fairness and reviewer defense | yes for completed classical rows; official external rows only after provenance passes |
| M5 Four-domain validation | robustness evidence | yes when 100neg outputs, audits, and paired tests are complete |
| M6 Complete recommender system | future full-system roadmap | no until official baselines, Shadow, LoRA, and generated-title verification are completed |

The `*_style_*` LLM-rec rows are paper-style supplementary rows, not official
reproductions. Official external-baseline claims require the
`*_official_qwen3base_*` family plus provenance and score coverage.

Internal Shadow/Light/LoRA ablations cannot substitute for missing external
baselines. If official external rows are unfinished, the paper wording must
state the baseline gap and keep those rows out of main official claims.

## Main table eligibility

A row can enter the main ranking table only if all are true:

- `artifact_class == completed_result`
- `status_label` matches the row type, for example
  `same_schema_external_baseline`, `same_schema_internal_method`, or a
  documented completed ablation label.
- Same-schema data, prompt, candidate construction, and metric definitions are
  used.
- External baseline rows follow the declared comparison variant, usually
  `official_code_qwen3base_default_hparams_declared_adaptation`.
- Official external rows require `implementation_status=official_completed`;
  `style_adapter_only` and `partial_official_adapter_exists` rows stay
  supplementary.
- Internal formal rows require exact score export and import. C-CRP uses
  `status_label=same_schema_internal_method`; SRPD uses
  `status_label=same_schema_internal_ablation` unless all trainable-framework
  gates are satisfied.
- C-CRP formal rows must include validation-only selection provenance, input
  hashes, exact score coverage, and score-degeneracy audit output.
- SRPD rows that use rank-order fallback scores are internal ablations only;
  they must not be presented as native trainable scorer results or promoted to
  the main method table.
- Runner or plan filenames containing `official` are not enough. The row must
  have unblocked provenance from the pinned official repo and pass score/import
  gates.
- Baseline hyperparameters come from the official default/recommended setting,
  unless an override is explicitly recorded for protocol compatibility.
- Our method's gates, weights, and hyperparameters are selected on validation or
  fixed before test.
- Full-finetune and retuned-baseline variants are not mixed into the primary
  main comparison table.
- External score files have exact unique `source_event_id,user_id,item_id`
  coverage with finite numeric scores.
- Calibration method and C-CRP weights are selected on validation or fixed
  before test.
- `calibration_split_metadata.csv` has zero user overlap unless the table is
  explicitly labeled non-main.
- Candidate protocol audit exists for the domain/split.
- Statistical tests provide paired confidence intervals for the supported claim.

## Safe wording

Use "observed best" for results whose confidence interval crosses zero or whose
paired test is not significant after correction. Use "winner" only for results
that pass the configured statistical rule.

## C-CRP v3 Progress (2026-05-31)

### Multi-Domain Results

C-CRP v3 uses a profile-enhanced prompt that asks the LLM to infer user
preferences before scoring each candidate. All domains use 10k users,
101 candidates (1 positive + 100 negative), Qwen3-8B via vLLM.

| Domain | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR | vs Best Baseline |
|--------|------|-------|-------|--------|---------|---------|-----|------------------|
| beauty (973u) | 0.157 | 0.229 | — | — | 0.134 | — | 0.128 | #2 (ProEx=0.253) |
| books | 0.374 | **0.476** | — | — | **0.333** | — | 0.306 | **SOTA** (+0.8% vs LLMEmb) |
| electronics | 0.218 | **0.299** | — | — | **0.183** | — | 0.168 | **SOTA** (+22% vs LLMEmb) |
| movies | 0.145 | 0.208 | — | — | 0.128 | — | 0.127 | #5 (LLMEmb=0.334) |
| sports | 0.275 | 0.382 | 0.517 | 0.198 | 0.233 | 0.267 | 0.208 | baselines pending |
| toys | 0.317 | 0.396 | 0.506 | 0.245 | 0.271 | 0.298 | 0.250 | baselines pending |
| home | 0.156 | 0.226 | 0.351 | 0.110 | 0.132 | 0.164 | 0.126 | baselines pending |
| tools | 0.194 | 0.270 | 0.393 | 0.142 | 0.166 | 0.197 | 0.156 | baselines pending |

Status: `completed_result` for beauty/books/electronics/movies/sports/toys/home/tools.
The C-CRP v3 batch completed without FAILED/OOM/Traceback markers in
`ccrp_v3_all_domains.log`; new-domain official baselines have not started.
For sports/toys/home/tools, each report records `n_users=10000`,
`n_prompts=1010000`, the test same-candidate `data_path`, `scores.csv` with
1,010,000 scored candidate rows plus header, and `user_ranks.jsonl` with
10,000 rows.
Fields marked `—` for beauty/books/electronics/movies are currently missing
from the server `outputs/*ccrp_v3/report.json` lookup and must be recovered
from archived artifacts before final paper tables.

### Strategy for SOTA

C-CRP v3 achieves SOTA on books and electronics. For sports, toys, home, and
tools, do not claim SOTA until the canonical 8 official baselines finish and
paired same-candidate tests pass. Current values are candidate results awaiting
external-baseline comparison.

### Remaining for paper submission

1. Reconcile `scripts/run_baselines_new_domains.sh` with the canonical 8
   official baselines; SETRec is marked blocked/supplementary in
   `configs/official_external_baselines.yaml` unless future gates pass.
2. Run 8 official baselines on sports/toys/home/tools after the script is
   aligned and disk/GPU state is rechecked.
3. Full @5/@10/@20 comparison table across all domains
4. Statistical significance tests (paired t-test, 20+ seeds or bootstrap)
5. Paper writing
6. GPT-5.5/Codex review cycle (target: 8/10)
