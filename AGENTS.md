# Agent Operating Contract

This file is the first-read contract for future Codex/agent work in this
repository. It exists because chat memory is not a research artifact.

## Start Here Every Time

Before changing code, documents, experiments, or claims, read:

1. `README.md`
2. `docs/milestones/README.md`
3. `docs/paper_claims_and_status.md`
4. `docs/top_conference_review_gate.md`
5. `docs/server_runbook.md`

For baseline or experiment implementation, also read:

1. `docs/baseline_protocol.md`
2. `OFFICIAL_EXTERNAL_BASELINE_UPGRADE_PLAN_2026-05-07.md`
3. `configs/official_external_baselines.yaml`
4. `PROJECT_LINEAGE_AND_FILE_INDEX_2026-05-06.md`

Legacy Week8 handoff/roadmap files are historical context only. Do not start
from them, and do not treat their roadmap items as completed evidence.

## Project Direction

The project is not a toy demo and not a loose collection of scripts. The
research spine is:

```text
Week1-4 / pony12 observation
-> Pony framework
-> Light series boundary tests
-> Shadow task-grounded uncertainty
-> same-candidate baseline system
-> small-domain to four-domain 100neg validation
-> complete recommendation-system roadmap
```

The defended paper claim is narrower than the full roadmap:

```text
Task-grounded calibrated uncertainty improves controlled candidate ranking /
reranking reliability under same-schema evaluation.
```

Do not silently expand this into a full-catalog recommender SOTA claim, a LoRA
distillation claim, or a generative-title recommender claim. Those are future
modules unless their own protocol, baselines, audits, and claims are completed.

## Senior Baseline Advice

The adopted main comparison policy follows the senior advice:

```text
official or official-code-level implementations
+ our same-candidate dataset/protocol
+ unified Qwen3-8B base model
+ baseline official/default or recommended hyperparameters
+ our method validation-selected or pre-fixed hyperparameters
+ no test-set model or hyperparameter selection
```

This is a normal academic fairness setting. We do not need to fully retune every
baseline for the primary table, but we must state the policy cleanly. If a
baseline-retuned variant is run, label it as sensitivity/robustness, not as the
primary default-hyperparameter table.

LoRA/full fine-tuning policy must be explicit. The primary official external
baseline variant uses Qwen3-8B base plus the method-declared adapter,
identifier, representation learner, graph/intent module, or downstream
checkpoint required by that baseline's official algorithm. Full fine-tuning,
original-backbone, and retuned-baseline rows are supplementary/sensitivity
variants unless a new experiment-wide policy is declared.

## Non-Toy Experiment Standard

Do not replace hard baseline work with toy shortcuts. A result is not
paper-facing unless it has:

- real task data, not synthetic placeholders;
- the same train/valid/test discipline used by the claim;
- the same candidate rows and metric importer for comparable rows;
- score/provenance coverage checks;
- paired tests or a clear statement that it is diagnostic only;
- status labels that match the evidence level.

Small smoke tests are useful for code health, but they are never main evidence.

## Official External Baseline Guardrails

1. An `official` filename, wrapper, plan row, or `--plan_stage run` is not
   official evidence. Only provenance with `implementation_status=official_completed`,
   `blockers=[]`, and passing score/provenance/import/paired-test gates can be
   called an official baseline result.
2. Current `main_run_*_official_same_candidate_adapter.py` files are thin
   wrappers. If the unified runner writes `official_blocked`,
   `runner_support_level=inspect_scaffold`, or blocker
   `run_stage_not_implemented_for_method`, do not import the row as
   `completed_result`, do not put it in a main table, and do not call it an
   official reproduction.
3. `style_adapter_only` and `partial_official_adapter_exists` are not official
   main-table rows. `*_style_*`, scaffold scorers, hash/centroid embeddings, and
   adapter smoke-test scores are supplementary/pilot/protocol checks only.
4. Official rows must come from the pinned repo/commit in
   `configs/official_external_baselines.yaml`. Missing repos, missing commits,
   dirty/unverified checkouts, and commit mismatches are blockers.
5. Preserve the official architecture, loss/objective, train/eval entrypoint,
   adapter/representation/graph/intent/identifier module, and scoring head. The
   allowed changes are data adapters, Qwen3-8B representation bridge, and exact
   same-candidate score export.
6. Provenance is not "there is a file." Required fields must be filled or
   explicitly marked `none` with a reason: official entrypoint, repo commit,
   default hparam source, overrides/reason, adaptation mode,
   adapter/checkpoint path or not-applicable reason, task sources, score path,
   score sha256, and score coverage.
7. The plan generator is not a truth source. `implementation_status` passed on
   the CLI cannot override blocked provenance.
8. Score gates must pass before import: exact `source_event_id,user_id,item_id`
   key match to `candidate_items.csv`, no missing/extra/duplicate keys, finite
   numeric scores, and `score_coverage_rate=1.0`. Candidate order, row index,
   constant scores, or tie-only scaffolds cannot be method scores.
9. Full-catalog metrics from official repos do not enter same-candidate main
   tables. Use official repos to produce method-native candidate scores; compute
   final metrics through `main_import_same_candidate_baseline_scores.py`.
10. If documents, runner output, score files, and provenance conflict, use the
    most conservative status: blocked > partial > style/scaffold >
    official_completed.

## Multi-Agent Collaboration

Use multi-agent work for broad research/engineering tasks. At minimum:

- one implementation/engineering agent;
- one literature/protocol scout when claims or baselines are involved;
- one reviewer/auditor agent for overclaim, fairness, and table eligibility.

Each sub-agent handoff should report:

- milestone touched;
- files inspected or changed;
- claim status and table eligibility;
- blockers;
- server commands needed;
- whether results can enter main, supplementary, or diagnostic tables.

Reviewer/auditor findings can veto wording and table inclusion. Do not average
away a serious reviewer objection.

## Server Collaboration

Agents normally cannot see the server. Do not guess server state. Give the user
copy-paste commands, then use the pasted output to decide the next action.

Ask for or provide commands that reveal:

```bash
git status --short --branch
python main_project_readiness_check.py
python main_audit_official_fairness_policy.py
python main_audit_official_external_repos.py
tail -n 80 <LOG>
ps -p $(cat <PID_FILE>) -o pid=,etime=,cmd=
ls -lh <expected output paths>
```

If a long job is needed, use `nohup` or the runbook pattern. Record the log path
and PID path. The user will paste logs or errors; update code/docs and push
fixes as needed.

## GitHub And Documentation Hygiene

After meaningful code/config/doc changes:

1. run the relevant smoke/readiness checks;
2. update canonical docs if the plan, status, or command surface changed;
3. stage only related files;
4. commit with a clear milestone/status message;
5. push to GitHub.

Do not push `outputs/` bulk artifacts, private keys, raw logs, model weights, or
large local data. Push code, configs, manifests, provenance schema, and concise
docs. If generated summaries are important but ignored by git, describe the
server path and regeneration command instead.

## Change Discipline

Do real read/modify/verify work. Do not only add Markdown when code/config is
needed, and do not only add code when the claim boundary or server runbook has
changed. Use `rg` first, inspect existing patterns, preserve user changes, and
avoid broad deletions unless they are explicitly requested and audited.

When in doubt, downgrade the claim, not the evidence standard.
