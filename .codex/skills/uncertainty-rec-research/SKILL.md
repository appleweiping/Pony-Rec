---
name: uncertainty-rec-research
description: "Use when Codex is asked to work on the Task-Grounded Uncertainty recommendation research repository: code changes, experiment or baseline implementation, server commands, claim and table wording, C-CRP or SRPD work, same-candidate evaluation, official external baselines, project readiness checks, or research/audit reviews."
---

# Uncertainty Rec Research

## Purpose

Use this skill as the project-local operating guide for the Task-Grounded
Uncertainty recommendation repository. Treat repository docs as the source of
truth; this skill tells you which evidence rules and files to load before
acting.

## Start Every Task

1. Locate the repository root, usually the current working directory containing
   `AGENTS.md` and `README.md`.
2. Read these first, before changing code, docs, configs, experiments, or
   claims:

   ```text
   README.md
   docs/milestones/README.md
   docs/paper_claims_and_status.md
   docs/top_conference_review_gate.md
   docs/server_runbook.md
   ```

3. For complex tasks, run an `rg` discovery pass for the subsystem instead of
   relying only on the fixed first-read list. Cover project route, task plan,
   implementation details, and execution/evidence rules.
4. Check `git status --short --branch` before edits and preserve unrelated user
   changes.

## Route By Task

- Claims, paper text, tables, or result status: read
  `docs/paper_claims_and_status.md`, `docs/top_conference_review_gate.md`, and
  the relevant `docs/milestones/M*.md`.
- Official external baselines: read `docs/baseline_protocol.md`,
  `OFFICIAL_EXTERNAL_BASELINE_UPGRADE_PLAN_2026-05-07.md`,
  `configs/official_external_baselines.yaml`, and
  `PROJECT_LINEAGE_AND_FILE_INDEX_2026-05-06.md`.
- C-CRP, Shadow, or task-grounded uncertainty: read `docs/shadow_method.md`,
  relevant `configs/shadow/*.yaml`, and the candidate-score importer/exporter
  scripts touched by the task.
- SRPD or LoRA training: read the formal SRPD configs in `configs/srpd/` and
  `configs/lora/`, plus `docs/server_runbook.md` for leakage, teacher, and
  weighted-loss gates.
- Server or long-running experiment work: use `docs/server_runbook.md` as the
  command source. Do not guess server state; ask for pasted logs, PIDs, audit
  output, and file listings.
- Reviews or audits: lead with risks, table eligibility, overclaiming,
  fairness, leakage, reproducibility, and missing tests.

## Evidence Rules

- Keep the defended claim narrow: task-grounded calibrated uncertainty improves
  controlled candidate ranking/reranking reliability under same-schema
  evaluation.
- Do not expand results into full-catalog recommender SOTA, generative-title
  recommendation, LoRA distillation, or universal winner claims unless their
  own protocols and audits are completed.
- Main-table rows require `completed_result`, a valid status label, same
  train/valid/test discipline, exact same-candidate score coverage, finite
  scores, provenance, candidate audit, and paired tests.
- Official external-baseline rows require pinned official or official-code-level
  repos, preserved official algorithm/loss/scoring head, Qwen3-8B policy
  provenance, `implementation_status=official_completed`, `blockers=[]`, exact
  score coverage, import, and paired-test gates.
- Treat `style_adapter_only`, `partial_official_adapter_exists`, scaffold
  scorers, hash/centroid embeddings, and `run_stage_not_implemented_for_method`
  as supplementary, partial, or blocked evidence.
- C-CRP is the main internal task-grounded uncertainty method only after
  validation-only selection, exact score export, same-schema import, score
  degeneracy audit, and paired tests.
- SRPD is a trainable framework/ablation line. Require validation-side teacher
  data, no test-derived teachers, leakage audit, sample weights when claimed,
  exact score export, and `same_schema_internal_ablation` unless all stronger
  gates are explicitly completed.

## Server Workflow

When server state matters, give copy-paste commands and continue only from the
user's pasted output. Useful probes include:

```bash
git status --short --branch
python main_project_readiness_check.py
python main_audit_official_fairness_policy.py
python main_audit_official_external_repos.py
tail -n 80 <LOG>
ps -p $(cat <PID_FILE>) -o pid=,etime=,stat=,cmd=
ls -lh <expected output paths>
```

For storage-heavy official baselines, prefer one method-domain production loop:
run one row, verify unblocked provenance and exact coverage, import multi-k
metrics, package a lightweight evidence artifact, have the user confirm a local
copy, then clean only documented intermediates before starting the next domain.

## Change Discipline

- Use `rg` and existing local patterns before editing.
- Keep changes scoped to the requested subsystem and avoid moving historical
  files during active server work.
- Do not push bulk `outputs/`, raw logs, datasets, model weights, checkpoints,
  private keys, or local-only artifacts.
- Update canonical docs when the plan, status labels, command surface, claim
  boundary, or server runbook changes.
- Run the relevant smoke/readiness checks after meaningful changes. For skill
  edits, run the skill validator; for project changes, also run
  `python main_project_readiness_check.py` when feasible.
- If complex work calls for multiple research roles, cover implementation,
  literature/protocol, and reviewer/auditor perspectives. Use actual subagents
  only when the active environment and user request allow them; otherwise do the
  same checks locally and report blockers explicitly.

## Final Handoff

End substantial work with:

- what changed or was learned;
- files inspected or changed;
- claim status and table eligibility;
- blockers or missing evidence;
- next server command, code/doc action, audit, or stopping condition.

When enough gates are complete for the defended claim, say that the experiment
phase is basically closed and the project should move to writing and artifact
packaging. When gates remain, name the minimum remaining gates, not an
open-ended wishlist.
