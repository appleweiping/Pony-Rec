# OPENCODE.md — Uncertainty Project

Read `AGENTS.md` first. It is the authoritative operating contract.

This file provides OpenCode-specific context for operating within the Uncertainty project.

## Project Identity

**Title**: Task-Grounded Uncertainty for LLM-based Recommendation
**Core Claim**: Task-grounded calibrated uncertainty improves controlled candidate ranking/reranking reliability under same-schema evaluation.
**Stage**: Between M4 (baseline system) and M5 (four-domain 100-neg validation)
**Server**: `pony-rec-gpu` (SSH key-based auth, directly accessible by agents)
- SSH command: `ssh pony-rec-gpu`
- Host: `125.71.97.70:15302`, User: `ajifang`
- GPU: NVIDIA RTX 4090 (49GB VRAM)
- Server project path: `~/projects/pony-rec-rescue-shadow-v6`
- Local project path: `D:\Research\Uncertainty`

## Methods

- **C-CRP** (Calibrated Candidate Relevance Posterior): Main method. Task-grounded uncertainty with boundary ambiguity, calibration gap, and evidence insufficiency.
- **SRPD**: Trainable framework/ablation line. Supplementary unless full protocol passes.

## Baselines (Official External)

9 methods configured: ELMRec, IRLLRec, LLM2Rec, LLMEmb, LLMESR, ProEx, ProMax, RLMRec, SetRec.
All use Qwen3-8B backbone, official code, default hyperparameters, same-candidate protocol.

## Key Files to Read Before Acting

1. `AGENTS.md` — operating contract, baseline guardrails, experiment standards
2. `docs/milestones/README.md` — M0-M6 milestone map
3. `docs/paper_claims_and_status.md` — frozen claims and status rules
4. `docs/top_conference_review_gate.md` — reviewer gate
5. `docs/server_runbook.md` — server execution patterns
6. `configs/official_external_baselines.yaml` — baseline configurations

## Source Structure

```
src/
  shadow/       - C-CRP implementation (main method)
  uncertainty/  - Uncertainty estimation (calibration, confidence, consistency)
  baselines/    - Baseline implementations + official_runner/ (9 external)
  methods/      - Ranking methods (uncertainty ranker, reranker, reweighting)
  eval/         - Evaluation metrics and protocols
  training/     - SRPD/LoRA training
  data/         - Data loading
  backends/     - LLM backend interfaces
  analysis/     - Result analysis
```

## OpenCode Operating Rules

### What OpenCode Can Do Here

- Read and modify code, configs, docs
- Run local Python scripts (preprocessing, analysis, comparison builders)
- **Directly execute commands on the GPU server via `ssh pony-rec-gpu "<command>"`**
- Monitor running experiments, check logs, tail outputs on server
- Audit results, provenance, and table eligibility
- Build comparison tables from imported scores

### What OpenCode Cannot Do Here

- Declare a baseline "official_completed" without full provenance audit

### Server Interaction Pattern

Agents now have direct SSH access to the server:
```
ssh pony-rec-gpu "<command>"
```
- Can run training/inference, monitor GPU, check logs, manage files directly
- Use `nohup` for long-running tasks, track PIDs
- Check progress: `wc -l` on output files, `nvidia-smi`, `ps aux | grep python`

### Evidence Discipline

- Only `completed_result` rows enter main tables
- Score gates: exact key match to candidate_items.csv, no missing/extra/duplicate, finite scores, coverage=1.0
- Paired statistical tests required before claiming significance
- When in doubt, downgrade the claim, not the evidence standard

## Coordination with CC Family

When Opus or Sonnet reviews this project via Claude Code:
- They read `AGENTS.md` and `.codex/skills/uncertainty-rec-research/SKILL.md`
- They follow the same milestone map and evidence rules
- Their reviews can veto table inclusion and wording

OpenCode and CC family share the same operating contract. The only difference is the entry point and execution environment.
