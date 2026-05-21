# CLAUDE.md

Read `AGENTS.md` first. It is the authoritative operating contract for this repository.

This is the Uncertainty project: Task-Grounded Uncertainty for LLM-based Recommendation.

## Quick Orientation

- **Stage**: Between M4 and M5 (four-domain 100-neg validation is the active gate)
- **Core Claim**: Task-grounded calibrated uncertainty improves controlled candidate ranking/reranking reliability under same-schema evaluation.
- **Methods**: C-CRP (main), SRPD (ablation/supplementary)
- **Baselines**: 9 official external (ELMRec, IRLLRec, LLM2Rec, LLMEmb, LLMESR, ProEx, ProMax, RLMRec, SetRec)
- **Server**: `pony-rec-gpu` (SSH accessible via `ssh pony-rec-gpu`, key-based auth configured)
  - Server project path: `~/projects/pony-rec-rescue-shadow-v6`
  - Local project path: `D:\Research\Uncertainty`
  - SSH config: `125.71.97.70:15302`, user `ajifang`
  - GPU: RTX 4090 (49GB VRAM)

## Your Role (Claude Code / Opus / Sonnet)

When invoked by Codex or OpenCode as a reviewer:
- You are **read-only**. Do not edit files, stage, commit, or push.
- Challenge rigor, novelty, technical depth, ablation completeness, baseline coverage, leakage risk, table eligibility.
- Reviewer objections remain blockers until addressed.
- Apply top-conference reviewer standards (RecSys/SIGIR/WSDM/KDD/NeurIPS).

When the user opens Claude Code directly with write scope:
- Follow `AGENTS.md` operating contract fully.
- Update canonical docs after meaningful changes.
- Run readiness checks before pushing.

## Key Docs (Read Order)

1. `AGENTS.md` — full operating contract (280 lines)
2. `docs/milestones/README.md` — M0-M6 milestone map
3. `docs/paper_claims_and_status.md` — frozen claims, status labels
4. `docs/top_conference_review_gate.md` — standing reviewer gate
5. `docs/server_runbook.md` — server execution patterns
6. `configs/official_external_baselines.yaml` — baseline configs

## Evidence Rules (Summary)

- Only `completed_result` rows enter main tables
- Score gates: exact key match, no missing/extra/duplicate, finite scores, coverage=1.0
- Paired statistical tests required for significance claims
- `official_completed` requires: pinned repo/commit, default hparams, full provenance, score coverage=1.0
- When in doubt, downgrade the claim, not the evidence standard

## Multi-Agent System

This project participates in the 6-agent collaboration system:

| Agent | Role in this project |
|-------|---------------------|
| OpenCode (像素猫) | Implementation, analysis, comparison builders, doc updates |
| Codex | Coordination, parallel execution, server command generation |
| Opus | Architecture review, complex reasoning, security/leakage audit |
| Sonnet | Code review, table eligibility checks, second opinion |
| Haiku | Quick lint, format checks |
| DeepSeek (鲸鱼) | Translation, bulk text, Chinese content |

Agent Hub daemon on port 9800 provides real-time collaboration when running.
