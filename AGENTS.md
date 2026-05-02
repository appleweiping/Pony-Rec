# AGENTS.md

This repository is a top-conference research codebase for uncertainty-aware LLM4Rec.

Do not implement toy-only code except under tests/ or examples/.
Do not fabricate results.
Do not mark smoke/mock outputs as paper evidence.
Do not preserve old code if it is duplicated, leaky, untestable, or harmful.
Prefer clean modular packages under src/ over many one-off main_*.py files.
All experiments must be config-driven, reproducible, and seedable.

Validation data may be used for calibration and model selection.
Test data must never be used for fitting calibrators, selecting hyperparameters, pruning training data, or designing reranking thresholds.

Raw data must be the source of truth.
Existing processed data may be used only after verification.
Data splits must be deterministic and leakage-free.

DeepSeek is the primary API backend.
API experiments must be cached, resumable, and log raw responses, token usage, latency, retries, failures, and config hashes.
Mock backend is only for tests and smoke runs.
Distinguish artifact classes clearly:
- smoke: tiny mock/local pipeline checks, never paper evidence
- pilot: small real-backend cost and failure-mode checks
- diagnostic: debugging/analysis outputs
- paper-result: full raw-data, approved-config, non-mock evidence

Every result file must include:
- dataset
- domain
- split
- seed
- method
- model/backend
- prompt ID
- config hash
- timestamp

Before completing any task:
- run pytest
- run a small end-to-end smoke experiment when relevant
- update README/docs if commands change
- report exact commands and outputs
- report remaining blockers honestly