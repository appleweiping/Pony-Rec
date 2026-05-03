# AGENTS.md

This repository is a top-conference research codebase for uncertainty-aware LLM4Rec (working name: CARE-Rec).
All work must serve the central claim: recommendation quality is not enough; confidence reliability and confidence-driven exposure risk must be modeled together.

## 0) Required reading (current roadmap docs)

Always align implementations and reports with:

- `docs/CODEX_TOPCONF_UNCERTAINTY_LLM4REC_PLAN.md`
- `docs/METHOD.md`
- `docs/PILOT_CARE_LORA_DEBUG.md`
- `docs/PILOT_CARE_RERANK.md`
- `docs/CALIBRATION_DIAGNOSTICS.md`
- `docs/PILOT_100U_PROTOCOL.md`
- `docs/PILOT_DEEPSEEK_100U.md`
- `docs/PILOT_CARE_RERANK_100U.md`
- `docs/PILOT_RECBOLE_100U.md`
- `docs/PILOT_CARE_LORA_100U.md`

## 1) Non-negotiable objective

Build a coherent framework, not a loose stack of "LLM + calibration + reranking".
Each implementation round should support at least one of:

1. Confidence-correctness reliability measurement.
2. Confidence-popularity and echo-chamber risk diagnosis.
3. CARE inference policy (expected utility under uncertainty and exposure risk).
4. CARE-LoRA training policy (uncertainty-aware, recommendation-specific).
5. Reproducible protocols and reviewer-safe evidence.

Do not write paper claims before experiment evidence is ready.

## 2) Hard red lines

- Do not fabricate data, metrics, logs, or conclusions.
- Do not mark smoke/mock outputs as paper evidence.
- Do not use test split for calibrator fitting, threshold tuning, hyperparameter/model selection, sample pruning, or rerank weight design.
- Do not silently mix incompatible protocols (candidate-set eval vs full-sort eval).
- Do not claim exact reproduction when an approximation is used; label it `Approx-*`.
- Do not preserve duplicated, leaky, untestable, or harmful legacy code.

## 3) Data and split protocol rules

- Raw data is source-of-truth when available.
- At current stage, processed source tables are accepted input only after verification.
- Current official domains under `data/processed`:
  - `amazon_beauty`
  - `amazon_books`
  - `amazon_electronics`
  - `amazon_movies`
- Each domain should have exactly:
  - `interactions.csv`
  - `items.csv`
  - `popularity_stats.csv`
  - `users.csv`
- Recompute splits/candidates/popularity from source tables; do not reuse old split artifacts as evidence.
- Splits must be deterministic, seedable, and leakage-free.
- Preferred split protocol now: per-user temporal leave-one-out.

### Leakage prevention

- Validation can be used for calibration and model selection.
- Test is final evaluation only.
- Negatives must exclude target item and user history.
- Popularity features must be train-only when required by protocol.

## 4) Methods boundary (what counts as in-scope)

### CARE inference framing

Expected utility should combine relevance, calibrated reliability, uncertainty penalty, and exposure-harm penalty.
If implemented as weighted terms, weights must be ablated and documented.

### CARE-LoRA framing

Training policy should distinguish:

- high-confidence correct anchors,
- high-confidence wrong head-item risks,
- low-confidence correct tail signals (not always noise),
- invalid outputs.

Do not collapse this into naive "drop uncertain samples" only.

## 5) Baseline and comparison rules

- Baselines should include classical/sequential + API LLM + local LoRA variants where feasible.
- Confidence-like signals for non-LLM baselines can come from score margin/entropy/calibrated probability.
- Keep candidate sets identical for methods compared under the same protocol.
- If RecBole internal full-sort metrics are reported, separate them clearly from sampled candidate-set results.

## 6) Backend and runtime rules

- DeepSeek is primary API backend.
- API experiments must be cached, resumable, and auditable.
- Log raw responses and structured metadata: token usage, latency, retries, failures, and config hash.
- Mock backend is only for unit tests and smoke checks.

## 7) Artifact classes and evidence policy

All outputs must carry explicit artifact class:

- `smoke`: tiny local/mock pipeline checks, never paper evidence.
- `pilot`: small-scale real-backend feasibility/failure-mode checks.
- `diagnostic`: debugging/analysis outputs.
- `paper-result`: full protocol, approved config, non-mock evidence.

Until explicitly promoted, new outputs default to:

- `is_paper_result=false`
- pilot/diagnostic class as appropriate.

## 8) Required metadata for result files

Every result artifact must include enough metadata for replay and audit:

- dataset/domain
- split protocol
- candidate protocol
- seed
- method name
- model/backend
- prompt ID/version (if LLM prompting involved)
- config hash
- timestamp
- run type (`smoke` / `pilot` / `full`)
- backend type (`api` / `baseline` / `lora`)
- source protocol (`processed_source_reprocess` / `raw_rebuild`)

## 9) Execution order for current project phase

Given current repo status, follow this staged priority:

1. LoRA debug pass (small-scale, one-domain, short-step).
2. Uncertainty feature extraction.
3. Calibration + diagnostics.
4. CARE rerank pilot.
5. CARE-LoRA pilot.

Do not jump to full-scale experiments before each prior stage is stable and documented.

## 10) Code quality and project structure

- Prefer modular implementation under `src/` over many one-off entry scripts.
- Keep all experiments config-driven and seedable.
- Add/maintain tests for leakage, calibration boundaries, and metric correctness.
- Keep docs in sync with command or protocol changes.
- Keep changes minimal but complete; remove dead paths when replacing old logic.

## 11) Task completion checklist (must pass before closing a task)

- Run `pytest`.
- Run a small end-to-end smoke or pilot check when relevant.
- Update docs/README if behavior, commands, or protocol changed.
- Report exact commands executed and key outputs.
- Report blockers and limitations honestly.

If any checklist item is skipped, explain why explicitly.