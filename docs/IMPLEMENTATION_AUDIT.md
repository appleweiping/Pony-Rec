# Implementation Audit

## What Exists

- A `src/` tree with useful fragments for raw Amazon loading, k-core filtering, pointwise/listwise prompting, LLM parsing, calibration, reranking, exposure diagnostics, local HF inference, and LoRA scaffolding.
- Many YAML configs under legacy groups such as `configs/data`, `configs/model`, `configs/exp`, `configs/srpd`, and `configs/lora`.
- Prompt files for pointwise, pairwise, candidate ranking, and several shadow/structured-risk variants.
- A DeepSeek wrapper via the older `src.llm` backend path, plus generic OpenAI-compatible API support.
- One existing pytest file for shadow-v6 bridge logic.

## What Is Broken Or Risky

- The main interface is fragmented across roughly 50 `main_*.py` scripts. This makes experiment provenance, reproducibility, and review difficult.
- The old data pipeline defaults to small negative counts and legacy `3-core` configs in places, while the paper protocol requires configurable 3/5/10-core and 19/99/199/499 candidates.
- Existing processed data is absent in this checkout and must not be trusted as source of truth.
- Raw Beauty/Electronics/Books/Movie files are absent in this checkout, so full-domain preprocessing cannot be verified locally.
- Legacy parsers sometimes normalize invalid outputs into usable fields. The paper requires bad outputs to remain measurable.
- Calibration code exists, but the project did not have enough tests proving validation-only fitting and test-only application.
- Backend logging/caching/resume behavior was incomplete for a paper-scale DeepSeek run.
- Baseline integration is incomplete for RecBole and several LLM4Rec baselines; approximate baselines must not be overclaimed.

## What Should Be Kept

- Existing `src/data/raw_loaders.py` ideas for Amazon JSONL field normalization.
- Existing calibration components for isotonic and Platt scaling.
- Existing local HF and LoRA scaffolding, but only behind clean config-driven CLI entrypoints.
- Existing ranking/parser work where it preserves invalid-output flags.
- Existing analysis utilities after they are connected to saved artifact metadata.

## What Should Be Deleted Or Archived

- Legacy `main_*.py` scripts should be archived after the new `python -m src.cli.*` path fully covers their use cases.
- Week-specific config names and summary scripts should be moved out of the main paper path.
- Mock/smoke outputs under `outputs/smoke_*` must never be promoted to paper evidence.

## What Was Rewritten In This Pass

- Added a clean data protocol in `src/data/protocol.py`.
- Added required backend files in `src/backends/`.
- Added prompt templates/parsers in `src/prompts/`.
- Added uncertainty estimator interface in `src/uncertainty/interface.py`.
- Added paper metrics, reranking formulas, echo-chamber diagnostics, baseline sanity runners, and CLI entrypoints.
- Added smoke fixtures and tests for leakage, deterministic candidates, parser failures, metrics, reranking, mock backend, calibration leakage guard, and LoRA data formatting.

## Missing Core Modules

- Full RecBole atomic conversion and scheduler.
- Official P5/OpenP5, CoLLM, LLM2Rec/LLMEmb, SLMRec, LLM-ESR, AGRec, CoVE, GUIDER integrations.
- Dense/BM25/collaborative retrieval candidate generators beyond lightweight BM25 sanity ranking.
- Beta calibration and temperature scaling for backend-provided logits/logprobs.
- Full uncertainty-pruned/weighted/curriculum LoRA data builders.

## Top-Conference Risks

- Novelty risk if framed as only verbalized confidence. The implemented architecture must emphasize recommendation-specific calibration, exposure, abstention, reranking, long-tail coverage, and noise-aware training.
- Leakage risk from processed data or calibration on test predictions. The new CLI explicitly fits on `valid` and applies to `test`.
- Baseline risk if approximations are presented as official methods. `docs/BASELINES.md` must track official commits and deviations.
- Evidence risk if mock/smoke artifacts are mixed with paper results. The output policy distinguishes smoke, pilot, diagnostic, and paper-result artifacts.

## Final Architecture Proposal

Use raw data as the source of truth:

`raw files -> src.cli.preprocess -> src.cli.build_candidates -> src.cli.infer -> src.cli.calibrate -> src.cli.rerank -> src.cli.evaluate -> src.cli.aggregate -> src.cli.export_paper_tables`

All experiment commands are YAML-configured. DeepSeek is the primary API backend; mock is only for tests and smoke. Local LoRA uses `src.cli.train_lora` and `src.cli.eval_lora` with server-side Transformers/PEFT dependencies.
