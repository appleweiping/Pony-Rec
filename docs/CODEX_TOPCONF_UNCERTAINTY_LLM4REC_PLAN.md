# CODEX / top-conference research engineering plan (anchor)

The full narrative plan (CARE-Rec framing, reviewer-style risks, baseline matrix, file-level roadmap) should live **in this file** in the repo so Codex/agents and humans share one source of truth.

**If you only have the plan on another machine** (for example `Downloads/CODEX_TOPCONF_UNCERTAINTY_LLM4REC_PLAN.md`), copy its contents here and commit.

## Section 13 — Codex execution order (executive summary)

1. **LoRA debug** — end-to-end Qwen + PEFT on reprocessed candidates (`run_lora_debug_reprocessed`, pilot outputs).
2. **Uncertainty feature extraction** — `python -m src.cli.run_uncertainty_probe` on `rank_predictions.jsonl`; see `src/uncertainty/features.py`.
3. **Calibration diagnostics** — reliability curves / ECE slices conditioned on popularity and head–tail (build on probe + `src/cli/evaluate` metrics).
4. **CARE rerank pilot** — reranker using extracted signals (separate CLI under `src/methods/care_*` when added).
5. **CARE-LoRA pilot** — training policy that uses uncertainty / exposure objectives (separate from vanilla LoRA SFT).

## Related docs

- `docs/PILOT_LORA_QWEN3_DEBUG.md` — completed LoRA debug pilot notes.
