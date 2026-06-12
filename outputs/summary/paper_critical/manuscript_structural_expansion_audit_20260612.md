# Manuscript Structural Expansion Audit

Status: `FIRST_STRUCTURE_EXPANSION_PASS_COMPLETE`

Final submission ready: `false`

## What Changed

- Expanded `Paper/sections/method.tex` with the reranking-scope contract, C-CRP
  posterior notation, uncertainty decomposition, risk-adjusted ranking,
  validation-only selection, and fail-closed score gates.
- Expanded `Paper/sections/experiments.tex` with the same-candidate protocol,
  Qwen3-8B fairness policy, official/default baseline policy, provenance
  requirements, paired-test family, and evidence packaging rules.
- Expanded `Paper/sections/results.tex` with an explicit uncertainty
  stratification table, observation row-count gates, component-ablation deltas,
  and hyperparameter-stability values.
- Added `Paper/tables/uncertainty_stratification.tex`.
- Updated `Paper/sections/analysis.tex` so limitations are scientific scope
  boundaries rather than the stale citation-repair blocker.
- Updated `Paper/CLAIM_MAP.md` so future agents do not repeat the obsolete
  "replace placeholder BibTeX" task.

## Verification

- Server preflight: no matching Pony/C-CRP/baseline/uncertainty Python process;
  GPU `0%`, `15 MiB / 49140 MiB`; disk about `17G` free / `92%` used.
- LaTeX: `pdflatex -> bibtex -> pdflatex -> pdflatex` succeeded.
- PDF: `Paper/main.pdf`, 8 pages, 533021 bytes.
- BibTeX: `Paper/main.blg` reports `warning$ -- 0`.
- Paper-critical tests: 66 passed.
- Readiness/bootstrap: `project_readiness_ok=True`, `bootstrap_ok=true`.
- Stale paper scan: no old calibration, old-domain, SRPD, or full-SOTA narrative
  found in `Paper/`.
- Full pytest is not used as this checkpoint gate because collection is blocked
  by pre-existing local import-path issues in historical official-runner tests;
  no paper-critical test failed.

## Remaining Blockers

- Run final ARIS paper-claim-audit on the expanded manuscript text.
- Run final ARIS citation spot-check before any submission-readiness claim.
- Run section-level top-conference review and apply its edits.
- Keep the claim boundary strict: no full-catalog SOTA, no every-component
  necessity, and no positive-risk-penalty necessity.
