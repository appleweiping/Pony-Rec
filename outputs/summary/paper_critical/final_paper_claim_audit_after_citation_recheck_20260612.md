# Final Paper-Facing Claim Audit

- Generated UTC: `2026-06-12T05:27:49+00:00`
- Verdict: `READY_FOR_MANUSCRIPT_LEVEL_CLAIM_AND_CITATION_AUDIT`
- Evidence ready for drafting: `True`
- Final submission ready: `False`
- Will start experiment: `False`

## Evidence Gates

- `comparison_ready`: `True`
- `observation_ready`: `True`
- `component_ready`: `True`
- `hyperparameter_ready`: `True`
- `framework_ready`: `True`

## Claim Status Counts

- `CONTRADICTED`: `2`
- `SUPPORTED`: `6`
- `UNSUPPORTED`: `2`

## Supported Or Allowed Claims

- `C1` C-CRP v3 ranks first in the four tested same-candidate new-domain certificates against eight official-code-level baselines.
- `C2` The paper-facing ledger is complete for the declared four-domain same-candidate comparison.
- `C3` Use as descriptive motivation evidence that C-CRP uncertainty stratifies same-candidate ranking reliability.
- `C4` Report leave-one-component-out diagnostics; several components are neutral, weak, redundant, or mixed.
- `C6` Eta and weight-grid choices are stable within the preregistered tolerance on NDCG@10 across the four domains.
- `C8` Use the figure to explain the pipeline and where uncertainty/calibration and the ranking-family risk term enter.

## Overclaim Guards

- `C5` `CONTRADICTED`: necessary; essential; each component contributes; removing any component hurts
- `C7` `CONTRADICTED`: eta must be positive; risk penalty is necessary; risk penalty uniformly improves
- `C9` `UNSUPPORTED`: submission-ready; final READY; cleared for camera-ready
- `C10` `UNSUPPORTED`: full-catalog SOTA; universal cross-domain winner beyond tested same-candidate domains

## Citation Audit

- Status: `READY_FOR_MANUSCRIPT_AND_BIBTEX_AUDIT`
- Verdict: `NEEDS_MANUAL_REFERENCE_COMPLETENESS_CHECK`
- Manuscript files: `1`
- BibTeX files: `1`

Must-add citation categories before final submission:
- LLM-based recommendation and sequential recommendation foundations
- uncertainty estimation, calibration, and selective/risk-aware prediction
- all eight official baselines used in the main same-candidate table
- same-candidate/reranking evaluation protocol and paired significance testing
- recent 2025-2026 LLM recommender and uncertainty-aware recommender work

## Failures

- none

## Next Actions

- Draft the paper using only SUPPORTED allowed wording and the explicit forbidden-wording guardrails.
- Run ARIS paper-claim-audit and aris-citation-audit on the existing manuscript and bibliography, then revise text to the allowed wording in this audit.
- Run GPT-5.5/Codex review on each written section before calling the paper submission-ready.
