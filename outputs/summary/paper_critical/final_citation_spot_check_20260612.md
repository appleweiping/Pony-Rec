# Final Citation Spot-Check

ARIS skill: `aris-citation-audit`

Overall citation health: `HEALTHY_WITH_PROCEEDINGS_METADATA_CAUTION`

Final submission ready: `false`

## Automated Checks

- Cited keys: 19
- Bibliography entries: 19
- Missing citation keys: 0
- Uncited bibliography entries: 0
- Placeholder hits: 0
- BibTeX warnings: 0 (`Paper/main.blg` reports `warning$ -- 0`)
- Unresolved citations: 0

## Completeness

Must-add citations: 0.

All eight official baselines are cited: ELMRec, IRLLRec, LLM2Rec, LLMEmb,
LLM-ESR, ProEx, ProMax, and RLMRec. Foundational categories are also covered:
sequential recommendation, calibration, LLM confidence/uncertainty, Bayesian or
distribution-free uncertainty, and sampled/same-candidate recommendation
evaluation.

## Fairness Repairs

- Reworded ELMRec as high-order interaction-aware LLM recommendation.
- Reworded IRLLRec as intent representation learning with LLMs.
- Reworded ProEx as profile extrapolation with LLMs.
- Reworded ProMax as LLM-derived profiles with distribution shaping.
- Reworded RLMRec as representation learning with LLMs, removing the incorrect
  reinforcement-learning description.

## BibTeX Repairs

- Added the arXiv DOI for ProMax.
- Added arXiv eprint/DOI fields for Lin et al. while keeping TMLR as a note to
  avoid ACM BibTeX volume/page warnings.

## External Spot-Checks

- ProEx: ACM DOI `10.1145/3770854.3780284`.
- ProMax: arXiv `2604.26231`.
- RLMRec: ACM DOI `10.1145/3589334.3645458`.
- ELMRec: ACL Anthology `2024.emnlp-main.653`.
- Lin et al.: OpenReview/TMLR plus arXiv `2205.14334`.

## Remaining Cautions

- ProEx and ProMax are 2026 entries; re-check proceedings metadata immediately
  before final submission.
- Citation spot-check does not replace section-level top-conference review.
