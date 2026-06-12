# Citation Audit Repair Report

- Generated UTC: `2026-06-12T03:50:00Z`
- Skill used: `aris-citation-audit`
- Manuscript: `Paper/main.tex`
- Bibliography: `Paper/references.bib`
- BibTeX log: `Paper/main.blg`
- Overall citation health: `MINOR_ISSUES_AFTER_REPAIR`

## Completeness

| Category | Expected citation | Present? | Priority |
|---|---|---:|---|
| Official baseline | LLM2Rec | Yes | Satisfied |
| Official baseline | LLMEmb | Yes | Satisfied |
| Official baseline | LLM-ESR | Yes | Satisfied |
| Official baseline | IRLLRec | Yes | Satisfied |
| Official baseline | ProEx | Yes | Satisfied |
| Official baseline | ProMax | Yes | Satisfied |
| Official baseline | ELMRec | Yes | Satisfied |
| Official baseline | RLMRec | Yes | Satisfied |
| Sequential recommender foundation | SASRec | Yes | Satisfied |
| Calibration foundation | Guo et al.; Niculescu-Mizil and Caruana; Platt; Zadrozny and Elkan | Yes | Satisfied |
| LLM uncertainty | Kadavath et al.; Tian et al.; Lin et al. | Yes | Satisfied |
| Uncertainty foundations | Bayesian deep learning; conformal prediction | Yes | Satisfied |
| Same-candidate/sampled evaluation | Krichene and Rendle | Yes | Satisfied |

## Fairness

- All eight methods in the main official baseline set are cited and listed in
  `Paper/sections/experiments.tex`.
- The related-work text distinguishes embedding, sequential, profile, graph,
  and intent-based baselines without claiming they are weak or obsolete.
- No `Anonymous`, `TODO`, or placeholder author entries remain in
  `Paper/references.bib`.
- Remaining fairness caution: ProEx and ProMax are 2026 accepted/future
  proceedings entries as of 2026-06-12; re-check final proceedings metadata
  before submission.

## Recency

- Total active bibliography entries: `19`
- Citations from 2024-2026: `8` (`42.1%`)
- Citations from 2022-2026: `11` (`57.9%`)
- Most recent citation year: `2026`
- Oldest citation year: `1999`
- Recency verdict: `Good`

## BibTeX Quality

- `bibtex main`: PASS
- `Paper/main.blg`: `warning$ -- 0`
- Placeholder/anonymous bibliography authors: `0`
- Unresolved citation keys in current log: `0`
- Unused old placeholder references removed from active bibliography.

## Remaining Actions

1. Re-run this audit after any new related-work paragraph or baseline is added.
2. Spot-check ProEx/ProMax final proceedings metadata before submission because
   the manuscript date is 2026-06-12 and those entries may still be evolving.
3. Continue manuscript section review; citation repair alone does not make the
   paper submission-ready.
