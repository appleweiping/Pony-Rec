# Manuscript Claim And Citation Audit

- Generated UTC: `2026-06-12T03:46:00Z`
- Manuscript: `paper/main.tex`
- Evidence gate: `outputs/summary/paper_critical/final_paper_claim_audit_20260612.{json,md,csv}`
- Compile check: `pdflatex -> bibtex -> pdflatex -> pdflatex` completed and
  produced `paper/main.pdf` (6 pages, 504041 bytes)
- BibTeX check: `Paper/main.blg` reports `warning$ -- 0`
- Final submission ready: `false`

## Verdict

`NEEDS_SECTION_REVIEW_BEFORE_SUBMISSION`.

The rewritten manuscript now follows the current C-CRP same-candidate evidence
spine instead of the older calibration/domain narrative. Citation repair and
visible result-table supplementation are complete enough for the next strict
ARIS claim/citation audit, but the manuscript still needs section-level review
and expansion before any final submission-ready claim.

## Claim Audit

Supported in the current draft:

- C-CRP v3 ranks first against eight official-code-level baselines on Sports,
  Toys, Home, and Tools under the 10k-user/101-candidate same-candidate
  protocol.
- Each domain has 56/56 positive Holm-significant C-CRP-vs-official paired
  metric tests.
- Event-level C-CRP uncertainty stratifies ranking reliability as descriptive
  motivation evidence.
- Component ablations are complete and show weak, redundant, or mixed
  component behavior rather than universal component necessity.
- Hyperparameter evidence supports NDCG@10 stability/sensitivity for `eta` and
  the uncertainty weight grid.
- The framework figure is usable as a pipeline overview.
- Visible result tables now include the main best-baseline table, a paired-test
  summary table, a complete NDCG@10 ranking over all eight official baselines
  per domain, and the module-evidence table.

Forbidden or contradicted wording:

- Do not claim full-catalog recommender SOTA.
- Do not claim universal recommender SOTA beyond the tested same-candidate
  domains.
- Do not claim every C-CRP component is necessary.
- Do not claim positive `eta` or risk penalty is necessary or uniformly
  beneficial.
- Do not claim all-metric hyperparameter robustness.
- Do not reintroduce the older calibration/ECE or older-domain SOTA narrative
  without a fresh current evidence gate.

## Citation Audit

`REPAIRED_BIBTEX_NEEDS_FINAL_ARIS_SPOT_CHECK`.

The placeholder bibliography was repaired: the active `.bib` now contains 19
used references, no `Anonymous` author placeholders, and `bibtex main` reports
zero bibliography warnings.

- The eight official baseline entries have verified non-placeholder authors,
  venues, years, pages or source identifiers, and DOI/URL fields where
  available.
- ProEx/ProMax are 2026 accepted/future-proceedings citations as of 2026-06-12;
  re-check final proceedings metadata before submission.
- A final ARIS citation spot-check is still required before any submission
  readiness claim.

No fabricated BibTeX entries should be added.

## Review Status

- GPT-5.5 xhigh pre-rewrite sidecar review: evidence `CONDITIONAL PASS,
  8.2/10`; old manuscript `NEEDS REVISION, 6.4/10`.
- GPT-5.5 xhigh post-rewrite re-review: claim safety `8.3/10`,
  writing/top-conference readiness `7.2/10`, combined manuscript score
  `7.7/10` before citation repair and table supplementation.
- Claude reviewer: attempted twice in this session; both failed with
  `Claude CLI did not return JSON output`.

## Required Next Actions

1. Run final ARIS citation-audit/spot-check on the repaired bibliography.
2. Run a fresh ARIS paper-claim-audit on the current manuscript after citation
   and table repair.
3. Expand/rebalance the compressed six-page draft into a fuller top-conference
   manuscript.
4. Run GPT-5.5/Codex section-level review before marking the paper
   submission-ready.
