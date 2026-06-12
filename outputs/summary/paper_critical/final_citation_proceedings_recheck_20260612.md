# Final Citation Proceedings Recheck

ARIS skill: `aris-citation-audit`

- Generated UTC: `2026-06-12T05:17:23Z`
- Overall citation health: `HEALTHY_WITH_PROCEEDINGS_PAGE_CAUTION`
- Final submission ready: `false`

## Automated Checks

- Cited keys: `21`
- Bibliography entries: `21`
- Missing citation keys: `0`
- Uncited bibliography entries: `0`
- Placeholder hits: `0`

## Proceedings Metadata Recheck

### ProEx

- Status: `proceedings_metadata_retained_with_booktitle_volume_detail`
- DOI: `10.1145/3770854.3780284`
- Venue: `Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1`
- Pages: `1940--1951`

Repair applied: changed the ProEx BibTeX booktitle to include `V.1`.

### ProMax

- Status: `upgraded_from_arxiv_misc_to_sigir_2026_inproceedings_with_acm_doi`
- DOI: `10.1145/3805712.3809600`
- Venue: `Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval`
- `numpages`: `11` from arXiv metadata
- arXiv: `2604.26231`

Repair applied: changed ProMax from arXiv-only `@misc` to SIGIR 2026
`@inproceedings`, added ACM DOI `10.1145/3805712.3809600`, and retained the
arXiv eprint fields. The BibTeX uses `numpages = {11}` until final ACM page
ranges are visible.

## Source Notes

- The arXiv API reports ProMax as `11 pages, 8 figures, accepted by SIGIR
  2026`.
- The arXiv HTML page exposes DOI `10.1145/3805712.3809600`.
- ACM DOI pages were Cloudflare-gated in this environment.
- Crossref returned `404` for the checked 2026 ACM DOI records, so final
  submission should still verify ACM-visible pages and registry details.

## Remaining Cautions

- Do not mark the paper final-submission-ready from this audit alone.
- Confirm ProMax final ACM page range and ACM/Crossref registry details immediately
  before submission.
- Confirm ProEx ACM-visible metadata immediately before submission because
  Crossref returned `404` in this environment.
- This citation audit does not replace final full-panel manuscript review.

## Verification

- LaTeX: `pdflatex -> bibtex -> pdflatex -> pdflatex` passed.
- BibTeX: `warning$ -- 0`.
- PDF: `Paper/main.pdf`, `546621` bytes.
