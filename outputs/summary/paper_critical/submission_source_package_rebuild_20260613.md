# Submission Source Package Rebuild Audit

Generated: 2026-06-12T22:37:04.172561+00:00

- OK: `true`
- Submission source package rebuild ready: `true`
- Final submission ready: `false`
- Work dir: `artifacts\submission_source_package_rebuild_20260613\work`
- Verified files: `21`
- PDF: `artifacts\submission_source_package_rebuild_20260613\work\Paper\main.pdf`, `9` pages, `546716` bytes
- BibTeX warnings: `0`
- Overfull hbox warnings: `0`
- Underfull hbox/vbox warnings: `6` / `8`

## Commands

- `pdflatex -interaction=nonstopmode -halt-on-error main.tex` -> `0`
- `bibtex main` -> `0`
- `pdflatex -interaction=nonstopmode -halt-on-error main.tex` -> `0`
- `pdflatex -interaction=nonstopmode -halt-on-error main.tex` -> `0`

## Remaining Blockers

- ProMax final ACM page range and ACM/Crossref registry visibility must be rechecked immediately before submission.
- promax:final_page_range_missing_in_bib
- promax:crossref_registry_not_visible:status=404
- promax:doi_resolver_not_visible:status=404
- Final manual submission-system metadata/format checklist is not closed.

## Failures

- None

## Warnings

- `rebuilt_underfull_layout_warnings:hbox=6,vbox=8`

## Next Actions

- Treat this as a local rebuildability gate, not as final submission readiness.
- Re-run source package staging and this rebuild audit after any paper/source/BibTeX/figure/PDF change.
- Keep final_submission_ready=false until ProMax proceedings metadata and private manual submission-system gates close.
