# Submission Source Package Rebuild Audit

Generated: 2026-06-14T21:21:55.473736+00:00

- OK: `false`
- Submission source package rebuild ready: `false`
- Final submission ready: `false`
- Work dir: `artifacts\submission_source_package_rebuild_20260614\work`
- Verified files: `0`
- PDF: `None`, `None` pages, `None` bytes
- BibTeX warnings: `None`
- Overfull hbox warnings: `None`
- Underfull hbox/vbox warnings: `None` / `None`

## Commands

- None

## Remaining Blockers

- ProMax final ACM page range and ACM/Crossref registry visibility must be rechecked immediately before submission.
- Final submission package still needs the external submission-target-specific formatting pass.
- promax:final_page_range_missing_in_bib
- promax:crossref_registry_not_visible:status=404
- promax:doi_resolver_not_visible:status=404
- Final manual submission-system metadata/format checklist is not closed.

## Failures

- `source_package_not_ok`
- `source_package_not_ready`
- `source_package_failures_not_empty_or_missing`
- `copied_manifest_missing_files`

## Warnings

- None

## Next Actions

- Treat this as a local rebuildability gate, not as final submission readiness.
- Re-run source package staging and this rebuild audit after any paper/source/BibTeX/figure/PDF change.
- Keep final_submission_ready=false until ProMax proceedings metadata and private manual submission-system gates close.
