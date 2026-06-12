# Final Submission Gate

Generated: 2026-06-12T10:14:25.092376+00:00

- Verdict: `LOCAL_PACKAGE_READY_BUT_EXTERNAL_OR_MANUAL_BLOCKED`
- OK: `true`
- Local artifact gates OK: `true`
- External proceedings metadata ready: `false`
- Manual submission system ready: `false`
- Final submission ready: `false`

## Gate Summary

- `submission_package`: ok=`true`, ready=`true`, final_ready=`false`, path=`outputs\summary\paper_critical\submission_package_audit_20260612.json`
- `submission_metadata_packet`: ok=`true`, ready=`true`, final_ready=`false`, path=`outputs\summary\paper_critical\submission_metadata_packet_20260612.json`
- `external_proceedings_metadata`: ok=`true`, ready=`false`, final_ready=`false`, path=`outputs\summary\paper_critical\external_proceedings_metadata_recheck_20260612.json`
- `manual_submission_checklist`: ok=`true`, ready=`false`, final_ready=`false`, path=`outputs\summary\paper_critical\manual_submission_checklist_20260612.json`

## Remaining Blockers

- ProMax final ACM page range and ACM/Crossref registry visibility must be rechecked immediately before submission.
- promax:final_page_range_missing_in_bib
- promax:crossref_registry_not_visible:status=404
- promax:doi_resolver_not_visible:status=404
- Final manual submission-system metadata/format checklist is not closed.
- external_proceedings_metadata_not_ready
- manual_submission_system_not_ready
- confirm_external_proceedings_metadata:external_proceedings_metadata_ready_not_closed
- manual_submission_system_items_not_confirmed

## Failures

- None

## Warnings

- `submission_package:underfull_layout_warnings:hbox=6,vbox=8`
- `external_proceedings_metadata:proex:crossref_not_visible:status=404`
- `external_proceedings_metadata:proex:doi_resolver_not_visible:status=404`

## Next Actions

- Rerun external proceedings metadata recheck immediately before final submission.
- Rerun submission package, metadata packet, manual checklist, and this final gate after any paper/source/BibTeX change.
- Complete private author/COI/reviewer/declaration fields only inside the submission system.
- Keep final_submission_ready=false until external proceedings metadata and manual submission-system gates are both ready.
