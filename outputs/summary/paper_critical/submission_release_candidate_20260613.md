# Submission Release-Candidate Packet

Generated: 2026-06-12T22:37:09.997755+00:00

- Verdict: `LOCAL_RELEASE_CANDIDATE_READY_FINAL_BLOCKED`
- OK: `true`
- Local release candidate ready: `true`
- Final submission ready: `false`
- Final-ready source: `outputs\summary\paper_critical\final_submission_gate_20260613.json`

## Status Boundary

- Local RC: Repo-side audited package, source rebuild, metadata packet, manual checklist, external audit health, and freshness index are internally consistent.
- Final submission: Submission may proceed only when the final submission gate also reports true; external proceedings metadata and private submission-system confirmation remain final blockers.

## Gate Summary

- `final_submission_gate`: ok=`true`, ready=`true`, final_ready=`false`, path=`outputs\summary\paper_critical\final_submission_gate_20260613.json`
- `pre_submission_refresh_freshness`: ok=`true`, ready=`true`, final_ready=`false`, path=`outputs\summary\paper_critical\pre_submission_gate_refresh_freshness_20260613.json`
- `submission_source_package`: ok=`true`, ready=`true`, final_ready=`false`, path=`outputs\summary\paper_critical\submission_source_package_20260613.json`
- `submission_source_package_rebuild`: ok=`true`, ready=`true`, final_ready=`false`, path=`outputs\summary\paper_critical\submission_source_package_rebuild_20260613.json`
- `submission_metadata_packet`: ok=`true`, ready=`true`, final_ready=`false`, path=`outputs\summary\paper_critical\submission_metadata_packet_20260613.json`
- `manual_submission_checklist`: ok=`true`, ready=`true`, final_ready=`false`, path=`outputs\summary\paper_critical\manual_submission_checklist_20260613.json`
- `external_proceedings_metadata`: ok=`true`, ready=`n/a`, final_ready=`false`, path=`outputs\summary\paper_critical\external_proceedings_metadata_recheck_20260613.json`

## Package Summary

- Source files: `21`
- Source bytes: `652763`
- Source manifest sha256: `795aa65895cf06452a3cf48b6519bece473cd61fc2ee33c2becb7c2d8b2e79cc`
- Rebuild PDF pages: `9`
- Rebuild PDF bytes: `546716`
- BibTeX warnings: `0`
- Overfull hbox warnings: `0`

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

- `final_submission_gate:submission_package:underfull_layout_warnings:hbox=6,vbox=8`
- `final_submission_gate:submission_source_package_rebuild:rebuilt_underfull_layout_warnings:hbox=6,vbox=8`
- `final_submission_gate:external_proceedings_metadata:proex:crossref_not_visible:status=404`
- `final_submission_gate:external_proceedings_metadata:proex:doi_resolver_not_visible:status=404`
- `final_submission_gate:external_proceedings_metadata:proex:crossref_discovery_alternate_doi_candidates_present`
- `final_submission_gate:external_proceedings_metadata:promax:crossref_discovery_alternate_doi_candidates_present`
- `submission_source_package_rebuild:rebuilt_underfull_layout_warnings:hbox=6,vbox=8`
- `external_proceedings_metadata:proex:crossref_not_visible:status=404`
- `external_proceedings_metadata:proex:doi_resolver_not_visible:status=404`
- `external_proceedings_metadata:proex:crossref_discovery_alternate_doi_candidates_present`
- `external_proceedings_metadata:promax:crossref_discovery_alternate_doi_candidates_present`

## Next Actions

- Use this packet as a local handoff index, not as final submission approval.
- Resolve ProMax final page range and DOI/Crossref visibility, then rerun the external metadata audit and refresh stack.
- Complete private submission-system fields with an untracked confirmation JSON, then rerun the manual checklist and refresh stack.
- Rerun the refresh, freshness audit, final gate, and release-candidate packet after any paper/source/BibTeX/package change.
