# Pre-Submission Refresh Freshness Audit

Generated: 2026-06-13T02:51:34.496421+00:00

- OK: `true`
- Refresh artifact fresh: `true`
- Final submission ready: `false`
- Refresh JSON: `outputs\summary\paper_critical\pre_submission_gate_refresh_20260613.json`
- Refresh verdict: `LOCAL_PACKAGE_READY_BUT_EXTERNAL_MANUAL_OR_REVIEW_BLOCKED`
- Input fingerprints checked: `23`
- Generated step files checked: `14`
- Input mismatches: `0`
- Generated output mismatches: `0`

## Git Provenance Policy

Git HEAD is provenance for the code/input state that generated the refresh. Freshness is decided by current file fingerprints and generated gate hashes, because committing generated artifacts necessarily changes HEAD.

- Refresh generation HEAD: `e509a070a26747f01668ceee6047c7eea00e6da7`
- Current HEAD: `e509a070a26747f01668ceee6047c7eea00e6da7`
- HEAD changed since refresh generation: `false`

## Failures

- None

## Warnings

- `refresh_recorded_tracked_dirty_inputs_before_generation`

## Mismatched Files

- None

## Remaining Blockers

- promax:final_page_range_missing_in_bib
- promax:crossref_registry_not_visible:status=404
- promax:doi_resolver_not_visible:status=404
- Final manual submission-system metadata/format checklist is not closed.
- ProMax final ACM page range and ACM/Crossref registry visibility must be rechecked immediately before submission.
- confirm_external_proceedings_metadata:external_proceedings_metadata_ready_not_closed
- manual_submission_system_items_not_confirmed
- external_proceedings_metadata_not_ready
- manual_submission_system_not_ready
- review_panel_coverage_not_complete
- promax:crossref_registry_not_visible
- promax:doi_resolver_not_visible
- explicit_claude_opus_review

## Next Actions

- If any fingerprint mismatch is present, rerun main_refresh_pre_submission_gates.
- Use final_submission_gate as the semantic readiness summary after freshness passes.
- Keep final_submission_ready=false until external proceedings metadata and private manual submission-system checks are closed.
