# Submission Release-Candidate Stack Refresh

Generated: 2026-06-13T05:19:42.950183+00:00

- OK: `true`
- Local release candidate ready: `true`
- Readiness scope: `local_artifacts_only`
- Blocking status: `external_manual_or_review_blocked`
- Final submission ready: `false`
- Stamp: `20260613`

## Steps

- `pre_submission_gate_refresh`: ok=`true`, ready=`true`, json=`outputs\summary\paper_critical\pre_submission_gate_refresh_20260613.json`
- `pre_submission_refresh_freshness`: ok=`true`, ready=`true`, json=`outputs\summary\paper_critical\pre_submission_gate_refresh_freshness_20260613.json`
- `submission_release_candidate`: ok=`true`, ready=`true`, json=`outputs\summary\paper_critical\submission_release_candidate_20260613.json`

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

## Failures

- None

## Warnings

- `pre_submission_gate_refresh:proex:crossref_not_visible:status=404`
- `pre_submission_gate_refresh:proex:doi_resolver_not_visible:status=404`
- `pre_submission_gate_refresh:proex:crossref_discovery_alternate_doi_candidates_present`
- `pre_submission_gate_refresh:promax:crossref_discovery_alternate_doi_candidates_present`
- `pre_submission_gate_refresh:underfull_layout_warnings:hbox=6,vbox=8`
- `pre_submission_gate_refresh:rebuilt_underfull_layout_warnings:hbox=6,vbox=8`
- `pre_submission_gate_refresh:acm_dl_not_accessible:status=403`
- `pre_submission_gate_refresh:refresh_recorded_tracked_dirty_inputs_before_generation`
- `pre_submission_refresh_freshness:refresh_recorded_tracked_dirty_inputs_before_generation`
- `submission_release_candidate:underfull_layout_warnings:hbox=6,vbox=8`
- `submission_release_candidate:rebuilt_underfull_layout_warnings:hbox=6,vbox=8`
- `submission_release_candidate:proex:crossref_not_visible:status=404`
- `submission_release_candidate:proex:doi_resolver_not_visible:status=404`
- `submission_release_candidate:proex:crossref_discovery_alternate_doi_candidates_present`
- `submission_release_candidate:promax:crossref_discovery_alternate_doi_candidates_present`
- `submission_release_candidate:acm_dl_not_accessible:status=403`
- `submission_release_candidate:refresh_recorded_tracked_dirty_inputs_before_generation`

## Next Actions

- Use this stack refresh as the preferred one-command local pre-submission handoff.
- Keep final_submission_ready=false until the final submission gate reports true.
- Resolve ProMax public page-range/DOI metadata and private manual submission-system confirmation before final submission.
