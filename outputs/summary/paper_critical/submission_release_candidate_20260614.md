# Submission Release-Candidate Packet

Generated: 2026-06-14T21:21:57.468412+00:00

- Verdict: `LOCAL_RELEASE_CANDIDATE_NEEDS_REPAIR`
- OK: `false`
- Local release candidate ready: `false`
- Final submission ready: `false`
- Final-ready source: `outputs\summary\paper_critical\final_submission_gate_20260614.json`

## Status Boundary

- Local RC: Repo-side audited package, source rebuild, metadata packet, manual checklist, external audit health, and freshness index are internally consistent.
- Final submission: Submission may proceed only when the final submission gate also reports true; external proceedings metadata, private submission-system confirmation, and final review-panel coverage remain final blockers.

## Gate Summary

- `final_submission_gate`: ok=`false`, ready=`false`, final_ready=`false`, path=`outputs\summary\paper_critical\final_submission_gate_20260614.json`
- `pre_submission_refresh_freshness`: ok=`true`, ready=`true`, final_ready=`false`, path=`outputs\summary\paper_critical\pre_submission_gate_refresh_freshness_20260614.json`
- `submission_source_package`: ok=`false`, ready=`false`, final_ready=`false`, path=`outputs\summary\paper_critical\submission_source_package_20260614.json`
- `submission_source_package_rebuild`: ok=`false`, ready=`false`, final_ready=`false`, path=`outputs\summary\paper_critical\submission_source_package_rebuild_20260614.json`
- `submission_metadata_packet`: ok=`false`, ready=`false`, final_ready=`false`, path=`outputs\summary\paper_critical\submission_metadata_packet_20260614.json`
- `manual_submission_checklist`: ok=`false`, ready=`false`, final_ready=`false`, path=`outputs\summary\paper_critical\manual_submission_checklist_20260614.json`
- `external_proceedings_metadata`: ok=`true`, ready=`n/a`, final_ready=`false`, path=`outputs\summary\paper_critical\external_proceedings_metadata_recheck_20260614.json`

## Package Summary

- Source files: `0`
- Source bytes: `0`
- Source manifest sha256: ``
- Rebuild PDF pages: `0`
- Rebuild PDF bytes: `0`
- BibTeX warnings: `0`
- Overfull hbox warnings: `0`

## Remaining Blockers

- ProMax final ACM page range and ACM/Crossref registry visibility must be rechecked immediately before submission.
- Final submission package still needs the external submission-target-specific formatting pass.
- promax:final_page_range_missing_in_bib
- promax:crossref_registry_not_visible:status=404
- promax:doi_resolver_not_visible:status=404
- Final manual submission-system metadata/format checklist is not closed.
- external_proceedings_metadata_not_ready
- manual_submission_system_not_ready
- confirm_anonymous_shell:target_formatting_profile_not_ok
- confirm_anonymous_shell:target_profile_not_ok
- confirm_external_proceedings_metadata:external_proceedings_metadata_ready_not_closed
- manual_submission_system_items_not_confirmed
- review_panel_coverage_not_complete
- promax:crossref_registry_not_visible
- promax:doi_resolver_not_visible
- explicit_claude_opus_review

## Failures

- `final_submission_gate:not_ok`
- `final_submission_gate:submission_package:not_ok`
- `final_submission_gate:submission_package:not_ready`
- `final_submission_gate:submission_package:page_count_exceeds_limit:15 > 9`
- `final_submission_gate:submission_package:overfull_hbox_count:8 > 0`
- `final_submission_gate:submission_package:target_profile:target_profile_page_count_exceeds_limit:15 > 9`
- `final_submission_gate:submission_package:target_profile:target_profile_requires_no_overfull_hbox`
- `final_submission_gate:submission_metadata_packet:not_ok`
- `final_submission_gate:submission_metadata_packet:not_ready`
- `final_submission_gate:submission_metadata_packet:submission_package_audit_not_ok`
- `final_submission_gate:submission_metadata_packet:submission_package_not_ready_for_target_formatting`
- `final_submission_gate:submission_metadata_packet:target_profile_not_ok`
- `final_submission_gate:submission_source_package_rebuild:not_ok`
- `final_submission_gate:submission_source_package_rebuild:not_ready`
- `final_submission_gate:submission_source_package_rebuild:source_package_not_ok`
- `final_submission_gate:submission_source_package_rebuild:source_package_not_ready`
- `final_submission_gate:submission_source_package_rebuild:source_package_failures_not_empty_or_missing`
- `final_submission_gate:submission_source_package_rebuild:copied_manifest_missing_files`
- `final_submission_gate:manual_submission_checklist:not_ok`
- `final_submission_gate:manual_submission_checklist:submission_metadata_packet_not_ok`
- `final_submission_gate:manual_submission_checklist:submission_metadata_packet_not_ready`
- `final_submission_gate:manual_submission_checklist:submission_package_audit_not_ok`
- `final_submission_gate:manual_submission_checklist:submission_package_not_ready_for_target_formatting`
- `submission_source_package:not_ok`
- `submission_source_package:submission_package_audit_not_ok`
- `submission_source_package:submission_package_not_ready_for_target_formatting`
- `submission_source_package:submission_package_audit_failures_not_empty_or_missing`
- `submission_source_package_rebuild:not_ok`
- `submission_source_package_rebuild:source_package_not_ok`
- `submission_source_package_rebuild:source_package_not_ready`
- `submission_source_package_rebuild:source_package_failures_not_empty_or_missing`
- `submission_source_package_rebuild:copied_manifest_missing_files`
- `submission_metadata_packet:not_ok`
- `submission_metadata_packet:submission_package_audit_not_ok`
- `submission_metadata_packet:submission_package_not_ready_for_target_formatting`
- `submission_metadata_packet:target_profile_not_ok`
- `manual_submission_checklist:not_ok`
- `manual_submission_checklist:submission_metadata_packet_not_ok`
- `manual_submission_checklist:submission_metadata_packet_not_ready`
- `manual_submission_checklist:submission_package_audit_not_ok`
- `manual_submission_checklist:submission_package_not_ready_for_target_formatting`
- `final_submission_gate:all_local_artifact_gates_ok_not_true`
- `submission_source_package:submission_source_package_ready_not_true`
- `submission_source_package_rebuild:submission_source_package_rebuild_ready_not_true`
- `submission_metadata_packet:submission_metadata_packet_ready_not_true`
- `manual_submission_checklist:manual_submission_checklist_ready_not_true`
- `submission_source_package_rebuild:build_commands_not_all_zero`
- `source_manifest_sha256_values_do_not_all_match`

## Warnings

- `final_submission_gate:underfull_layout_warnings:hbox=10,vbox=12`
- `final_submission_gate:abstract_word_count_outside_common_range:419`
- `final_submission_gate:proex:crossref_not_visible:status=404`
- `final_submission_gate:proex:doi_resolver_not_visible:status=404`
- `final_submission_gate:proex:crossref_discovery_alternate_doi_candidates_present`
- `final_submission_gate:promax:crossref_discovery_alternate_doi_candidates_present`
- `final_submission_gate:underfull_layout_warnings:hbox=6,vbox=8`
- `final_submission_gate:rebuilt_underfull_layout_warnings:hbox=6,vbox=8`
- `final_submission_gate:acm_dl_not_accessible:status=403`
- `final_submission_gate:refresh_recorded_tracked_dirty_inputs_before_generation`
- `pre_submission_refresh_freshness:refresh_recorded_tracked_dirty_inputs_before_generation`
- `submission_metadata_packet:abstract_word_count_outside_common_range:419`
- `external_proceedings_metadata:proex:crossref_not_visible:status=404`
- `external_proceedings_metadata:proex:doi_resolver_not_visible:status=404`
- `external_proceedings_metadata:proex:crossref_discovery_alternate_doi_candidates_present`
- `external_proceedings_metadata:promax:crossref_discovery_alternate_doi_candidates_present`

## Next Actions

- Use this packet as a local handoff index, not as final submission approval.
- Resolve ProMax final page range and DOI/Crossref visibility, then rerun the external metadata audit and refresh stack.
- Attach a valid explicit Claude Opus review, rerun review-continuation, then rerun the final gate and release-candidate packet.
- Complete private submission-system fields with an untracked confirmation JSON, then rerun the manual checklist and refresh stack.
- Rerun the refresh, freshness audit, final gate, and release-candidate packet after any paper/source/BibTeX/package change.
