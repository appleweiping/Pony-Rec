# Final Submission Blocker Closure Packet

Generated: 2026-06-13T00:52:22.413117+00:00

- OK: `true`
- Closure packet ready: `true`
- Local release candidate ready: `true`
- Final submission ready: `false`
- External proceedings metadata ready: `false`
- Manual submission system ready: `false`
- Remaining blocker count: `9`

## Closure Groups

### local_artifact_handoff

- Status: `ready`
- Public safe: `true`
- Can close without private data: `true`

Remaining blockers:
- None

Closure conditions:
- Keep the stack artifact fresh after any paper, bibliography, package, or metadata change.

Next commands:
- `python -m scripts.audit.main_refresh_submission_release_candidate_stack --stamp YYYYMMDD --output-json outputs/summary/paper_critical/submission_release_candidate_stack_refresh_YYYYMMDD.json --output-md outputs/summary/paper_critical/submission_release_candidate_stack_refresh_YYYYMMDD.md`

### external_proceedings_metadata

- Status: `blocked`
- Public safe: `true`
- Can close without private data: `true`

Remaining blockers:
- ProMax final ACM page range and ACM/Crossref registry visibility must be rechecked immediately before submission.
- promax:final_page_range_missing_in_bib
- promax:crossref_registry_not_visible:status=404
- promax:doi_resolver_not_visible:status=404
- external_proceedings_metadata_not_ready
- confirm_external_proceedings_metadata:external_proceedings_metadata_ready_not_closed

Closure conditions:
- Add the final ProMax ACM page range to Paper/references.bib when it is public.
- Direct Crossref /works lookup for DOI 10.1145/3805712.3809600 must return 200 with matching DOI metadata.
- The DOI resolver https://doi.org/10.1145/3805712.3809600 must resolve successfully.
- Rerun the external proceedings metadata audit and the release-candidate stack after metadata changes.

Next commands:
- `python -m scripts.audit.main_probe_promax_public_metadata --network-mode live --timeout-seconds 45 --output-json outputs/summary/paper_critical/promax_public_metadata_probe_YYYYMMDD.json --output-md outputs/summary/paper_critical/promax_public_metadata_probe_YYYYMMDD.md`
- `python -m scripts.audit.main_audit_external_proceedings_metadata --network-mode live --timeout-seconds 45 --output-json outputs/summary/paper_critical/external_proceedings_metadata_recheck_YYYYMMDD.json --output-md outputs/summary/paper_critical/external_proceedings_metadata_recheck_YYYYMMDD.md`
- `python -m scripts.audit.main_refresh_submission_release_candidate_stack --stamp YYYYMMDD --external-timeout-seconds 45 --output-json outputs/summary/paper_critical/submission_release_candidate_stack_refresh_YYYYMMDD.json --output-md outputs/summary/paper_critical/submission_release_candidate_stack_refresh_YYYYMMDD.md`

Current ProMax evidence:
- DOI: `10.1145/3805712.3809600`
- Pages: ``
- Num pages: `11`
- ISBN: `979-8-4007-2599-9`
- Location: `Melbourne, VIC, Australia`
- Crossref status: `404`
- DOI resolver status: `404`

Latest public probe:
- Created: `2026-06-13T00:49:05.508136+00:00`
- ProMax public metadata ready: `false`
- Crossref status: `404`
- DOI resolver status: `404`
- ACM DL status: `403`

Latest public source probes:
- `arxiv_html_promax_acm_metadata`: ok=`true`, status=`200`, missing=``
- `sigir2026_accepted_papers_promax`: ok=`true`, status=`200`, missing=``
- `uq_author_profile_promax_sigir2026`: ok=`true`, status=`200`, missing=``
- `author_publications_promax_sigir2026`: ok=`true`, status=`200`, missing=``
- `uq_experts_profile_promax_sigir2026`: ok=`true`, status=`200`, missing=``

### manual_submission_system

- Status: `manual_private_pending`
- Public safe: `false`
- Can close without private data: `false`

Remaining blockers:
- Final manual submission-system metadata/format checklist is not closed.
- manual_submission_system_not_ready
- manual_submission_system_items_not_confirmed

Closure conditions:
- A human must complete the submission-system fields outside git.
- Create an untracked private confirmation JSON from configs/paper_manual_submission_private_confirmation.template.json.
- Set confirmed_in_submission_system=true, private_fields_completed_in_submission_system=true, no_private_fields_stored=true, source_manifest_sha256 to the current value, and completed_item_ids to the required checklist IDs.
- Do not store author identities, conflicts, reviewer preferences, declarations, account metadata, or other private payloads in git.

Next commands:
- `python -m scripts.audit.main_build_manual_submission_checklist --private-confirmation-json path/to/untracked_private_confirmation.json --output-json outputs/summary/paper_critical/manual_submission_checklist_YYYYMMDD.json --output-md outputs/summary/paper_critical/manual_submission_checklist_YYYYMMDD.md`
- `python -m scripts.audit.main_refresh_submission_release_candidate_stack --stamp YYYYMMDD --manual-private-confirmation-json path/to/untracked_private_confirmation.json --external-timeout-seconds 45 --output-json outputs/summary/paper_critical/submission_release_candidate_stack_refresh_YYYYMMDD.json --output-md outputs/summary/paper_critical/submission_release_candidate_stack_refresh_YYYYMMDD.md`

Manual confirmation safe fields:
- Source manifest sha256: `91d1d6495fe3fa85608d7711fb5873730d907237242b3b3fa489c6f1ed516424`
- Unconfirmed item IDs: `select_track_and_paper_type, paste_title, paste_abstract, paste_keywords, select_topic_areas, upload_pdf, upload_source_if_required, confirm_anonymous_shell, enter_authors, complete_conflicts, complete_reviewer_preferences, complete_declarations, final_preview_and_submit`
- Private item IDs: `enter_authors, complete_conflicts, complete_reviewer_preferences, complete_declarations, final_preview_and_submit`

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

## Next Actions

- Monitor or recheck ProMax public ACM/Crossref/DOI metadata; update BibTeX only after final public page range is available.
- Prepare the private manual submission confirmation outside git after the submission-system fields are completed.
- Rerun the release-candidate stack and this closure packet after either blocker group changes.
