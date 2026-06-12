# Manual Submission Checklist

Generated: 2026-06-12T19:08:32.619815+00:00

- OK: `true`
- Manual submission checklist ready: `true`
- Manual submission system ready: `false`
- Final submission ready: `false`
- Target profile: `sigir2026_full_paper_acm_anonymous`
- Checklist ID: `sigir2026_anonymous_full_paper_manual_submission`
- Items: `14`
- Source manifest sha256: `4f2a9856f722c98ffaf6b7073af27f6890c3086fffe23fa596ebe9fc62aa3cfa`

## Private Confirmation

- Provided: `false`
- Exists: `false`
- SHA256: ``
- Completed item count: `0`

## Items

### `select_track_and_paper_type`

- Category: `submission_system`
- Label: Select the target track and full research paper type
- Status: `manual_pending`
- Private: `false`
- Storage policy: `manual_confirmation_only`
- Blockers: None

### `paste_title`

- Category: `metadata`
- Label: Paste and verify the title from the metadata packet
- Status: `manual_pending`
- Private: `false`
- Storage policy: `repo_prefill_public_value`
- Prefill source: `submission_fields.title`
- Prefill summary: Actionable Uncertainty for LLM-Based Recommendation
- Blockers: None

### `paste_abstract`

- Category: `metadata`
- Label: Paste and verify the abstract from the metadata packet
- Status: `manual_pending`
- Private: `false`
- Storage policy: `repo_prefill_public_value`
- Prefill source: `submission_fields.abstract`
- Prefill summary: Large language models (LLMs) are increasingly used as scoring and representation engines for recommendation, but high ranking scores do not by themselves tell us when a recommendation decision is reliable. We study this problem under a controlled same-candidate protocol: every method ranks the sa...
- Blockers: None

### `paste_keywords`

- Category: `metadata`
- Label: Paste and verify keywords
- Status: `manual_pending`
- Private: `false`
- Storage policy: `repo_prefill_public_value`
- Prefill source: `submission_fields.keywords`
- Prefill summary: LLM-based recommendation, uncertainty estimation, calibration, candidate reranking, same-candidate evaluation
- Blockers: None

### `select_topic_areas`

- Category: `metadata`
- Label: Select topic areas
- Status: `manual_pending`
- Private: `false`
- Storage policy: `repo_prefill_public_value`
- Prefill source: `submission_fields.topic_areas`
- Prefill summary: Recommender systems, Evaluation and reproducibility, Large language models, Uncertainty and calibration
- Blockers: None

### `upload_pdf`

- Category: `files`
- Label: Upload the audited anonymous PDF
- Status: `manual_pending`
- Private: `false`
- Storage policy: `repo_evidence_path_only`
- Prefill source: `package_crosscheck.pdf_path`
- Prefill summary: Paper\main.pdf
- Blockers: None

### `upload_source_if_required`

- Category: `files`
- Label: Upload source package if required by the submission system
- Status: `manual_pending`
- Private: `false`
- Storage policy: `repo_evidence_manifest_only`
- Prefill source: `package_crosscheck.source_manifest_sha256`
- Prefill summary: 4f2a9856f722c98ffaf6b7073af27f6890c3086fffe23fa596ebe9fc62aa3cfa
- Blockers: None

### `confirm_anonymous_shell`

- Category: `formatting`
- Label: Confirm anonymous PDF/source shell in the submission system
- Status: `manual_pending`
- Private: `false`
- Storage policy: `repo_evidence_plus_manual_confirmation`
- Blockers: None

### `enter_authors`

- Category: `private_metadata`
- Label: Enter authors, affiliations, and order inside the submission system
- Status: `manual_private_not_stored`
- Private: `true`
- Storage policy: `not_stored_in_repo`
- Blockers: None

### `complete_conflicts`

- Category: `private_metadata`
- Label: Complete conflict-of-interest fields inside the submission system
- Status: `manual_private_not_stored`
- Private: `true`
- Storage policy: `not_stored_in_repo`
- Blockers: None

### `complete_reviewer_preferences`

- Category: `private_metadata`
- Label: Complete reviewer suggestions/exclusions only inside the submission system if requested
- Status: `manual_private_not_stored`
- Private: `true`
- Storage policy: `not_stored_in_repo`
- Blockers: None

### `complete_declarations`

- Category: `declarations`
- Label: Complete submission-system declarations, ethics, artifact, and policy checkboxes
- Status: `manual_private_not_stored`
- Private: `true`
- Storage policy: `manual_confirmation_only`
- Blockers: None

### `confirm_external_proceedings_metadata`

- Category: `external_metadata`
- Label: Confirm external proceedings metadata blockers are closed
- Status: `blocked`
- Private: `false`
- Storage policy: `repo_external_audit_plus_manual_confirmation`
- Blockers: `confirm_external_proceedings_metadata:external_proceedings_metadata_ready_not_closed`

### `final_preview_and_submit`

- Category: `final_review`
- Label: Review final submission preview and submit
- Status: `manual_private_not_stored`
- Private: `true`
- Storage policy: `manual_confirmation_only`
- Blockers: None

## Manual Private Fields Not Stored

- author names and affiliations
- author order and contribution declarations
- conflicts of interest
- reviewer suggestions or exclusions
- submission-system declarations
- private submission account metadata

## Remaining Blockers

- ProMax final ACM page range and ACM/Crossref registry visibility must be rechecked immediately before submission.
- promax:final_page_range_missing_in_bib
- promax:crossref_registry_not_visible:status=404
- promax:doi_resolver_not_visible:status=404
- Final manual submission-system metadata/format checklist is not closed.
- confirm_external_proceedings_metadata:external_proceedings_metadata_ready_not_closed
- manual_submission_system_items_not_confirmed

## Failures

- None

## Warnings

- None

## Next Actions

- Use this checklist while filling the submission system; do not copy private author/COI/reviewer data into the repository.
- Optionally pass --private-confirmation-json to audit a local untracked confirmation file after the submission-system fields are completed.
- Rerun the external proceedings metadata audit immediately before final submission.
- Rerun submission package, metadata packet, and this checklist after any paper/source/BibTeX change.
- Keep final_submission_ready=false until manual submission-system items and external metadata blockers are all closed.
