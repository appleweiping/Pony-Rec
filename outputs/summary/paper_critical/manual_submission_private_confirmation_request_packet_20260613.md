# Manual Submission Private Confirmation Request Packet

- Created UTC: `2026-06-13T03:28:25.688585+00:00`
- OK: `true`
- Request packet ready: `true`
- Manual confirmation needed: `true`
- Manual submission system ready: `false`
- Final submission ready: `false`
- Target profile: `sigir2026_full_paper_acm_anonymous`
- Checklist ID: `sigir2026_anonymous_full_paper_manual_submission`
- Recommended untracked path: `artifacts/private/manual_submission_private_confirmation_20260613.json`
- Source manifest sha256: `91d1d6495fe3fa85608d7711fb5873730d907237242b3b3fa489c6f1ed516424`

## Safe Confirmation Skeleton

```json
{
  "checklist_id": "sigir2026_anonymous_full_paper_manual_submission",
  "completed_item_ids": [
    "select_track_and_paper_type",
    "paste_title",
    "paste_abstract",
    "paste_keywords",
    "select_topic_areas",
    "upload_pdf",
    "upload_source_if_required",
    "confirm_anonymous_shell",
    "enter_authors",
    "complete_conflicts",
    "complete_reviewer_preferences",
    "complete_declarations",
    "confirm_external_proceedings_metadata",
    "final_preview_and_submit"
  ],
  "confirmed_in_submission_system": true,
  "no_private_fields_stored": true,
  "private_fields_completed_in_submission_system": true,
  "schema_version": "2026-06-12.manual_submission_private_confirmation.v1",
  "source_manifest_sha256": "91d1d6495fe3fa85608d7711fb5873730d907237242b3b3fa489c6f1ed516424",
  "target_profile_id": "sigir2026_full_paper_acm_anonymous"
}
```

## Item IDs

Full manual gate item IDs:
- `select_track_and_paper_type`
- `paste_title`
- `paste_abstract`
- `paste_keywords`
- `select_topic_areas`
- `upload_pdf`
- `upload_source_if_required`
- `confirm_anonymous_shell`
- `enter_authors`
- `complete_conflicts`
- `complete_reviewer_preferences`
- `complete_declarations`
- `confirm_external_proceedings_metadata`
- `final_preview_and_submit`

Currently unconfirmed required item IDs:
- `select_track_and_paper_type`
- `paste_title`
- `paste_abstract`
- `paste_keywords`
- `select_topic_areas`
- `upload_pdf`
- `upload_source_if_required`
- `confirm_anonymous_shell`
- `enter_authors`
- `complete_conflicts`
- `complete_reviewer_preferences`
- `complete_declarations`
- `final_preview_and_submit`

Currently blocked item IDs:
- `confirm_external_proceedings_metadata`

## Forbidden Private Fields

- author names and affiliations
- author order and contribution declarations
- conflicts of interest
- reviewer suggestions or exclusions
- submission-system declarations
- private submission account metadata

Rejected JSON keys:
- `authors`
- `author_names`
- `affiliations`
- `conflicts`
- `conflicts_of_interest`
- `reviewer_suggestions`
- `reviewer_exclusions`
- `submission_account`
- `account_metadata`
- `private_payload`

## Privacy Rules

- Keep the filled confirmation JSON under an ignored path such as artifacts/private/.
- Do not commit the filled confirmation JSON.
- Do not store author names, affiliations, COI details, reviewer preferences, declarations, account metadata, or submission-account data.
- Only record booleans, source_manifest_sha256, completed_item_ids, and non-sensitive notes if absolutely needed.
- Do not set completed_item_ids until the corresponding action is genuinely complete in the submission system.
- This request packet does not close ProMax public metadata or Claude Opus review blockers.

## Follow-Up Commands

```bash
python -m scripts.audit.main_build_manual_submission_checklist --private-confirmation-json artifacts/private/manual_submission_private_confirmation_20260613.json --output-json outputs/summary/paper_critical/manual_submission_checklist_20260613.json --output-md outputs/summary/paper_critical/manual_submission_checklist_20260613.md
```

```bash
python -m scripts.audit.main_refresh_submission_release_candidate_stack --stamp 20260613 --manual-private-confirmation-json artifacts/private/manual_submission_private_confirmation_20260613.json --external-timeout-seconds 45 --output-json outputs/summary/paper_critical/submission_release_candidate_stack_refresh_20260613.json --output-md outputs/summary/paper_critical/submission_release_candidate_stack_refresh_20260613.md
```

```bash
python -m scripts.audit.main_build_final_submission_gate --manual-checklist-json outputs/summary/paper_critical/manual_submission_checklist_20260613.json --output-json outputs/summary/paper_critical/final_submission_gate_YYYYMMDD.json --output-md outputs/summary/paper_critical/final_submission_gate_YYYYMMDD.md
```

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
