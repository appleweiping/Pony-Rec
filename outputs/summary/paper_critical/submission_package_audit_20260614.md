# Submission Package Audit

Generated: 2026-06-14T21:21:55.305703+00:00

- Verdict: `NEEDS_SUBMISSION_PACKAGE_REPAIR`
- OK: `false`
- Submission package ready for target formatting: `false`
- Final submission ready: `false`
- PDF pages: `15`
- PDF bytes: `850448`
- BibTeX warnings: `0`
- Overfull hbox warnings: `8`
- Underfull hbox/vbox warnings: `10` / `12`
- Cited keys: `32`
- Framework overview: `inline_tikz`
- Panel score floor: `8.0`
- Anonymous source leak scan: `true` (emails=`0`, orcid=`0`, acks=`0`, local_paths=`0`)
- External proceedings metadata ready: `False`
- Target formatting profile: `sigir2026_full_paper_acm_anonymous` ok=`false`
- Source package manifest files: `26`
- Source package manifest sha256: `2acac6e54318be410e9e216429195cad580fd870b91ef95a6bddb9f361909a08`

## Remaining Blockers

- ProMax final ACM page range and ACM/Crossref registry visibility must be rechecked immediately before submission.
- Final submission package still needs the external submission-target-specific formatting pass.
- promax:final_page_range_missing_in_bib
- promax:crossref_registry_not_visible:status=404
- promax:doi_resolver_not_visible:status=404
- Final manual submission-system metadata/format checklist is not closed.

## Failures

- `page_count_exceeds_limit:15 > 9`
- `overfull_hbox_count:8 > 0`
- `target_profile:target_profile_page_count_exceeds_limit:15 > 9`
- `target_profile:target_profile_requires_no_overfull_hbox`

## Warnings

- `underfull_layout_warnings:hbox=10,vbox=12`

## Next Actions

- Run the final manual submission-system metadata/format checklist on the audited Paper package.
- Recheck ProMax final ACM page range and ACM/Crossref visibility immediately before submission.
- Keep final_submission_ready=false until external metadata and formatting blockers are closed.
