# External Proceedings Metadata Audit

Generated: 2026-06-12T17:54:49.674670+00:00

- OK: `true`
- External proceedings metadata ready: `false`
- Final submission ready: `false`
- Network mode: `live`
- Checked entries: `2`

## Entry Checks

### ProEx (`proex`)

- OK: `true`
- External metadata ready: `true`
- DOI: `10.1145/3770854.3780284`
- Pages: `1940--1951`
- Num pages: ``
- arXiv: ``
- Crossref: ok=`false`, status=`404`
- DOI resolver: ok=`false`, status=`404`
- Source `dblp_search_proex_kdd_2026`: ok=`true`, status=`200`, missing_patterns=[]

Blockers:
- None

Warnings:
- `proex:crossref_not_visible:status=404`
- `proex:doi_resolver_not_visible:status=404`

### ProMax (`promax`)

- OK: `true`
- External metadata ready: `false`
- DOI: `10.1145/3805712.3809600`
- Pages: ``
- Num pages: `11`
- arXiv: `2604.26231`
- Crossref: ok=`false`, status=`404`
- DOI resolver: ok=`false`, status=`404`
- Source `arxiv_2604_26231`: ok=`true`, status=`200`, missing_patterns=[]
- Source `sigir2026_accepted_papers_promax`: ok=`true`, status=`200`, missing_patterns=[]

Blockers:
- `promax:final_page_range_missing_in_bib`
- `promax:crossref_registry_not_visible:status=404`
- `promax:doi_resolver_not_visible:status=404`

Warnings:
- None

## Remaining Blockers

- promax:final_page_range_missing_in_bib
- promax:crossref_registry_not_visible:status=404
- promax:doi_resolver_not_visible:status=404
- Final manual submission-system metadata/format checklist is not closed.

## Failures

- None

## Warnings

- `proex:crossref_not_visible:status=404`
- `proex:doi_resolver_not_visible:status=404`

## Next Actions

- If ProMax DOI/Crossref/page-range visibility becomes public, update Paper/references.bib and rerun this audit.
- Rerun the submission package and metadata packet audits after this external metadata audit changes.
- Keep final_submission_ready=false until this audit, target formatting, and manual submission-system checks are all closed.
