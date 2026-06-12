# External Proceedings Metadata Audit

Generated: 2026-06-12T23:21:34.118026+00:00

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
- ISBN: ``
- Location: ``
- arXiv: ``
- Crossref: ok=`false`, status=`404`
- DOI resolver: ok=`false`, status=`404`
- Source `dblp_search_proex_kdd_2026`: ok=`true`, status=`200`, missing_patterns=[]
- Crossref discovery candidates: `5`
- Discovery exact-DOI candidates: `0`
- Discovery exact-DOI candidates with pages: `0`
- Discovery alternate-DOI candidates: `5`
- Discovery policy: Discovery candidates are advisory only. They help detect newly public or changed metadata, but they do not by themselves satisfy the exact BibTeX page-range, DOI resolver, or Crossref final-readiness gates.
- Discovery query `crossref_title_search_proex`: ok=`true`, status=`200`, candidates=`5`, error=``
  - candidate doi=`10.2139/ssrn.5841223`, year=`None`, pages=``, expected_doi_match=`false`, title=`A Unified Reinforcement Learning Framework for Dynamic User Profiling and Predictive Recommendation`
  - candidate doi=`10.20944/preprints202510.1143.v1`, year=`None`, pages=``, expected_doi_match=`false`, title=`A Unified Reinforcement Learning Framework for Dynamic User Profiling and Predictive Recommendation`
  - candidate doi=`10.3403/30434665`, year=`None`, pages=``, expected_doi_match=`false`, title=`Information technology. Object Management Group Unified Architecture Framework (OMG UAF)`

Blockers:
- None

Warnings:
- `proex:crossref_not_visible:status=404`
- `proex:doi_resolver_not_visible:status=404`
- `proex:crossref_discovery_alternate_doi_candidates_present`

### ProMax (`promax`)

- OK: `true`
- External metadata ready: `false`
- DOI: `10.1145/3805712.3809600`
- Pages: ``
- Num pages: `11`
- ISBN: `979-8-4007-2599-9`
- Location: `Melbourne, VIC, Australia`
- arXiv: `2604.26231`
- Crossref: ok=`false`, status=`404`
- DOI resolver: ok=`false`, status=`404`
- Source `arxiv_2604_26231`: ok=`true`, status=`200`, missing_patterns=[]
- Source `arxiv_html_promax_acm_metadata`: ok=`true`, status=`200`, missing_patterns=[]
- Source `sigir2026_accepted_papers_promax`: ok=`true`, status=`200`, missing_patterns=[]
- Crossref discovery candidates: `5`
- Discovery exact-DOI candidates: `0`
- Discovery exact-DOI candidates with pages: `0`
- Discovery alternate-DOI candidates: `5`
- Discovery policy: Discovery candidates are advisory only. They help detect newly public or changed metadata, but they do not by themselves satisfy the exact BibTeX page-range, DOI resolver, or Crossref final-readiness gates.
- Discovery query `crossref_title_search_promax`: ok=`true`, status=`200`, candidates=`5`, error=``
  - candidate doi=`10.26481/dis.20220324sh`, year=`None`, pages=``, expected_doi_match=`false`, title=`Health recommender systems for behavior change`
  - candidate doi=`10.1145/3701716.3717734`, year=`2025`, pages=`2102-2111`, expected_doi_match=`false`, title=`Improving LLM-Based Recommender Systems with User-Controllable Profiles`
  - candidate doi=`10.1007/978-3-032-01152-7_4`, year=`2025`, pages=`99-129`, expected_doi_match=`false`, title=`LLM as Recommender`

Blockers:
- `promax:final_page_range_missing_in_bib`
- `promax:crossref_registry_not_visible:status=404`
- `promax:doi_resolver_not_visible:status=404`

Warnings:
- `promax:crossref_discovery_alternate_doi_candidates_present`

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
- `proex:crossref_discovery_alternate_doi_candidates_present`
- `promax:crossref_discovery_alternate_doi_candidates_present`

## Next Actions

- If ProMax DOI/Crossref/page-range visibility becomes public, update Paper/references.bib and rerun this audit.
- Rerun the submission package and metadata packet audits after this external metadata audit changes.
- Keep final_submission_ready=false until this audit, target formatting, and manual submission-system checks are all closed.
