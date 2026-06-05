# Local/Server Evidence Consistency Audit

- ok: `true`
- mode: `local_server_evidence_consistency_audit`
- read_only: `true`
- domains: `sports, toys, home, tools`
- rows: `32/32` ok
- failures: `0`

| domain | method | ok | failures | server large artifacts | checked local files |
| --- | --- | --- | --- | --- | --- |
| sports | proex_profile | `true` |  | `2` | `13` |
| sports | promax_profile | `true` |  | `2` | `13` |
| sports | elmrec_graph | `true` |  | `2` | `13` |
| sports | llmemb | `true` |  | `2` | `13` |
| sports | irllrec_intent | `true` |  | `2` | `13` |
| sports | rlmrec_graphcl | `true` |  | `2` | `13` |
| sports | llm2rec_sasrec | `true` |  | `1` | `15` |
| sports | llmesr_sasrec | `true` |  | `2` | `13` |
| toys | proex_profile | `true` |  | `2` | `12` |
| toys | promax_profile | `true` |  | `2` | `12` |
| toys | elmrec_graph | `true` |  | `2` | `12` |
| toys | llmemb | `true` |  | `2` | `12` |
| toys | irllrec_intent | `true` |  | `2` | `12` |
| toys | rlmrec_graphcl | `true` |  | `2` | `12` |
| toys | llm2rec_sasrec | `true` |  | `1` | `14` |
| toys | llmesr_sasrec | `true` |  | `2` | `12` |
| home | proex_profile | `true` |  | `2` | `12` |
| home | promax_profile | `true` |  | `2` | `13` |
| home | elmrec_graph | `true` |  | `2` | `13` |
| home | llmemb | `true` |  | `2` | `12` |
| home | irllrec_intent | `true` |  | `2` | `12` |
| home | rlmrec_graphcl | `true` |  | `3` | `12` |
| home | llm2rec_sasrec | `true` |  | `3` | `13` |
| home | llmesr_sasrec | `true` |  | `3` | `12` |
| tools | proex_profile | `true` |  | `3` | `12` |
| tools | promax_profile | `true` |  | `3` | `13` |
| tools | elmrec_graph | `true` |  | `3` | `13` |
| tools | llmemb | `true` |  | `3` | `12` |
| tools | irllrec_intent | `true` |  | `3` | `12` |
| tools | rlmrec_graphcl | `true` |  | `3` | `13` |
| tools | llm2rec_sasrec | `true` |  | `3` | `13` |
| tools | llmesr_sasrec | `true` |  | `3` | `12` |
