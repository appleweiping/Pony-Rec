# Local/Server Evidence Consistency Audit

- ok: `true`
- mode: `local_server_evidence_consistency_audit`
- read_only: `true`
- domains: `tools`
- rows: `8/8` ok
- failures: `0`

| domain | method | ok | failures | server large artifacts | checked local files |
| --- | --- | --- | --- | --- | --- |
| tools | proex_profile | `true` |  | `3` | `12` |
| tools | promax_profile | `true` |  | `3` | `13` |
| tools | elmrec_graph | `true` |  | `3` | `13` |
| tools | llmemb | `true` |  | `3` | `12` |
| tools | irllrec_intent | `true` |  | `3` | `12` |
| tools | rlmrec_graphcl | `true` |  | `3` | `13` |
| tools | llm2rec_sasrec | `true` |  | `3` | `13` |
| tools | llmesr_sasrec | `true` |  | `3` | `12` |
