# Local/Server Evidence Consistency Audit

- ok: `false`
- mode: `local_server_evidence_consistency_audit`
- read_only: `true`
- domains: `sports, toys, home, tools`
- rows: `11/32` ok
- failures: `51`

| domain | method | ok | failures | server large artifacts | checked local files |
| --- | --- | --- | --- | --- | --- |
| sports | proex_profile | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_required_local_file:server_large_artifact_manifest.sha256<br>missing_server_large_artifact_manifest_json | `` | `11` |
| sports | promax_profile | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_required_local_file:server_large_artifact_manifest.sha256<br>missing_server_large_artifact_manifest_json | `` | `11` |
| sports | elmrec_graph | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_required_local_file:server_large_artifact_manifest.sha256<br>missing_server_large_artifact_manifest_json | `` | `11` |
| sports | llmemb | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_required_local_file:server_large_artifact_manifest.sha256<br>missing_server_large_artifact_manifest_json | `` | `11` |
| sports | irllrec_intent | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_required_local_file:server_large_artifact_manifest.sha256<br>missing_server_large_artifact_manifest_json | `` | `11` |
| sports | rlmrec_graphcl | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_required_local_file:server_large_artifact_manifest.sha256<br>missing_server_large_artifact_manifest_json | `` | `11` |
| sports | llm2rec_sasrec | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_required_local_file:server_large_artifact_manifest.sha256<br>missing_server_large_artifact_manifest_json | `` | `12` |
| sports | llmesr_sasrec | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_required_local_file:server_large_artifact_manifest.sha256<br>missing_server_large_artifact_manifest_json | `` | `11` |
| toys | proex_profile | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_server_large_artifact_manifest_json | `` | `11` |
| toys | promax_profile | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_server_large_artifact_manifest_json | `` | `11` |
| toys | elmrec_graph | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_server_large_artifact_manifest_json | `` | `11` |
| toys | llmemb | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_server_large_artifact_manifest_json | `` | `11` |
| toys | irllrec_intent | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_server_large_artifact_manifest_json | `` | `11` |
| toys | rlmrec_graphcl | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_server_large_artifact_manifest_json | `` | `11` |
| toys | llm2rec_sasrec | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_required_local_file:server_large_artifact_manifest.sha256<br>missing_server_large_artifact_manifest_json | `` | `11` |
| toys | llmesr_sasrec | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_server_large_artifact_manifest_json | `` | `11` |
| home | proex_profile | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_server_large_artifact_manifest_json | `` | `11` |
| home | promax_profile | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_server_large_artifact_manifest_json | `` | `11` |
| home | elmrec_graph | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_server_large_artifact_manifest_json | `` | `11` |
| home | llmemb | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_server_large_artifact_manifest_json | `` | `11` |
| home | irllrec_intent | `false` | missing_required_local_file:server_large_artifact_manifest.json<br>missing_server_large_artifact_manifest_json | `` | `11` |
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

## Failures

- `sports/proex_profile:missing_required_local_file:server_large_artifact_manifest.json`
- `sports/proex_profile:missing_required_local_file:server_large_artifact_manifest.sha256`
- `sports/proex_profile:missing_server_large_artifact_manifest_json`
- `sports/promax_profile:missing_required_local_file:server_large_artifact_manifest.json`
- `sports/promax_profile:missing_required_local_file:server_large_artifact_manifest.sha256`
- `sports/promax_profile:missing_server_large_artifact_manifest_json`
- `sports/elmrec_graph:missing_required_local_file:server_large_artifact_manifest.json`
- `sports/elmrec_graph:missing_required_local_file:server_large_artifact_manifest.sha256`
- `sports/elmrec_graph:missing_server_large_artifact_manifest_json`
- `sports/llmemb:missing_required_local_file:server_large_artifact_manifest.json`
- `sports/llmemb:missing_required_local_file:server_large_artifact_manifest.sha256`
- `sports/llmemb:missing_server_large_artifact_manifest_json`
- `sports/irllrec_intent:missing_required_local_file:server_large_artifact_manifest.json`
- `sports/irllrec_intent:missing_required_local_file:server_large_artifact_manifest.sha256`
- `sports/irllrec_intent:missing_server_large_artifact_manifest_json`
- `sports/rlmrec_graphcl:missing_required_local_file:server_large_artifact_manifest.json`
- `sports/rlmrec_graphcl:missing_required_local_file:server_large_artifact_manifest.sha256`
- `sports/rlmrec_graphcl:missing_server_large_artifact_manifest_json`
- `sports/llm2rec_sasrec:missing_required_local_file:server_large_artifact_manifest.json`
- `sports/llm2rec_sasrec:missing_required_local_file:server_large_artifact_manifest.sha256`
- `sports/llm2rec_sasrec:missing_server_large_artifact_manifest_json`
- `sports/llmesr_sasrec:missing_required_local_file:server_large_artifact_manifest.json`
- `sports/llmesr_sasrec:missing_required_local_file:server_large_artifact_manifest.sha256`
- `sports/llmesr_sasrec:missing_server_large_artifact_manifest_json`
- `toys/proex_profile:missing_required_local_file:server_large_artifact_manifest.json`
- `toys/proex_profile:missing_server_large_artifact_manifest_json`
- `toys/promax_profile:missing_required_local_file:server_large_artifact_manifest.json`
- `toys/promax_profile:missing_server_large_artifact_manifest_json`
- `toys/elmrec_graph:missing_required_local_file:server_large_artifact_manifest.json`
- `toys/elmrec_graph:missing_server_large_artifact_manifest_json`
- `toys/llmemb:missing_required_local_file:server_large_artifact_manifest.json`
- `toys/llmemb:missing_server_large_artifact_manifest_json`
- `toys/irllrec_intent:missing_required_local_file:server_large_artifact_manifest.json`
- `toys/irllrec_intent:missing_server_large_artifact_manifest_json`
- `toys/rlmrec_graphcl:missing_required_local_file:server_large_artifact_manifest.json`
- `toys/rlmrec_graphcl:missing_server_large_artifact_manifest_json`
- `toys/llm2rec_sasrec:missing_required_local_file:server_large_artifact_manifest.json`
- `toys/llm2rec_sasrec:missing_required_local_file:server_large_artifact_manifest.sha256`
- `toys/llm2rec_sasrec:missing_server_large_artifact_manifest_json`
- `toys/llmesr_sasrec:missing_required_local_file:server_large_artifact_manifest.json`
- `toys/llmesr_sasrec:missing_server_large_artifact_manifest_json`
- `home/proex_profile:missing_required_local_file:server_large_artifact_manifest.json`
- `home/proex_profile:missing_server_large_artifact_manifest_json`
- `home/promax_profile:missing_required_local_file:server_large_artifact_manifest.json`
- `home/promax_profile:missing_server_large_artifact_manifest_json`
- `home/elmrec_graph:missing_required_local_file:server_large_artifact_manifest.json`
- `home/elmrec_graph:missing_server_large_artifact_manifest_json`
- `home/llmemb:missing_required_local_file:server_large_artifact_manifest.json`
- `home/llmemb:missing_server_large_artifact_manifest_json`
- `home/irllrec_intent:missing_required_local_file:server_large_artifact_manifest.json`
- `home/irllrec_intent:missing_server_large_artifact_manifest_json`
