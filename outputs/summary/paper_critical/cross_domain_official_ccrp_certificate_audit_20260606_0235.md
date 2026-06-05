# Cross-Domain Official+C-CRP Certificate Audit

- Generated UTC: `2026-06-05T18:34:56+00:00`
- OK: `True`
- Comparison certificate ready: `True`
- Paper ready: `False`
- Read only: `True`
- Will start experiment: `False`

## Supported Claim

On Sports, Toys, Home, and Tools 10k-user/101-candidate same-candidate domains, C-CRP v3 ranks first against eight official-code-level baselines, with all 56 per-domain C-CRP-vs-official metric tests positive and Holm-significant.

## Domain Gates

| Domain | OK | Official rows | C-CRP rank-all | Positive Holm tests | Min delta | Max Holm p |
|---|---:|---:|---:|---:|---:|---:|
| sports | `True` | 8 | `True` | 56/56 | 0.027200 | 1.21913e-06 |
| toys | `True` | 8 | `True` | 56/56 | 0.019300 | 0.000406112 |
| home | `True` | 8 | `True` | 56/56 | 0.033600 | 1.02169e-08 |
| tools | `True` | 8 | `True` | 56/56 | 0.029400 | 9.71723e-07 |

## Evidence Consistency

- Local/server evidence consistency OK: `True`
- Official local-light rows OK: `32/32`
- Method rows audited: `36/36`

## Paper Boundary

- Paper-critical audit paper_ready: `False`
- Signal rows available: `False`
- Phase 2.5 storage launch allowed: `False`

## Failures

- none

## Warnings

- sports/ccrp_v3_qwen3base_pointwise:local_event_restat_not_self_contained
- toys/ccrp_v3_qwen3base_pointwise:local_event_restat_not_self_contained
