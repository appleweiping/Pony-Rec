# New-Domain Paper-Facing Evidence Ledger

- Generated UTC: `2026-06-05T18:40:19+00:00`
- OK: `True`
- Comparison ledger ready: `True`
- Paper ready: `False`
- Rows: `36`
- Official rows: `32`
- C-CRP rows: `4`
- Certificate audit OK: `True`
- Local/server evidence consistency OK: `True`

## Supported Claim

On Sports, Toys, Home, and Tools 10k-user/101-candidate same-candidate domains, C-CRP v3 ranks first against eight official-code-level baselines, with all 56 per-domain C-CRP-vs-official metric tests positive and Holm-significant.

## Domain Counts

| Domain | Rows | Official | C-CRP | Gate OK | Positive Holm Tests |
|---|---:|---:|---:|---:|---:|
| sports | 9 | 8 | 1 | `True` | 56/56 |
| toys | 9 | 8 | 1 | `True` | 56/56 |
| home | 9 | 8 | 1 | `True` | 56/56 |
| tools | 9 | 8 | 1 | `True` | 56/56 |

## Failures

- none

## Warnings

- sports/ccrp_v3_qwen3base_pointwise:compact_certificate_not_self_contained_for_event_restat;missing_local_ccrp_user_ranks
- toys/ccrp_v3_qwen3base_pointwise:compact_certificate_not_self_contained_for_event_restat;missing_local_ccrp_user_ranks

## Boundary

This ledger supports the four new-domain same-candidate official comparison table. It does not make the paper ready because observation/motivation, C-CRP component ablation, and hyperparameter-curve modules still need full-scale uncertainty signal rows.
