# Four New-Domain Official+C-CRP Gate Summary

Created UTC: 2026-06-05T14:10:46.779820+00:00

## ARIS Verdict

CONDITIONAL_PASS_FOR_COMPACT_FOUR_NEW_DOMAIN_GATE_SUMMARY. This is a compact comparison-gate certificate, not a self-contained raw reproduction package and not a paper-readiness verdict.

Supported wording:

> On Sports, Toys, Home, and Tools 10k-user/101-candidate same-candidate domains, C-CRP v3 ranks first against eight official-code-level baselines, with all 56 per-domain C-CRP-vs-official metric tests positive and Holm-significant.

Unsupported wording: paper-ready SOTA; full-catalog recommender SOTA; large practical effect; universal winner; uncertainty mechanism proven.

## Domain Results

| domain | official ok | C-CRP rank | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR | paired tests | min delta | min CI low | max Holm p | min |dz| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sports | 8 | 1 | 0.2745 | 0.3819 | 0.5172 | 0.198451 | 0.232862 | 0.267006 | 0.207566 | 56/56 | 0.0272 | 0.0164 | 1.22e-06 | 0.049 |
| toys | 8 | 1 | 0.3172 | 0.3964 | 0.5059 | 0.245190 | 0.270799 | 0.298341 | 0.250305 | 56/56 | 0.0193 | 0.0083 | 0.000406 | 0.035 |
| home | 8 | 1 | 0.1561 | 0.2264 | 0.3505 | 0.109780 | 0.132394 | 0.163511 | 0.125948 | 56/56 | 0.0336 | 0.0216 | 1.02e-08 | 0.057 |
| tools | 8 | 1 | 0.1937 | 0.2696 | 0.3931 | 0.141864 | 0.166116 | 0.197040 | 0.155859 | 56/56 | 0.0294 | 0.0173 | 9.72e-07 | 0.049 |

## Artifact Gaps

- non_blocking_for_compact_certificate: sports ccrp_v3_qwen3base_pointwise - local raw C-CRP event rows are missing or incomplete; compact summary relies on copied gate/comparison/paired certificates
- non_blocking_for_compact_certificate: toys ccrp_v3_qwen3base_pointwise - local raw C-CRP event rows are missing or incomplete; compact summary relies on copied gate/comparison/paired certificates

## Next Gates

- observation/motivation study with real event-level uncertainty fields and paper-ready figure/table
- full C-CRP component ablation over nontrivial design components
- hyperparameter/stability curves over real controls with validation/test separation
- final framework overview figure refinement
- GPT-5.5/Codex xhigh review cycle reaching at least 8/10
