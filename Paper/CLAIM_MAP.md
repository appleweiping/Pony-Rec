# Claim-Evidence Map

Current source of truth: `outputs/summary/paper_critical/final_paper_claim_audit_20260612.*`.

This map records what the manuscript may claim after the 2026-06-12 evidence
consolidation. It replaces the older pre-Phase-2.5 calibration narrative.

## Supported Claims

| ID | Claim | Evidence | Boundary |
|---|---|---|---|
| C1 | C-CRP v3 ranks first against eight official-code-level baselines on Sports, Toys, Home, and Tools under 10k-user/101-candidate same-candidate evaluation, with all 56 per-domain paired tests positive and Holm-significant. | `outputs/summary/paper_critical/cross_domain_official_ccrp_certificate_audit_post_ccrp_backfill_20260606_0255.json` | Same-candidate reranking only; not full-catalog SOTA. Paired-test gate holds for these four domains. |
| C1b | Across all eight domains (Beauty, Books, Electronics, Movies + the above four), C-CRP v3 ranks first in six (Books, Electronics, Sports, Toys, Home, Tools; +21.6% to +53.2% NDCG@10 over the strongest baseline) and is competitive but not first in two (Beauty rank 2, −11% NDCG@10 vs ProEx; Movies rank 5, −24% vs LLMEmb). | `outputs/ccrp_v3_formal/main_comparison_table.csv` (Beauty/Books/Electronics/Movies); `outputs/summary/new_domains_official_ccrp_cross_domain_20260605_method_rows.csv` (Sports/Toys/Home/Tools) | Same-candidate reranking only. The two non-winning domains are reported, not dropped. Paired-test Holm gate only verified for the four new domains. |
| C2 | The comparison ledger has 64 official rows and 8 C-CRP rows (eight baselines x eight domains) with complete metrics, row counts, score coverage, provenance, and local/server evidence checks. | `outputs/ccrp_v3_formal/main_comparison_table.csv`; `outputs/summary/new_domains_official_ccrp_cross_domain_20260605_method_rows.csv` | Ledger supports table drafting, not final submission readiness alone. |
| C3 | C-CRP event-level uncertainty stratifies same-candidate ranking reliability in all four domains. | `outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/observation_four_domain/` | Descriptive motivation only; no causal/statistical bin-effect claim. |
| C4 | Component ablation packages are complete and support a diagnostic discussion of weak, redundant, or mixed components. | `outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/ccrp_component_ablation_four_domain/` | Supplementary diagnostic evidence only. |
| C5 | Eta and uncertainty weight choices are stable within tolerance on NDCG@10 across the four domains. | `outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/ccrp_hyperparameter_four_domain/` | Stability/sensitivity only; not all-metric robustness or universal optimum. |
| C6 | The framework figure can explain the same-candidate pipeline, calibration, uncertainty, and the ranking-family risk term. | `outputs/summary/paper_critical/framework_overview/` | Figure is not evidence that modules are complete. |

## Contradicted Or Forbidden Claims

| Claim | Status | Why forbidden |
|---|---|---|
| Every C-CRP component is necessary. | Contradicted | Leave-one-component-out diagnostics show neutral or nonworse removals, including boundary uncertainty and risk/counterevidence terms. |
| Positive eta or risk penalty is necessary. | Contradicted | The hyperparameter sweep reports test-best eta = 0 in all four domains. |
| The project is final submission-ready. | Unsupported | The evidence is ready for manuscript-level claim/citation audit, and BibTeX warnings are repaired, but the manuscript still needs final ARIS spot-checks, section-level top-conference review, and balance/clarity passes. |
| C-CRP is full-catalog or universal recommender SOTA. | Unsupported | All formal claims are scoped to controlled same-candidate reranking. |
| C-CRP ranks first on all seven metrics in every domain. | Forbidden (cherry-picking) | False over the full eight-domain set: C-CRP is rank 2 on Beauty and rank 5 on Movies. The headline must read "first in 6 of 8 domains," and Beauty/Movies must be reported, not dropped. |

## Next Manuscript Gates

1. Re-run ARIS paper-claim-audit on the actual manuscript text after each
   major rewrite.
2. Run the final ARIS citation spot-check before any submission-readiness
   claim, even though the 2026-06-12 repair removed BibTeX warnings.
3. Run section-level top-conference review after the compressed method,
   protocol, result, and limitation sections have been expanded and balanced.
