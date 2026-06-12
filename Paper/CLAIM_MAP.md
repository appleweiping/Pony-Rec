# Claim-Evidence Map

Current source of truth: `outputs/summary/paper_critical/final_paper_claim_audit_20260612.*`.

This map records what the manuscript may claim after the 2026-06-12 evidence
consolidation. It replaces the older pre-Phase-2.5 calibration narrative.

## Supported Claims

| ID | Claim | Evidence | Boundary |
|---|---|---|---|
| C1 | C-CRP v3 ranks first against eight official-code-level baselines on Sports, Toys, Home, and Tools under 10k-user/101-candidate same-candidate evaluation. | `outputs/summary/paper_critical/cross_domain_official_ccrp_certificate_audit_post_ccrp_backfill_20260606_0255.json` | Same-candidate reranking only; not full-catalog SOTA. |
| C2 | The comparison ledger has 32 official rows and 4 C-CRP rows with complete metrics, row counts, score coverage, provenance, and local/server evidence checks. | `outputs/summary/paper_critical/new_domains_paper_facing_full_metric_evidence_ledger_post_ccrp_backfill_20260606_0255.json` | Ledger supports table drafting, not final submission readiness alone. |
| C3 | C-CRP event-level uncertainty stratifies same-candidate ranking reliability in all four domains. | `outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/observation_four_domain/` | Descriptive motivation only; no causal/statistical bin-effect claim. |
| C4 | Component ablation packages are complete and support a diagnostic discussion of weak, redundant, or mixed components. | `outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/ccrp_component_ablation_four_domain/` | Supplementary diagnostic evidence only. |
| C5 | Eta and uncertainty weight choices are stable within tolerance on NDCG@10 across the four domains. | `outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/ccrp_hyperparameter_four_domain/` | Stability/sensitivity only; not all-metric robustness or universal optimum. |
| C6 | The framework figure can explain the same-candidate pipeline, calibration, uncertainty, and risk-adjusted ranking. | `outputs/summary/paper_critical/framework_overview/` | Figure is not evidence that modules are complete. |

## Contradicted Or Forbidden Claims

| Claim | Status | Why forbidden |
|---|---|---|
| Every C-CRP component is necessary. | Contradicted | Leave-one-component-out diagnostics show neutral or nonworse removals, including boundary uncertainty and risk/counterevidence terms. |
| Positive eta or risk penalty is necessary. | Contradicted | The hyperparameter sweep reports test-best eta = 0 in all four domains. |
| The project is final submission-ready. | Unsupported | The evidence is ready for manuscript-level claim/citation audit, but references still need verified non-placeholder BibTeX and full section review. |
| C-CRP is full-catalog or universal recommender SOTA. | Unsupported | All formal claims are scoped to controlled same-candidate reranking. |

## Next Manuscript Gates

1. Replace placeholder BibTeX entries with verified citations for every
   official baseline and key related-work category.
2. Re-run ARIS paper-claim-audit on the actual manuscript text after each
   major rewrite.
3. Run ARIS citation-audit on `paper/references.bib` before any submission
   readiness claim.
