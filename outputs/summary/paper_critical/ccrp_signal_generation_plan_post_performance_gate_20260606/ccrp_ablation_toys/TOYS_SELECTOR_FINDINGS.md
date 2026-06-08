# Toys C-CRP Selector — Results & Findings (Phase 2.5)

Date: 2026-06-08. Config: preregistered principled (full/full, eta=1.0, weights
0.5,0.3,0.2, cw=0.7). Leakage-safe selector (same reviewed 8/10 design as Sports).

## Main test metrics (10,000 events x 101 candidates)
HR@5/10/20  = 0.3154 / 0.3987 / 0.5138
NDCG@5/10/20 = 0.2449 / 0.2717 / 0.3006
MRR = 0.2511
Gates: degeneracy_audit_ok=True, tie_pair_rate=0.317, constant_score_event_rate=0.0008.

## Component ablation (test NDCG@10, frozen at main config)
- full (main):                 0.2717
- without_boundary_uncertainty: 0.2717  (identical)
- without_calibration_gap:      0.2695  (slightly worse, -0.8%)
- without_evidence_support:     0.2712
- without_counterevidence:      0.2724  (slightly BETTER)
- without_risk_penalty:         0.2717
- confidence_only:              0.2717  (= full)
- evidence_only:                0.2365  (much worse; degeneracy_audit_ok=False)
- confidence_plus_evidence:     0.2717

## Finding — REPLICATES Sports
The near-inert-components pattern holds on toys: leave-one-out of every uncertainty/
risk component moves NDCG@10 by <=0.8% (mix of neutral/slightly-better/slightly-worse),
confidence_only == full, and only evidence_only (dropping the calibrated probability)
collapses. So on BOTH Sports and toys the C-CRP ranking is driven by the calibrated
relevance probability alone; the boundary-uncertainty / calibration-gap /
counterevidence / risk-penalty components add ~nothing under the coarse zero-heavy
LLM signal. Two of four new domains now agree. Home and tools still pending before a
final cross-domain verdict, but the trend is strong and consistent.

Comparison to Sports (NDCG@10): Sports main 0.2281, toys main 0.2717 (toys higher,
consistent with toys being a denser/easier domain). Both show the same component
inertness.
