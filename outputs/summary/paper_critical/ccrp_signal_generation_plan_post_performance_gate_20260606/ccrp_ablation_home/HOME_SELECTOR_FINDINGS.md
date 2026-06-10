# Home C-CRP Selector — Results & Findings (Phase 2.5)

Date: 2026-06-10. Config: preregistered principled (full/full, eta=1.0, weights
0.5,0.3,0.2, cw=0.7). Same reviewed selector as Sports/toys.

## Main test metrics (10,000 events x 101 candidates)
HR@5/10/20  = 0.1597 / 0.2368 / 0.3613
NDCG@5/10/20 = 0.1116 / 0.1364 / 0.1676
MRR = 0.1278
Gates: degeneracy_audit_ok=True, tie_pair_rate=0.412, constant_score_event_rate=0.0005.

## Component ablation (test NDCG@10, frozen at main config)
- full (main):                  0.1364
- without_boundary_uncertainty:  0.1364 (identical)
- without_calibration_gap:       0.1349 (-1.1%)
- without_evidence_support:      0.1363
- without_counterevidence:       0.1381 (+1.3%, slightly BETTER)
- without_risk_penalty:          0.1374 (slightly BETTER)
- confidence_only:               0.1374 (= full, slightly better)
- evidence_only:                 0.1056 (much worse; degeneracy_audit_ok=False)
- confidence_plus_evidence:      0.1363

## Finding — REPLICATES Sports + toys (3rd domain)
The near-inert-components pattern holds again on home: leave-one-out of every
uncertainty/risk component moves NDCG@10 by <=1.3% (mix of neutral / slightly
better / slightly worse), confidence_only == full (actually marginally above),
and only evidence_only collapses. Three of four new domains (sports, toys, home)
now agree: C-CRP ranking is driven by the calibrated relevance probability alone;
the boundary-uncertainty / calibration-gap / counterevidence / risk-penalty
components add no measurable ranking value under the coarse zero-heavy LLM signal.
Home is the sparsest/hardest domain (lowest absolute metrics: NDCG@10 0.136 vs
sports 0.228, toys 0.272) yet shows the same inertness. Tools pending for the
final cross-domain verdict.
