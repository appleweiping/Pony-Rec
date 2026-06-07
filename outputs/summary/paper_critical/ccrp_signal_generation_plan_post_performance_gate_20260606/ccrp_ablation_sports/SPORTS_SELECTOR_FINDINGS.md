# Sports C-CRP Selector — Results & Honest Findings (Phase 2.5)

Date: 2026-06-07. Config: preregistered principled (score_mode=full, ablation=full,
eta=1.0, weights 0.5,0.3,0.2, confidence_weight=0.7). Leakage-safe selector
(design reviewed 8/10 PASS). Evidence dir: ccrp_ablation_sports/.

## Main test metrics (10,000 events x 101 candidates, same-candidate protocol)
HR@5/10/20  = 0.2672 / 0.3793 / 0.5183
NDCG@5/10/20 = 0.1920 / 0.2281 / 0.2631
MRR = 0.2025
Gates: degeneracy_audit_ok=True, tie_pair_rate=0.3896 (gate 0.70),
constant_score_event_rate=0.0002 (gate 0.02). Consistent with the earlier formal
C-CRP run (NDCG@10 ~0.233), confirming correctness.

## HONEST FINDING — component ablations are near-inert on Sports
Leave-one-component-out at the frozen main config (test NDCG@10):
- full (main):                0.2281
- without_boundary_uncertainty:0.2281  (identical)
- without_calibration_gap:     0.2283  (slightly BETTER)
- without_evidence_support:    0.2279
- without_counterevidence:     0.2295  (BETTER)
- without_risk_penalty:        0.2287  (slightly BETTER)
- confidence_only:             0.2287  (slightly BETTER)
- evidence_only:               0.1811  (much worse; degeneracy_audit_ok=False)
- confidence_plus_evidence:    0.2277

Interpretation (must be reported honestly, not hidden): on Sports, removing the
uncertainty/risk components (boundary uncertainty, calibration gap, counterevidence,
risk penalty) is NEUTRAL-to-slightly-POSITIVE — the ranking is essentially driven
by the calibrated relevance probability alone (confidence_only ~= full). Only the
evidence channel without confidence (evidence_only) collapses. Likely cause: the
LLM-verbalized relevance signal is coarse and heavily zero-quantized (median
relevance_probability = 0.0, only 19 distinct values over 1.01M rows), so the
calibrated-posterior + risk terms have little additional signal to exploit on this
domain.

## Implications for the paper (do NOT overclaim)
- This is a single domain; the multi-domain Phase 2.5 (toys/home/tools) is needed
  before drawing a general conclusion about component value.
- If the pattern holds across domains, the paper must either (a) honestly report
  the components as weak/redundant under coarse LLM signals (a finding in itself,
  consistent with the project's miscalibration motivation), or (b) revisit the
  component design / signal granularity. Do NOT present the full C-CRP as
  component-justified on Sports alone.
- The risk-adjusted formula base*(1-u)^eta is near-identity here because eta=1.0
  with small u; the eta hyperparameter study (next module) will probe this directly.

## Next
- Component-ablation summary module + observation/motivation + hyperparameter
  sweep consume ccrp_selected_test_scored_rows.csv (server-side, 377MB).
- Extend the full chain to toys/home/tools; compare whether components are inert
  everywhere or Sports-specific.
- Statistical: feed ccrp_selected_test_scores.csv through the same-candidate
  importer + main_build_domain_official_comparison.py for Holm-corrected paired
  tests vs the 8 official baselines.
