# Claim-Evidence Map

Every claim in the paper mapped to its supporting evidence.

## Main Claims

| # | Claim | Evidence | Source | Strength |
|---|-------|----------|--------|----------|
| C1 | All baselines exhibit severe miscalibration (ECE 0.31-0.66) | Baseline calibration diagnostic | `outputs/baseline_calibration_diagnostic/` | Strong (16 measurements, 4 methods × 4 domains) |
| C2 | MCE > 0.88 universally | Same as C1 | Same | Strong |
| C3 | C-CRP v3 achieves SOTA on Books (NDCG@10 +21.5%) | Main comparison table | `outputs/ccrp_v3_formal/books/report.json` | Strong (10,000 users) |
| C4 | C-CRP v3 achieves SOTA on Electronics (NDCG@10 +52.5%) | Main comparison table | `outputs/ccrp_v3_formal/electronics/report.json` | Strong (10,000 users) |
| C5 | C-CRP v3 is competitive on Beauty (#2) | Main comparison table | `outputs/ccrp_v3_formal/beauty/report.json` | Strong (973 users) |
| C6 | C-CRP v3 produces better-calibrated scores (ECE 0.24) | C-CRP diagnostic | `outputs/beauty_supplementary_smallerN_100neg_qwen3_shadow_v1/tables/diagnostic_metrics.csv` | Moderate (beauty only for our method's ECE) |
| C7 | Performance correlates with text information density | Domain analysis | Title length + NDCG correlation | Moderate (observational) |

## Ablation Claims

| # | Claim | Evidence | Source | Strength |
|---|-------|----------|--------|----------|
| A1 | Uncertainty decomposition hurts performance | Formal C-CRP beauty result | `outputs/beauty_ccrp_formal_selected_same_candidate/` | Strong (same data, same users) |
| A2 | Enhanced prompt (v4) hurts movies | V4 movies result | `outputs/ccrp_v4_enhanced/movies/report.json` | Strong (10,000 users) |
| A3 | Temperature scaling is mathematically ineffective for ranking | Mathematical proof | Monotonic transformation preserves order | Definitive |

## Hedged Claims (need careful wording)

| # | Claim | Issue | Recommended Wording |
|---|-------|-------|---------------------|
| H1 | "Best-calibrated scores across all domains" | Only have our ECE for beauty, not 3 other domains from diagnostic | "significantly better-calibrated on the tested domain" |
| H2 | "Universal phenomenon" | 4 baselines × 4 domains, not all 8 | "observed across all tested methods and domains" |
| H3 | Movies performance | Not SOTA | "competitive performance" not "state-of-the-art" |

## Missing Evidence (limitations to acknowledge)

| Item | Status | Impact |
|------|--------|--------|
| Statistical significance (multiple seeds) | Not done | Must acknowledge in limitations |
| Our method's ECE for books/electronics/movies | Diagnostic running but raw_scores exist | Can compute offline |
| Full-catalog evaluation | Not done | Acknowledge as future work |
| Other LLM backbones | Not done | Acknowledge as future work |
