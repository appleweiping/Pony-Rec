# Significance prose edits (8-domain paired tests) — human to apply

Computed 2026-06-14 (agent:cc). The #1 gap-to-8 blocker from both reviewers
(paired Holm-corrected bootstrap on only 4/8 domains) is now closed: paired tests
exist for ALL 8 domains. Numbers below come from
`outputs/summary/paper_critical/significance_all8/` (per-domain `*_paired_summary.json`
+ `all8_domains_significance_summary.{csv,json}`), and the new
`Paper/tables/significance_summary.tex` (8-domain signed version, already replaced
and verified to compile).

Method is IDENTICAL to the new-4 gate: paired per-event bootstrap, 2000 resamples,
95% percentile CI, two-sided Wilcoxon signed-rank (Pratt) p-values, Holm correction
at alpha=0.05 over the 56-test family per domain (8 baselines x 7 metrics).

C-CRP per-event source for the original-4 = `outputs/ccrp_v3_formal/<domain>/user_ranks.jsonl`
(its `pos_rank` is 0-indexed; +1 reproduces the published `main_comparison_table.csv`
metrics EXACTLY — verified per domain, `ccrp_event_means_match_published=true`).
Each baseline dir was selected by matching its `tables/ranking_metrics.csv` NDCG@10
and HR@10 to the authoritative published numbers (all 8x4 matched to <1e-6; 0 errors).

## The authoritative 8-domain signed counts (out of 56 tests each)

| Domain      | C-CRP rank (NDCG@10) | C-CRP ahead (Holm-sig.) | C-CRP behind (Holm-sig.) | n.s. |
|-------------|----------------------|-------------------------|--------------------------|------|
| Books       | 1 | 54/56 | 1/56  | 1/56  |
| Electronics | 1 | 56/56 | 0/56  | 0/56  |
| Sports      | 1 | 56/56 | 0/56  | 0/56  |
| Toys        | 1 | 56/56 | 0/56  | 0/56  |
| Home        | 1 | 56/56 | 0/56  | 0/56  |
| Tools       | 1 | 56/56 | 0/56  | 0/56  |
| Beauty      | 2 | 14/56 | 0/56  | 42/56 |
| Movies      | 5 | 16/56 | 22/56 | 18/56 |

Signed-label rule (conservative):
- **ahead** = Holm-significant AND delta>0 AND bootstrap 95% CI lower bound > 0.
- **behind** = Holm-significant AND delta<0 AND bootstrap 95% CI upper bound < 0.
- **n.s.** = not Holm-significant, OR Holm-significant but CI straddles 0 (negligible effect).

Key honest facts to preserve:
- **Books**: 54/56 ahead. The 1 "behind" = HR@20 vs LLMEmb (delta=-0.039). The 1 n.s. = HR@10 vs LLMEmb. (C-CRP still rank-1 by NDCG@10.)
- **Electronics**: clean 56/56 ahead.
- **Beauty** (rank-2, behind ProEx): C-CRP trails ProEx on all 7 metrics, but NONE of those gaps is Holm-significant (all CIs straddle 0) -> **zero significant losses**. C-CRP is significantly ahead of the 7 weaker baselines on 14 pairs.
- **Movies** (rank-5): C-CRP is significantly **behind** the 4 stronger baselines (LLMEmb, LLM2Rec, IRLLRec, RLMRec) on **22** metric/baseline pairs, ahead of weaker baselines on 16. The n.s. column (18) includes 5 tests that are Wilcoxon-significant but whose bootstrap CI straddles 0 (negligible effect, conservatively non-significant).

---

## EDIT 1 — `sections/results.tex` (the "For the four domains..." passage, ~lines 24-30)

REPLACE this sentence (verbatim OLD):

> For the four domains with full per-event signal rows
> (Sports, Toys, Home, Tools), the paired test family contains 56 comparisons
> (eight baselines times seven metrics) and all 56 deltas are positive and
> Holm-significant in every one of them. Table~\ref{tab:significance_summary}
> summarizes that paired-test gate, and Table~\ref{tab:full_official_ndcg10}

WITH (NEW):

> For every one of the eight domains the paired-test family contains 56
> comparisons (eight baselines times seven metrics). On the six domains where
> \method{} ranks first (Sports, Toys, Home, Tools, Electronics: 56/56;
> Books: 54/56), almost all deltas are positive and Holm-significant; the only
> exceptions on Books are HR@20 versus LLMEmb (a significant loss) and HR@10
> versus LLMEmb (not significant). On the two domains where \method{} does not
> rank first we report the losses with the same paired uncertainty: on Beauty
> (rank~2) \method{} trails ProEx on every metric but none of those gaps is
> Holm-significant, while it is significantly ahead of the seven weaker baselines
> on 14 pairs; on Movies (rank~5) \method{} is significantly behind the four
> stronger baselines on 22 metric/baseline pairs and significantly ahead of the
> weaker baselines on 16. Table~\ref{tab:significance_summary}
> summarizes the signed paired-test outcomes for all eight domains, and
> Table~\ref{tab:full_official_ndcg10}

(Note: the trailing "shows the complete per-domain NDCG@10 ranking..." sentence is unchanged.)

---

## EDIT 2 — `sections/experiments.tex` (Metrics and Statistics paragraph, ~lines 99-103)

REPLACE (verbatim OLD):

> family of 56 tests in that domain. The Holm-corrected paired-test gate is
> reported for the four domains where \method{} ranks first on every metric
> (Sports, Toys, Home, Tools); for the four original domains we report ranks and
> per-metric improvements directly.

WITH (NEW):

> family of 56 tests in that domain. The Holm-corrected paired tests are
> reported for all eight domains, including the two where \method{} does not
> rank first (Beauty, Movies), so that the losses carry the same paired
> uncertainty as the wins; Table~\ref{tab:significance_summary} gives the signed
> counts (ahead / behind / non-significant) out of 56 per domain.

---

## EDIT 3 — `sections/introduction.tex` (the "On the four domains..." sentence, ~lines 90-92)

REPLACE (verbatim OLD):

> two non-winning domains explicitly (Beauty rank~2, $-11\%$; Movies rank~5,
> $-24\%$) instead of dropping them. On the four domains with full per-event
> signal rows, all 56 per-domain paired tests are positive and
> Holm-significant. The claim is scoped to same-candidate reranking, not

WITH (NEW):

> two non-winning domains explicitly (Beauty rank~2, $-11\%$; Movies rank~5,
> $-24\%$) instead of dropping them. We compute paired Holm-corrected bootstrap
> tests on all eight domains: on the six winning domains \method{} is
> significantly ahead on all 56 metric/baseline pairs (54/56 on Books), while on
> the two losing domains the paired tests quantify the gap honestly --- on Beauty
> none of the losses to ProEx is Holm-significant, and on Movies \method{} is
> significantly behind the stronger baselines on 22 of 56 pairs. The claim is
> scoped to same-candidate reranking, not

(Optional, line 22 / line 79: the phrase "paired Holm-corrected bootstrap tests"
already in the contributions list now holds for all 8 domains — no number change
needed, but you may drop any implicit "four-domain" qualifier if present nearby.)

Also at intro line ~88 the phrase "leading on essentially all seven metrics in
those six domains" is fine for Electronics/Sports/Toys/Home/Tools (56/56) but is
slightly loose for Books (HR@20 is a significant loss, HR@10 n.s.). Consider:
"leading on all seven metrics in five of those six domains and on five of seven in
Books (HR@10/HR@20 versus LLMEmb being the exceptions)."

---

## Files produced (small, committed)

- `Paper/tables/significance_summary.tex` — REPLACED with 8-domain signed table (compiles; booktabs only; same `\label{tab:significance_summary}`).
- `outputs/summary/paper_critical/significance_all8/` — per-domain JSON+CSV for original-4 (beauty/books/electronics/movies) + `all8_domains_significance_summary.{csv,json}`.
- `scripts/experiments/main_build_original_domain_official_comparison.py` — the generalized compute script (server-side, CPU).
