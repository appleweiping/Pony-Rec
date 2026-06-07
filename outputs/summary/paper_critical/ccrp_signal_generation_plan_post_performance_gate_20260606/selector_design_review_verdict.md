# Selector Design Review — Verdict Log (Sports Phase 2.5)

Date: 2026-06-07. Gate: ARIS >=8/10 design-review-before-execution (mandatory).
Request: selector_design_review_request.md.

## Reviewer 1 — Claude Opus 4.8 (independent adversarial)
Score: **4/10 — FAIL**. Blocking fixes required:
1. **Ablations must not compete in selection.** Code confirmed: the selection loop
   `for ablation in ablations: ... best = argmax NDCG@10` lets a leave-one-out
   variant be crowned the main C-CRP. Fix: freeze main method to `ablation=full`,
   select hyperparameters only within `full`, then evaluate each LOO ablation at
   the frozen config in a SEPARATE table (ablations are diagnostics, not selection
   candidates).
2. **Tie-break leakage.** Code confirmed: ranking tie-break is original candidate
   index (`sorted(..., key=lambda i: (-i[1], i[2]))`) and the degeneracy gate
   allows up to `max_tie_pair_rate=0.98`. A near-constant scorer can inherit
   HR/NDCG from candidate order. Fix: (i) verify candidate order is randomized
   independently of the positive (seed in provenance); (ii) random tie-break with
   fixed seed (or average over permutations), not index; (iii) tighten
   `max_tie_pair_rate` well below 0.98; report per-config tie_pair_rate.
3. **No paired significance testing.** Add per-event paired tests (paired bootstrap
   CI or Wilcoxon) vs each of 8 baselines on primary metric(s) over 10,000 events,
   Holm/Bonferroni corrected; report effect sizes + CIs.
4. **Tuning-budget unfairness.** C-CRP selected over ~430 valid configs vs baselines
   at official defaults (zero tuning). Fix one of: give baselines equal valid-tuning
   budget; OR freeze C-CRP to one pre-registered principled config (full, eta=1,
   default weights) as main result + demote grid to sensitivity appendix; OR
   explicitly disclose+justify the asymmetric budget. (Option B safest.)
5. **Pre-register selection protocol.** Declare NDCG@10 as sole selection metric +
   exact grid before seeing test; report all secondary metrics (no k cherry-pick);
   record pre-registration in provenance.
Optional: dedupe redundant grid cells (confidence_only/evidence_only ignore
weights+eta yet enumerate them); document inter-config tie-break; report
valid->test generalization gap; persist candidate-order/tie-break seed.

## Reviewer 2 — Codex (GPT xhigh): UNAVAILABLE THIS SESSION (tooling blocker)
The local Codex CLI (v0.122.0-alpha.1) cannot run: its config sets
`service_tier=priority` which the CLI rejects ("unknown variant, expected fast or
flex"); overriding to `flex` is rejected by the API ("Unsupported service_tier");
overriding to `fast` yields "gpt-5.5 requires a newer version of Codex". So the
GPT-xhigh perspective could not be obtained. Per the review-cadence rule
(reviewer availability varies per session — use available reviewers and record
the missing perspective), substituting a second independent Claude Opus 4.8
reviewer with a distinct adversarial framing. ACTION ITEM: fix/upgrade the Codex
CLI (or its endpoint's service_tier support) so GPT-xhigh review is available for
later modules.

## Reviewer 2b — second Claude Opus 4.8 (independent, substitute for Codex)
Score: **4/10 — FAIL**. Independently CONFIRMED reviewer 1's #1/#3/#4/#5 and
REFUTED the in-script valid->test leakage (selection reads valid paths only; test
touched once; `best` carries scalar hyperparameters only; scoring is stateless).
Added a sharper finding: the selection grid ALSO lets degenerate score-modes
`confidence_only`/`evidence_only` (which bypass the risk term, ccrp.py:138-139)
win the argmax — so selection mixes the real method, its ablations, AND
non-method sub-models, then reports the argmax as "C-CRP". Compounds #1. Agrees
#1 is a true blocker. Also: tie-break is not a directional leak (positive index
uniform) but `max_tie_pair_rate=0.98` is required-fix (too loose given coarse LLM
probabilities -> heavy ties); selection-metric ties currently favor the
first-iterated `confidence_only` config. Optional: verify upstream calibrator was
fit on train/valid only (outside this script).

## Consensus
Both independent Opus 4.8 reviewers: 4/10 FAIL, same #1 blocker. GPT-xhigh
(Codex) unavailable this session (tooling). Required blocking fixes (merged):
1. Freeze `ablation=full` AND `score_mode=full` out of the selection argmax;
   select hyperparameters only within the real method; evaluate LOO ablations and
   confidence_only/evidence_only as SEPARATE diagnostics.
2. Main reported C-CRP = pre-registered principled config (full, eta=1.0, weights
   0.5/0.3/0.2); demote the grid to a sensitivity/robustness study (fixes the
   tuning-budget asymmetry vs untuned baselines).
3. Tighten max_tie_pair_rate (<=0.5), report per-event tie/degeneracy stats,
   seeded/averaged tie-break (or document the positive-index-uniformity
   justification for deterministic order).
4. Per-event Holm-corrected paired tests (bootstrap/Wilcoxon) vs each baseline,
   with CIs, for the test row to be paper-eligible.
5. Pre-register selection metric (NDCG@10) + full grid + rule (commit hash/time)
   before any test metric is computed.

## Disposition
Design FAILS the >=8/10 gate. Do NOT execute the selector as-is. Implement the
five blocking fixes, then re-review before execution.

## RESOLUTION — fixes implemented + re-review PASS (2026-06-07)
Selector rewritten (scripts/misc/main_select_ccrp_variant_on_valid.py):
1. Valid sweep restricted to score_mode=full, ablation=full only (eta x weight_grid;
   confidence_weight is a no-op under full so it is fixed, not swept). Ablations +
   degenerate score-modes moved to a SEPARATE ccrp_ablation_diagnostics.csv at the
   frozen main config (row_kind main vs diagnostic), never selection candidates.
2. Seeded order-independent tie-break (sha256 of seed:event:user:item) replaces the
   original-candidate-index key; --max_tie_pair_rate default tightened to 0.5 and
   threaded into audit_score_degeneracy; seed + threshold recorded in provenance.
3. --main_config_mode defaults to `preregistered` (full/full, eta=1.0, weights
   0.5,0.3,0.2, cw=0.7) as the MAIN reported result; the valid grid is a disclosed
   sensitivity study (fixes tuning-budget asymmetry vs untuned baselines).
4. Selector emits importable same-candidate scores; downstream
   main_build_domain_official_comparison.py provides per-event Wilcoxon + Holm +
   bootstrap CIs for the paper-eligible test row (verified wired).
5. ccrp_selection_preregistration.json written BEFORE the test evaluation.
Re-review by independent Claude Opus 4.8: **8/10 PASS**, all five issues PASS, no
validity-threatening regression (main path computed+written before the diagnostics
try/except; main_cfg vs best handled cleanly). Cosmetic nits (no-op cw sweep, dead
--score_modes/--ablations args, dead _parse_csv_list) also fixed. Verified:
py_compile OK; ccrp test suite 6 passed; end-to-end smoke run produces all 5
artifacts with valid sweep correctly restricted to full/full.
GPT-xhigh (Codex) remained unavailable (tooling) — ACTION ITEM stands.
Design now CLEARED for execution on real Sports data.
