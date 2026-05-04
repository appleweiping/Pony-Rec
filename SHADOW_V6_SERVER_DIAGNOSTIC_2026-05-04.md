# Shadow V6 Server Diagnostic - 2026-05-04

This note records the first four-domain `shadow_v6` diagnostic run on the
server rescue checkout. It is not a paper-result promotion.

## Server Checkout

- Code checkout:
  `~/projects/pony-rec-rescue-shadow-v6`
- Branch:
  `codex/pre-cursor-shadow-v6-rescue`
- Code commit at checkout:
  `5d8f89e Record server experiment asset audit`
- Artifact class used:
  `diagnostic`
- Winner signal used:
  `shadow_v1`

The run reused existing server outputs instead of rerunning ranking or
pointwise shadow inference.

## Inputs

Ranking anchor inputs:

- `~/projects/uncertainty-llm4rec-codex-apr12-preserve-local/outputs/{domain}_qwen3_shadow_rank_full_replay/predictions/rank_predictions.jsonl`

Winner-signal inputs:

- `~/projects/uncertainty-llm4rec-codex-apr12-preserve-local/outputs/{domain}_qwen3_shadow_v1_full_replay_pointwise/calibrated/test_calibrated.jsonl`

Checked input sizes:

- Beauty: ranking `973` events, signal `5838` rows.
- Books: ranking `500` events, signal `3000` rows.
- Electronics: ranking `500` events, signal `3000` rows.
- Movies: ranking `500` events, signal `3000` rows.

## Outputs

Outputs were written under:

- `~/projects/pony-rec-rescue-shadow-v6/outputs/{domain}_qwen3_shadow_v6_full_replay_structured_risk`

Per-domain output files include:

- `reranked/shadow_v6_bridge_rows.jsonl`
- `reranked/shadow_v6_decision_reranked.jsonl`
- `tables/shadow_v6_bridge_rows.csv`
- `tables/shadow_v6_bridge_summary.csv`
- `tables/rerank_results.csv`

## Bridge Quality

All four domains had complete signal coverage:

| domain | bridge_rows | matched_signal_rate | fallback_rate | mean_correction_gate | mean_signal_uncertainty | mean_anchor_disagreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| beauty | 5838 | 1.0 | 0.0 | 0.3935716389 | 0.5064973562 | 0.4182071645 |
| books | 3000 | 1.0 | 0.029 | 0.4376150273 | 0.4304605754 | 0.4317589309 |
| electronics | 3000 | 1.0 | 0.0 | 0.4394579472 | 0.4418843327 | 0.4246298447 |
| movies | 3000 | 1.0 | 0.0 | 0.4143146052 | 0.4746745378 | 0.4224805862 |

Interpretation:

- The bridge did not suffer missing-signal fallback.
- Beauty is almost unchanged because only about one event was reranked.
- Books has the largest intervention rate and largest metric gain.
- Electronics and Movies show smaller but positive diagnostic movement.

## Ranking Metrics

| domain | method | sample_count | NDCG@10 | MRR | changed_ranking_fraction | avg_position_shift |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| beauty | direct_candidate_ranking | 973 | 0.6353658183 | 0.5184480987 | 0.0 | 0.0 |
| beauty | shadow_v6_decision_bridge | 973 | 0.6353973143 | 0.5184823570 | 0.0010277492 | 0.0006851662 |
| books | direct_candidate_ranking | 500 | 0.6365719712 | 0.5195000000 | 0.0 | 0.0 |
| books | shadow_v6_decision_bridge | 500 | 0.6568907475 | 0.5460666667 | 0.3040000000 | 0.1160000000 |
| electronics | direct_candidate_ranking | 500 | 0.6574286221 | 0.5460333333 | 0.0 | 0.0 |
| electronics | shadow_v6_decision_bridge | 500 | 0.6631285871 | 0.5534666667 | 0.1080000000 | 0.0426666667 |
| movies | direct_candidate_ranking | 500 | 0.5723849700 | 0.4380333333 | 0.0 | 0.0 |
| movies | shadow_v6_decision_bridge | 500 | 0.5733843398 | 0.4393333333 | 0.0360000000 | 0.0110000000 |

Metric deltas:

| domain | delta_NDCG@10 | delta_MRR |
| --- | ---: | ---: |
| beauty | +0.0000314960 | +0.0000342583 |
| books | +0.0203187763 | +0.0265666667 |
| electronics | +0.0056999649 | +0.0074333333 |
| movies | +0.0009993699 | +0.0013000000 |

## Current Interpretation

This run supports `shadow_v6` as a conservative bridge from full-replay
`shadow_v1` signal into ranking decisions.

The strongest diagnostic evidence is Books, where v6 improves NDCG@10 from
`0.6365719712` to `0.6568907475` and MRR from `0.5195` to `0.5460666667`.
Electronics also improves beyond the direct anchor. Beauty is effectively a
non-destructive sanity check. Movies shows a very small positive movement.

Do not treat this as a paper result yet. It is a same-split diagnostic over
existing full-replay shadow assets, with `artifact_class=diagnostic`.

## Next Gates

Before promotion beyond diagnostic:

1. Compare v6 against the week7.7 SRPD and structured-risk tables in one
   unified matrix.
2. Verify whether the Books direct anchor differs from the earlier week7.7
   direct ranking baseline and document the reason if so.
3. Run a threshold sensitivity sweep for `gate_threshold`,
   `uncertainty_threshold`, and `anchor_conflict_penalty`.
4. Decide whether Beauty/Books small-prior `shadow_v2` or `shadow_v5` should be
   tested as alternative signal sources, without replacing `shadow_v1` as the
   conservative full-replay default.
5. Preserve the server output folders and command context before any cleanup.
