# Unified Method Matrix Interpretation - 2026-05-04

This note interprets the server-generated
`outputs/summary/unified_method_matrix_week77_shadow.csv`.

The table compares:

- Week7.7 direct candidate ranking.
- Week7.7 structured-risk rerank.
- Week7.7 SRPD LoRA ranker variants.
- Week7.9 full-replay `shadow_v1`.
- Week7.9 diagnostic `shadow_v6`.

## Main Finding

`shadow_v6` is promising but should remain a diagnostic bridge for now.

It is competitive in Beauty, Electronics, and Movies, and it improves over the
shadow full-replay direct anchor in all four domains. However, it is not a
clear replacement for the self-trained SRPD framework because Books still
strongly favors SRPD-v2.

## Domain-Level Readout

| domain | best checked row | NDCG@10 | MRR | current interpretation |
| --- | --- | ---: | ---: | --- |
| beauty | `shadow_v6` / `shadow_v1` diagnostic, essentially tied with `SRPD-v2` | 0.6353973143 | 0.5184823570 | v6 is non-destructive and comparable, but the margin over SRPD-v2 is tiny and diagnostic-scoped. |
| books | `SRPD-v2` | 0.7057074243 | 0.6117666667 | SRPD remains much stronger; v6 is useful but not main-method competitive here. |
| electronics | `shadow_v6` diagnostic | 0.6631285871 | 0.5534666667 | v6 slightly exceeds SRPD-v5, but needs aligned protocol/statistics before promotion. |
| movies | `shadow_v6` diagnostic | 0.5733843398 | 0.4393333333 | v6 slightly exceeds structured-risk, but the gain is very small and diagnostic-scoped. |

## Current Method Roles

### SRPD

SRPD remains the current self-trained framework line.

Evidence:

- Beauty: `SRPD-v2` NDCG@10 `0.6353658183`, MRR `0.5184480987`.
- Books: `SRPD-v2` NDCG@10 `0.7057074243`, MRR `0.6117666667`.
- Electronics: `SRPD-v5` NDCG@10 `0.6621001801`, MRR `0.5528333333`.
- Movies: SRPD does not beat structured-risk; best SRPD row is `SRPD-v4`,
  NDCG@10 `0.5463885554`, MRR `0.4043666667`.

Interpretation:

- SRPD is the strongest current trainable model/framework evidence on
  Beauty/Books/Electronics.
- SRPD is not a universal winner because Movies remains weak.

### Structured Risk

Structured-risk rerank remains the strongest non-trained uncertainty-aware
reference in the Week7.7 export.

Evidence:

- Movies structured-risk NDCG@10 `0.5731826886`, MRR `0.4391000000`, stronger
  than all checked SRPD rows.
- On Electronics, structured-risk is tied with direct ranking at NDCG@10
  `0.6583007652`, while SRPD-v5 and v6 are higher.

Interpretation:

- Structured-risk is still the safest handcrafted baseline/reference.
- It is the main fallback for domains where SRPD does not transfer.

### Shadow V1

`shadow_v1` is the current full-replay winner signal source.

Evidence:

- Four-domain full replay is `ready_with_noisy`.
- Its ranking metrics are close to the shadow direct anchor and form the signal
  source for v6.

Interpretation:

- `shadow_v1` should be treated as a signal source, not a final ranking method.
- It is the right conservative input to v6.

### Shadow V6

`shadow_v6` is the current signal-to-decision diagnostic bridge.

Evidence:

- Beauty: NDCG@10 `0.6353973143`, MRR `0.5184823570`.
- Books: NDCG@10 `0.6568907475`, MRR `0.5460666667`.
- Electronics: NDCG@10 `0.6631285871`, MRR `0.5534666667`.
- Movies: NDCG@10 `0.5733843398`, MRR `0.4393333333`.

Interpretation:

- v6 is not merely a smoke test anymore; it has real same-data diagnostic value.
- v6 should not yet be promoted as the main method because it is diagnostic
  scope and loses to SRPD on Books.
- v6 is a strong bridge/ablation candidate and a possible next trainable
  decision-supervision source.

## Paper-Safe Claim

Safe wording:

> SRPD is currently the strongest self-trained ranking framework line on three
> of four checked domains, while shadow_v6 provides a conservative
> signal-to-decision bridge that is competitive on Beauty/Electronics/Movies
> but remains diagnostic until aligned protocol and external baselines are
> completed.

Unsafe wording:

> shadow_v6 is the new main method and beats all baselines.

Why unsafe:

- Books is clearly stronger under SRPD-v2.
- v6 rows are `week7_9_shadow_diagnostic`, not Week7.7 paper-candidate export.
- External paper baselines are not yet same-schema reproduced.
- Statistical significance has not been checked.

## Next Required Gates

1. Align v6 to the Week7.7 protocol or explicitly document why the current
   shadow full-replay anchor differs from the Week7.7 direct/SRPD anchor.
2. Run threshold sensitivity for v6:
   `gate_threshold`, `uncertainty_threshold`, and `anchor_conflict_penalty`.
3. Add same-candidate external baselines:
   SASRec, BERT4Rec, GRU4Rec, LightGCN or selected runnable methods from
   `Paper/BASELINE/NH` and `Paper/BASELINE/NR`.
4. Run paired statistical tests before calling any row a winner.
5. Keep SRPD as the trainable-framework main line while treating v6 as the
   bridge toward future Decision LoRA or signal-derived training targets.
