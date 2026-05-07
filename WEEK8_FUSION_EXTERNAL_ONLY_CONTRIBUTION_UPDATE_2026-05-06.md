# Week8 Fusion and External-Only Contribution Update - 2026-05-06

This note updates the current conclusion after adding six senior-recommended
LLM-rec paper-style same-schema baselines, the ours+external rank-fusion
diagnostic, and the external-only phenomenon diagnostic.

## Current Conclusion

The framework should not be positioned as a universal standalone SOTA winner.
The stronger and safer conclusion is:

```text
Our framework contributes a complementary risk/decision signal that can improve
or explain strong external LLM-rec baselines under the same-candidate protocol.
```

The six-paper paper-style external block is now strong enough that the story should move
from "ours simply beats baselines" to:

- broad same-schema baseline coverage,
- observed complementarity with strong external methods,
- event-level uncertainty/complementarity phenomenon,
- larger-scale validation under a harder sampled-ranking protocol.

## Six-Paper Paper-Style Baseline Readout

Completed same-candidate external paper-style baselines:

```text
LLM2Rec-style Qwen3-8B Emb. + SASRec
LLM-ESR-style Qwen3-8B Emb. + LLMESR-SASRec
LLMEmb-style Qwen3-8B Emb. + SASRec
RLMRec-style Qwen3-8B GraphCL
IRLLRec-style Qwen3-8B IntentRep
SETRec-style Qwen3-8B Identifier
```

Strongest observed paper-project rows:

| domain | strongest observed external row | NDCG@10 |
| --- | --- | ---: |
| beauty | IRLLRec-style Qwen3-8B IntentRep | 0.662061 |
| books | IRLLRec-style Qwen3-8B IntentRep | 0.716744 |
| electronics | RLMRec-style Qwen3-8B GraphCL | 0.628100 |
| movies | IRLLRec-style Qwen3-8B IntentRep | 0.707149 |

This means the external paper-style baselines are genuinely competitive. Avoid
claiming a clean standalone win for our framework across all domains, and do
not label these rows as official external reproductions.

## Ours + External Fusion Diagnostic

Rank-fusion best-by-domain rows:

| domain | fusion pair | ours weight | fused NDCG@10 | gain over best constituent |
| --- | --- | ---: | ---: | ---: |
| beauty | SRPD-best + IRLLRec-style | 0.3 | 0.663963 | +0.001902 |
| books | SRPD-best + IRLLRec-style | 0.3 | 0.730990 | +0.014247 |
| electronics | structured-risk + RLMRec-style | 0.7 | 0.676842 | +0.018542 |
| movies | structured-risk + IRLLRec-style | 0.3 | 0.708930 | +0.001781 |

Interpretation:

- The framework signal is complementary to strong external paper-project rows.
- Books and Electronics show the clearest observed fusion gain.
- Beauty and Movies show small but positive diagnostic gains.
- Because the weight sweep was selected on test, this is an upper-bound
  diagnostic unless rerun with fixed/validation-selected weights.

Safe wording:

```text
A diagnostic rank-fusion analysis suggests that our risk-aware/SRPD signal is
complementary to strong external LLM-rec baselines; in all four domains, the
best observed fusion row is higher than both of its constituents.
```

Unsafe wording:

```text
The fused method is a new main SOTA method selected on the test set.
```

## External-Only Phenomenon

When structured-risk and SRPD are removed from the candidate method set, the six
external paper-project baselines still show event-level complementarity:

| domain | best single external | best single NDCG@10 | external oracle NDCG@10 | oracle gain |
| --- | --- | ---: | ---: | ---: |
| beauty | IRLLRec-style | 0.662061 | 0.886395 | +0.224334 |
| books | IRLLRec-style | 0.716744 | 0.890521 | +0.173777 |
| electronics | RLMRec-style | 0.628100 | 0.828486 | +0.200385 |
| movies | IRLLRec-style | 0.707149 | 0.890900 | +0.183751 |

Base-rank bins show the earlier phenomenon clearly:

- rank 1: oracle gain is 0 because the best single row already ranks the
  positive first.
- rank 2-3: oracle gains are large, about +0.234 to +0.304.
- rank 4-6: oracle gains are even larger, about +0.318 to +0.381.

Disagreement bins also support the diagnostic:

- high-disagreement bins generally have larger oracle gains than low-disagreement
  bins.
- external method disagreement is a useful proxy for hard/uncertain events.
- popularity bins are mixed, so do not overclaim long-tail from this diagnostic
  alone.

Safe wording:

```text
Without using our structured-risk/SRPD method as a candidate, the external
paper-project baselines still show event-level complementarity. The external
oracle is observed higher than the best single external row, especially in
harder base-rank and high-disagreement bins.
```

Unsafe wording:

```text
Their methods implement our uncertainty mechanism.
```

## Updated Contribution Framing

The current contribution stack should be:

1. Same-schema evaluation protocol: all rows use the same split, candidate set,
   metric implementation, score import audit, and paired-test path.
2. Broad baseline coverage: four classical recommenders plus six
   senior-recommended LLM-rec paper-project same-backbone/style baselines.
3. Risk/complementarity insight: our framework is not merely another scorer;
   it captures a signal that can combine with strong external methods.
4. Event-level phenomenon: external-only oracle and disagreement bins show that
   recommendation uncertainty/complementarity exists even without our method in
   the candidate set.
5. Next large-scale validation: move Books/Electronics/Movies from 500-event
   six-candidate checks to 10k users with 100 sampled negatives plus one
   positive candidate.

## Decision

The framework is not "finished" in the sense of final paper evidence. It is
finished as a small/medium same-candidate proof block, but the next necessary
step is a larger, more standard sampled-ranking validation.

Use the Week8 large-scale 10k/100neg protocol as the next main robustness gate.
