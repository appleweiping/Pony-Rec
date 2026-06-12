# Manuscript Claim Audit After Structural Expansion

ARIS skill: `aris-paper-claim-audit`

Claim-text verdict: `READY_WITH_SCOPE_GUARDS`

Submission-gate verdict: `NEEDS_SECTION_REVIEW_BEFORE_SUBMISSION`

Final submission ready: `false`

## Summary

- Claims extracted: 12
- Supported: 11
- Weakly supported: 1
- Unsupported: 0
- Overclaimed: 0
- Contradicted: 0

## Supported Claim Families

- Same-candidate protocol: all methods rank the same 101 candidates per user
  under the same importer, metrics, provenance checks, and paired tests.
- Method description: C-CRP decomposes uncertainty into boundary ambiguity,
  calibration gap, evidence support, and counterevidence, then applies
  risk-adjusted reranking.
- Main comparison: C-CRP ranks first against eight official-code-level baselines
  across Sports, Toys, Home, and Tools on HR@5/@10/@20, NDCG@5/@10/@20, and MRR.
- Statistical gate: all 56 per-domain C-CRP-vs-baseline tests are positive and
  Holm-significant in each domain.
- Motivation diagnostic: event-level uncertainty stratifies ranking reliability
  in all four domains.
- Component diagnostic: several uncertainty terms are weak or redundant rather
  than uniformly necessary.
- Hyperparameter diagnostic: validation-selected controls are stable within the
  pre-registered tolerance on NDCG@10.
- Scope boundary: the manuscript repeatedly denies full-catalog or universal
  recommender SOTA claims.

## Weak Claim

- The broad motivation that LLM recommendation scores do not by themselves
  reveal decision reliability is contextually supported by the framing and
  related work, but it is not a standalone theorem or exhaustive empirical
  statement. Current wording is acceptable as motivation.

## Red-Flag Terms

- `significant` is tied to Holm-corrected paired tests.
- `all` is limited to enumerated four-domain/eight-baseline/56-test statements
  or explicit negations.
- `necessary` appears mainly in denials of component/risk-penalty necessity.
- `SOTA`, `causal`, and `optimal` appear only in disclaimers or non-claim
  boundaries.

## Required Next Steps

- Run final citation spot-check against `Paper/references.bib` before any
  submission-readiness claim.
- Run section-level top-conference review and apply requested edits.
- Rerun this audit after any future performance, novelty, or generality claims
  are added.
