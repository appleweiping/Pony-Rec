# Manuscript Claim Audit After Section Review

ARIS skill: `aris-paper-claim-audit`

Claim-text verdict: `READY_WITH_SCOPE_GUARDS`

Submission gate verdict: `NEEDS_FINAL_PANEL_REVIEW_BEFORE_SUBMISSION`

Final submission ready: `false`

## Summary Statistics

- Total claims extracted: 12
- Supported: 12
- Weakly supported: 0
- Unsupported: 0
- Overclaimed: 0
- Contradicted: 0

## Claim Table

| ID | Claim | Status | Evidence |
| --- | --- | --- | --- |
| M1 | Same 101-candidate score schema/importer/metrics/provenance/paired-test protocol for all methods. | SUPPORTED | Ledger, same-candidate gates, evidence packaging audits. |
| M2 | Unified Qwen3-8B use whenever an LLM or LLM-derived representation is required. | SUPPORTED | Official baseline provenance and fairness policy. |
| M3 | C-CRP is a calibrated posterior with uncertainty components and a validation-controlled ranking family. | SUPPORTED | Method equations, signal provenance, selector grid, fail-closed gates. |
| M4 | C-CRP ranks first against eight official-code-level baselines on all HR/NDCG/MRR metrics across four domains. | SUPPORTED | Cross-domain certificate, ledger, main table, rank-by-metric table. |
| M5 | All 56 per-domain paired tests are positive and Holm-significant in each domain. | SUPPORTED | Paired-test summary, 2,000 paired event bootstrap samples, Holm alpha 0.05. |
| M6 | C-CRP event-level uncertainty stratifies ranking reliability in all four domains. | SUPPORTED | Observation four-domain provenance and uncertainty stratification table. |
| M7 | Component ablations show several terms are weak, neutral, or redundant. | SUPPORTED | Component four-domain provenance and ablation NDCG@10 summary. |
| M8 | Eta and uncertainty-weight settings are stable within the pre-registered NDCG@10 tolerance. | SUPPORTED | Hyperparameter four-domain provenance and stability rows. |
| M9 | Positive risk penalty is not necessary or uniformly beneficial in current sweeps. | SUPPORTED | Eta test-best zero in all four domains and risk-penalty ablation. |
| M10 | The result is controlled same-candidate reranking, not full-catalog SOTA. | SUPPORTED | Protocol and limitations text. |
| M11 | Official-code-level baselines preserve declared mechanisms and use default/recommended hyperparameters under the unified score contract. | SUPPORTED | Baseline provenance table and evidence ledger. |
| M12 | Limitations include LLM cost/latency, prompt/parser dependence, sampled negatives, external adapter policy, and no full-catalog claim. | SUPPORTED | Discussion limitations. |

## Red-Flag Scan

- `significant` is tied to paired Holm tests and is backed.
- `SOTA` appears only as a negated full-catalog boundary.
- No first/novel priority claim is used.
- Component and risk-penalty necessity claims are explicitly negated.
- The text uses hyperparameter stability/sensitivity rather than broad
  robustness.

## Consistency

Abstract, introduction, method, experiments, results, discussion, and
conclusion are consistent on the protocol, baseline count, metrics, uncertainty
diagnostic scope, and risk-penalty caveat.

## Remaining Process Blockers

- Run another section-level top-conference review on the latest draft.
- Claude Opus reviewer perspective remains unavailable in this session.
- Re-check ProEx and ProMax proceedings metadata immediately before final
  submission.
- Do not convert this claim-text pass into final submission readiness without
  the remaining review process.
