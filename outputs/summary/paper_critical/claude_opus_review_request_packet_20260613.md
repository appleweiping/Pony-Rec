# Claude Opus Review Request Packet

- Created UTC: `2026-06-13T05:08:28.936887+00:00`
- OK: `true`
- Claude review needed: `true`
- Existing score floor: `8.0`
- Failed Claude attempts: `10`
- Prompt sha256: `3b01c2c9d3622aa8bfc9a29db57b496d0463f6c5595aa42b9fcd4cc7043153bd`

## Missing Perspectives

- `explicit_claude_opus_review`

## Expected Review JSON

- Recommended path: `outputs/summary/paper_critical/claude_opus_review_20260613.json`

```json
{
  "reviewer": "Must include claude and opus, e.g. claude-opus.",
  "created_at_utc": "ISO-8601 UTC timestamp.",
  "source": "Tool, thread, or manual channel that produced the review.",
  "score_0_to_10": "Numeric top-conference review score.",
  "verdict": "ACCEPT, WEAK_ACCEPT, CONDITIONAL_PASS, BORDERLINE, WEAK_REJECT, or REJECT.",
  "claim_boundary_ok": "Boolean; true only if scoped same-candidate claim is respected.",
  "final_submission_ready_claim_allowed": "Boolean; must stay false while final gates are open.",
  "kill_argument": "The strongest remaining rejection argument.",
  "major_concerns": "Array of concrete concerns.",
  "required_changes": "Array of concrete changes before submission/final-ready claim.",
  "remaining_blockers_acknowledged": "Array naming external/manual blockers the review did not waive.",
  "valid_review_evidence": "Boolean; true only for a complete substantive review."
}
```

## Prompt

```json
{
  "review_role": "Claude Opus independent hostile top-conference reviewer",
  "claim_scope": "Task-grounded calibrated uncertainty improves controlled same-candidate candidate ranking/reranking reliability. Do not expand this into a full-catalog recommender SOTA claim.",
  "evidence_summary": {
    "official_same_candidate_rows": "32 official baseline rows + 4 C-CRP rows on sports/toys/home/tools",
    "protocol": "10k users per new domain, 101 same candidates, Qwen3-8B backbone, full @5/@10/@20 + MRR metrics",
    "comparison_result": "C-CRP rank 1 on all 7 metrics with 56/56 positive Holm-significant paired tests per domain",
    "phase_2_5_limits": [
      "observation module is motivation-only, not causal or SOTA evidence",
      "component ablation is supplementary diagnostic-only; do not claim every component is necessary",
      "hyperparameter analysis is supplementary stability/sensitivity evidence"
    ]
  },
  "panel_state": {
    "existing_panel_score_floor": 8.0,
    "existing_reviewers": [
      "Faraday existing subagent",
      "Avicenna existing subagent",
      "Meitner existing subagent"
    ],
    "panel_consensus": {
      "score_floor": "8.0/10",
      "score_ceiling": "8.3/10",
      "verdict": "weak_accept_conditional_pass_under_scope_guards",
      "new_experiment_required": false,
      "claim_boundary_ok": true,
      "final_submission_ready": false
    },
    "explicit_claude_opus_present": false,
    "missing_perspectives": [
      "explicit_claude_opus_review"
    ],
    "failed_claude_attempts": {
      "count": 10,
      "unique_errors": [
        "Claude CLI did not return JSON output"
      ],
      "sources": [
        "mcp__claude_review.review_start job 999276b352b940c0b957e3be212f9919",
        "mcp__claude_review.review_start job b9757b214eb84142bce54dd28f7e258c",
        "mcp__claude_review.review_start job 2d4e39f665de4a848138ca2fc9630357",
        "mcp__claude_review.review synchronous call with tools disabled",
        "mcp__claude_review.review minimal JSON-oriented no-tools call",
        "mcp__claude_review.review synchronous JSON-only call with model=opus and tools disabled",
        "mcp__claude_review.review synchronous JSON-only call with model=opus, tools disabled, seventh attempt",
        "mcp__claude_review.review_start async job b1b88420168a4e498029a00a8695098a with model=opus and tools disabled",
        "mcp__claude_review.review_start async job a3863723466147e9b9b849cf994ca8fd with model=opus and tools disabled",
        "mcp__claude_review.review_start async job b6e19654680c457d8be4845e168ce251 with model=opus and tools disabled"
      ]
    },
    "gpt55_verdict": "CONDITIONAL_PASS",
    "gpt55_score_0_to_10": 8.0,
    "gpt55_critical_blockers": [
      "Final submission gate is explicitly false: external proceedings metadata and manual submission-system gates remain open.",
      "ProMax final ACM page range is missing; Crossref /works and DOI resolver for 10.1145/3805712.3809600 still return 404.",
      "Private manual submission-system confirmation is not completed; author/COI/reviewer/declaration/final-preview fields remain manual/private pending."
    ]
  },
  "claim_audit_state": {
    "ok": true,
    "paper_evidence_ready_for_drafting": true,
    "final_submission_ready": false,
    "claim_status_counts": {
      "SUPPORTED": 6,
      "CONTRADICTED": 2,
      "UNSUPPORTED": 2
    },
    "non_supported_or_contradicted_claims": [
      {
        "id": "C5",
        "status": "CONTRADICTED",
        "allowed_wording": "Do not make this claim.",
        "forbidden_wording": "necessary; essential; each component contributes; removing any component hurts"
      },
      {
        "id": "C7",
        "status": "CONTRADICTED",
        "allowed_wording": "Do not claim risk-penalty necessity; discuss eta as stable/sensitivity-only.",
        "forbidden_wording": "eta must be positive; risk penalty is necessary; risk penalty uniformly improves"
      },
      {
        "id": "C9",
        "status": "UNSUPPORTED",
        "allowed_wording": "Evidence is ready for strict paper drafting and subsequent manuscript-level claim/citation review.",
        "forbidden_wording": "submission-ready; final READY; cleared for camera-ready"
      },
      {
        "id": "C10",
        "status": "UNSUPPORTED",
        "allowed_wording": "Controlled same-candidate ranking/reranking reliability under the tested protocol.",
        "forbidden_wording": "full-catalog SOTA; universal cross-domain winner beyond tested same-candidate domains"
      }
    ]
  },
  "current_gates": {
    "review_continuation_ready": true,
    "local_release_candidate_ready": true,
    "final_submission_ready": false,
    "gate_summary": {
      "claim_audit_ok": true,
      "closure_other_blockers": [],
      "closure_packet_ok": true,
      "external_proceedings_metadata_ready": false,
      "manual_submission_system_ready": false,
      "panel_ok": true,
      "promax_probe_expected_blocked": true,
      "promax_probe_ok": true,
      "promax_public_metadata_ready": false,
      "release_candidate_stack_ok": true,
      "submission_package_audit_ok": true
    },
    "classified_remaining_blockers": {
      "external_proceedings_metadata": [
        "ProMax final ACM page range and ACM/Crossref registry visibility must be rechecked immediately before submission.",
        "promax:final_page_range_missing_in_bib",
        "promax:crossref_registry_not_visible:status=404",
        "promax:doi_resolver_not_visible:status=404",
        "external_proceedings_metadata_not_ready",
        "confirm_external_proceedings_metadata:external_proceedings_metadata_ready_not_closed",
        "promax:crossref_registry_not_visible",
        "promax:doi_resolver_not_visible"
      ],
      "manual_submission_system": [
        "Final manual submission-system metadata/format checklist is not closed.",
        "manual_submission_system_not_ready",
        "manual_submission_system_items_not_confirmed"
      ],
      "other": [],
      "review_panel_coverage": [
        "review_panel_coverage_not_complete",
        "explicit_claude_opus_review"
      ]
    }
  },
  "review_instructions": [
    "Return a single JSON object matching the requested schema; do not include markdown.",
    "Be strict and identify the strongest remaining kill argument.",
    "Do not waive ProMax public proceedings metadata blockers.",
    "Do not waive private manual submission-system blockers.",
    "Set final_submission_ready_claim_allowed=false unless all final gates are genuinely closed.",
    "A score below 8.0 is allowed if the evidence does not satisfy top-conference standards."
  ],
  "required_output_schema": {
    "reviewer": "Must include claude and opus, e.g. claude-opus.",
    "created_at_utc": "ISO-8601 UTC timestamp.",
    "source": "Tool, thread, or manual channel that produced the review.",
    "score_0_to_10": "Numeric top-conference review score.",
    "verdict": "ACCEPT, WEAK_ACCEPT, CONDITIONAL_PASS, BORDERLINE, WEAK_REJECT, or REJECT.",
    "claim_boundary_ok": "Boolean; true only if scoped same-candidate claim is respected.",
    "final_submission_ready_claim_allowed": "Boolean; must stay false while final gates are open.",
    "kill_argument": "The strongest remaining rejection argument.",
    "major_concerns": "Array of concrete concerns.",
    "required_changes": "Array of concrete changes before submission/final-ready claim.",
    "remaining_blockers_acknowledged": "Array naming external/manual blockers the review did not waive.",
    "valid_review_evidence": "Boolean; true only for a complete substantive review."
  }
}
```

## Validation Command Before Attach

```bash
python -m scripts.audit.main_validate_claude_opus_review_json --review-json outputs/summary/paper_critical/claude_opus_review_20260613.json --output-json outputs/summary/paper_critical/claude_opus_review_validation_20260613.json --output-md outputs/summary/paper_critical/claude_opus_review_validation_20260613.md
```

## Follow-Up Command

```bash
python -m scripts.audit.main_build_review_continuation_packet --additional-review-json outputs/summary/paper_critical/claude_opus_review_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_retry_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_third_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_sync_notools_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_minimal_json_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_sixth_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_seventh_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_eighth_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_ninth_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_tenth_20260613.json
```
