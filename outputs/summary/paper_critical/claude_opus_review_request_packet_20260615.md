# Claude Opus Review Request Packet

- Created UTC: `2026-06-14T23:31:54.160320+00:00`
- OK: `true`
- Claude review needed: `true`
- Existing score floor: `8.0`
- Failed Claude attempts: `15`
- Prompt sha256: `96f0877f8597640db261f81e6ed93c0372cb2c9f5c1772eed675ec51b8d2322a`

## Warnings

- `review_continuation_packet_not_ready_request_allowed_for_missing_claude_coverage`

## Missing Perspectives

- `explicit_claude_opus_review`

## Expected Review JSON

- Recommended path: `outputs/summary/paper_critical/claude_opus_review_20260615.json`

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

## Response Template

This template is intentionally not ready to attach until the reviewer fills it and sets `valid_review_evidence=true`.

```json
{
  "reviewer": "claude-opus",
  "created_at_utc": "YYYY-MM-DDTHH:MM:SS+00:00",
  "source": "external Claude Opus review channel or tool/job id",
  "score_0_to_10": null,
  "verdict": "CONDITIONAL_PASS",
  "claim_boundary_ok": true,
  "final_submission_ready_claim_allowed": false,
  "kill_argument": "Replace with the strongest remaining rejection argument.",
  "major_concerns": [
    "Replace with substantive concern 1.",
    "Replace with substantive concern 2."
  ],
  "required_changes": [
    "Replace with concrete required change 1.",
    "Replace with concrete required change 2."
  ],
  "remaining_blockers_acknowledged": [
    "promax_public_metadata: final page range, Crossref, and DOI resolver visibility remain open",
    "manual_submission_system: private submission-system confirmation remains open"
  ],
  "valid_review_evidence": false
}
```

## Prompt

```json
{
  "review_role": "Claude Opus independent hostile top-conference reviewer",
  "claim_scope": "Controlled same-candidate LLM recommendation reranking. The current honest claim is dual: a task-grounded pointwise LLM relevance posterior is the working ranking signal and ranks first in 6 of 8 domains, while the calibrated-uncertainty/risk decomposition is a characterized negative result that does not improve ranking. Do not claim full-catalog SOTA, universal cross-domain dominance, calibration guarantees, or that uncertainty improves ranking.",
  "evidence_summary": {
    "official_same_candidate_rows": "8 Amazon domains, 8 official-code-level baselines, shared same-candidate importer, C-CRP/pointwise-posterior rows.",
    "protocol": "10k users per full domain where eligible, 101 same candidates, shared Qwen3-8B backbone when an LLM is required, full HR/NDCG @5/@10/@20 plus MRR metrics, paired Holm-corrected bootstrap tests.",
    "comparison_result": "Pointwise posterior ranks first in 6 of 8 domains; Beauty is rank 2 behind ProEx and Movies is rank 5 behind LLMEmb, both reported rather than dropped. On Sports/Toys/Home/Tools, per-event signal rows support full paired-test evidence.",
    "negative_uncertainty_result": "Risk/uncertainty adjustment does not improve ranking over the bare posterior: eta=0 is test-best in four diagnostic domains, confidence-only can match or exceed the full family, and boundary uncertainty is inert.",
    "phase_2_5_limits": [
      "observation module is motivation-only, not causal or SOTA evidence",
      "component ablation is supplementary diagnostic-only; do not claim every component is necessary or beneficial",
      "hyperparameter analysis is supplementary stability/sensitivity evidence",
      "current target-formatting gate is not closed: 15 pages against the 9-page profile and 8 overfull hbox warnings"
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
      "count": 15,
      "unique_errors": [
        "Claude CLI did not return JSON output",
        "Claude CLI returned empty output. stderr: 'WEAK_ACCEPT' is not recognized as an internal or external command, operable program or batch file.",
        "Claude CLI returned empty output. stderr: The command line is too long."
      ],
      "sources": [
        "mcp__claude_review.review_start job 999276b352b940c0b957e3be212f9919",
        "mcp__claude_review.review_start async job b1b88420168a4e498029a00a8695098a with model=opus and tools disabled",
        "mcp__claude_review.review_start async job bf4b6b8145404ffa881cd99ed3c73429 with model=opus and tools disabled",
        "mcp__claude_review.review direct opus no-tools compressed current-claim prompt",
        "mcp__claude_review.review direct opus no-tools current 20260615 full request packet prompt",
        "mcp__claude_review.review minimal JSON-oriented no-tools call",
        "mcp__claude_review.review_start async job a3863723466147e9b9b849cf994ca8fd with model=opus and tools disabled",
        "mcp__claude_review.review_start job b9757b214eb84142bce54dd28f7e258c",
        "mcp__claude_review.review synchronous JSON-only call with model=opus, tools disabled, seventh attempt",
        "mcp__claude_review.review synchronous JSON-only call with model=opus and tools disabled",
        "mcp__claude_review.review synchronous call with tools disabled",
        "mcp__claude_review.review_start async job b6e19654680c457d8be4845e168ce251 with model=opus and tools disabled",
        "mcp__claude_review.review_start job 2d4e39f665de4a848138ca2fc9630357",
        "mcp__claude_review.review direct opus no-tools safe-prompt continuation attempt 2026-06-13",
        "mcp__claude_review.review direct opus no-tools continuation attempt 2026-06-13"
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
    "review_continuation_ready": false,
    "local_release_candidate_ready": false,
    "final_submission_ready": false,
    "gate_summary": {
      "claim_audit_ok": true,
      "closure_other_blockers": [
        "Final submission package still needs the external submission-target-specific formatting pass.",
        "confirm_anonymous_shell:target_formatting_profile_not_ok",
        "confirm_anonymous_shell:target_profile_not_ok"
      ],
      "closure_packet_ok": false,
      "external_proceedings_metadata_ready": false,
      "manual_submission_system_ready": false,
      "panel_ok": true,
      "promax_probe_expected_blocked": true,
      "promax_probe_ok": true,
      "promax_public_metadata_ready": false,
      "release_candidate_stack_ok": false,
      "submission_package_audit_ok": false
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
      "other": [
        "Final submission package still needs the external submission-target-specific formatting pass.",
        "confirm_anonymous_shell:target_formatting_profile_not_ok",
        "confirm_anonymous_shell:target_profile_not_ok"
      ],
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
    "Do not waive target-formatting blockers or explicit Claude Opus coverage requirements.",
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

## Connector Health Command Before Another Retry

```bash
python -m scripts.audit.main_audit_claude_review_connector_health --output-json outputs/summary/paper_critical/claude_review_connector_health_20260615.json --output-md outputs/summary/paper_critical/claude_review_connector_health_20260615.md
```

## Validation Command Before Attach

```bash
python -m scripts.audit.main_validate_claude_opus_review_json --review-json outputs/summary/paper_critical/claude_opus_review_20260615.json --review-request-packet-json outputs/summary/paper_critical/claude_opus_review_request_packet_20260615.json --review-continuation-packet-json outputs/summary/paper_critical/review_continuation_packet_20260615.json --output-json outputs/summary/paper_critical/claude_opus_review_validation_20260615.json --output-md outputs/summary/paper_critical/claude_opus_review_validation_20260615.md
```

## Follow-Up Command

```bash
python -m scripts.audit.main_build_review_continuation_packet --additional-review-json outputs/summary/paper_critical/claude_opus_review_20260615.json --output-json outputs/summary/paper_critical/review_continuation_packet_20260615.json --output-md outputs/summary/paper_critical/review_continuation_packet_20260615.md --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_eighth_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_eleventh_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_fifteenth_20260615.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_fourteenth_20260615.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_minimal_json_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_ninth_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_retry_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_seventh_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_sixth_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_sync_notools_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_tenth_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_third_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_thirteenth_20260613.json --failed-review-attempt-json outputs/summary/paper_critical/claude_opus_review_attempt_twelfth_20260613.json
```
