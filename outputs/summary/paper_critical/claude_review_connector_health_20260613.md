# Claude Review Connector Health

- Created UTC: `2026-06-13T06:05:27.117341+00:00`
- OK: `true`
- Final submission ready: `false`
- Failed attempt count: `11`
- Valid review evidence count: `0`
- Last error: `Claude CLI did not return JSON output`
- Same-error tail streak: `11`
- Connector unhealthy: `true`
- Same route retry recommended: `false`
- Recommended next route: `external_claude_opus_json_via_request_packet_and_validator`

## Error Counts

- `Claude CLI did not return JSON output`: `11`

## Warnings

- same_connector_error_repeated:11:Claude_CLI_did_not_return_JSON_output

## Next Actions

- If connector_unhealthy=true, do not keep retrying the same mcp__claude_review route unless the connector/tooling changes.
- Use claude_opus_review_request_packet_20260613.{json,md} to obtain a substantive external Claude Opus JSON.
- Run main_validate_claude_opus_review_json.py before attaching any returned Claude Opus JSON with --additional-review-json.
- Keep final_submission_ready=false until the final submission gate reports true.
