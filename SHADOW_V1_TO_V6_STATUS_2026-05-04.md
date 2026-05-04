# Shadow V1-V6 Status Matrix - 2026-05-04

This note records the server-side `shadow_v1` to `shadow_v6` status matrix
generated from the rescue branch. It separates signal candidates from the
decision bridge so later work does not overclaim the shadow line.

## Source

Server checkout:

- `~/projects/pony-rec-rescue-shadow-v6`
- Branch: `codex/pre-cursor-shadow-v6-rescue`
- Script commit: `375ac2f Fill fallback shadow matrix metrics`

Command:

```bash
python main_summarize_shadow_v1_to_v6.py \
  --shadow_summary_root ~/projects/uncertainty-llm4rec-codex-apr12-preserve-local/outputs/summary \
  --shadow_output_root ~/projects/uncertainty-llm4rec-codex-apr12-preserve-local/outputs \
  --v6_output_root outputs \
  --output_root outputs/summary \
  --output_name shadow_v1_to_v6_status_matrix
```

Generated files on server:

- `outputs/summary/shadow_v1_to_v6_status_matrix.csv`
- `outputs/summary/shadow_v1_to_v6_status_matrix.md`

Matrix count:

- `44` rows total.
- `40` signal-candidate rows.
- `4` decision-bridge rows.
- Status counts: `26` missing, `10` ready_with_noisy, `4` ready_clean,
  `4` ready.

## Status Definition

- `signal_candidate`: `shadow_v1` to `shadow_v5`; these are uncertainty signal
  candidates.
- `decision_bridge`: `shadow_v6`; this uses a winner signal to form ranking
  decisions.
- `ready_with_noisy`: clean pointwise, calibration, rerank, noisy pointwise,
  and noisy rerank are all present.
- `ready_clean`: clean pointwise, calibration, and rerank are present, but
  noisy robustness is not present.
- `ready`: `shadow_v6` bridge output is present.
- `missing`: expected row not found in summary or raw output folders.

## Full-Replay Results

Full replay has a conservative four-domain path:

- `shadow_v1` is ready_with_noisy on Beauty, Books, Electronics, and Movies.
- `shadow_v6` is ready on Beauty, Books, Electronics, and Movies, using
  `shadow_v1` as `winner_signal_variant`.
- Full-replay `shadow_v2` to `shadow_v5` are not present in the checked matrix.

| domain | variant | status | AUROC | NDCG@10 | MRR | delta_NDCG@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| beauty | shadow_v1 | ready_with_noisy | 0.5723036899 | 0.6353973143 | 0.5184823570 |  |
| beauty | shadow_v6 | ready |  | 0.6353973143 | 0.5184823570 | +0.0000314960 |
| books | shadow_v1 | ready_with_noisy | 0.7133176000 | 0.6362591123 | 0.5191000000 |  |
| books | shadow_v6 | ready |  | 0.6568907475 | 0.5460666667 | +0.0203187763 |
| electronics | shadow_v1 | ready_with_noisy | 0.6230092000 | 0.6574286221 | 0.5460333333 |  |
| electronics | shadow_v6 | ready |  | 0.6631285871 | 0.5534666667 | +0.0056999649 |
| movies | shadow_v1 | ready_with_noisy | 0.5223908000 | 0.5725075524 | 0.4381666667 |  |
| movies | shadow_v6 | ready |  | 0.5733843398 | 0.4393333333 | +0.0009993699 |

Interpretation:

- `shadow_v1` is the only four-domain full-replay signal candidate currently
  supported by complete clean plus noisy evidence.
- `shadow_v6` is a four-domain diagnostic bridge, not a paper result.
- Books is the strongest v6 diagnostic gain. Electronics is modestly positive.
  Beauty and Movies are nearly non-destructive sanity checks.

## Small-Prior Results

Small prior is complete only for Beauty and Books in the checked matrix.

| domain | variant | status | AUROC | NDCG@10 | MRR |
| --- | --- | --- | ---: | ---: | ---: |
| beauty | shadow_v1 | ready_with_noisy | 0.5837600000 | 0.6394303493 | 0.5236666667 |
| beauty | shadow_v2 | ready_with_noisy | 0.5892100000 | 0.6394303493 | 0.5236666667 |
| beauty | shadow_v3 | ready_clean | 0.5755000000 | 0.6394303493 | 0.5236666667 |
| beauty | shadow_v4 | ready_clean | 0.5720000000 | 0.6394303493 | 0.5236666667 |
| beauty | shadow_v5 | ready_with_noisy | 0.5811850000 | 0.6394303493 | 0.5236666667 |
| books | shadow_v1 | ready_with_noisy | 0.7246266667 | 0.6469384300 | 0.5336666667 |
| books | shadow_v2 | ready_with_noisy | 0.6918275000 | 0.6469384300 | 0.5336666667 |
| books | shadow_v3 | ready_clean | 0.5727175000 | 0.6284849177 | 0.5086666667 |
| books | shadow_v4 | ready_clean | 0.6760700000 | 0.6469384300 | 0.5336666667 |
| books | shadow_v5 | ready_with_noisy | 0.7002650000 | 0.6516282885 | 0.5400555556 |

Interpretation:

- Beauty small-prior rerank metrics are tied across v1-v5 in the checked
  matrix, while AUROC differs slightly.
- Books small-prior favors `shadow_v5` on rerank metrics, while `shadow_v1`
  has the highest pointwise AUROC.
- `shadow_v3` and `shadow_v4` are clean-only for Beauty/Books and should not be
  promoted as robust variants without noisy completion.
- Electronics and Movies small-prior rows are missing in this checked matrix.

## Working Decision

For the next mainline step:

1. Use full-replay `shadow_v1` as the conservative winner signal source.
2. Treat `shadow_v6` as the active signal-to-decision diagnostic bridge.
3. Do not claim that `shadow_v2` to `shadow_v5` are four-domain full-replay
   winners.
4. Do not promote small-prior `shadow_v3` or `shadow_v4` beyond clean-only
   status.
5. Compare `shadow_v6` against week7.7 SRPD and structured-risk methods in a
   unified matrix before deciding whether it becomes a paper-facing method or
   remains a bridge/ablation.
