# Shadow Method: C-CRP

The main shadow method is C-CRP: Calibrated Candidate Relevance Posterior.

Shadow v2-v6 are design variants and appendix-only unless they are rerun under
the same split, seed, candidate set, and statistical protocol.

## Main method

C-CRP estimates a calibrated candidate relevance posterior and decomposes
uncertainty into three components:

```text
U = alpha * U_boundary
  + beta  * U_calibration_gap
  + gamma * U_evidence
```

Where:

- `U_boundary = 4 * p_cal * (1 - p_cal)`
- `U_calibration_gap = abs(p_raw - p_cal)`
- `U_evidence = 1 - clamp(evidence_support - counterevidence_strength, 0, 1)`

The risk-adjusted ranking score is:

```text
score = p_cal * (1 - U)^eta
```

## Formal Method Variants

The formal internal method track evaluates three score families under the same
candidate rows and importer used by external baselines:

| variant | score source | paper role |
| --- | --- | --- |
| `confidence_only` | calibrated relevance probability | confidence baseline; must pass confidence-collapse diagnostics |
| `evidence_only` | `clamp(evidence_support - counterevidence_strength, 0, 1)` | evidence-only C-CRP ablation |
| `confidence_plus_evidence` / `full` | calibrated probability plus evidence with C-CRP uncertainty penalty | main C-CRP candidate |

Mode, weights, `eta`, and ablations must be fixed before test or selected on
validation only. The selected test row must be exported as:

```text
source_event_id,user_id,item_id,score
```

and imported through `main_import_same_candidate_baseline_scores.py` with
`status_label=same_schema_internal_method`.

The formal selector records the full validation grid, selected weights, ranking
input hashes, signal hashes, candidate hashes, exact score coverage, and a
score-degeneracy audit. A C-CRP row is not paper-facing if the exported score
file has constant or near-tie-only scores for candidate events, exceeds the
declared tie-pair threshold, or if the selected configuration was chosen from
test behavior.

## Weight rule

`alpha`, `beta`, and `gamma` must be fixed before test or selected on
validation only. The default fixed weights are:

```text
alpha = 0.5
beta  = 0.3
gamma = 0.2
```

Additional weight triples may be searched only through the validation selector
and must be recorded in `valid_ccrp_sweep.csv` and
`ccrp_internal_provenance.json`.

## Required ablations

The main method table may include these ablations:

- C-CRP without calibration gap
- C-CRP without evidence support
- C-CRP without counterevidence
- C-CRP without risk penalty

## Entry point

```bash
python main_select_ccrp_variant_on_valid.py \
  --domain books \
  --valid_ranking_path outputs/baselines/external_tasks/books_large10000_100neg_valid_same_candidate/ranking_valid.jsonl \
  --test_ranking_path outputs/baselines/external_tasks/books_large10000_100neg_test_same_candidate/ranking_test.jsonl \
  --valid_candidate_items_path outputs/baselines/external_tasks/books_large10000_100neg_valid_same_candidate/candidate_items.csv \
  --test_candidate_items_path outputs/baselines/external_tasks/books_large10000_100neg_test_same_candidate/candidate_items.csv \
  --valid_signal_path outputs/books_large10000_100neg_qwen3_shadow_v1/calibrated/valid_calibrated.jsonl \
  --test_signal_path outputs/books_large10000_100neg_qwen3_shadow_v1/calibrated/test_calibrated.jsonl \
  --output_dir outputs/summary/week8_large10000_100neg_ccrp_formal/books \
  --import_scores
```

`main_shadow_ccrp_eval.py` remains useful for local diagnostics. Prompt-only
shadow diagnostics are not ranking main-table evidence.

## Outputs

- C-CRP scored records
- C-CRP diagnostic summary
- risk-coverage curve
- optional ranked records when pointwise candidate rows are available
- selected validation config
- exact same-candidate score CSV
- internal provenance JSON
- imported same-candidate summary when `--import_scores` is used
