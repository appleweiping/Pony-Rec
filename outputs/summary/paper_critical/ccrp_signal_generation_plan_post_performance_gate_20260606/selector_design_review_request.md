# Design Review Request — Sports C-CRP Selector Module (Phase 2.5)

You are a top-conference (RecSys/SIGIR/KDD/NeurIPS) reviewer. Score this experiment DESIGN 0-10 (8 = top-conference-submission level). Be adversarial: hunt for leakage, unfairness, overclaim, statistical invalidity. Output: numeric score, then PASS(>=8)/FAIL, then concrete required fixes.

## Context
Project "Actionable Uncertainty for LLM-based Recommendation". Method C-CRP (calibrated candidate relevance posterior with boundary uncertainty, calibration gap, evidence support, counterevidence; risk-adjusted ranking). Eight-domain narrative; Sports is first of the four new domains. Same-candidate protocol: 10,000 eval events, 101 candidates/event. Backbone Qwen3-8B.

## Inputs (already audited, this session)
- valid + test C-CRP signal rows: each 1,010,000 rows, coverage 1.0, 0 dup keys, parse-fail 0, status=recomputable_signal_rows. Columns: source_event_id,user_id,candidate_item_id,item_id,candidate_idx,relevance_probability,calibrated_relevance_probability,evidence_support,counterevidence_strength,reason,parse_success.
- Signal generation prompt excludes positive_item_index/labels (leakage guard verified).

## The selector design (scripts/misc/main_select_ccrp_variant_on_valid.py)
1. Grid over: score_modes {confidence_only, evidence_only, confidence_plus_evidence, full}; ablations {full, without_boundary_uncertainty, without_calibration_gap, without_evidence_support, without_counterevidence, without_risk_penalty}; eta {0.5,1.0,2.0}; confidence_weight {0.5,0.7,0.9} (only varied for confidence_plus_evidence mode, else fixed to first); weight_grid (boundary,calibration_gap,evidence) {0.5,0.3,0.2; 0.7,0.2,0.1; 0.4,0.4,0.2; 0.4,0.2,0.4}.
2. For each config, compute C-CRP risk-adjusted score per candidate from the signal rows, rank candidates per event, compute HR@5/10/20, NDCG@5/10/20, MRR on the VALID split.
3. Selection: keep only configs passing audit_ok AND degeneracy_audit_ok (score-degeneracy guard); pick the one maximizing selection_metric=NDCG@10 on VALID.
4. Apply that single fixed config ONCE to the TEST split; export ccrp_selected_test_scores.csv (schema source_event_id,user_id,item_id,score), ccrp_selected_test_scored_rows.csv (per-candidate uncertainty + risk score, feeds observation/ablation), selected_test_metrics.csv, ccrp_internal_provenance.json (records both signal sha256s, ranking sha256s, candidate sha256s, the full grid, selected config).
5. --import_scores imports the test scores through the same same-candidate importer used by the 8 official baselines, producing comparable HR/NDCG/MRR rows.

## Questions for the reviewer
- Is the valid-only selection -> fixed-config test application leakage-clean and fair vs the official baselines (which use official default hyperparameters)?
- Is the grid principled and broad enough, or is it over/under-searched (multiple-comparison risk on valid)?
- Is selecting on NDCG@10 then reporting all of HR/NDCG@5/10/20+MRR defensible?
- Does folding the leave-one-component-out ablations into the SAME selection grid create a problem (selecting an ablation variant as "best" vs reporting ablations separately)? Should ablation be selection-frozen to full and evaluated separately?
- Any degeneracy/tie-breaking/coverage concern with risk-adjusted scoring on 101 candidates?
- Statistical reporting: what paired-test discipline is needed for the test row to be paper-eligible?
