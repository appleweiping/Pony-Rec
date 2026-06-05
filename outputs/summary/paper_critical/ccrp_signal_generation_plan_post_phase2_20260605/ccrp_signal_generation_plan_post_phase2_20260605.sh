#!/usr/bin/env bash
set -euo pipefail

echo 'GUARDED PLAN ONLY: fill TODO signal paths and remove this exit after all gates pass.'
echo 'This file is intentionally non-runnable as generated.'
exit 2

cd /home/ajifang/projects/pony-rec-rescue-shadow-v6

# 1. Header discovery for candidate signal/scored-row artifacts.
python scripts/audit/main_remote_discover_ccrp_uncertainty_sources.py \
  --root . \
  --domain sports \
  --domain toys \
  --domain home \
  --domain tools \
  --name_token ccrp \
  --name_token shadow \
  --name_token signal \
  --name_token calibrated \
  --name_token scored \
  --name_token rows \
  --expected_events 10000 \
  --expected_candidates_per_event 101 \
  --output_json outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_signal_source_discovery.json \
  --output_csv outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_signal_source_discovery.csv \
  --quiet

# sports: audit the chosen signal artifact after replacing TODO paths.
python scripts/audit/main_audit_ccrp_uncertainty_sources.py \
  --candidate_items_path outputs/baselines/external_tasks/sports_large10000_100neg_test_same_candidate/candidate_items.csv \
  --source candidate_signal=TODO_TEST_SPORTS_CCRP_SIGNAL_JSONL_OR_CSV \
  --expected_events 10000 \
  --expected_candidates_per_event 101 \
  --output_json outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_signal_source_audit_sports.json \
  --output_csv outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_signal_source_audit_sports.csv

# sports: validation-select C-CRP components/hyperparameters and export test rows.
python scripts/misc/main_select_ccrp_variant_on_valid.py \
  --domain sports \
  --valid_ranking_path outputs/baselines/external_tasks/sports_large10000_100neg_valid_same_candidate/ranking_task.jsonl \
  --test_ranking_path outputs/baselines/external_tasks/sports_large10000_100neg_test_same_candidate/ranking_task.jsonl \
  --valid_candidate_items_path outputs/baselines/external_tasks/sports_large10000_100neg_valid_same_candidate/candidate_items.csv \
  --test_candidate_items_path outputs/baselines/external_tasks/sports_large10000_100neg_test_same_candidate/candidate_items.csv \
  --valid_signal_path TODO_VALID_SPORTS_CCRP_SIGNAL_JSONL_OR_CSV \
  --test_signal_path TODO_TEST_SPORTS_CCRP_SIGNAL_JSONL_OR_CSV \
  --output_dir outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_ablation_sports \
  --score_modes confidence_only,evidence_only,confidence_plus_evidence,full \
  --ablations full,without_boundary_uncertainty,without_calibration_gap,without_evidence_support,without_counterevidence,without_risk_penalty \
  --etas 0.5,1.0,2.0 \
  --confidence_weights 0.5,0.7,0.9 \
  --weight_grid '0.5,0.3,0.2;0.7,0.2,0.1;0.4,0.4,0.2;0.4,0.2,0.4' \
  --selection_metric NDCG@10 \
  --import_scores

# sports: build observation/motivation table and figure.
python scripts/analysis/main_build_uncertainty_observation_study.py \
  --domain sports \
  --uncertainty_scores_path outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_ablation_sports/ccrp_selected_test_scored_rows.csv \
  --ccrp_eval_path outputs/sports_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate/tables/ranking_eval_records.csv \
  --method_eval llmemb=outputs/sports_large10000_100neg_llmemb_official_qwen3base_same_candidate/tables/ranking_eval_records.csv \
  --method_eval proex=outputs/sports_large10000_100neg_proex_profile_official_qwen3base_same_candidate/tables/ranking_eval_records.csv \
  --method_eval rlmrec=outputs/sports_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate/tables/ranking_eval_records.csv \
  --output_dir outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/observation_sports \
  --expected_events 10000 \
  --min_join_rate 0.999

# sports: plot validation hyperparameter curves.
python scripts/analysis/main_plot_ccrp_hyperparameter_sweep.py \
  --sweep_csv outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_ablation_sports/valid_ccrp_sweep.csv \
  --output_dir outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_hyperparameter_sports \
  --domain sports \
  --metric NDCG@10 \
  --score_mode full \
  --ablation full

# toys: audit the chosen signal artifact after replacing TODO paths.
python scripts/audit/main_audit_ccrp_uncertainty_sources.py \
  --candidate_items_path outputs/baselines/external_tasks/toys_large10000_100neg_test_same_candidate/candidate_items.csv \
  --source candidate_signal=TODO_TEST_TOYS_CCRP_SIGNAL_JSONL_OR_CSV \
  --expected_events 10000 \
  --expected_candidates_per_event 101 \
  --output_json outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_signal_source_audit_toys.json \
  --output_csv outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_signal_source_audit_toys.csv

# toys: validation-select C-CRP components/hyperparameters and export test rows.
python scripts/misc/main_select_ccrp_variant_on_valid.py \
  --domain toys \
  --valid_ranking_path outputs/baselines/external_tasks/toys_large10000_100neg_valid_same_candidate/ranking_task.jsonl \
  --test_ranking_path outputs/baselines/external_tasks/toys_large10000_100neg_test_same_candidate/ranking_task.jsonl \
  --valid_candidate_items_path outputs/baselines/external_tasks/toys_large10000_100neg_valid_same_candidate/candidate_items.csv \
  --test_candidate_items_path outputs/baselines/external_tasks/toys_large10000_100neg_test_same_candidate/candidate_items.csv \
  --valid_signal_path TODO_VALID_TOYS_CCRP_SIGNAL_JSONL_OR_CSV \
  --test_signal_path TODO_TEST_TOYS_CCRP_SIGNAL_JSONL_OR_CSV \
  --output_dir outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_ablation_toys \
  --score_modes confidence_only,evidence_only,confidence_plus_evidence,full \
  --ablations full,without_boundary_uncertainty,without_calibration_gap,without_evidence_support,without_counterevidence,without_risk_penalty \
  --etas 0.5,1.0,2.0 \
  --confidence_weights 0.5,0.7,0.9 \
  --weight_grid '0.5,0.3,0.2;0.7,0.2,0.1;0.4,0.4,0.2;0.4,0.2,0.4' \
  --selection_metric NDCG@10 \
  --import_scores

# toys: build observation/motivation table and figure.
python scripts/analysis/main_build_uncertainty_observation_study.py \
  --domain toys \
  --uncertainty_scores_path outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_ablation_toys/ccrp_selected_test_scored_rows.csv \
  --ccrp_eval_path outputs/toys_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate/tables/ranking_eval_records.csv \
  --method_eval llmemb=outputs/toys_large10000_100neg_llmemb_official_qwen3base_same_candidate/tables/ranking_eval_records.csv \
  --method_eval proex=outputs/toys_large10000_100neg_proex_profile_official_qwen3base_same_candidate/tables/ranking_eval_records.csv \
  --method_eval rlmrec=outputs/toys_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate/tables/ranking_eval_records.csv \
  --output_dir outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/observation_toys \
  --expected_events 10000 \
  --min_join_rate 0.999

# toys: plot validation hyperparameter curves.
python scripts/analysis/main_plot_ccrp_hyperparameter_sweep.py \
  --sweep_csv outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_ablation_toys/valid_ccrp_sweep.csv \
  --output_dir outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_hyperparameter_toys \
  --domain toys \
  --metric NDCG@10 \
  --score_mode full \
  --ablation full

# home: audit the chosen signal artifact after replacing TODO paths.
python scripts/audit/main_audit_ccrp_uncertainty_sources.py \
  --candidate_items_path outputs/baselines/external_tasks/home_large10000_100neg_test_same_candidate/candidate_items.csv \
  --source candidate_signal=TODO_TEST_HOME_CCRP_SIGNAL_JSONL_OR_CSV \
  --expected_events 10000 \
  --expected_candidates_per_event 101 \
  --output_json outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_signal_source_audit_home.json \
  --output_csv outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_signal_source_audit_home.csv

# home: validation-select C-CRP components/hyperparameters and export test rows.
python scripts/misc/main_select_ccrp_variant_on_valid.py \
  --domain home \
  --valid_ranking_path outputs/baselines/external_tasks/home_large10000_100neg_valid_same_candidate/ranking_task.jsonl \
  --test_ranking_path outputs/baselines/external_tasks/home_large10000_100neg_test_same_candidate/ranking_task.jsonl \
  --valid_candidate_items_path outputs/baselines/external_tasks/home_large10000_100neg_valid_same_candidate/candidate_items.csv \
  --test_candidate_items_path outputs/baselines/external_tasks/home_large10000_100neg_test_same_candidate/candidate_items.csv \
  --valid_signal_path TODO_VALID_HOME_CCRP_SIGNAL_JSONL_OR_CSV \
  --test_signal_path TODO_TEST_HOME_CCRP_SIGNAL_JSONL_OR_CSV \
  --output_dir outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_ablation_home \
  --score_modes confidence_only,evidence_only,confidence_plus_evidence,full \
  --ablations full,without_boundary_uncertainty,without_calibration_gap,without_evidence_support,without_counterevidence,without_risk_penalty \
  --etas 0.5,1.0,2.0 \
  --confidence_weights 0.5,0.7,0.9 \
  --weight_grid '0.5,0.3,0.2;0.7,0.2,0.1;0.4,0.4,0.2;0.4,0.2,0.4' \
  --selection_metric NDCG@10 \
  --import_scores

# home: build observation/motivation table and figure.
python scripts/analysis/main_build_uncertainty_observation_study.py \
  --domain home \
  --uncertainty_scores_path outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_ablation_home/ccrp_selected_test_scored_rows.csv \
  --ccrp_eval_path outputs/home_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate/tables/ranking_eval_records.csv \
  --method_eval llmemb=outputs/home_large10000_100neg_llmemb_official_qwen3base_same_candidate/tables/ranking_eval_records.csv \
  --method_eval proex=outputs/home_large10000_100neg_proex_profile_official_qwen3base_same_candidate/tables/ranking_eval_records.csv \
  --method_eval rlmrec=outputs/home_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate/tables/ranking_eval_records.csv \
  --output_dir outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/observation_home \
  --expected_events 10000 \
  --min_join_rate 0.999

# home: plot validation hyperparameter curves.
python scripts/analysis/main_plot_ccrp_hyperparameter_sweep.py \
  --sweep_csv outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_ablation_home/valid_ccrp_sweep.csv \
  --output_dir outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_hyperparameter_home \
  --domain home \
  --metric NDCG@10 \
  --score_mode full \
  --ablation full

# tools: audit the chosen signal artifact after replacing TODO paths.
python scripts/audit/main_audit_ccrp_uncertainty_sources.py \
  --candidate_items_path outputs/baselines/external_tasks/tools_large10000_100neg_test_same_candidate/candidate_items.csv \
  --source candidate_signal=TODO_TEST_TOOLS_CCRP_SIGNAL_JSONL_OR_CSV \
  --expected_events 10000 \
  --expected_candidates_per_event 101 \
  --output_json outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_signal_source_audit_tools.json \
  --output_csv outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_signal_source_audit_tools.csv

# tools: validation-select C-CRP components/hyperparameters and export test rows.
python scripts/misc/main_select_ccrp_variant_on_valid.py \
  --domain tools \
  --valid_ranking_path outputs/baselines/external_tasks/tools_large10000_100neg_valid_same_candidate/ranking_task.jsonl \
  --test_ranking_path outputs/baselines/external_tasks/tools_large10000_100neg_test_same_candidate/ranking_task.jsonl \
  --valid_candidate_items_path outputs/baselines/external_tasks/tools_large10000_100neg_valid_same_candidate/candidate_items.csv \
  --test_candidate_items_path outputs/baselines/external_tasks/tools_large10000_100neg_test_same_candidate/candidate_items.csv \
  --valid_signal_path TODO_VALID_TOOLS_CCRP_SIGNAL_JSONL_OR_CSV \
  --test_signal_path TODO_TEST_TOOLS_CCRP_SIGNAL_JSONL_OR_CSV \
  --output_dir outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_ablation_tools \
  --score_modes confidence_only,evidence_only,confidence_plus_evidence,full \
  --ablations full,without_boundary_uncertainty,without_calibration_gap,without_evidence_support,without_counterevidence,without_risk_penalty \
  --etas 0.5,1.0,2.0 \
  --confidence_weights 0.5,0.7,0.9 \
  --weight_grid '0.5,0.3,0.2;0.7,0.2,0.1;0.4,0.4,0.2;0.4,0.2,0.4' \
  --selection_metric NDCG@10 \
  --import_scores

# tools: build observation/motivation table and figure.
python scripts/analysis/main_build_uncertainty_observation_study.py \
  --domain tools \
  --uncertainty_scores_path outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_ablation_tools/ccrp_selected_test_scored_rows.csv \
  --ccrp_eval_path outputs/tools_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate/tables/ranking_eval_records.csv \
  --method_eval llmemb=outputs/tools_large10000_100neg_llmemb_official_qwen3base_same_candidate/tables/ranking_eval_records.csv \
  --method_eval proex=outputs/tools_large10000_100neg_proex_profile_official_qwen3base_same_candidate/tables/ranking_eval_records.csv \
  --method_eval rlmrec=outputs/tools_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate/tables/ranking_eval_records.csv \
  --output_dir outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/observation_tools \
  --expected_events 10000 \
  --min_join_rate 0.999

# tools: plot validation hyperparameter curves.
python scripts/analysis/main_plot_ccrp_hyperparameter_sweep.py \
  --sweep_csv outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_ablation_tools/valid_ccrp_sweep.csv \
  --output_dir outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/ccrp_hyperparameter_tools \
  --domain tools \
  --metric NDCG@10 \
  --score_mode full \
  --ablation full
