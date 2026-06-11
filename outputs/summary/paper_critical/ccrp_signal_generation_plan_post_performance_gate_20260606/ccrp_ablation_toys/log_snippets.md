# Toys Component-Ablation Evidence Log Snippets

- 2026-06-11: design re-review passed with GPT-5.5 xhigh sidecar verdict PROCEED, average score 8.18; execution used existing full-scale server signal rows only, with no new LLM query.
- 2026-06-11: ran `scripts/analysis/main_build_ccrp_component_ablation_summary.py` on `pony-rec-gpu` for `toys` with the preregistered main C-CRP config (`eta=1.0`, weights `0.5,0.3,0.2`, `tie_break_seed=20260607`); builder returned `ok=true` and `failures=[]`.
- 2026-06-11: imported frozen `ccrp_selected_test_scores.csv` through `scripts/misc/main_import_same_candidate_baseline_scores.py` as `toys_ccrp_component_full`, `status_label=same_schema_internal_ablation`, `artifact_class=completed_result`, and `--tie_break_seed 20260607`; score coverage remained `1.000000`.
- 2026-06-11: synchronized the lightweight local evidence package tables, figures, selected config/provenance, and component summary files; server `predictions/rank_predictions.jsonl`, signal rows, `ccrp_selected_test_scored_rows.csv`, and full `ccrp_selected_test_scores.csv` were not copied into the local component package.
