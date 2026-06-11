# Home Component-Ablation Evidence Log Snippets

- 2026-06-11: rebuilt `component_ablation_summary.{csv,json}` and `component_ablation_provenance.json` on `pony-rec-gpu` with the preregistered main C-CRP config (`eta=1.0`, `tie_break_seed=20260607`); builder returned `ok=true` and `failures=[]`.
- 2026-06-11: imported `ccrp_selected_test_scores.csv` through `scripts/misc/main_import_same_candidate_baseline_scores.py` with `--tie_break_seed 20260607`; imported `NDCG@10=0.13635503352938705`, matching `selected_test_metrics.csv`.
- 2026-06-11: synchronized the lightweight local evidence package tables, figures, selected config/provenance, and component summary files; server `predictions/rank_predictions.jsonl` and full `ccrp_selected_test_scores.csv` were not copied into the local package.
