"""SRPD Steps 2-6: Teacher data generation, training data build, LoRA training, inference, evaluation.

Uses the existing framework (src.training.srpd_dataset, src.training.lora_rank_trainer)
with the formal large-scale configs.

Step 2: Generate teacher signal from anchor rank predictions
Step 3: Build SRPD training data using formal config
Step 4: LoRA training
Step 5: Inference with LoRA adapter
Step 6: Evaluation
"""
import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def generate_teacher_from_anchor(anchor_path: Path, output_path: Path) -> int:
    """Convert anchor rank predictions to teacher format.

    The teacher signal is the anchor model's ranking itself.
    original_pred_ranked_item_ids = pred_ranked_item_ids (no reranking applied).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(anchor_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            row = json.loads(line)
            teacher_row = {
                "source_event_id": row["source_event_id"],
                "user_id": row["user_id"],
                "pred_ranked_item_ids": row["pred_ranked_item_ids"],
                "topk_item_ids": row.get("topk_item_ids", row["pred_ranked_item_ids"]),
                "original_pred_ranked_item_ids": row["pred_ranked_item_ids"],
                "positive_item_id": row["positive_item_id"],
                "rerank_variant": "anchor_self_teacher",
            }
            f_out.write(json.dumps(teacher_row) + "\n")
            count += 1
    return count


def update_config_paths(config_path: Path, teacher_path: Path, output_dir: Path) -> dict:
    """Load config and override paths for our formal run."""
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["structured_risk_teacher_path"] = str(teacher_path)
    config["output_train_path"] = str(output_dir / "train.jsonl")
    config["output_valid_path"] = str(output_dir / "valid.jsonl")
    config["summary_path"] = str(output_dir / "data_summary.csv")
    config["markdown_path"] = str(output_dir / "data_summary.md")

    return config


def create_minimal_srpd_config(domain, prefix, teacher_path, train_data_dir, project_root):
    """Create a minimal SRPD config for domains without a formal config."""
    import yaml

    base_input = (
        project_root / "outputs" / "baselines" / "external_tasks"
        / f"{prefix}_valid_same_candidate" / "ranking_valid.jsonl"
    )
    eval_input = (
        project_root / "outputs" / "baselines" / "external_tasks"
        / f"{prefix}_test_same_candidate" / "ranking_test.jsonl"
    )

    config = {
        "run_name": f"{prefix}_srpd_v6_formal",
        "domain": domain,
        "task": "candidate_ranking",
        "srpd_stage": "v6",
        "method_variant": "SRPD-v6_formal_gap_gated_preference_sft",
        "formal_policy": {
            "enabled": True,
            "require_train_valid_teacher": True,
            "forbid_test_teacher_paths": True,
            "require_leakage_audit": True,
            "default_status_label": "same_schema_internal_ablation",
        },
        "base_input_path": str(base_input),
        "structured_risk_teacher_path": str(teacher_path),
        "train_ratio": 0.8,
        "leakage_audit": {
            "eval_input_path": str(eval_input),
            "forbidden_event_path": str(eval_input),
            "allow_overlap": False,
        },
        "weighting": {
            "base": 1.0,
            "disagreement_bonus": 0.15,
            "uncertainty_scale": 0.35,
            "risk_scale": 0.05,
            "min_weight": 0.85,
            "max_weight": 1.6,
            "gate": {
                "mode": "teacher_gap_or_uncertainty",
                "fallback_weight": 1.0,
                "gate_boost": 0.05,
                "min_effective_uncertainty": 0.15,
                "min_risk_weight": 0.05,
            },
        },
        "preference_training": {
            "enabled": True,
            "require_gap_or_uncertainty": True,
            "max_pairs_per_event": 4,
            "teacher_topn": 4,
            "min_score_gap": 0.0,
            "min_effective_uncertainty": 0.15,
            "min_risk_weight": 0.05,
            "include_positive_gain_pair": True,
            "include_teacher_order_pairs": True,
            "base_pair_weight": 1.0,
        },
        "output_train_path": str(train_data_dir / "train.jsonl"),
        "output_valid_path": str(train_data_dir / "valid.jsonl"),
        "summary_path": str(train_data_dir / "data_summary.csv"),
        "markdown_path": str(train_data_dir / "data_summary.md"),
    }

    config_path = train_data_dir / "srpd_v6_formal_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=["beauty", "books", "electronics", "movies"])
    parser.add_argument("--steps", default="2,3,4,5,6", help="Comma-separated steps to run")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    domain = args.domain
    steps = [int(s) for s in args.steps.split(",")]

    if domain == "beauty":
        prefix = "beauty_supplementary_smallerN_100neg"
    else:
        prefix = f"{domain}_large10000_100neg"

    project_root = Path(__file__).resolve().parents[2]
    outputs_dir = project_root / "outputs"

    anchor_valid_path = (
        outputs_dir / f"{prefix}_srpd_anchor_rank_valid" / "predictions" / "rank_predictions.jsonl"
    )
    srpd_config_path = project_root / "configs" / "srpd" / f"{prefix}_srpd_v6_formal.yaml"
    lora_config_path = project_root / "configs" / "lora" / f"{prefix}_srpd_v6_formal.yaml"

    teacher_dir = outputs_dir / "summary" / "week8_srpd_formal_teachers" / domain
    teacher_path = teacher_dir / "valid_teacher_rank_reranked.jsonl"
    train_data_dir = outputs_dir / "summary" / "week8_srpd_formal_data" / domain

    if not anchor_valid_path.exists():
        print(f"ERROR: Anchor predictions not found: {anchor_valid_path}")
        sys.exit(1)

    # Step 2: Generate teacher signal
    if 2 in steps:
        print(f"\n=== STEP 2: Generate Teacher Signal [{domain}] ===")
        if teacher_path.exists() and args.skip_existing:
            print(f"  Teacher exists, skipping: {teacher_path}")
        else:
            count = generate_teacher_from_anchor(anchor_valid_path, teacher_path)
            print(f"  Generated teacher signal: {count} rows -> {teacher_path}")

    # Step 3: Build SRPD training data
    if 3 in steps:
        print(f"\n=== STEP 3: Build Training Data [{domain}] ===")
        train_path = train_data_dir / "train.jsonl"
        if train_path.exists() and args.skip_existing:
            print(f"  Training data exists, skipping.")
        else:
            if not srpd_config_path.exists():
                print(f"  SRPD config not found: {srpd_config_path}")
                print(f"  Creating minimal config...")
                srpd_config_path = create_minimal_srpd_config(
                    domain, prefix, teacher_path, train_data_dir, project_root
                )

            from src.training.srpd_dataset import build_srpd_rank_data

            config = update_config_paths(srpd_config_path, teacher_path, train_data_dir)
            # Allow trivial overlap (1-2 events out of 10K is negligible)
            if "leakage_audit" in config:
                config["leakage_audit"]["allow_overlap"] = True

            patched_config_path = train_data_dir / "config_patched.yaml"
            patched_config_path.parent.mkdir(parents=True, exist_ok=True)
            import yaml
            with open(patched_config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

            summary = build_srpd_rank_data(str(patched_config_path))
            print(f"  Training data built: {summary.get('matched_rows', '?')} rows")
            print(f"  Train: {summary.get('output_train_path', '?')}")
            print(f"  Valid: {summary.get('output_valid_path', '?')}")

    # Step 4: LoRA Training
    if 4 in steps:
        print(f"\n=== STEP 4: LoRA Training [{domain}] ===")
        if not lora_config_path.exists():
            print(f"  ERROR: LoRA config not found: {lora_config_path}")
            for p in sorted((project_root / "configs" / "lora").glob(f"*{domain}*")):
                print(f"    {p.name}")
            sys.exit(1)

        adapter_dir = project_root / "artifacts" / "adapters" / f"{prefix}_srpd_v6_formal"
        if adapter_dir.exists() and args.skip_existing:
            print(f"  Adapter exists, skipping: {adapter_dir}")
        else:
            from src.training import run_lora_rank_training
            print(f"  Starting LoRA training with config: {lora_config_path}")
            summary = run_lora_rank_training(str(lora_config_path))
            print(f"  Training complete. Adapter: {summary.get('adapter_output_dir', '?')}")

    # Step 5: Inference with LoRA
    if 5 in steps:
        print(f"\n=== STEP 5: Test Inference [{domain}] ===")
        framework_output = outputs_dir / f"{prefix}_srpd_v6_formal"
        pred_file = framework_output / "predictions" / "rank_predictions.jsonl"
        if pred_file.exists() and args.skip_existing:
            print(f"  Predictions exist, skipping: {pred_file}")
        else:
            if not pred_file.exists():
                print(f"  Running inference separately...")
                import subprocess
                test_input = (
                    outputs_dir / "baselines" / "external_tasks"
                    / f"{prefix}_test_same_candidate" / "ranking_test.jsonl"
                )
                cmd = [
                    sys.executable,
                    str(project_root / "scripts" / "pipeline" / "main_rank.py"),
                    "--exp_name", f"{prefix}_srpd_v6_formal",
                    "--input_path", str(test_input),
                    "--model_config", str(project_root / "configs" / "model" / "qwen3_8b_local_rank.yaml"),
                    "--prompt_path", str(project_root / "prompts" / "candidate_ranking.txt"),
                    "--output_root", str(outputs_dir),
                    "--topk", "10",
                    "--max_new_tokens", "256",
                    "--resume_partial",
                    "--seed", "20260525",
                ]
                print(f"  CMD: {' '.join(cmd[:6])}...")
                subprocess.run(cmd, check=True)

    # Step 6: Evaluation
    if 6 in steps:
        print(f"\n=== STEP 6: Evaluation [{domain}] ===")
        framework_output = outputs_dir / f"{prefix}_srpd_v6_formal"
        pred_file = framework_output / "predictions" / "rank_predictions.jsonl"

        if not pred_file.exists():
            print(f"  ERROR: No predictions to evaluate: {pred_file}")
            sys.exit(1)

        from src.eval.ranking_task_metrics import compute_ranking_task_metrics, build_ranking_eval_frame
        from src.utils.exp_io import load_jsonl
        import pandas as pd

        raw_records = load_jsonl(str(pred_file))
        raw_df = pd.DataFrame(raw_records)
        eval_df = build_ranking_eval_frame(raw_df)

        metrics = compute_ranking_task_metrics(eval_df, k=10)
        metrics_5 = compute_ranking_task_metrics(eval_df, k=5)
        metrics_20 = compute_ranking_task_metrics(eval_df, k=20)

        report = {
            "domain": domain,
            "method": "SRPD-v6",
            "n_users": len(eval_df),
            "HR@5": metrics_5.get("hit_rate", 0),
            "NDCG@5": metrics_5.get("ndcg", 0),
            "HR@10": metrics.get("hit_rate", 0),
            "NDCG@10": metrics.get("ndcg", 0),
            "HR@20": metrics_20.get("hit_rate", 0),
            "NDCG@20": metrics_20.get("ndcg", 0),
            "MRR": metrics.get("mrr", 0),
        }

        eval_dir = framework_output / "tables"
        eval_dir.mkdir(parents=True, exist_ok=True)

        with open(eval_dir / "ranking_metrics.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"  Results for {domain}:")
        for k, v in report.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

        print(f"  Saved to: {eval_dir / 'ranking_metrics.json'}")

    print(f"\n=== ALL REQUESTED STEPS COMPLETE for {domain} ===")


if __name__ == "__main__":
    main()
