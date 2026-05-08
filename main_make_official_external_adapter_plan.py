from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from src.utils.exp_io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate official external adapter implementation plan commands.")
    parser.add_argument("--config", default="configs/official_external_baselines.yaml")
    parser.add_argument("--domains", default="beauty,books,electronics,movies")
    parser.add_argument("--output_path", default="outputs/summary/official_external_adapter_plan.sh")
    parser.add_argument(
        "--plan_stage",
        choices=["inspect", "run"],
        default="inspect",
        help="inspect writes official provenance/blocker reports; run executes implemented adapters and imports scores.",
    )
    return parser.parse_args()


def _q(value: Any) -> str:
    return shlex.quote(str(value))


def _exp_prefix(domain: str) -> str:
    if domain == "beauty":
        return "beauty_supplementary_smallerN_100neg"
    return f"{domain}_large10000_100neg"


def _dataset(domain: str) -> str:
    return {
        "beauty": "amazon_beauty",
        "books": "amazon_books",
        "electronics": "amazon_electronics",
        "movies": "amazon_movies",
    }[domain]


def _parse_domains(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _runner_name(method: str) -> str:
    return f"main_run_{method}_official_same_candidate_adapter.py"


def _plan_rows(cfg: dict[str, Any], domains: list[str], *, plan_stage: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    policy = cfg.get("fairness_policy", {}) or {}
    backbone = cfg.get("backbone_policy", {}) or {}
    policy_id = str(policy.get("policy_id", ""))
    comparison_variant = str(policy.get("primary_table_variant", ""))
    backbone_path = str(policy.get("unified_backbone_path") or backbone.get("base_model_path") or "")
    backbone_family = str(policy.get("unified_backbone_family") or backbone.get("base_model_family") or "")
    for method, method_cfg in (cfg.get("official_baselines") or {}).items():
        target_name = str(method_cfg["target_baseline_name"])
        contract = method_cfg.get("fairness_contract", {}) or {}
        for domain in domains:
            exp = _exp_prefix(domain)
            task = f"outputs/baselines/external_tasks/{exp}_test_same_candidate"
            ranking = f"{task}/ranking_test.jsonl"
            adapter_dir = f"outputs/baselines/official_adapters/{exp}_{method}_official"
            scores = f"{adapter_dir}/{method}_official_scores.csv"
            provenance = f"{adapter_dir}/fairness_provenance.json"
            imported_exp = f"{exp}_{target_name}_same_candidate"
            rows.append(
                {
                    "method": method,
                    "domain": domain,
                    "runner": _runner_name(method),
                    "task_dir": task,
                    "ranking_input_path": ranking,
                    "scores_path": scores,
                    "provenance_path": provenance,
                    "imported_exp": imported_exp,
                    "target_baseline_name": target_name,
                    "dataset_name": _dataset(domain),
                    "fairness_policy_id": policy_id,
                    "comparison_variant": comparison_variant,
                    "comparison_tier": str(contract.get("comparison_tier", "")),
                    "backbone_family": backbone_family,
                    "backbone_path": backbone_path,
                    "hparam_policy": str(contract.get("hparam_policy", "")),
                    "implementation_status": "official_completed" if plan_stage == "run" else "official_inspection_ready",
                    "plan_stage": plan_stage,
                }
            )
    return rows


def _commands(rows: list[dict[str, str]], *, plan_stage: str) -> list[str]:
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Official external adapter plan.",
        "# Default stage is inspect: write fairness provenance/blocker reports without importing scores.",
        "# Use --plan_stage run only after a method adapter has a real official-code implementation.",
        "",
    ]
    for row in rows:
        commands.extend(
            [
                f"# {row['domain']} / {row['method']}",
                " ".join(
                    [
                        "python",
                        row["runner"],
                        "--stage",
                        _q(row["plan_stage"]),
                        "--task_dir",
                        _q(row["task_dir"]),
                        "--domain",
                        _q(row["domain"]),
                        "--output_scores_path",
                        _q(row["scores_path"]),
                        "--fairness_policy_id",
                        _q(row["fairness_policy_id"]),
                        "--comparison_variant",
                        _q(row["comparison_variant"]),
                        "--backbone_path",
                        _q(row["backbone_path"]),
                        "--hparam_policy",
                        _q(row["hparam_policy"]),
                        "--provenance_output_path",
                        _q(row["provenance_path"]),
                        "--allow_blocked_exit_zero" if plan_stage == "inspect" else "",
                    ]
                ),
                "",
            ]
        )
        if plan_stage == "run":
            commands.extend(
                [
                    " ".join(
                        [
                            "python main_audit_same_candidate_score_file.py",
                            "--candidate_items_path",
                            _q(f"{row['task_dir']}/candidate_items.csv"),
                            "--scores_path",
                            _q(row["scores_path"]),
                        ]
                    ),
                    " ".join(
                        [
                            "python main_import_same_candidate_baseline_scores.py",
                            "--baseline_name",
                            _q(row["target_baseline_name"]),
                            "--exp_name",
                            _q(row["imported_exp"]),
                            "--domain",
                            _q(row["domain"]),
                            "--ranking_input_path",
                            _q(row["ranking_input_path"]),
                            "--scores_path",
                            _q(row["scores_path"]),
                            "--status_label same_schema_external_baseline",
                            "--artifact_class completed_result",
                            "--fairness_policy_id",
                            _q(row["fairness_policy_id"]),
                            "--comparison_variant",
                            _q(row["comparison_variant"]),
                            "--implementation_status",
                            _q(row["implementation_status"]),
                            "--provenance_path",
                            _q(row["provenance_path"]),
                            "--require_fairness_provenance",
                        ]
                    ),
                    "",
                ]
            )
    return commands


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    rows = _plan_rows(cfg, _parse_domains(args.domains), plan_stage=args.plan_stage)
    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(_commands(rows, plan_stage=args.plan_stage)) + "\n", encoding="utf-8")
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved command plan: {output_path}")
    print(f"Saved manifest: {manifest_path}")
    print(f"planned_rows={len(rows)}")


if __name__ == "__main__":
    main()
