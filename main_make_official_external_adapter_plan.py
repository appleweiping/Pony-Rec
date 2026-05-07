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


def _plan_rows(cfg: dict[str, Any], domains: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for method, method_cfg in (cfg.get("official_baselines") or {}).items():
        target_name = str(method_cfg["target_baseline_name"])
        for domain in domains:
            exp = _exp_prefix(domain)
            task = f"outputs/baselines/external_tasks/{exp}_test_same_candidate"
            ranking = f"{task}/ranking_test.jsonl"
            scores = f"outputs/baselines/official_adapters/{exp}_{method}_official/{method}_official_scores.csv"
            imported_exp = f"{exp}_{target_name}_same_candidate"
            rows.append(
                {
                    "method": method,
                    "domain": domain,
                    "runner": _runner_name(method),
                    "task_dir": task,
                    "ranking_input_path": ranking,
                    "scores_path": scores,
                    "imported_exp": imported_exp,
                    "target_baseline_name": target_name,
                    "dataset_name": _dataset(domain),
                }
            )
    return rows


def _commands(rows: list[dict[str, str]]) -> list[str]:
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# This file is an implementation plan, not yet a runnable official baseline suite.",
        "# Each main_run_*_official_same_candidate_adapter.py runner must preserve the",
        "# official algorithm and emit source_event_id,user_id,item_id,score.",
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
                        "--task_dir",
                        _q(row["task_dir"]),
                        "--domain",
                        _q(row["domain"]),
                        "--output_scores_path",
                        _q(row["scores_path"]),
                    ]
                ),
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
                    ]
                ),
                "",
            ]
        )
    return commands


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    rows = _plan_rows(cfg, _parse_domains(args.domains))
    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(_commands(rows)) + "\n", encoding="utf-8")
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved command plan: {output_path}")
    print(f"Saved manifest: {manifest_path}")
    print(f"planned_rows={len(rows)}")


if __name__ == "__main__":
    main()
