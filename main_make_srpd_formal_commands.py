from __future__ import annotations

import argparse
import posixpath
import shlex
from pathlib import Path
from typing import Any


DEFAULT_DOMAINS = ("books", "electronics", "movies")


def _q(value: Any) -> str:
    return shlex.quote(str(value).replace("\\", "/"))


def _p(*parts: Any) -> str:
    return posixpath.join(*(str(part).replace("\\", "/") for part in parts if str(part)))


def _domain_list(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(DEFAULT_DOMAINS)
    return [item.strip() for item in value.split(",") if item.strip()]


def _exp(domain: str) -> str:
    return f"{domain}_large10000_100neg"


def _srpd_config(domain: str) -> str:
    return f"configs/srpd/{domain}_large10000_100neg_srpd_v6_formal.yaml"


def _lora_config(domain: str) -> str:
    return f"configs/lora/{domain}_large10000_100neg_srpd_v6_formal.yaml"


def build_commands(domains: list[str], *, stage: str, startup_check_only: bool) -> list[str]:
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'cd "${PONY_REC_ROOT:-$HOME/projects/pony-rec-rescue-shadow-v6}"',
        "mkdir -p outputs/summary/logs",
    ]
    for domain in domains:
        exp = _exp(domain)
        task_test = f"outputs/baselines/external_tasks/{exp}_test_same_candidate"
        prediction_path = f"outputs/{exp}_srpd_v6_formal/predictions/rank_predictions.jsonl"
        scores_dir = f"outputs/summary/week8_srpd_formal/{domain}"
        commands.append("")
        commands.append(f"# SRPD formal: {domain}")
        if stage in {"all", "data"} or (stage == "train" and startup_check_only):
            commands.append(f"python main_build_srpd_rank_data.py --config {_q(_srpd_config(domain))}")
        if stage in {"all", "train"}:
            train_cmd = ["python main_lora_train_rank.py", "--config", _q(_lora_config(domain))]
            if startup_check_only:
                train_cmd.append("--startup_check")
            commands.append(" ".join(train_cmd))
        if stage in {"all", "eval"} and not startup_check_only:
            commands.append(f"python main_eval_lora_rank.py --config {_q(_lora_config(domain))} --resume_partial")
            commands.append(
                " ".join(
                    [
                        "python main_export_srpd_scores_from_predictions.py",
                        "--ranking_input_path", _q(_p(task_test, "ranking_test.jsonl")),
                        "--candidate_items_path", _q(_p(task_test, "candidate_items.csv")),
                        "--prediction_path", _q(prediction_path),
                        "--output_scores_path", _q(_p(scores_dir, "srpd_scores.csv")),
                        "--provenance_output_path", _q(_p(scores_dir, "srpd_internal_provenance.json")),
                        "--method_variant", "SRPD-v6_formal_gap_gated_preference_sft",
                        "--status_label", "same_schema_internal_ablation",
                        "--artifact_class", "completed_result",
                    ]
                )
            )
            commands.append(
                " ".join(
                    [
                        "python main_import_same_candidate_baseline_scores.py",
                        "--baseline_name", _q(f"{domain}_srpd_v6_formal"),
                        "--exp_name", _q(f"{domain}_srpd_v6_formal_same_candidate"),
                        "--domain", _q(domain),
                        "--ranking_input_path", _q(_p(task_test, "ranking_test.jsonl")),
                        "--scores_path", _q(_p(scores_dir, "srpd_scores.csv")),
                        "--status_label", "same_schema_internal_ablation",
                        "--artifact_class", "completed_result",
                    ]
                )
            )
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fail-fast SRPD formal server commands.")
    parser.add_argument("--domains", default="all")
    parser.add_argument("--stage", choices=["all", "data", "train", "eval"], default="all")
    parser.add_argument("--startup_check_only", action="store_true")
    parser.add_argument("--output_path", default="outputs/summary/week8_srpd_formal_commands.sh")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = build_commands(_domain_list(args.domains), stage=args.stage, startup_check_only=args.startup_check_only)
    path = Path(args.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"Saved SRPD formal command script: {path}")
    print(f"commands={sum(1 for line in commands if line.startswith('python '))}")


if __name__ == "__main__":
    main()
