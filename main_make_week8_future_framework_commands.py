from __future__ import annotations

import argparse
import posixpath
import shlex
from pathlib import PurePosixPath
from typing import Any

from src.utils.exp_io import load_yaml


def _server_text(value: Any) -> str:
    return str(value).replace("\\", "/")


def _q(value: Any) -> str:
    return shlex.quote(_server_text(value))


def _csv(values: Any) -> str:
    if isinstance(values, str):
        text = values.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return ",".join(item.strip() for item in text.split(",") if item.strip())
    return ",".join(str(value) for value in values)


def _p(*parts: Any) -> str:
    cleaned = [_server_text(part) for part in parts if str(part) != ""]
    if not cleaned:
        return ""
    return posixpath.join(*cleaned)


def _domain_list(cfg: dict[str, Any], selected: str) -> list[str]:
    domains = cfg.get("domains") or {}
    if selected == "all":
        return list(domains)
    return [item.strip() for item in selected.split(",") if item.strip()]


def _ranking_path(task_dir: str, split: str) -> str:
    return _p(task_dir, f"ranking_{split}.jsonl")


def _pointwise_path(root: str, domain: str, split: str, family: str) -> str:
    return _p(root, f"{domain}_{family}_{split}_pointwise.jsonl")


def _shadow_exp(domain: str, family: str) -> str:
    return f"{domain}_large10000_100neg_{family}_shadow_v1"


def _light_exp(domain: str) -> str:
    return f"{domain}_large10000_100neg_light_verbalized_confidence"


def build_shadow_commands(cfg: dict[str, Any], domains: list[str], *, max_events: int | None) -> list[str]:
    output_root = str(cfg.get("output_root", "outputs"))
    model_config = str(cfg.get("model_config", "configs/model/qwen3_8b_local.yaml"))
    seed = str(cfg.get("seed", 20260506))
    topk = int(cfg.get("topk", 10))
    shadow_cfg = cfg["shadow"]
    pointwise_root = str(shadow_cfg["pointwise_output_dir"])
    commands: list[str] = []

    for domain in domains:
        domain_cfg = cfg["domains"][domain]
        valid_rank = _ranking_path(domain_cfg["task_valid_dir"], "valid")
        test_rank = _ranking_path(domain_cfg["task_test_dir"], "test")
        valid_pointwise = _pointwise_path(pointwise_root, domain, "valid", "shadow")
        test_pointwise = _pointwise_path(pointwise_root, domain, "test", "shadow")
        exp_name = _shadow_exp(domain, "qwen3")
        rank_valid_exp = f"{domain}_large10000_100neg_shadow_anchor_rank_valid"
        rank_test_exp = f"{domain}_large10000_100neg_shadow_anchor_rank_test"
        gate_dir = _p(shadow_cfg["gate_sweep_output_dir"], domain)

        for split, rank_path, pointwise_path in [
            ("valid", valid_rank, valid_pointwise),
            ("test", test_rank, test_pointwise),
        ]:
            commands.append(
                " ".join(
                    [
                        "python main_build_week8_same_candidate_pointwise_inputs.py",
                        "--ranking_input_path", _q(rank_path),
                        "--output_path", _q(pointwise_path),
                        "--domain", _q(domain),
                        "--split_name", split,
                        "--task_name", _q(PurePosixPath(rank_path).parent.name),
                        *(["--max_events", str(max_events)] if max_events else []),
                    ]
                )
            )

        for split, input_path in [("valid", valid_pointwise), ("test", test_pointwise)]:
            commands.append(
                " ".join(
                    [
                        "python main_infer.py",
                        "--exp_name", _q(exp_name),
                        "--input_path", _q(input_path),
                        "--split_name", split,
                        "--model_config", _q(model_config),
                        "--prompt_path", _q(shadow_cfg["signal_prompt_path"]),
                        "--output_root", _q(output_root),
                        "--response_schema shadow",
                        "--shadow_variant", _q(shadow_cfg["winner_signal_variant"]),
                        "--resume_partial",
                        "--checkpoint_every_batches 1",
                        "--seed", seed,
                    ]
                )
            )
        commands.append(
            f"python main_eval_shadow.py --exp_name {_q(exp_name)} --output_root {_q(output_root)} --score_col shadow_score --seed {seed}"
        )
        commands.append(
            " ".join(
                [
                    "python main_calibrate_shadow.py",
                    "--exp_name", _q(exp_name),
                    "--shadow_variant", _q(shadow_cfg["winner_signal_variant"]),
                    "--output_root", _q(output_root),
                    "--score_col shadow_score",
                    "--method isotonic",
                ]
            )
        )

        commands.append(
            " ".join(
                [
                    "python main_rank.py",
                    "--exp_name", _q(rank_valid_exp),
                    "--input_path", _q(valid_rank),
                    "--model_config", _q(model_config),
                    "--prompt_path prompts/candidate_ranking.txt",
                    "--output_root", _q(output_root),
                    "--topk", str(topk),
                    "--max_new_tokens 256",
                    "--resume_partial",
                    "--seed", seed,
                ]
            )
        )
        commands.append(
            " ".join(
                [
                    "python main_rank.py",
                    "--exp_name", _q(rank_test_exp),
                    "--input_path", _q(test_rank),
                    "--model_config", _q(model_config),
                    "--prompt_path prompts/candidate_ranking.txt",
                    "--output_root", _q(output_root),
                    "--topk", str(topk),
                    "--max_new_tokens 256",
                    "--resume_partial",
                    "--seed", seed,
                ]
            )
        )
        commands.append(
            " ".join(
                [
                    "python main_run_week8_shadow_v6_gate_sweep.py",
                    "--valid_rank_input_path", _q(_p(output_root, rank_valid_exp, "predictions", "rank_predictions.jsonl")),
                    "--test_rank_input_path", _q(_p(output_root, rank_test_exp, "predictions", "rank_predictions.jsonl")),
                    "--valid_signal_input_path", _q(_p(output_root, exp_name, "calibrated", "valid_calibrated.jsonl")),
                    "--test_signal_input_path", _q(_p(output_root, exp_name, "calibrated", "test_calibrated.jsonl")),
                    "--output_dir", _q(gate_dir),
                    "--domain", _q(domain),
                    "--winner_signal_variant", _q(shadow_cfg["winner_signal_variant"]),
                    "--gate_thresholds", _q(_csv(shadow_cfg["gate_thresholds"])),
                    "--uncertainty_thresholds", _q(_csv(shadow_cfg["uncertainty_thresholds"])),
                    "--anchor_conflict_penalties", _q(_csv(shadow_cfg["anchor_conflict_penalties"])),
                    "--artifact_class", _q(shadow_cfg.get("artifact_class", "diagnostic")),
                ]
            )
        )
    return commands


def build_light_commands(cfg: dict[str, Any], domains: list[str], *, max_events: int | None) -> list[str]:
    output_root = str(cfg.get("output_root", "outputs"))
    model_config = str(cfg.get("model_config", "configs/model/qwen3_8b_local.yaml"))
    seed = str(cfg.get("seed", 20260506))
    light_cfg = cfg["light"]
    pointwise_root = str(light_cfg["pointwise_output_dir"])
    commands: list[str] = []
    for domain in domains:
        domain_cfg = cfg["domains"][domain]
        valid_rank = _ranking_path(domain_cfg["task_valid_dir"], "valid")
        test_rank = _ranking_path(domain_cfg["task_test_dir"], "test")
        valid_pointwise = _pointwise_path(pointwise_root, domain, "valid", "light")
        test_pointwise = _pointwise_path(pointwise_root, domain, "test", "light")
        exp_name = _light_exp(domain)
        for split, rank_path, pointwise_path in [
            ("valid", valid_rank, valid_pointwise),
            ("test", test_rank, test_pointwise),
        ]:
            commands.append(
                " ".join(
                    [
                        "python main_build_week8_same_candidate_pointwise_inputs.py",
                        "--ranking_input_path", _q(rank_path),
                        "--output_path", _q(pointwise_path),
                        "--domain", _q(domain),
                        "--split_name", split,
                        "--task_name", _q(PurePosixPath(rank_path).parent.name),
                        *(["--max_events", str(max_events)] if max_events else []),
                    ]
                )
            )
            commands.append(
                " ".join(
                    [
                        "python main_infer.py",
                        "--exp_name", _q(exp_name),
                        "--input_path", _q(pointwise_path),
                        "--split_name", split,
                        "--model_config", _q(model_config),
                        "--prompt_path", _q(light_cfg["prompt_path"]),
                        "--output_root", _q(output_root),
                        "--response_schema pointwise_yesno",
                        "--resume_partial",
                        "--checkpoint_every_batches 1",
                        "--seed", seed,
                    ]
                )
            )
        commands.append(f"python main_eval.py --exp_name {_q(exp_name)} --output_root {_q(output_root)}")
        commands.append(
            " ".join(
                [
                    "python main_calibrate.py",
                    "--exp_name", _q(exp_name),
                    "--output_root", _q(output_root),
                    "--method isotonic",
                    "--allow_user_overlap true",
                    "--allow_item_overlap true",
                ]
            )
        )
    return commands


def build_generated_title_commands(cfg: dict[str, Any], domains: list[str], *, max_events: int | None) -> list[str]:
    out_root = str(cfg["generated_title"]["scaffold_output_dir"])
    commands: list[str] = []
    for domain in domains:
        domain_cfg = cfg["domains"][domain]
        for split, task_key in [("valid", "task_valid_dir"), ("test", "task_test_dir")]:
            rank_path = _ranking_path(domain_cfg[task_key], split)
            output_path = _p(out_root, domain, f"{split}_verification_scaffold.jsonl")
            commands.append(
                " ".join(
                    [
                        "python main_build_week8_generated_title_verification_scaffold.py",
                        "--ranking_input_path", _q(rank_path),
                        "--output_path", _q(output_path),
                        "--domain", _q(domain),
                        "--split_name", split,
                        *(["--max_events", str(max_events)] if max_events else []),
                    ]
                )
            )
    return commands


def build_lora_scaffold_commands(cfg_path: str, domains: str) -> list[str]:
    return [
        " ".join(
            [
                "python main_build_week8_lora_framework_scaffold.py",
                "--config", _q(cfg_path),
                "--domains", _q(domains),
                "--output_dir configs/week8_future_lora",
            ]
        )
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate future Week8 framework commands without running them.")
    parser.add_argument("--config", default="configs/week8_large_scale_future_framework.yaml")
    parser.add_argument("--domains", default="all")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "shadow", "light", "generated_title"],
        help="Which command block to emit.",
    )
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--max_events", type=int, default=None, help="Optional smoke cap for generated commands.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    domains = _domain_list(cfg, args.domains)
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", "cd \"${PONY_REC_ROOT:-$HOME/projects/pony-rec-rescue-shadow-v6}\"", "mkdir -p outputs/summary/logs"]
    if args.stage in {"all", "shadow"}:
        commands.append("")
        commands.append("# Shadow v1 signal + validation-selected v6 bridge")
        commands.extend(build_shadow_commands(cfg, domains, max_events=args.max_events))
    if args.stage in {"all", "light"}:
        commands.append("")
        commands.append("# Old light/verbalized-confidence large-scale negative-control ablation")
        commands.extend(build_light_commands(cfg, domains, max_events=args.max_events))
    if args.stage in {"all", "generated_title"}:
        commands.append("")
        commands.append("# Catalog-grounded generated-title verification scaffold")
        commands.extend(build_generated_title_commands(cfg, domains, max_events=args.max_events))
    if args.stage == "all":
        commands.append("")
        commands.append("# LoRA config scaffolds, not training")
        commands.extend(build_lora_scaffold_commands(args.config, args.domains))

    output_path = args.output_path
    if output_path is None:
        if args.stage == "shadow":
            output_path = cfg["shadow"]["command_output_path"]
        elif args.stage == "light":
            output_path = cfg["light"]["command_output_path"]
        else:
            output_path = "outputs/summary/week8_large10000_100neg_future_framework_commands.sh"
    from pathlib import Path

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"Saved command script: {path}")
    print(f"commands={sum(1 for line in commands if line.startswith('python '))}")


if __name__ == "__main__":
    main()
