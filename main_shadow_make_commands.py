from __future__ import annotations

import argparse
from pathlib import Path
import shlex
from typing import Any

from src.utils.exp_io import load_yaml


def _shell_join(parts: list[str]) -> str:
    return " ".join(str(part) for part in parts if str(part) != "")


def _exp_prefix(domain: str, variant: str, scenario: str) -> str:
    return f"{domain}_qwen3_{variant}_{scenario}"


def _variant_list(cfg: dict[str, Any], selected: str) -> list[str]:
    if selected == "all":
        return sorted((cfg.get("variants") or {}).keys())
    return [item.strip() for item in selected.split(",") if item.strip()]


def _domain_list(scenario_cfg: dict[str, Any], selected: str) -> list[str]:
    domains = scenario_cfg.get("domains") or {}
    if selected == "all":
        return sorted(domains.keys())
    return [item.strip() for item in selected.split(",") if item.strip()]


def _max_arg(name: str, value: Any) -> list[str]:
    if value is None:
        return []
    return [name, str(value)]


def _quote(value: str | Path) -> str:
    text = str(value)
    if text == "~" or text.startswith("~/"):
        return text
    return shlex.quote(text)


def _noisy_build_command(clean_path: str, noisy_path: str, *, seed: str, python_cmd: str) -> str:
    return _shell_join(
        [
            "test -f", _quote(noisy_path),
            f"|| {python_cmd} main_generate_noisy.py",
            "--input_path", _quote(clean_path),
            "--output_path", _quote(noisy_path),
            "--noise_level 0.1",
            "--seed", seed,
        ]
    )


def _noisy_prep_commands(domain_cfg: dict[str, Any], *, seed: str, python_cmd: str) -> list[str]:
    pairs = [
        ("pointwise_valid_path", "noisy_valid_path"),
        ("pointwise_test_path", "noisy_test_path"),
        ("ranking_test_path", "noisy_ranking_test_path"),
    ]
    commands: list[str] = []
    for clean_key, noisy_key in pairs:
        if clean_key in domain_cfg and noisy_key in domain_cfg:
            commands.append(
                _noisy_build_command(
                    str(domain_cfg[clean_key]),
                    str(domain_cfg[noisy_key]),
                    seed=seed,
                    python_cmd=python_cmd,
                )
            )
    return commands


def build_commands(
    cfg: dict[str, Any],
    *,
    scenario: str,
    variants: list[str],
    domains: list[str],
    include_noisy: bool,
    project_dir: str,
    prepare_noisy_inputs: bool,
    winner_signal_variant: str,
    python_cmd: str,
) -> list[str]:
    scenario_cfg = cfg["scenarios"][scenario]
    output_root = str(cfg.get("output_root", "outputs"))
    seed = str(cfg.get("seed", 42))
    topk = str(cfg.get("default_topk", 10))
    max_new_tokens = str(cfg.get("default_rank_max_new_tokens", 96))
    rerank_variant = str(cfg.get("default_rerank_variant", "nonlinear_structured_risk_rerank"))
    lambda_penalty = str(cfg.get("default_lambda_penalty", 0.5))
    commands: list[str] = [f"cd {_quote(project_dir)}", "mkdir -p outputs/logs outputs/summary"]

    for domain in domains:
        domain_cfg = scenario_cfg["domains"][domain]
        if include_noisy and prepare_noisy_inputs:
            commands.extend(_noisy_prep_commands(domain_cfg, seed=seed, python_cmd=python_cmd))

        rank_exp = f"{domain}_qwen3_shadow_rank_{scenario}"
        commands.append(
            _shell_join(
                [
                    f"{python_cmd} main_rank.py",
                    "--exp_name", rank_exp,
                    "--input_path", domain_cfg["ranking_test_path"],
                    "--model_config", domain_cfg["rank_model_config"],
                    "--prompt_path prompts/candidate_ranking.txt",
                    "--output_root", output_root,
                    "--topk", topk,
                    "--max_new_tokens", max_new_tokens,
                    *_max_arg("--max_samples", domain_cfg.get("max_rank_samples")),
                    "--resume_partial",
                    "--seed", seed,
                ]
            )
        )

        noisy_rank_exp = f"{domain}_qwen3_shadow_rank_{scenario}_noisy_nl10"
        if include_noisy:
            commands.append(
                _shell_join(
                    [
                        f"{python_cmd} main_rank.py",
                        "--exp_name", noisy_rank_exp,
                        "--input_path", domain_cfg["noisy_ranking_test_path"],
                        "--model_config", domain_cfg["rank_model_config"],
                        "--prompt_path prompts/candidate_ranking.txt",
                        "--output_root", output_root,
                        "--topk", topk,
                        "--max_new_tokens", max_new_tokens,
                        *_max_arg("--max_samples", domain_cfg.get("max_rank_samples")),
                        "--resume_partial",
                        "--seed", seed,
                    ]
                )
            )

        for variant in variants:
            if variant == "shadow_v6":
                signal_pointwise_exp = f"{_exp_prefix(domain, winner_signal_variant, scenario)}_pointwise"
                bridge_exp = f"{_exp_prefix(domain, variant, scenario)}_structured_risk"
                commands.append(
                    _shell_join(
                        [
                            f"{python_cmd} main_build_shadow_v6_bridge.py",
                            "--exp_name", bridge_exp,
                            "--rank_input_path", f"{output_root}/{rank_exp}/predictions/rank_predictions.jsonl",
                            "--signal_input_path", f"{output_root}/{signal_pointwise_exp}/calibrated/test_calibrated.jsonl",
                            "--output_root", output_root,
                            "--winner_signal_variant", winner_signal_variant,
                            "--signal_score_col shadow_calibrated_score",
                            "--signal_uncertainty_col shadow_uncertainty",
                            "--k", topk,
                            "--seed", seed,
                        ]
                    )
                )
                if include_noisy:
                    noisy_signal_pointwise_exp = f"{signal_pointwise_exp}_noisy_nl10"
                    noisy_bridge_exp = f"{bridge_exp}_noisy_nl10"
                    commands.append(
                        _shell_join(
                            [
                                f"{python_cmd} main_build_shadow_v6_bridge.py",
                                "--exp_name", noisy_bridge_exp,
                                "--rank_input_path", f"{output_root}/{noisy_rank_exp}/predictions/rank_predictions.jsonl",
                                "--signal_input_path", f"{output_root}/{noisy_signal_pointwise_exp}/calibrated/test_calibrated.jsonl",
                                "--output_root", output_root,
                                "--winner_signal_variant", winner_signal_variant,
                                "--signal_score_col shadow_calibrated_score",
                                "--signal_uncertainty_col shadow_uncertainty",
                                "--k", topk,
                                "--seed", seed,
                            ]
                        )
                    )
                continue

            prompt_path = cfg["variants"][variant]["prompt_path"]
            pointwise_exp = f"{_exp_prefix(domain, variant, scenario)}_pointwise"
            commands.extend(
                [
                    _shell_join(
                        [
                            f"{python_cmd} main_infer.py",
                            "--exp_name", pointwise_exp,
                            "--input_path", domain_cfg["pointwise_valid_path"],
                            "--split_name valid",
                            "--model_config", domain_cfg["model_config"],
                            "--prompt_path", prompt_path,
                            "--output_root", output_root,
                            "--response_schema shadow",
                            "--shadow_variant", variant,
                            *_max_arg("--max_samples", domain_cfg.get("max_pointwise_samples")),
                            "--resume_partial",
                            "--checkpoint_every_batches 1",
                            "--seed", seed,
                        ]
                    ),
                    _shell_join(
                        [
                            f"{python_cmd} main_infer.py",
                            "--exp_name", pointwise_exp,
                            "--input_path", domain_cfg["pointwise_test_path"],
                            "--split_name test",
                            "--model_config", domain_cfg["model_config"],
                            "--prompt_path", prompt_path,
                            "--output_root", output_root,
                            "--response_schema shadow",
                            "--shadow_variant", variant,
                            *_max_arg("--max_samples", domain_cfg.get("max_pointwise_samples")),
                            "--resume_partial",
                            "--checkpoint_every_batches 1",
                            "--seed", seed,
                        ]
                    ),
                    f"{python_cmd} main_eval_shadow.py --exp_name {pointwise_exp} --output_root {output_root} --score_col shadow_score --seed {seed}",
                    f"{python_cmd} main_calibrate_shadow.py --exp_name {pointwise_exp} --shadow_variant {variant} --output_root {output_root} --score_col shadow_score --method isotonic",
                ]
            )

            rerank_exp = f"{_exp_prefix(domain, variant, scenario)}_structured_risk"
            commands.append(
                _shell_join(
                    [
                        f"{python_cmd} main_rank_rerank.py",
                        "--exp_name", rank_exp,
                        "--new_exp_name", rerank_exp,
                        "--uncertainty_exp_name", pointwise_exp,
                        "--uncertainty_input_path", f"{output_root}/{pointwise_exp}/calibrated/test_calibrated.jsonl",
                        "--uncertainty_col shadow_uncertainty",
                        "--uncertainty_confidence_col shadow_calibrated_score",
                        "--output_root", output_root,
                        "--rerank_variant", rerank_variant,
                        "--lambda_penalty", lambda_penalty,
                        "--k", topk,
                        "--seed", seed,
                    ]
                )
            )

            if not include_noisy:
                continue
            noisy_pointwise_exp = f"{pointwise_exp}_noisy_nl10"
            commands.extend(
                [
                    _shell_join(
                        [
                            f"{python_cmd} main_infer.py",
                            "--exp_name", noisy_pointwise_exp,
                            "--input_path", domain_cfg["noisy_valid_path"],
                            "--split_name valid",
                            "--model_config", domain_cfg["model_config"],
                            "--prompt_path", prompt_path,
                            "--output_root", output_root,
                            "--response_schema shadow",
                            "--shadow_variant", variant,
                            *_max_arg("--max_samples", domain_cfg.get("max_pointwise_samples")),
                            "--resume_partial",
                            "--checkpoint_every_batches 1",
                            "--seed", seed,
                        ]
                    ),
                    _shell_join(
                        [
                            f"{python_cmd} main_infer.py",
                            "--exp_name", noisy_pointwise_exp,
                            "--input_path", domain_cfg["noisy_test_path"],
                            "--split_name test",
                            "--model_config", domain_cfg["model_config"],
                            "--prompt_path", prompt_path,
                            "--output_root", output_root,
                            "--response_schema shadow",
                            "--shadow_variant", variant,
                            *_max_arg("--max_samples", domain_cfg.get("max_pointwise_samples")),
                            "--resume_partial",
                            "--checkpoint_every_batches 1",
                            "--seed", seed,
                        ]
                    ),
                    f"{python_cmd} main_eval_shadow.py --exp_name {noisy_pointwise_exp} --output_root {output_root} --score_col shadow_score --seed {seed}",
                    f"{python_cmd} main_calibrate_shadow.py --exp_name {noisy_pointwise_exp} --shadow_variant {variant} --output_root {output_root} --score_col shadow_score --method isotonic",
                ]
            )
            noisy_rerank_exp = f"{rerank_exp}_noisy_nl10"
            commands.append(
                _shell_join(
                    [
                        f"{python_cmd} main_rank_rerank.py",
                        "--exp_name", noisy_rank_exp,
                        "--new_exp_name", noisy_rerank_exp,
                        "--uncertainty_exp_name", noisy_pointwise_exp,
                        "--uncertainty_input_path", f"{output_root}/{noisy_pointwise_exp}/calibrated/test_calibrated.jsonl",
                        "--uncertainty_col shadow_uncertainty",
                        "--uncertainty_confidence_col shadow_calibrated_score",
                        "--output_root", output_root,
                        "--rerank_variant", rerank_variant,
                        "--lambda_penalty", lambda_penalty,
                        "--k", topk,
                        "--seed", seed,
                    ]
                )
            )

    commands.append(
        f"{python_cmd} main_compare_shadow_line.py --scenario {scenario} --variants {','.join(variants)} --domains {','.join(domains)}"
    )
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/shadow/week7_9_shadow_runtime.yaml")
    parser.add_argument("--scenario", choices=["small_prior", "full_replay", "formal_full_domains"], default="small_prior")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--domains", default="all")
    parser.add_argument("--include_noisy", action="store_true")
    parser.add_argument(
        "--project_dir",
        default=None,
        help="Directory to cd into at the top of the generated bash script. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--no_prepare_noisy_inputs",
        action="store_true",
        help="Do not prepend commands that materialize missing *_noisy_nl10 inputs with main_generate_noisy.py.",
    )
    parser.add_argument(
        "--winner_signal_variant",
        default=None,
        help="Winner signal used by the shadow_v6 bridge. Defaults to manifest winner_signal_variant or shadow_v1.",
    )
    parser.add_argument(
        "--python_cmd",
        default=None,
        help="Python executable used in generated bash commands. Defaults to manifest python_cmd or python3.12.",
    )
    parser.add_argument("--output_path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.manifest)
    scenario_cfg = cfg["scenarios"][args.scenario]
    variants = _variant_list(cfg, args.variants)
    domains = _domain_list(scenario_cfg, args.domains)
    project_dir = args.project_dir or str(Path.cwd())
    winner_signal_variant = args.winner_signal_variant or str(cfg.get("winner_signal_variant", "shadow_v1"))
    python_cmd = args.python_cmd or str(cfg.get("python_cmd", "python3.12"))
    commands = build_commands(
        cfg,
        scenario=args.scenario,
        variants=variants,
        domains=domains,
        include_noisy=args.include_noisy,
        project_dir=project_dir,
        prepare_noisy_inputs=not args.no_prepare_noisy_inputs,
        winner_signal_variant=winner_signal_variant,
        python_cmd=python_cmd,
    )
    text = "\n".join(commands) + "\n"
    if args.output_path:
        path = Path(args.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
