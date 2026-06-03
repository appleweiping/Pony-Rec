from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOMAINS = ("sports", "toys")
DEFAULT_NAME_TOKENS = ("ccrp", "shadow", "signal", "calibrated", "scored", "rows")
DEFAULT_OUTPUT_DIR = "outputs/summary/paper_critical/ccrp_signal_generation_plan"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a guarded, non-executing plan for the paper-critical C-CRP "
            "signal-row evidence flow. The output records discovery, audit, "
            "selection, observation, and hyperparameter commands with explicit "
            "placeholders. It does not start a server experiment."
        )
    )
    parser.add_argument("--domain", action="append", default=[], help="Domain to include. Defaults to sports and toys.")
    parser.add_argument("--remote_project", default="/home/ajifang/projects/pony-rec-rescue-shadow-v6")
    parser.add_argument("--remote_root", default=".")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected_events", type=int, default=10000)
    parser.add_argument("--expected_candidates_per_event", type=int, default=101)
    parser.add_argument("--selection_metric", default="NDCG@10")
    parser.add_argument("--plan_id", default="ccrp_signal_generation_plan")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_sh", default="")
    return parser.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _git_status_short(paths: list[str] | None = None) -> str:
    try:
        cmd = ["git", "status", "--short"]
        if paths:
            cmd.extend(["--", *paths])
        return subprocess.check_output(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _unique_domains(domains: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for domain in domains or list(DEFAULT_DOMAINS):
        value = str(domain).strip().lower()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out or list(DEFAULT_DOMAINS)


def _cmd(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part))


def _line_command(parts: list[str], indent: str = "  ") -> str:
    if len(parts) <= 4:
        return _cmd(parts)
    first = " ".join(parts[:2]) if len(parts) >= 2 else parts[0]
    tokens: list[str] = []
    idx = 2 if len(parts) >= 2 else 1
    while idx < len(parts):
        token = parts[idx]
        if token.startswith("--") and idx + 1 < len(parts) and not parts[idx + 1].startswith("--"):
            tokens.append(f"{shlex.quote(token)} {shlex.quote(parts[idx + 1])}")
            idx += 2
        else:
            tokens.append(shlex.quote(token))
            idx += 1
    lines = [_cmd(parts[:2]) + " \\"]
    for idx, token in enumerate(tokens):
        suffix = " \\" if idx < len(tokens) - 1 else ""
        lines.append(f"{indent}{token}{suffix}")
    return "\n".join(lines)


def _task_prefix(domain: str) -> str:
    return f"{domain}_large10000_100neg"


def _candidate_path(domain: str, split: str) -> str:
    return f"outputs/baselines/external_tasks/{_task_prefix(domain)}_{split}_same_candidate/candidate_items.csv"


def _ranking_path(domain: str, split: str) -> str:
    return f"outputs/baselines/external_tasks/{_task_prefix(domain)}_{split}_same_candidate/ranking_task.jsonl"


def _valid_signal_placeholder(domain: str) -> str:
    return f"TODO_VALID_{domain.upper()}_CCRP_SIGNAL_JSONL_OR_CSV"


def _test_signal_placeholder(domain: str) -> str:
    return f"TODO_TEST_{domain.upper()}_CCRP_SIGNAL_JSONL_OR_CSV"


def _plan_output_dir(base_output_dir: str, domain: str, name: str) -> str:
    return f"{base_output_dir.rstrip('/')}/{name}_{domain}"


def _discovery_outputs(output_dir: str) -> dict[str, str]:
    base = output_dir.rstrip("/")
    return {
        "json": f"{base}/ccrp_signal_source_discovery.json",
        "csv": f"{base}/ccrp_signal_source_discovery.csv",
    }


def _full_audit_outputs(output_dir: str, domain: str) -> dict[str, str]:
    base = output_dir.rstrip("/")
    return {
        "json": f"{base}/ccrp_signal_source_audit_{domain}.json",
        "csv": f"{base}/ccrp_signal_source_audit_{domain}.csv",
    }


def build_domain_plan(
    domain: str,
    *,
    output_dir: str,
    expected_events: int,
    expected_candidates_per_event: int,
    selection_metric: str,
) -> dict[str, Any]:
    expected_rows = expected_events * expected_candidates_per_event
    audit_out = _full_audit_outputs(output_dir, domain)
    selector_out = _plan_output_dir(output_dir, domain, "ccrp_ablation")
    observation_out = _plan_output_dir(output_dir, domain, "observation")
    hyper_out = _plan_output_dir(output_dir, domain, "ccrp_hyperparameter")
    prefix = _task_prefix(domain)
    valid_signal = _valid_signal_placeholder(domain)
    test_signal = _test_signal_placeholder(domain)
    audit_source_placeholder = f"candidate_signal={test_signal}"
    ccrp_eval_candidates = [
        f"outputs/{prefix}_ccrp_v3_qwen3base_pointwise_same_candidate/tables/ranking_eval_records.csv",
        f"outputs/{prefix}_ccrp_v3_same_candidate/tables/ranking_eval_records.csv",
        f"outputs/{prefix}_ccrp_v3/tables/ranking_eval_records.csv",
    ]
    representative_method_eval_candidates = {
        "llmemb": f"outputs/{prefix}_llmemb_official_qwen3base_same_candidate/tables/ranking_eval_records.csv",
        "proex": f"outputs/{prefix}_proex_profile_official_qwen3base_same_candidate/tables/ranking_eval_records.csv",
        "rlmrec": f"outputs/{prefix}_rlmrec_graphcl_official_qwen3base_same_candidate/tables/ranking_eval_records.csv",
    }
    return {
        "domain": domain,
        "expected_events": expected_events,
        "expected_candidates_per_event": expected_candidates_per_event,
        "expected_score_rows": expected_rows,
        "paths": {
            "valid_candidate_items": _candidate_path(domain, "valid"),
            "test_candidate_items": _candidate_path(domain, "test"),
            "valid_ranking": _ranking_path(domain, "valid"),
            "test_ranking": _ranking_path(domain, "test"),
            "valid_signal_placeholder": valid_signal,
            "test_signal_placeholder": test_signal,
            "selector_output_dir": selector_out,
            "observation_output_dir": observation_out,
            "hyperparameter_output_dir": hyper_out,
            "ccrp_eval_candidates": ccrp_eval_candidates,
            "representative_method_eval_candidates": representative_method_eval_candidates,
        },
        "commands": {
            "full_audit_candidate_signal_template": _line_command(
                [
                    "python",
                    "scripts/audit/main_audit_ccrp_uncertainty_sources.py",
                    "--candidate_items_path",
                    _candidate_path(domain, "test"),
                    "--source",
                    audit_source_placeholder,
                    "--expected_events",
                    str(expected_events),
                    "--expected_candidates_per_event",
                    str(expected_candidates_per_event),
                    "--output_json",
                    audit_out["json"],
                    "--output_csv",
                    audit_out["csv"],
                ]
            ),
            "select_ccrp_ablation_and_scores_template": _line_command(
                [
                    "python",
                    "scripts/misc/main_select_ccrp_variant_on_valid.py",
                    "--domain",
                    domain,
                    "--valid_ranking_path",
                    _ranking_path(domain, "valid"),
                    "--test_ranking_path",
                    _ranking_path(domain, "test"),
                    "--valid_candidate_items_path",
                    _candidate_path(domain, "valid"),
                    "--test_candidate_items_path",
                    _candidate_path(domain, "test"),
                    "--valid_signal_path",
                    valid_signal,
                    "--test_signal_path",
                    test_signal,
                    "--output_dir",
                    selector_out,
                    "--score_modes",
                    "confidence_only,evidence_only,confidence_plus_evidence,full",
                    "--ablations",
                    "full,without_boundary_uncertainty,without_calibration_gap,without_evidence_support,without_counterevidence,without_risk_penalty",
                    "--etas",
                    "0.5,1.0,2.0",
                    "--confidence_weights",
                    "0.5,0.7,0.9",
                    "--weight_grid",
                    "0.5,0.3,0.2;0.7,0.2,0.1;0.4,0.4,0.2;0.4,0.2,0.4",
                    "--selection_metric",
                    selection_metric,
                    "--import_scores",
                ]
            ),
            "build_observation_study_template": _line_command(
                [
                    "python",
                    "scripts/analysis/main_build_uncertainty_observation_study.py",
                    "--domain",
                    domain,
                    "--uncertainty_scores_path",
                    f"{selector_out}/ccrp_selected_test_scored_rows.csv",
                    "--ccrp_eval_path",
                    ccrp_eval_candidates[0],
                    "--method_eval",
                    f"llmemb={representative_method_eval_candidates['llmemb']}",
                    "--method_eval",
                    f"proex={representative_method_eval_candidates['proex']}",
                    "--method_eval",
                    f"rlmrec={representative_method_eval_candidates['rlmrec']}",
                    "--output_dir",
                    observation_out,
                    "--expected_events",
                    str(expected_events),
                    "--min_join_rate",
                    "0.999",
                ]
            ),
            "plot_hyperparameter_curves_template": _line_command(
                [
                    "python",
                    "scripts/analysis/main_plot_ccrp_hyperparameter_sweep.py",
                    "--sweep_csv",
                    f"{selector_out}/valid_ccrp_sweep.csv",
                    "--output_dir",
                    hyper_out,
                    "--domain",
                    domain,
                    "--metric",
                    selection_metric,
                    "--score_mode",
                    "full",
                    "--ablation",
                    "full",
                ]
            ),
        },
        "execution_gates": [
            "Do not run while Home rlmrec_graphcl or any other baseline process is active.",
            "Replace signal placeholders only with artifacts classified as recomputable_signal_rows or paper_ready_uncertainty_rows.",
            "Do not use formal C-CRP scores.csv as the uncertainty source when it lacks uncertainty columns.",
            "Use validation-only selection; test rows must be transformed/evaluated only after selection is fixed.",
            "Require exact same-candidate score coverage before importing or reporting table metrics.",
        ],
    }


def build_plan(
    *,
    domains: list[str],
    remote_project: str,
    remote_root: str,
    output_dir: str,
    expected_events: int,
    expected_candidates_per_event: int,
    selection_metric: str,
    plan_id: str,
) -> dict[str, Any]:
    selected_domains = _unique_domains(domains)
    discovery_out = _discovery_outputs(output_dir)
    discovery_parts = [
        "python",
        "scripts/audit/main_remote_discover_ccrp_uncertainty_sources.py",
        "--root",
        remote_root,
    ]
    for domain in selected_domains:
        discovery_parts.extend(["--domain", domain])
    for token in DEFAULT_NAME_TOKENS:
        discovery_parts.extend(["--name_token", token])
    discovery_parts.extend(
        [
            "--expected_events",
            str(expected_events),
            "--expected_candidates_per_event",
            str(expected_candidates_per_event),
            "--output_json",
            discovery_out["json"],
            "--output_csv",
            discovery_out["csv"],
            "--quiet",
        ]
    )
    domain_plans = [
        build_domain_plan(
            domain,
            output_dir=output_dir,
            expected_events=expected_events,
            expected_candidates_per_event=expected_candidates_per_event,
            selection_metric=selection_metric,
        )
        for domain in selected_domains
    ]
    return {
        "plan_id": plan_id,
        "artifact_class": "paper_critical_guarded_execution_plan",
        "status_label": "planning_only_not_executed",
        "will_start_experiment": False,
        "git_commit": _git_commit(),
        "git_worktree_dirty_at_generation": bool(_git_status_short()),
        "git_status_short_relevant_at_generation": _git_status_short(
            [
                "scripts/audit/main_plan_ccrp_signal_generation.py",
                "tests/test_plan_ccrp_signal_generation.py",
                "docs/active_todo_pony_uncertainty.md",
                "docs/paper_claims_and_status.md",
                "docs/paper_critical_experiment_plan_2026-06-03.md",
                output_dir,
            ]
        ),
        "remote_project": remote_project,
        "remote_root": remote_root,
        "domains": selected_domains,
        "output_dir": output_dir,
        "current_blocker": (
            "Full-scale C-CRP Sports/Toys/Home/Tools artifacts visible so far are score-only; "
            "valid/test signal rows with evidence and counterevidence must be located or regenerated "
            "from saved non-test-selected inputs before observation, ablation, or hyperparameter claims."
        ),
        "global_commands": {
            "remote_header_discovery": _line_command(discovery_parts),
        },
        "domain_plans": domain_plans,
        "required_status_before_execution": [
            "Active Home rlmrec_graphcl official row completed or failed with an audited recovery decision.",
            "Official-row server_final and local_light gates complete before any next baseline or C-CRP evidence run.",
            "No matching Python process that would create duplicate-run or GPU-memory risk.",
            "Server disk is above the project danger threshold after any necessary audited cleanup.",
        ],
        "paper_readiness_links": [
            "observation_motivation_study",
            "ccrp_component_ablation",
            "ccrp_hyperparameter_analysis",
        ],
    }


def guarded_shell_script(plan: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "echo 'GUARDED PLAN ONLY: fill TODO signal paths and remove this exit after all gates pass.'",
        "echo 'This file is intentionally non-runnable as generated.'",
        "exit 2",
        "",
        f"cd {plan['remote_project']}",
        "",
        "# 1. Header discovery for candidate signal/scored-row artifacts.",
        plan["global_commands"]["remote_header_discovery"],
        "",
    ]
    for domain_plan in plan["domain_plans"]:
        domain = domain_plan["domain"]
        lines.extend(
            [
                f"# {domain}: audit the chosen signal artifact after replacing TODO paths.",
                domain_plan["commands"]["full_audit_candidate_signal_template"],
                "",
                f"# {domain}: validation-select C-CRP components/hyperparameters and export test rows.",
                domain_plan["commands"]["select_ccrp_ablation_and_scores_template"],
                "",
                f"# {domain}: build observation/motivation table and figure.",
                domain_plan["commands"]["build_observation_study_template"],
                "",
                f"# {domain}: plot validation hyperparameter curves.",
                domain_plan["commands"]["plot_hyperparameter_curves_template"],
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_plan_files(plan: dict[str, Any], *, output_json: str | Path, output_sh: str | Path) -> dict[str, str]:
    json_path = Path(output_json)
    sh_path = Path(output_sh)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    sh_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    sh_path.write_text(guarded_shell_script(plan), encoding="utf-8")
    return {"json": str(json_path), "shell": str(sh_path)}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.rstrip("/")
    output_json = args.output_json or f"{output_dir}/{args.plan_id}.json"
    output_sh = args.output_sh or f"{output_dir}/{args.plan_id}.sh"
    plan = build_plan(
        domains=args.domain,
        remote_project=args.remote_project,
        remote_root=args.remote_root,
        output_dir=output_dir,
        expected_events=args.expected_events,
        expected_candidates_per_event=args.expected_candidates_per_event,
        selection_metric=args.selection_metric,
        plan_id=args.plan_id,
    )
    outputs = write_plan_files(plan, output_json=output_json, output_sh=output_sh)
    print(json.dumps({"ok": True, "outputs": outputs, "will_start_experiment": False}, indent=2))


if __name__ == "__main__":
    main()
