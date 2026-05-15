#!/usr/bin/env python
"""Check that the canonical project roadmap and server entry points exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.utils.exp_io import load_yaml


REQUIRED_FILES = [
    "AGENTS.md",
    "README.md",
    "docs/milestones/README.md",
    "docs/milestones/M0_week1_4_pony12_observation.md",
    "docs/milestones/M1_pony_framework_week5_6.md",
    "docs/milestones/M2_light_series.md",
    "docs/milestones/M3_shadow_series.md",
    "docs/milestones/M4_baseline_system.md",
    "docs/milestones/M5_four_domain_same_candidate.md",
    "docs/milestones/M6_complete_recommender_system.md",
    "docs/top_conference_review_gate.md",
    "docs/server_runbook.md",
    "docs/paper_claims_and_status.md",
    "PROJECT_LINEAGE_AND_FILE_INDEX_2026-05-06.md",
    "docs/archive/legacy_root_reports/CODEX_HANDOFF_WEEK8_2026-05-06.md",
    "docs/archive/legacy_root_reports/WEEK8_LARGE_SCALE_10K_100NEG_PLAN_2026-05-06.md",
    "docs/archive/legacy_root_reports/WEEK8_FUTURE_FRAMEWORK_ROADMAP_2026-05-06.md",
    "OFFICIAL_EXTERNAL_BASELINE_UPGRADE_PLAN_2026-05-07.md",
    "configs/official_external_baselines.yaml",
    "configs/baseline/week8_external_same_candidate_manifest.yaml",
    "scripts/run_week8_large_scale_10k_100neg.sh",
    "scripts/run_week8_shadow_large_scale_diagnostic.sh",
    "scripts/run_week8_light_large_scale_ablation.sh",
    "scripts/run_week8_generated_title_verification_scaffold.sh",
    "main_audit_official_external_repos.py",
    "main_audit_official_fairness_policy.py",
    "main_make_official_external_adapter_plan.py",
    "main_run_official_same_candidate_adapter.py",
    "main_run_llm2rec_official_same_candidate_adapter.py",
    "main_run_llmesr_official_same_candidate_adapter.py",
    "main_run_llmemb_official_same_candidate_adapter.py",
    "main_run_rlmrec_official_same_candidate_adapter.py",
    "main_run_irllrec_official_same_candidate_adapter.py",
    "main_run_setrec_official_same_candidate_adapter.py",
    "main_run_elmrec_official_same_candidate_adapter.py",
    "main_run_proex_official_same_candidate_adapter.py",
    "main_run_promax_official_same_candidate_adapter.py",
    "main_train_score_elmrec_upstream_adapter.py",
    "main_import_same_candidate_baseline_scores.py",
    "main_project_bootstrap.py",
]

REQUIRED_TEXT = {
    "AGENTS.md": [
        "Senior Baseline Advice",
        "Official External Baseline Guardrails",
        "Multi-Agent Collaboration",
        "Server Collaboration",
        "GitHub And Documentation Hygiene",
    ],
    "docs/milestones/README.md": [
        "M0 Week1-4 / pony12 observation",
        "M6 Complete recommendation-system roadmap",
        "Milestone Architect",
        "Multi-Agent Handoff Rules",
    ],
    "docs/top_conference_review_gate.md": [
        "RecSys",
        "source_event_id,user_id,item_id,score",
        "Beauty marked as supplementary smaller-N",
    ],
    "docs/server_runbook.md": [
        "git pull --ff-only",
        "python main_project_readiness_check.py",
        "*_official_qwen3base_*",
        "--plan_stage run",
        "Paste-back template",
    ],
    "docs/paper_claims_and_status.md": [
        "Milestone claim eligibility",
        "*_style_*",
        "*_official_qwen3base_*",
    ],
    "configs/official_external_baselines.yaml": [
        "Unified Qwen3-8B base",
        "train_and_retain_per_official_algorithm",
        "official_code_qwen3base_default_hparams_declared_adaptation",
        "official_code_qwen3base_default_hparams_declared_adaptation_v1",
        "official_default_or_recommended",
        "source_event_id",
        "fairness_contract:",
    ],
}

REQUIRED_SCORE_SCHEMA = ["source_event_id", "user_id", "item_id", "score"]
PRIMARY_VARIANT = "official_code_qwen3base_default_hparams_declared_adaptation"


def _get_nested(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _as_schema(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    return []


def check_fairness_policy(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {"ok": False, "failures": ["missing_config"], "method_failures": {}}
    cfg = load_yaml(config_path)
    failures: list[str] = []
    method_failures: dict[str, list[str]] = {}

    expectations = {
        "fairness_policy.primary_table_variant": PRIMARY_VARIANT,
        "fairness_policy.official_code_required": True,
        "fairness_policy.unified_backbone_required": True,
        "fairness_policy.unified_backbone_family": "Qwen3-8B",
        "fairness_policy.baseline_hyperparameter_policy": "official_default_or_recommended",
        "fairness_policy.baseline_extra_tuning_allowed_in_primary_table": False,
        "fairness_policy.test_set_model_selection_allowed": False,
        "fairness_policy.import_full_catalog_metrics_allowed": False,
    }
    if not _get_nested(cfg, "fairness_policy.policy_id"):
        failures.append("missing fairness_policy.policy_id")
    for path, expected in expectations.items():
        value = _get_nested(cfg, path)
        if value != expected:
            failures.append(f"{path} expected {expected!r}, got {value!r}")
    if _as_schema(_get_nested(cfg, "fairness_policy.score_schema")) != REQUIRED_SCORE_SCHEMA:
        failures.append("fairness_policy.score_schema mismatch")

    for method, method_cfg in (cfg.get("official_baselines") or {}).items():
        contract = (method_cfg or {}).get("fairness_contract", {}) or {}
        current: list[str] = []
        if not contract:
            current.append("missing fairness_contract")
        if contract.get("comparison_tier") != PRIMARY_VARIANT:
            current.append("comparison_tier mismatch")
        if contract.get("official_code_required") is not True:
            current.append("official_code_required must be true")
        if contract.get("backbone_replacement_required") is not True:
            current.append("backbone_replacement_required must be true")
        if contract.get("backbone_family") != "Qwen3-8B":
            current.append("backbone_family must be Qwen3-8B")
        if contract.get("hparam_policy") != "official_default_or_recommended":
            current.append("hparam_policy mismatch")
        if contract.get("extra_baseline_tuning_allowed") is not False:
            current.append("extra_baseline_tuning_allowed must be false")
        if not contract.get("accepted_llm_adaptation_modes"):
            current.append("missing accepted_llm_adaptation_modes")
        if current:
            method_failures[str(method)] = current

    return {
        "ok": not failures and not method_failures,
        "failures": failures,
        "method_failures": method_failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write machine-readable JSON instead of text",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()

    missing_files = []
    for rel in REQUIRED_FILES:
        if not (root / rel).exists():
            missing_files.append(rel)

    missing_text = {}
    for rel, needles in REQUIRED_TEXT.items():
        path = root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        absent = [needle for needle in needles if needle not in text]
        if absent:
            missing_text[rel] = absent

    fairness_policy = check_fairness_policy(root / "configs/official_external_baselines.yaml")

    ok = not missing_files and not missing_text and fairness_policy["ok"]
    result = {
        "ok": ok,
        "root": str(root),
        "missing_files": missing_files,
        "missing_text": missing_text,
        "fairness_policy": fairness_policy,
        "next_reads": [
            "docs/milestones/README.md",
            "docs/server_runbook.md",
            "docs/top_conference_review_gate.md",
        ],
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"project_readiness_ok={ok}")
        if missing_files:
            print("missing_files:")
            for rel in missing_files:
                print(f"  - {rel}")
        if missing_text:
            print("missing_text:")
            for rel, absent in missing_text.items():
                print(f"  - {rel}: {', '.join(absent)}")
        if not fairness_policy["ok"]:
            print("fairness_policy_failures:")
            for failure in fairness_policy["failures"]:
                print(f"  - {failure}")
            for method, failures in fairness_policy["method_failures"].items():
                print(f"  - {method}: {', '.join(failures)}")
        print("next_reads:")
        for rel in result["next_reads"]:
            print(f"  - {rel}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
