#!/usr/bin/env python
"""Check that the canonical project roadmap and server entry points exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = [
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
    "CODEX_HANDOFF_WEEK8_2026-05-06.md",
    "WEEK8_LARGE_SCALE_10K_100NEG_PLAN_2026-05-06.md",
    "WEEK8_FUTURE_FRAMEWORK_ROADMAP_2026-05-06.md",
    "OFFICIAL_EXTERNAL_BASELINE_UPGRADE_PLAN_2026-05-07.md",
    "configs/official_external_baselines.yaml",
    "configs/baseline/week8_external_same_candidate_manifest.yaml",
    "scripts/run_week8_large_scale_10k_100neg.sh",
    "scripts/run_week8_shadow_large_scale_diagnostic.sh",
    "scripts/run_week8_light_large_scale_ablation.sh",
    "scripts/run_week8_generated_title_verification_scaffold.sh",
    "main_audit_official_external_repos.py",
    "main_make_official_external_adapter_plan.py",
    "main_import_same_candidate_baseline_scores.py",
    "main_project_bootstrap.py",
]

REQUIRED_TEXT = {
    "docs/milestones/README.md": [
        "M0 Week1-4 / pony12 observation",
        "M6 Complete recommendation-system roadmap",
        "Milestone Architect",
    ],
    "docs/top_conference_review_gate.md": [
        "RecSys",
        "source_event_id,user_id,item_id,score",
        "Beauty marked as supplementary smaller-N",
    ],
    "docs/server_runbook.md": [
        "git pull --ff-only",
        "python main_project_readiness_check.py",
        "*_official_qwen3_lora_*",
    ],
    "docs/paper_claims_and_status.md": [
        "Milestone claim eligibility",
        "*_style_*",
        "*_official_qwen3_lora_*",
    ],
    "configs/official_external_baselines.yaml": [
        "Unified Qwen3-8B base",
        "train_and_retain_per_official_algorithm",
        "source_event_id",
    ],
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

    ok = not missing_files and not missing_text
    result = {
        "ok": ok,
        "root": str(root),
        "missing_files": missing_files,
        "missing_text": missing_text,
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
        print("next_reads:")
        for rel in result["next_reads"]:
            print(f"  - {rel}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
