from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PAPER_CRITICAL_DIR = Path("outputs/summary/paper_critical")
FRAMEWORK_DIR = PAPER_CRITICAL_DIR / "framework_overview"
PLAN_DIR = PAPER_CRITICAL_DIR / "ccrp_signal_generation_plan"
DOMAINS = ("sports", "toys", "home", "tools")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    if not path.exists():
        return entries
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            entries[parts[-1]] = parts[0]
    return entries


def audit_framework_overview(root: Path) -> dict[str, Any]:
    rel_files = [
        "framework_overview.svg",
        "framework_overview.pdf",
        "framework_overview.png",
        "framework_overview_caption.md",
        "framework_overview_provenance.json",
        "framework_overview_manifest.sha256",
    ]
    base = root / FRAMEWORK_DIR
    files = {name: base / name for name in rel_files}
    failures: list[str] = []
    for name, path in files.items():
        if not path.exists():
            failures.append(f"missing:{name}")
        elif path.stat().st_size <= 0:
            failures.append(f"empty:{name}")

    manifest = _load_manifest(files["framework_overview_manifest.sha256"])
    manifest_checks: dict[str, Any] = {}
    for name in rel_files[:-1]:
        path = files[name]
        expected = manifest.get(name)
        actual = _sha256(path) if path.exists() and path.is_file() else ""
        manifest_checks[name] = {"expected": expected, "actual": actual, "ok": bool(expected and expected == actual)}
        if not manifest_checks[name]["ok"]:
            failures.append(f"manifest_mismatch:{name}")

    provenance: dict[str, Any] = {}
    if files["framework_overview_provenance.json"].exists():
        provenance = _read_json(files["framework_overview_provenance.json"])
        if provenance.get("status_label") != "paper_critical_framework_overview_draft":
            failures.append("framework_status_label_not_draft")
        if provenance.get("claim_boundary") != "controlled_same_candidate_ranking_not_full_catalog":
            failures.append("framework_claim_boundary_mismatch")

    return {
        "module": "framework_overview",
        "status": "draft_scaffold_ready" if not failures else "incomplete",
        "artifact_scaffold_ready": not failures,
        "paper_claim_ready": False,
        "remaining_blockers": ["final_paper_layout_and_reviewer_polish"] if not failures else failures,
        "files": {name: str(path) for name, path in files.items()},
        "manifest_checks": manifest_checks,
        "provenance": {
            "status_label": provenance.get("status_label", ""),
            "git_commit": provenance.get("git_commit", ""),
            "generated_at_utc": provenance.get("generated_at_utc", ""),
        },
    }


def audit_signal_source_state(root: Path) -> dict[str, Any]:
    audit_paths = {
        domain: root / PAPER_CRITICAL_DIR / f"ccrp_uncertainty_source_audit_{domain}_fixed_filter_20260604_0502.json"
        for domain in DOMAINS
    }
    domain_rows: dict[str, Any] = {}
    failures: list[str] = []
    paper_ready_count = 0
    recomputable_count = 0
    score_only_count = 0
    exact_coverage_count = 0
    for domain, path in audit_paths.items():
        if not path.exists():
            failures.append(f"missing_signal_source_audit:{domain}")
            domain_rows[domain] = {"path": str(path), "present": False}
            continue
        payload = _read_json(path)
        sources = payload.get("sources") or []
        ready = sum(1 for source in sources if source.get("audit_paper_ready_uncertainty_rows"))
        recomputable = sum(1 for source in sources if source.get("audit_recomputable_signal_rows"))
        score_only = sum(1 for source in sources if source.get("audit_status") == "score_only_not_uncertainty")
        coverage_ok = sum(1 for source in sources if source.get("audit_candidate_key_coverage_rate") == 1.0)
        paper_ready_count += ready
        recomputable_count += recomputable
        score_only_count += score_only
        exact_coverage_count += coverage_ok
        domain_rows[domain] = {
            "path": str(path),
            "present": True,
            "source_count": len(sources),
            "paper_ready_uncertainty_rows": ready,
            "recomputable_signal_rows": recomputable,
            "score_only_not_uncertainty": score_only,
            "exact_candidate_coverage_sources": coverage_ok,
        }

    trace_path = root / PAPER_CRITICAL_DIR / "ccrp_formal_signal_path_trace_20260604_0535.json"
    trace: dict[str, Any] = {}
    if trace_path.exists():
        trace = _read_json(trace_path)
    else:
        failures.append("missing_formal_signal_path_trace")

    blocked = paper_ready_count == 0 and recomputable_count == 0
    if blocked:
        failures.append("missing_full_scale_uncertainty_or_recomputable_signal_rows")

    return {
        "status": "blocked_missing_signal_rows" if blocked else "signal_rows_available_for_next_gate",
        "paper_ready_uncertainty_source_count": paper_ready_count,
        "recomputable_signal_source_count": recomputable_count,
        "score_only_source_count": score_only_count,
        "exact_coverage_source_count": exact_coverage_count,
        "domain_audits": domain_rows,
        "formal_trace": {
            "path": str(trace_path),
            "present": trace_path.exists(),
            "can_rebuild_from_formal_scores_only": trace.get(
                "can_rebuild_paper_ready_uncertainty_rows_from_formal_scores_only"
            ),
            "blockers": trace.get("blockers", []),
        },
        "failures": failures,
    }


def audit_guarded_signal_plan(root: Path) -> dict[str, Any]:
    plan_json = root / PLAN_DIR / "ccrp_signal_generation_plan_20260604.json"
    plan_sh = root / PLAN_DIR / "ccrp_signal_generation_plan_20260604.sh"
    failures: list[str] = []
    payload: dict[str, Any] = {}
    if not plan_json.exists():
        failures.append("missing_guarded_plan_json")
    else:
        payload = _read_json(plan_json)
        if payload.get("will_start_experiment") is not False:
            failures.append("plan_may_start_experiment")
        if payload.get("status_label") != "planning_only_not_executed":
            failures.append("plan_status_not_planning_only")
    if not plan_sh.exists():
        failures.append("missing_guarded_plan_shell")
    else:
        shell = plan_sh.read_text(encoding="utf-8", errors="replace")
        if "exit 2" not in shell:
            failures.append("guarded_shell_missing_exit")
        elif "cd " in shell and shell.index("exit 2") > shell.index("cd "):
            failures.append("guarded_shell_exit_after_commands")
        if "TODO_TEST_" not in shell or "TODO_VALID_" not in shell:
            failures.append("guarded_shell_missing_signal_placeholders")
        if "nohup" in shell:
            failures.append("guarded_shell_contains_nohup")
    return {
        "status": "guarded_plan_ready_not_executable" if not failures else "incomplete",
        "paper_claim_ready": False,
        "files": {"json": str(plan_json), "shell": str(plan_sh)},
        "domains": payload.get("domains", []),
        "current_blocker": payload.get("current_blocker", ""),
        "failures": failures,
    }


def build_module_audit(root: str | Path = ".") -> dict[str, Any]:
    repo = Path(root).resolve()
    signal_state = audit_signal_source_state(repo)
    guarded_plan = audit_guarded_signal_plan(repo)
    framework = audit_framework_overview(repo)

    signal_blockers = list(signal_state["failures"])
    modules = {
        "observation_motivation": {
            "status": "blocked_missing_signal_rows"
            if signal_state["status"] == "blocked_missing_signal_rows"
            else "ready_for_representative_run_planning",
            "paper_claim_ready": False,
            "required_next_gate": "locate_or_regenerate_valid_test_uncertainty_signal_rows",
            "blockers": signal_blockers,
            "script": "scripts/analysis/main_build_uncertainty_observation_study.py",
        },
        "component_ablation": {
            "status": "blocked_missing_signal_rows"
            if signal_state["status"] == "blocked_missing_signal_rows"
            else "ready_for_validation_selection_run",
            "paper_claim_ready": False,
            "required_next_gate": "run_leave_one_component_out_after_signal_rows_exist",
            "blockers": signal_blockers,
            "script": "scripts/misc/main_select_ccrp_variant_on_valid.py",
        },
        "hyperparameter_analysis": {
            "status": "blocked_missing_signal_rows"
            if signal_state["status"] == "blocked_missing_signal_rows"
            else "ready_for_curve_generation",
            "paper_claim_ready": False,
            "required_next_gate": "build_validation_and_test_curves_after_signal_rows_exist",
            "blockers": signal_blockers,
            "script": "scripts/analysis/main_plot_ccrp_hyperparameter_sweep.py",
        },
        "framework_overview": framework,
    }
    paper_ready = all(module.get("paper_claim_ready") is True for module in modules.values())
    return {
        "ok": framework["artifact_scaffold_ready"] and guarded_plan["status"] == "guarded_plan_ready_not_executable",
        "paper_ready": paper_ready,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(repo),
        "summary": {
            "framework_overview_scaffold_ready": framework["artifact_scaffold_ready"],
            "signal_rows_available": signal_state["status"] != "blocked_missing_signal_rows",
            "guarded_plan_ready": guarded_plan["status"] == "guarded_plan_ready_not_executable",
            "paper_claims_ready": paper_ready,
        },
        "signal_source_state": signal_state,
        "guarded_signal_plan": guarded_plan,
        "modules": modules,
        "next_action": (
            "Keep the active Home LLM2Rec run protected. For paper-critical C-CRP modules, locate or regenerate "
            "full-scale valid/test uncertainty signal rows before running observation, ablation, or hyperparameter claims."
        ),
    }


def write_markdown(path: str | Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Paper-Critical Module Audit",
        "",
        f"- Generated UTC: `{audit['generated_at_utc']}`",
        f"- Paper ready: `{audit['paper_ready']}`",
        f"- Signal rows available: `{audit['summary']['signal_rows_available']}`",
        f"- Framework overview scaffold ready: `{audit['summary']['framework_overview_scaffold_ready']}`",
        f"- Guarded plan ready: `{audit['summary']['guarded_plan_ready']}`",
        "",
        "## Module Status",
        "",
    ]
    for name, module in audit["modules"].items():
        lines.append(f"- `{name}`: `{module['status']}`; paper claim ready = `{module.get('paper_claim_ready')}`")
        blockers = module.get("blockers") or module.get("remaining_blockers") or []
        if blockers:
            lines.append(f"  blockers: {', '.join(str(item) for item in blockers)}")
    lines.extend(["", "## Next Action", "", audit["next_action"], ""])
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit paper-critical module readiness without running experiments.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_md", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_module_audit(args.root)
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(args.output_md, audit)
    print(json.dumps({"ok": audit["ok"], "paper_ready": audit["paper_ready"], "summary": audit["summary"]}, indent=2))


if __name__ == "__main__":
    main()
