from __future__ import annotations

import argparse
import struct
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PAPER_CRITICAL_DIR = Path("outputs/summary/paper_critical")
FRAMEWORK_DIR = PAPER_CRITICAL_DIR / "framework_overview"
PLAN_DIR = PAPER_CRITICAL_DIR / "ccrp_signal_generation_plan"
COMPONENT_INVENTORY_DIR = PAPER_CRITICAL_DIR / "ccrp_component_inventory"
OBSERVATION_BUILDER_SCRIPT = Path("scripts/analysis/main_build_uncertainty_observation_study.py")
SELECTOR_SCRIPT = Path("scripts/misc/main_select_ccrp_variant_on_valid.py")
COMPONENT_BUILDER_SCRIPT = Path("scripts/analysis/main_build_ccrp_component_ablation_summary.py")
HYPERPARAMETER_PLOTTER_SCRIPT = Path("scripts/analysis/main_plot_ccrp_hyperparameter_sweep.py")
MODULE_PACKAGE_AUDIT_SCRIPT = Path("scripts/audit/main_audit_phase2_5_module_package.py")
DOMAINS = ("sports", "toys", "home", "tools")
FRAMEWORK_REVIEW_READY_LABEL = "paper_critical_framework_overview_review_ready"
FRAMEWORK_REQUIRED_LABELS = (
    "Same-candidate task",
    "LLM signal extraction",
    "Calibration layer",
    "C-CRP uncertainty",
    "Risk-adjusted ranking",
    "Official baseline block",
    "Required method-evidence gates",
    "Shared evidence gates",
    "risk_score = base_score",
    "* (1 - uncertainty)^eta",
)
REQUIRED_FRAMEWORK_EVIDENCE_GATE_STATUS = {
    "observation_motivation": "required_not_claimed_by_figure",
    "component_ablation": "required_not_claimed_by_figure",
    "hyperparameter_analysis": "required_not_claimed_by_figure",
}
FRAMEWORK_REQUIRED_CAPTION_PHRASES = (
    "required gates before a paper-ready claim",
)
FRAMEWORK_REQUIRED_SVG_PHRASES = (
    "before paper-ready claim",
)
FRAMEWORK_REQUIRED_CLAIM_LIMITS = (
    "Does not make observation, ablation, or hyperparameter evidence complete.",
)
FRAMEWORK_FORBIDDEN_OVERCLAIM_PHRASES = (
    "observation complete",
    "motivation complete",
    "ablation complete",
    "ablations complete",
    "component ablation complete",
    "component ablations complete",
    "hyperparameter complete",
    "hyperparameter analysis complete",
    "modules complete",
    "completed modules",
)
DEFAULT_EVIDENCE_CONSISTENCY_GLOBS = (
    "local_server_evidence_consistency_new_domains_post_backfill_*.json",
    "local_server_evidence_consistency_new_domains_*.json",
)
DEFAULT_STORAGE_AUDIT_GLOBS = (
    "server_storage_phase2_5_retention_audit_current_*.json",
    "server_storage_phase2_5_retention_audit_ranked_*.json",
    "server_storage_phase2_5_retention_audit_*.json",
)
REQUIRED_GUARDED_PLAN_COMMAND_KEYS = (
    "select_ccrp_ablation_and_scores_template",
    "build_component_ablation_summary_template",
    "audit_component_ablation_package_template",
    "build_observation_study_template",
    "audit_observation_package_template",
    "plot_hyperparameter_curves_template",
    "audit_hyperparameter_package_template",
)


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


def _png_dimensions(path: Path) -> tuple[int | None, int | None]:
    if not path.exists() or path.stat().st_size < 24:
        return None, None
    with path.open("rb") as fh:
        header = fh.read(24)
    if not header.startswith(b"\x89PNG\r\n\x1a\n"):
        return None, None
    return struct.unpack(">II", header[16:24])


def _latest_matching_file(base: Path, patterns: tuple[str, ...]) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in base.glob(pattern) if path.is_file())
        if candidates:
            break
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


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

    caption_text = (
        files["framework_overview_caption.md"].read_text(encoding="utf-8", errors="replace")
        if files["framework_overview_caption.md"].exists()
        else ""
    )
    caption_lower = caption_text.lower()
    for phrase in FRAMEWORK_REQUIRED_CAPTION_PHRASES:
        if phrase not in caption_lower:
            failures.append(f"framework_caption_missing_required_phrase:{phrase}")
    for phrase in FRAMEWORK_FORBIDDEN_OVERCLAIM_PHRASES:
        if phrase in caption_lower:
            failures.append(f"framework_caption_overclaim_phrase:{phrase}")

    provenance: dict[str, Any] = {}
    if files["framework_overview_provenance.json"].exists():
        provenance = _read_json(files["framework_overview_provenance.json"])
        if provenance.get("status_label") != FRAMEWORK_REVIEW_READY_LABEL:
            failures.append("framework_status_label_not_review_ready")
        if provenance.get("claim_boundary") != "controlled_same_candidate_ranking_not_full_catalog":
            failures.append("framework_claim_boundary_mismatch")
        if provenance.get("paper_claim_ready") is not True:
            failures.append("framework_paper_claim_ready_not_true")
        if "not_substitute_for_observation_ablation_or_hyperparameter" not in provenance.get("module_scope", ""):
            failures.append("framework_scope_not_limited_to_figure")
        if provenance.get("formula_alignment", {}).get("matches_src_shadow_ccrp_multiplicative_form") is not True:
            failures.append("framework_formula_alignment_missing")
        evidence_gate_status = provenance.get("evidence_gate_status") or {}
        for module, expected_status in REQUIRED_FRAMEWORK_EVIDENCE_GATE_STATUS.items():
            if evidence_gate_status.get(module) != expected_status:
                failures.append(f"framework_evidence_gate_status_mismatch:{module}")
        if provenance.get("caption", "").strip() != caption_text.strip():
            failures.append("framework_caption_file_provenance_mismatch")
        claim_limits = set(provenance.get("claim_limits") or [])
        for required_limit in FRAMEWORK_REQUIRED_CLAIM_LIMITS:
            if required_limit not in claim_limits:
                failures.append("framework_claim_limit_missing:no_module_completion_claim")

    svg_text = files["framework_overview.svg"].read_text(encoding="utf-8", errors="replace") if files["framework_overview.svg"].exists() else ""
    svg_lower = svg_text.lower()
    missing_labels = [label for label in FRAMEWORK_REQUIRED_LABELS if label not in svg_text]
    if missing_labels:
        failures.append("framework_missing_required_labels:" + ",".join(missing_labels))
    for phrase in FRAMEWORK_REQUIRED_SVG_PHRASES:
        if phrase not in svg_lower:
            failures.append(f"framework_svg_missing_required_phrase:{phrase}")
    for phrase in FRAMEWORK_FORBIDDEN_OVERCLAIM_PHRASES:
        if phrase in svg_lower:
            failures.append(f"framework_svg_overclaim_phrase:{phrase}")
    png_width, png_height = _png_dimensions(files["framework_overview.png"])
    if not png_width or not png_height:
        failures.append("framework_png_not_valid")
    elif png_width < 1800 or png_height < 900:
        failures.append(f"framework_png_too_small:{png_width}x{png_height}")

    return {
        "module": "framework_overview",
        "status": "review_ready" if not failures else "incomplete",
        "artifact_scaffold_ready": not failures,
        "paper_claim_ready": not failures,
        "remaining_blockers": [] if not failures else failures,
        "files": {name: str(path) for name, path in files.items()},
        "manifest_checks": manifest_checks,
        "required_label_checks": {
            "required_labels": list(FRAMEWORK_REQUIRED_LABELS),
            "missing_labels": missing_labels,
            "ok": not missing_labels,
        },
        "png_dimensions": {"width": png_width, "height": png_height},
        "provenance": {
            "status_label": provenance.get("status_label", ""),
            "paper_claim_ready": provenance.get("paper_claim_ready"),
            "review_status": provenance.get("review_status", ""),
            "module_scope": provenance.get("module_scope", ""),
            "evidence_gate_status": provenance.get("evidence_gate_status", {}),
            "claim_limits": provenance.get("claim_limits", []),
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
        for domain_plan in payload.get("domain_plans", []) or []:
            domain = str(domain_plan.get("domain", "unknown"))
            commands = domain_plan.get("commands") or {}
            for key in REQUIRED_GUARDED_PLAN_COMMAND_KEYS:
                if not str(commands.get(key, "")).strip():
                    failures.append(f"guarded_plan_missing_command:{domain}:{key}")
            component_cmd = str(commands.get("build_component_ablation_summary_template", ""))
            if component_cmd and "main_build_ccrp_component_ablation_summary.py" not in component_cmd:
                failures.append(f"guarded_plan_component_builder_command_mismatch:{domain}")
            package_cmd = str(commands.get("audit_component_ablation_package_template", ""))
            if package_cmd and "main_audit_phase2_5_module_package.py" not in package_cmd:
                failures.append(f"guarded_plan_component_package_audit_command_mismatch:{domain}")
            observation_cmd = str(commands.get("build_observation_study_template", ""))
            if observation_cmd and "main_build_uncertainty_observation_study.py" not in observation_cmd:
                failures.append(f"guarded_plan_observation_command_mismatch:{domain}")
            hyperparameter_cmd = str(commands.get("plot_hyperparameter_curves_template", ""))
            if hyperparameter_cmd and "main_plot_ccrp_hyperparameter_sweep.py" not in hyperparameter_cmd:
                failures.append(f"guarded_plan_hyperparameter_command_mismatch:{domain}")
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
        if "main_build_uncertainty_observation_study.py" not in shell:
            failures.append("guarded_shell_missing_observation_builder")
        if "main_build_ccrp_component_ablation_summary.py" not in shell:
            failures.append("guarded_shell_missing_component_builder")
        if "main_plot_ccrp_hyperparameter_sweep.py" not in shell:
            failures.append("guarded_shell_missing_hyperparameter_plotter")
        if "main_audit_phase2_5_module_package.py" not in shell:
            failures.append("guarded_shell_missing_module_package_audit")
    return {
        "status": "guarded_plan_ready_not_executable" if not failures else "incomplete",
        "paper_claim_ready": False,
        "files": {"json": str(plan_json), "shell": str(plan_sh)},
        "domains": payload.get("domains", []),
        "current_blocker": payload.get("current_blocker", ""),
        "required_command_keys": list(REQUIRED_GUARDED_PLAN_COMMAND_KEYS),
        "failures": failures,
    }


def audit_observation_execution_support(root: Path) -> dict[str, Any]:
    files = {
        "builder": root / OBSERVATION_BUILDER_SCRIPT,
        "package_audit": root / MODULE_PACKAGE_AUDIT_SCRIPT,
    }
    failures: list[str] = []
    for label, path in files.items():
        if not path.exists():
            failures.append(f"missing_{label}_script:{path}")
        elif path.stat().st_size <= 0:
            failures.append(f"empty_{label}_script:{path}")

    builder_text = files["builder"].read_text(encoding="utf-8", errors="replace") if files["builder"].exists() else ""
    required_builder_snippets = {
        "full_metrics": "DEFAULT_KS = (5, 10, 20)",
        "artifact_class": "paper_critical_observation_motivation",
        "status_label": "paper_critical_observation_ready",
        "claim_scope": "motivation_only_not_main_table_sota",
        "join_report": "join_report",
        "candidate_count_guard": "expected_candidates_per_event",
        "join_rate_guard": "min_join_rate",
        "uncertainty_column_guard": "No uncertainty column found",
        "summary_csv": "observation_summary.csv",
        "provenance_json": "observation_provenance.json",
        "figure_outputs": "fig_uncertainty_motivation",
    }
    for name, snippet in required_builder_snippets.items():
        if snippet not in builder_text:
            failures.append(f"observation_builder_missing_guard:{name}")

    package_text = files["package_audit"].read_text(encoding="utf-8", errors="replace") if files["package_audit"].exists() else ""
    for snippet in ("observation_summary.csv", "observation_event_bins.csv", "observation_provenance.json"):
        if snippet not in package_text:
            failures.append(f"package_audit_missing_observation_requirement:{snippet}")

    return {
        "status": "observation_execution_support_ready" if not failures else "incomplete",
        "paper_claim_ready": False,
        "files": {label: str(path) for label, path in files.items()},
        "required_builder_checks": sorted(required_builder_snippets),
        "failures": failures,
    }


def audit_component_ablation_execution_support(root: Path) -> dict[str, Any]:
    files = {
        "selector": root / SELECTOR_SCRIPT,
        "builder": root / COMPONENT_BUILDER_SCRIPT,
        "package_audit": root / MODULE_PACKAGE_AUDIT_SCRIPT,
    }
    failures: list[str] = []
    for label, path in files.items():
        if not path.exists():
            failures.append(f"missing_{label}_script:{path}")
        elif path.stat().st_size <= 0:
            failures.append(f"empty_{label}_script:{path}")

    selector_text = files["selector"].read_text(encoding="utf-8", errors="replace") if files["selector"].exists() else ""
    if "FULL_REPORTING_KS = (5, 10, 20)" not in selector_text:
        failures.append("selector_missing_full_reporting_ks")
    if "ks=FULL_REPORTING_KS" not in selector_text:
        failures.append("selector_not_passing_full_reporting_ks")

    builder_text = files["builder"].read_text(encoding="utf-8", errors="replace") if files["builder"].exists() else ""
    required_builder_snippets = {
        "full_metrics": "FULL_KS = (5, 10, 20)",
        "summary_csv": "component_ablation_summary.csv",
        "provenance_json": "component_ablation_provenance.json",
        "valid_selection_guard": "selector_provenance_selected_on_not_valid",
        "full_score_mode_guard": "selected_score_mode_not_full_for_component_ablation",
        "valid_sweep_ablation_guard": "valid_sweep_missing_ablation",
        "score_evaluator_reuse": "_evaluate_candidate_scores(",
    }
    for name, snippet in required_builder_snippets.items():
        if snippet not in builder_text:
            failures.append(f"component_builder_missing_guard:{name}")

    package_text = files["package_audit"].read_text(encoding="utf-8", errors="replace") if files["package_audit"].exists() else ""
    for snippet in ("component_ablation_summary.csv", "valid_ccrp_sweep.csv", "selected_test_metrics.csv"):
        if snippet not in package_text:
            failures.append(f"package_audit_missing_component_requirement:{snippet}")

    return {
        "status": "component_ablation_execution_support_ready" if not failures else "incomplete",
        "paper_claim_ready": False,
        "files": {label: str(path) for label, path in files.items()},
        "required_builder_checks": sorted(required_builder_snippets),
        "failures": failures,
    }


def audit_hyperparameter_execution_support(root: Path) -> dict[str, Any]:
    files = {
        "plotter": root / HYPERPARAMETER_PLOTTER_SCRIPT,
        "package_audit": root / MODULE_PACKAGE_AUDIT_SCRIPT,
    }
    failures: list[str] = []
    for label, path in files.items():
        if not path.exists():
            failures.append(f"missing_{label}_script:{path}")
        elif path.stat().st_size <= 0:
            failures.append(f"empty_{label}_script:{path}")

    plotter_text = files["plotter"].read_text(encoding="utf-8", errors="replace") if files["plotter"].exists() else ""
    required_plotter_snippets = {
        "test_sweep_arg": "--test_sweep_csv",
        "audit_requirement": "--require_audit_ok",
        "ready_status": "paper_critical_hyperparameter_curve_ready",
        "claim_scope": "valid_and_test_stability_curve_candidate",
        "test_sweep_hash": "test_sweep_sha256",
        "audit_summary": "audit_summary",
        "summary_csv": "ccrp_hyperparameter_curve_summary.csv",
        "provenance_json": "ccrp_hyperparameter_curve_provenance.json",
        "figure_outputs": "fig_hyper_eta_curve",
        "default_controls": "eta,confidence_weight,weight_grid_label",
    }
    for name, snippet in required_plotter_snippets.items():
        if snippet not in plotter_text:
            failures.append(f"hyperparameter_plotter_missing_guard:{name}")

    package_text = files["package_audit"].read_text(encoding="utf-8", errors="replace") if files["package_audit"].exists() else ""
    for snippet in ("ccrp_hyperparameter_curve_summary.csv", "ccrp_hyperparameter_curve_provenance.json", "test_sweep_sha256"):
        if snippet not in package_text:
            failures.append(f"package_audit_missing_hyperparameter_requirement:{snippet}")

    return {
        "status": "hyperparameter_execution_support_ready" if not failures else "incomplete",
        "paper_claim_ready": False,
        "files": {label: str(path) for label, path in files.items()},
        "required_plotter_checks": sorted(required_plotter_snippets),
        "failures": failures,
    }


def audit_component_inventory(root: Path) -> dict[str, Any]:
    inventory_json = root / COMPONENT_INVENTORY_DIR / "ccrp_component_inventory_20260604.json"
    inventory_md = root / COMPONENT_INVENTORY_DIR / "ccrp_component_inventory_20260604.md"
    failures: list[str] = []
    payload: dict[str, Any] = {}
    if not inventory_json.exists():
        failures.append("missing_component_inventory_json")
    else:
        payload = _read_json(inventory_json)
        if payload.get("status_label") != "paper_critical_ccrp_component_inventory":
            failures.append("component_inventory_status_label_mismatch")
        if payload.get("paper_claim_ready") is not False:
            failures.append("component_inventory_should_not_mark_paper_ready")
        if payload.get("component_count", 0) < 10:
            failures.append("component_inventory_too_small")
        if payload.get("formula_alignment", {}).get("figure_formula_contains_multiplicative_form") is not True:
            failures.append("component_inventory_formula_alignment_failed")
        ids = {component.get("id") for component in payload.get("components", [])}
        required_ids = {
            "boundary_uncertainty",
            "calibration_gap",
            "evidence_support_insufficiency",
            "counterevidence",
            "risk_penalty",
            "eta_risk_exponent",
            "confidence_weight",
            "uncertainty_weight_triple",
            "raw_vs_calibrated_posterior",
            "temperature_prompt_variants",
        }
        missing = sorted(required_ids - ids)
        if missing:
            failures.append("component_inventory_missing:" + ",".join(missing))
    if not inventory_md.exists():
        failures.append("missing_component_inventory_markdown")
    elif inventory_md.stat().st_size <= 0:
        failures.append("empty_component_inventory_markdown")
    return {
        "status": "inventory_ready_not_executed" if not failures else "incomplete",
        "paper_claim_ready": False,
        "files": {"json": str(inventory_json), "markdown": str(inventory_md)},
        "component_count": payload.get("component_count", 0),
        "blocked_by": payload.get("blocked_by", []),
        "formula_alignment": payload.get("formula_alignment", {}),
        "failures": failures,
    }


def audit_evidence_consistency(root: Path, artifact_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(artifact_path) if artifact_path else _latest_matching_file(root / PAPER_CRITICAL_DIR, DEFAULT_EVIDENCE_CONSISTENCY_GLOBS)
    if path and not path.is_absolute():
        path = root / path
    if not path or not path.exists():
        return {
            "status": "missing_evidence_consistency_audit",
            "paper_claim_ready": False,
            "path": str(path) if path else "",
            "ok": False,
            "row_count": 0,
            "ok_count": 0,
            "failure_count": 0,
            "failures": ["missing_local_server_evidence_consistency_audit"],
        }
    payload = _read_json(path)
    failures = list(payload.get("failures") or [])
    if payload.get("ok") is not True:
        failures.append("local_server_evidence_consistency_not_ok")
    if int(payload.get("row_count") or 0) != 32:
        failures.append("local_server_evidence_row_count_not_32")
    if int(payload.get("ok_count") or 0) != 32:
        failures.append("local_server_evidence_ok_count_not_32")
    if int(payload.get("failure_count") or 0) != 0:
        failures.append("local_server_evidence_failure_count_nonzero")
    return {
        "status": "four_domain_evidence_consistent" if not failures else "evidence_consistency_incomplete",
        "paper_claim_ready": not failures,
        "path": str(path),
        "ok": payload.get("ok") is True and not failures,
        "row_count": int(payload.get("row_count") or 0),
        "ok_count": int(payload.get("ok_count") or 0),
        "failure_count": int(payload.get("failure_count") or 0),
        "failures": failures,
    }


def audit_storage_gate(root: Path, artifact_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(artifact_path) if artifact_path else _latest_matching_file(root / PAPER_CRITICAL_DIR, DEFAULT_STORAGE_AUDIT_GLOBS)
    if path and not path.is_absolute():
        path = root / path
    if not path or not path.exists():
        return {
            "status": "missing_storage_gate_audit",
            "path": str(path) if path else "",
            "experiment_launch_allowed": False,
            "current_free_bytes": 0,
            "required_free_bytes_min": 0,
            "deficit_to_min_free_bytes": 0,
            "used_pct": None,
            "safe_now_total_recoverable_bytes": 0,
            "recommended_approval_candidate": None,
            "failures": ["missing_phase2_5_storage_audit"],
        }
    payload = _read_json(path)
    gate = payload.get("phase2_5_disk_gate") or {}
    server = payload.get("server") or {}
    disk = server.get("disk") or {}
    active_processes = server.get("relevant_python_processes") or []
    current_free = int(gate.get("current_free_bytes") or disk.get("free_bytes") or 0)
    required_free = int(gate.get("required_free_bytes_min") or 0)
    deficit = int(gate.get("deficit_to_min_free_bytes") or max(required_free - current_free, 0))
    launch_allowed = gate.get("experiment_launch_allowed") is True
    failures: list[str] = []
    if active_processes:
        failures.append("active_project_python_processes_present")
    if current_free and required_free and current_free < required_free:
        failures.append("server_disk_below_phase2_5_floor")
    if disk.get("used_pct") is not None and int(disk["used_pct"]) >= 97:
        failures.append("server_disk_used_pct_at_or_above_97")
    if not launch_allowed:
        failures.append("phase2_5_experiment_launch_not_allowed")
    return {
        "status": "phase2_5_storage_launch_allowed" if launch_allowed else "blocked_phase2_5_storage_gate",
        "path": str(path),
        "experiment_launch_allowed": launch_allowed,
        "current_free_bytes": current_free,
        "required_free_bytes_min": required_free,
        "deficit_to_min_free_bytes": deficit,
        "used_pct": disk.get("used_pct"),
        "safe_now_total_recoverable_bytes": int(payload.get("safe_now_total_recoverable_bytes") or 0),
        "recommended_approval_candidate": payload.get("recommended_approval_candidate"),
        "failures": failures,
    }


def _storage_action_summary(storage_gate: dict[str, Any]) -> dict[str, Any]:
    candidate = storage_gate.get("recommended_approval_candidate")
    if not isinstance(candidate, dict):
        candidate = {}
    approval_required = candidate.get("approval_decision_required") is True
    would_clear = candidate.get("would_clear_min_free_gate") is True
    safe_now_bytes = int(storage_gate.get("safe_now_total_recoverable_bytes") or 0)
    return {
        "storage_safe_now_total_recoverable_bytes": safe_now_bytes,
        "storage_recommended_candidate_path": str(candidate.get("path") or ""),
        "storage_recommended_candidate_size_bytes": int(candidate.get("size_bytes") or 0),
        "storage_recommended_candidate_would_clear_min_free_gate": would_clear,
        "storage_approval_decision_required": approval_required,
        "storage_cleanup_decision_required": bool(
            not storage_gate.get("experiment_launch_allowed")
            and safe_now_bytes <= 0
            and approval_required
            and would_clear
        ),
    }


def _next_action(signal_state: dict[str, Any], storage_gate: dict[str, Any], storage_action: dict[str, Any]) -> str:
    if signal_state["status"] == "blocked_missing_signal_rows" and storage_action["storage_cleanup_decision_required"]:
        return (
            "Full-scale valid/test uncertainty signal rows remain missing and the Phase 2.5 storage gate is closed. "
            "The current storage audit found no safe-now recoverable bytes; the approval-required candidate "
            f"{storage_action['storage_recommended_candidate_path']} would clear the minimum disk gate. "
            "Record an explicit archive/retention decision before any delete command, then rerun the storage gate "
            "and only then launch guarded signal-row generation."
        )
    return (
        "With the official-baseline evidence package consistent, do not start paper-critical C-CRP modules until "
        "full-scale valid/test uncertainty signal rows are located or regenerated under the same-candidate protocol "
        "and the Phase 2.5 storage gate allows launch. Then run observation, ablation, and hyperparameter gates "
        "with validation-only selection and exact score-coverage audits."
    )


def build_module_audit(
    root: str | Path = ".",
    *,
    evidence_consistency_json: str | Path | None = None,
    storage_audit_json: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    signal_state = audit_signal_source_state(repo)
    guarded_plan = audit_guarded_signal_plan(repo)
    framework = audit_framework_overview(repo)
    component_inventory = audit_component_inventory(repo)
    observation_execution_support = audit_observation_execution_support(repo)
    component_execution_support = audit_component_ablation_execution_support(repo)
    hyperparameter_execution_support = audit_hyperparameter_execution_support(repo)
    evidence_consistency = audit_evidence_consistency(repo, evidence_consistency_json)
    storage_gate = audit_storage_gate(repo, storage_audit_json)
    storage_action = _storage_action_summary(storage_gate)

    signal_blockers = list(signal_state["failures"])
    launch_blockers = signal_blockers + storage_gate["failures"]
    modules = {
        "observation_motivation": {
            "status": "blocked_missing_signal_rows"
            if signal_state["status"] == "blocked_missing_signal_rows"
            else "ready_for_representative_run_planning",
            "paper_claim_ready": False,
            "required_next_gate": "locate_or_regenerate_valid_test_uncertainty_signal_rows",
            "blockers": launch_blockers + observation_execution_support["failures"],
            "script": "scripts/analysis/main_build_uncertainty_observation_study.py",
            "execution_support": observation_execution_support,
        },
        "component_ablation": {
            "status": "blocked_missing_signal_rows"
            if signal_state["status"] == "blocked_missing_signal_rows"
            else "ready_for_validation_selection_run",
            "paper_claim_ready": False,
            "required_next_gate": "run_leave_one_component_out_after_signal_rows_exist",
            "blockers": launch_blockers + component_inventory["failures"] + component_execution_support["failures"],
            "script": "scripts/analysis/main_build_ccrp_component_ablation_summary.py",
            "selector_script": "scripts/misc/main_select_ccrp_variant_on_valid.py",
            "inventory": component_inventory,
            "execution_support": component_execution_support,
        },
        "hyperparameter_analysis": {
            "status": "blocked_missing_signal_rows"
            if signal_state["status"] == "blocked_missing_signal_rows"
            else "ready_for_curve_generation",
            "paper_claim_ready": False,
            "required_next_gate": "build_validation_and_test_curves_after_signal_rows_exist",
            "blockers": launch_blockers + hyperparameter_execution_support["failures"],
            "script": "scripts/analysis/main_plot_ccrp_hyperparameter_sweep.py",
            "execution_support": hyperparameter_execution_support,
        },
        "framework_overview": framework,
    }
    paper_ready = all(module.get("paper_claim_ready") is True for module in modules.values())
    return {
        "ok": (
            framework["artifact_scaffold_ready"]
            and guarded_plan["status"] == "guarded_plan_ready_not_executable"
            and component_inventory["status"] == "inventory_ready_not_executed"
            and observation_execution_support["status"] == "observation_execution_support_ready"
            and component_execution_support["status"] == "component_ablation_execution_support_ready"
            and hyperparameter_execution_support["status"] == "hyperparameter_execution_support_ready"
        ),
        "paper_ready": paper_ready,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(repo),
        "summary": {
            "framework_overview_scaffold_ready": framework["artifact_scaffold_ready"],
            "component_inventory_ready": component_inventory["status"] == "inventory_ready_not_executed",
            "observation_execution_support_ready": (
                observation_execution_support["status"] == "observation_execution_support_ready"
            ),
            "component_ablation_execution_support_ready": (
                component_execution_support["status"] == "component_ablation_execution_support_ready"
            ),
            "hyperparameter_execution_support_ready": (
                hyperparameter_execution_support["status"] == "hyperparameter_execution_support_ready"
            ),
            "signal_rows_available": signal_state["status"] != "blocked_missing_signal_rows",
            "guarded_plan_ready": guarded_plan["status"] == "guarded_plan_ready_not_executable",
            "four_domain_evidence_consistent": evidence_consistency["status"] == "four_domain_evidence_consistent",
            "phase2_5_storage_launch_allowed": storage_gate["experiment_launch_allowed"],
            "paper_claims_ready": paper_ready,
            **storage_action,
        },
        "signal_source_state": signal_state,
        "guarded_signal_plan": guarded_plan,
        "component_inventory": component_inventory,
        "observation_execution_support": observation_execution_support,
        "component_ablation_execution_support": component_execution_support,
        "hyperparameter_execution_support": hyperparameter_execution_support,
        "evidence_consistency": evidence_consistency,
        "storage_gate": storage_gate,
        "storage_action": storage_action,
        "modules": modules,
        "next_action": _next_action(signal_state, storage_gate, storage_action),
    }


def write_markdown(path: str | Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Paper-Critical Module Audit",
        "",
        f"- Generated UTC: `{audit['generated_at_utc']}`",
        f"- Paper ready: `{audit['paper_ready']}`",
        f"- Signal rows available: `{audit['summary']['signal_rows_available']}`",
        f"- Framework overview scaffold ready: `{audit['summary']['framework_overview_scaffold_ready']}`",
        f"- Component inventory ready: `{audit['summary']['component_inventory_ready']}`",
        f"- Observation execution support ready: `{audit['summary']['observation_execution_support_ready']}`",
        f"- Component-ablation execution support ready: `{audit['summary']['component_ablation_execution_support_ready']}`",
        f"- Hyperparameter execution support ready: `{audit['summary']['hyperparameter_execution_support_ready']}`",
        f"- Guarded plan ready: `{audit['summary']['guarded_plan_ready']}`",
        f"- Four-domain evidence consistent: `{audit['summary']['four_domain_evidence_consistent']}`",
        f"- Phase 2.5 storage launch allowed: `{audit['summary']['phase2_5_storage_launch_allowed']}`",
        f"- Storage free bytes: `{audit['storage_gate']['current_free_bytes']}`",
        f"- Storage deficit bytes: `{audit['storage_gate']['deficit_to_min_free_bytes']}`",
        f"- Storage safe-now recoverable bytes: `{audit['summary']['storage_safe_now_total_recoverable_bytes']}`",
        f"- Storage approval decision required: `{audit['summary']['storage_approval_decision_required']}`",
        f"- Storage cleanup decision required: `{audit['summary']['storage_cleanup_decision_required']}`",
        f"- Storage recommended candidate: `{audit['summary']['storage_recommended_candidate_path']}`",
        f"- Candidate would clear minimum gate: `{audit['summary']['storage_recommended_candidate_would_clear_min_free_gate']}`",
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
    parser.add_argument("--evidence_consistency_json", default="")
    parser.add_argument("--storage_audit_json", default="")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_md", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_module_audit(
        args.root,
        evidence_consistency_json=args.evidence_consistency_json or None,
        storage_audit_json=args.storage_audit_json or None,
    )
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(args.output_md, audit)
    print(json.dumps({"ok": audit["ok"], "paper_ready": audit["paper_ready"], "summary": audit["summary"]}, indent=2))


if __name__ == "__main__":
    main()
