from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
repo_text = str(REPO_ROOT)
if repo_text not in sys.path:
    sys.path.insert(0, repo_text)

from src.shadow.ccrp import CCRP_ABLATIONS, CCRP_SCORE_MODES


DEFAULT_OUTPUT_DIR = Path("outputs/summary/paper_critical/ccrp_component_inventory")
SIGNAL_ROW_BLOCKER = "missing_full_scale_uncertainty_or_recomputable_signal_rows"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a paper-critical C-CRP component and ablation inventory. "
            "This is a read-only planning/audit artifact; it does not run an experiment."
        )
    )
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--inventory_id", default="ccrp_component_inventory_20260604")
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


def _line_ref(path: str, needle: str) -> dict[str, Any]:
    file_path = REPO_ROOT / path
    if not file_path.exists():
        return {"path": path, "line": None, "needle": needle, "present": False}
    for idx, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if needle in line:
            return {"path": path, "line": idx, "needle": needle, "present": True}
    return {"path": path, "line": None, "needle": needle, "present": False}


def _component(
    *,
    component_id: str,
    label: str,
    kind: str,
    role: str,
    selector_handle: str,
    source_refs: list[dict[str, Any]],
    required_signal_columns: list[str],
    generated_columns: list[str],
    validation_controls: dict[str, Any] | None = None,
    risk_notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": component_id,
        "label": label,
        "kind": kind,
        "role": role,
        "selector_handle": selector_handle,
        "implementation_status": "implemented_in_current_code",
        "paper_execution_status": "requires_full_scale_signal_rows",
        "current_blocker": SIGNAL_ROW_BLOCKER,
        "required_signal_columns": required_signal_columns,
        "generated_columns": generated_columns,
        "validation_controls": validation_controls or {},
        "source_refs": source_refs,
        "risk_notes": risk_notes or [],
    }


def build_inventory() -> dict[str, Any]:
    ablation_handles = sorted(CCRP_ABLATIONS)
    score_modes = sorted(CCRP_SCORE_MODES)
    components = [
        _component(
            component_id="calibrated_posterior_base",
            label="Calibrated relevance posterior",
            kind="base_signal",
            role="Provides the calibrated posterior used by full C-CRP and confidence-only rows.",
            selector_handle="score_mode=confidence_only or score_mode=full",
            source_refs=[
                _line_ref("src/shadow/ccrp.py", "calibrated_relevance_probability"),
                _line_ref("scripts/misc/main_select_ccrp_variant_on_valid.py", "--score_modes"),
            ],
            required_signal_columns=["relevance_probability", "calibrated_relevance_probability"],
            generated_columns=["ccrp_raw_probability", "ccrp_calibrated_probability", "ccrp_base_score"],
            validation_controls={"score_modes": score_modes},
            risk_notes=[
                "If calibrated_relevance_probability is absent, code falls back to calibrated_confidence or raw probability; provenance must report which source was used.",
            ],
        ),
        _component(
            component_id="score_mode_family",
            label="Score-mode family",
            kind="mode_comparison",
            role="Separates posterior-only, evidence-only, blended, and full C-CRP scoring behavior.",
            selector_handle="score_modes=confidence_only,evidence_only,confidence_plus_evidence,full",
            source_refs=[
                _line_ref("src/shadow/ccrp.py", "CCRP_SCORE_MODES"),
                _line_ref("scripts/misc/main_select_ccrp_variant_on_valid.py", "--score_modes"),
            ],
            required_signal_columns=["relevance_probability", "calibrated_relevance_probability", "evidence_support"],
            generated_columns=["ccrp_score_mode", "ccrp_base_score", "ccrp_risk_adjusted_score"],
            validation_controls={"score_modes": score_modes},
            risk_notes=[
                "confidence_only and evidence_only bypass the risk penalty in current code, so they are mode baselines rather than leave-one-risk-component ablations.",
            ],
        ),
        _component(
            component_id="boundary_uncertainty",
            label="Boundary ambiguity",
            kind="leave_one_component_out",
            role="Measures uncertainty near the calibrated posterior decision boundary.",
            selector_handle="ablation=without_boundary_uncertainty",
            source_refs=[
                _line_ref("src/shadow/ccrp.py", "boundary_uncertainty ="),
                _line_ref("src/shadow/ccrp.py", "without_boundary_uncertainty"),
            ],
            required_signal_columns=["calibrated_relevance_probability"],
            generated_columns=["ccrp_boundary_uncertainty", "ccrp_weight_boundary", "ccrp_uncertainty"],
            validation_controls={"ablation_handle": "without_boundary_uncertainty"},
            risk_notes=[
                "Current leave-one-out handling renormalizes the remaining uncertainty weights, so this is not an all-else-identical zeroing experiment.",
            ],
        ),
        _component(
            component_id="calibration_gap",
            label="Calibration gap",
            kind="leave_one_component_out",
            role="Penalizes disagreement between raw and calibrated relevance probability.",
            selector_handle="ablation=without_calibration_gap",
            source_refs=[
                _line_ref("src/shadow/ccrp.py", "calibration_gap ="),
                _line_ref("src/shadow/ccrp.py", "without_calibration_gap"),
            ],
            required_signal_columns=["relevance_probability", "calibrated_relevance_probability"],
            generated_columns=["ccrp_calibration_gap", "ccrp_weight_calibration_gap", "ccrp_uncertainty"],
            validation_controls={"ablation_handle": "without_calibration_gap"},
            risk_notes=[
                "This ablates the calibration-gap uncertainty term, not calibrated posterior use itself.",
                "Current leave-one-out handling renormalizes remaining uncertainty weights.",
            ],
        ),
        _component(
            component_id="evidence_support_insufficiency",
            label="Evidence support / insufficiency",
            kind="leave_one_component_out",
            role="Uses support minus counterevidence to model whether the posterior is backed by task-grounded evidence.",
            selector_handle="ablation=without_evidence_support",
            source_refs=[
                _line_ref("src/shadow/ccrp.py", "evidence_support ="),
                _line_ref("src/shadow/ccrp.py", "evidence_uncertainty ="),
                _line_ref("src/shadow/ccrp.py", "without_evidence_support"),
            ],
            required_signal_columns=["evidence_support"],
            generated_columns=["ccrp_evidence_score", "ccrp_evidence_uncertainty", "ccrp_weight_evidence"],
            validation_controls={"ablation_handle": "without_evidence_support"},
            risk_notes=[
                "If evidence_support is absent, code falls back to evidence or 0.0; a paper row must verify real support columns exist.",
                "In evidence_only and confidence_plus_evidence modes, evidence may still enter base_score; isolate evidence-insufficiency claims under a fixed full-mode protocol.",
            ],
        ),
        _component(
            component_id="counterevidence",
            label="Counterevidence",
            kind="leave_one_component_out",
            role="Subtracts contradictory evidence from support before evidence insufficiency is computed.",
            selector_handle="ablation=without_counterevidence",
            source_refs=[
                _line_ref("src/shadow/ccrp.py", "counterevidence ="),
                _line_ref("src/shadow/ccrp.py", "without_counterevidence"),
            ],
            required_signal_columns=["counterevidence_strength"],
            generated_columns=["ccrp_evidence_score", "ccrp_evidence_uncertainty", "ccrp_uncertainty"],
            validation_controls={"ablation_handle": "without_counterevidence"},
            risk_notes=[
                "The ablation is a no-op if source rows do not contain counterevidence_strength/counterevidence; audit source columns before claiming this component matters.",
            ],
        ),
        _component(
            component_id="risk_penalty",
            label="Risk penalty",
            kind="leave_one_component_out",
            role="Turns uncertainty into the final risk-adjusted ranking score.",
            selector_handle="ablation=without_risk_penalty",
            source_refs=[
                _line_ref("src/shadow/ccrp.py", "risk_adjusted_score = base_score *"),
                _line_ref("src/shadow/ccrp.py", "without_risk_penalty"),
            ],
            required_signal_columns=[
                "relevance_probability",
                "calibrated_relevance_probability",
                "evidence_support",
            ],
            generated_columns=["ccrp_uncertainty", "ccrp_base_score", "ccrp_risk_adjusted_score"],
            validation_controls={"ablation_handle": "without_risk_penalty"},
            risk_notes=[
                "Current implementation uses multiplicative risk adjustment: base_score * ((1 - uncertainty) ** eta). Paper figures/text must not describe it as a subtractive posterior - eta * uncertainty formula unless code changes.",
                "Eta has no ranking effect for confidence_only or evidence_only rows because those modes bypass the risk penalty.",
            ],
        ),
        _component(
            component_id="eta_risk_exponent",
            label="Eta risk exponent",
            kind="hyperparameter",
            role="Controls the strength of multiplicative uncertainty punishment.",
            selector_handle="--etas",
            source_refs=[
                _line_ref("src/shadow/ccrp.py", "max(0.0, float(eta))"),
                _line_ref("scripts/misc/main_select_ccrp_variant_on_valid.py", "--etas"),
            ],
            required_signal_columns=["ccrp_uncertainty_inputs"],
            generated_columns=["ccrp_risk_adjusted_score"],
            validation_controls={"default_grid": [0.5, 1.0, 2.0]},
            risk_notes=[
                "Eta is meaningful only when risk penalty is active; report mode/ablation filters for eta curves.",
            ],
        ),
        _component(
            component_id="confidence_weight",
            label="Confidence/evidence blend weight",
            kind="hyperparameter",
            role="Controls calibrated-posterior vs evidence-score blend for confidence_plus_evidence mode.",
            selector_handle="--confidence_weights",
            source_refs=[
                _line_ref("src/shadow/ccrp.py", "confidence_weight * calibrated_probability"),
                _line_ref("scripts/misc/main_select_ccrp_variant_on_valid.py", "--confidence_weights"),
            ],
            required_signal_columns=["calibrated_relevance_probability", "evidence_support"],
            generated_columns=["ccrp_confidence_weight", "ccrp_base_score"],
            validation_controls={"default_grid": [0.5, 0.7, 0.9]},
            risk_notes=[
                "Selector evaluates extra confidence_weight values only for confidence_plus_evidence mode.",
            ],
        ),
        _component(
            component_id="uncertainty_weight_triple",
            label="Uncertainty weight triple",
            kind="hyperparameter",
            role="Controls boundary, calibration-gap, and evidence-insufficiency contributions to total uncertainty.",
            selector_handle="--weight_grid",
            source_refs=[
                _line_ref("src/shadow/ccrp.py", "class CcrpWeights"),
                _line_ref("scripts/misc/main_select_ccrp_variant_on_valid.py", "--weight_grid"),
            ],
            required_signal_columns=[
                "calibrated_relevance_probability",
                "relevance_probability",
                "evidence_support",
            ],
            generated_columns=[
                "ccrp_weight_boundary",
                "ccrp_weight_calibration_gap",
                "ccrp_weight_evidence",
                "ccrp_uncertainty",
            ],
            validation_controls={
                "default": [0.5, 0.3, 0.2],
                "planned_grid": [[0.5, 0.3, 0.2], [0.7, 0.2, 0.1], [0.4, 0.4, 0.2], [0.4, 0.2, 0.4]],
            },
            risk_notes=[
                "Grid search must be validation-only; test-set selection would invalidate the component evidence.",
                "Removed-component ablations renormalize remaining weights, so interpret them as redistributed uncertainty mass.",
            ],
        ),
        _component(
            component_id="raw_vs_calibrated_posterior",
            label="Raw posterior versus calibrated posterior",
            kind="conceptual_not_currently_executable",
            role="Would test whether validation-only calibration itself is necessary for C-CRP.",
            selector_handle="no_current_cli_handle",
            source_refs=[
                _line_ref("src/shadow/ccrp.py", "raw_probability ="),
                _line_ref("src/shadow/ccrp.py", "calibrated_probability ="),
            ],
            required_signal_columns=["relevance_probability", "calibrated_relevance_probability"],
            generated_columns=["ccrp_raw_probability", "ccrp_calibrated_probability"],
            validation_controls={},
            risk_notes=[
                "No current selector handle replaces calibrated posterior with raw posterior throughout C-CRP; do not count this as an executed LOO ablation unless a handle is implemented and audited.",
            ],
        ),
        _component(
            component_id="temperature_prompt_variants",
            label="Temperature / prompt variants",
            kind="conceptual_or_separate_runner",
            role="Nearby diagnostic runners may vary generation settings, but they are not part of the current C-CRP uncertainty-decomposition selector.",
            selector_handle="not_part_of_main_select_ccrp_variant_on_valid",
            source_refs=[
                _line_ref("docs/paper_critical_experiment_plan_2026-06-03.md", "Temperature"),
                _line_ref("experiments/rsc/run_ccrp_v3_temperature.py", "temperature scaling"),
            ],
            required_signal_columns=["not_applicable_to_current_selector"],
            generated_columns=[],
            validation_controls={},
            risk_notes=[
                "Treat temperature/prompt studies as separate diagnostics unless they emit full-scale uncertainty signal rows and pass the same candidate/provenance gates.",
            ],
        ),
    ]

    formula_ref = _line_ref("src/shadow/ccrp.py", "risk_adjusted_score = base_score *")
    figure_ref = _line_ref("scripts/analysis/main_build_framework_overview_figure.py", "risk_score")
    figure_text = (REPO_ROOT / "scripts/analysis/main_build_framework_overview_figure.py").read_text(
        encoding="utf-8",
        errors="replace",
    )
    formula_alignment = {
        "code_formula": "base_score * ((1 - uncertainty) ** eta)",
        "figure_formula_contains_multiplicative_form": "(1 - uncertainty)" in figure_text and "base_score" in figure_text,
        "source_refs": [formula_ref, figure_ref],
    }

    overclaim_risks = [
        SIGNAL_ROW_BLOCKER,
        "Do not report component ablations from score-only formal C-CRP scores.",
        "Do not claim counterevidence contribution unless source rows contain a non-empty counterevidence column.",
        "Do not claim raw-vs-calibrated posterior or temperature/prompt variants as completed C-CRP LOO ablations without new audited handles.",
    ]
    if not formula_alignment["figure_formula_contains_multiplicative_form"]:
        overclaim_risks.append("framework_overview_formula_mismatch_with_current_ccrp_code")

    return {
        "status_label": "paper_critical_ccrp_component_inventory",
        "paper_claim_ready": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "component_count": len(components),
        "ablation_handles_from_code": ablation_handles,
        "score_modes_from_code": score_modes,
        "blocked_by": [SIGNAL_ROW_BLOCKER],
        "formula_alignment": formula_alignment,
        "components": components,
        "overclaim_risks": overclaim_risks,
        "next_gate": "locate_or_regenerate_full_scale_valid_test_signal_rows_then_run_validation_selector",
    }


def write_markdown(path: str | Path, inventory: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# C-CRP Component Inventory",
        "",
        f"- Status: `{inventory['status_label']}`",
        f"- Paper-claim ready: `{inventory['paper_claim_ready']}`",
        f"- Component count: `{inventory['component_count']}`",
        f"- Blocked by: `{', '.join(inventory['blocked_by'])}`",
        f"- Code risk formula: `{inventory['formula_alignment']['code_formula']}`",
        f"- Figure formula aligned: `{inventory['formula_alignment']['figure_formula_contains_multiplicative_form']}`",
        "",
        "## Components",
        "",
        "| Component | Kind | Selector handle | Execution status |",
        "| --- | --- | --- | --- |",
    ]
    for component in inventory["components"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{component['id']}`",
                    component["kind"],
                    f"`{component['selector_handle']}`",
                    component["paper_execution_status"],
                ]
            )
            + " |"
        )
    lines.extend(["", "## Overclaim Risks", ""])
    lines.extend(f"- {risk}" for risk in inventory["overclaim_risks"])
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory()
    json_path = output_dir / f"{args.inventory_id}.json"
    md_path = output_dir / f"{args.inventory_id}.md"
    json_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(md_path, inventory)
    print(json.dumps({"ok": True, "json": str(json_path), "markdown": str(md_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
