from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PACKAGE_ROOT = Path(
    "outputs/summary/paper_critical/"
    "ccrp_signal_generation_plan_post_performance_gate_20260606"
)
DEFAULT_CERTIFICATE_JSON = Path(
    "outputs/summary/paper_critical/"
    "cross_domain_official_ccrp_certificate_audit_post_ccrp_backfill_20260606_0255.json"
)
DEFAULT_LEDGER_JSON = Path(
    "outputs/summary/paper_critical/"
    "new_domains_paper_facing_full_metric_evidence_ledger_post_ccrp_backfill_20260606_0255.json"
)
DEFAULT_OBSERVATION_JSON = DEFAULT_PACKAGE_ROOT / "observation_four_domain" / "observation_four_domain_provenance.json"
DEFAULT_COMPONENT_JSON = (
    DEFAULT_PACKAGE_ROOT
    / "ccrp_component_ablation_four_domain"
    / "component_ablation_four_domain_provenance.json"
)
DEFAULT_COMPONENT_SUMMARY_CSV = (
    DEFAULT_PACKAGE_ROOT
    / "ccrp_component_ablation_four_domain"
    / "component_ablation_four_domain_component_summary.csv"
)
DEFAULT_HYPERPARAMETER_JSON = (
    DEFAULT_PACKAGE_ROOT
    / "ccrp_hyperparameter_four_domain"
    / "ccrp_hyperparameter_four_domain_provenance.json"
)
DEFAULT_HYPERPARAMETER_STABILITY_CSV = (
    DEFAULT_PACKAGE_ROOT
    / "ccrp_hyperparameter_four_domain"
    / "ccrp_hyperparameter_four_domain_stability_rows.csv"
)
DEFAULT_FRAMEWORK_JSON = Path(
    "outputs/summary/paper_critical/framework_overview/framework_overview_provenance.json"
)

EXPECTED_DOMAINS = ("sports", "toys", "home", "tools")
EXPECTED_METRICS = ("HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR")
EXPECTED_OFFICIAL_ROWS = 32
EXPECTED_METHOD_ROWS = 36


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _path_state(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def _discover_manuscript_files(root: Path) -> dict[str, Any]:
    search_roots = [
        root,
        root / "paper",
        root / "papers",
        root / "manuscript",
        root / "latex",
        root / "docs",
    ]
    tex_files: list[str] = []
    bib_files: list[str] = []
    for base in search_roots:
        if not base.exists() or not base.is_dir():
            continue
        for pattern, target in (("*.tex", tex_files), ("*.bib", bib_files)):
            for path in base.glob(pattern):
                if path.is_file():
                    target.append(str(path.relative_to(root)))
    return {
        "tex_files": sorted(set(tex_files)),
        "bib_files": sorted(set(bib_files)),
        "has_manuscript": bool(tex_files),
        "has_bibliography": bool(bib_files),
    }


def _status(flag: bool, *, supported: str = "SUPPORTED", failed: str = "UNSUPPORTED") -> str:
    return supported if flag else failed


def _claim(
    claim_id: str,
    claim: str,
    *,
    category: str,
    status: str,
    evidence_location: str,
    allowed_wording: str,
    forbidden_wording: str,
    notes: str = "",
) -> dict[str, str]:
    return {
        "id": claim_id,
        "claim": claim,
        "category": category,
        "status": status,
        "evidence_location": evidence_location,
        "allowed_wording": allowed_wording,
        "forbidden_wording": forbidden_wording,
        "notes": notes,
    }


def _component_negative_evidence(component_rows: list[dict[str, str]]) -> dict[str, Any]:
    ndcg_rows = [row for row in component_rows if row.get("metric") == "NDCG@10"]
    nonworse = [
        row
        for row in ndcg_rows
        if int(float(row.get("nonworse_domain_count") or 0)) >= 3
        or "harmful_or_redundant" in str(row.get("classification", ""))
    ]
    positive_removal = [
        row
        for row in ndcg_rows
        if float(row.get("mean_delta_removal_minus_full") or 0.0) > 0.0
    ]
    return {
        "ndcg10_row_count": len(ndcg_rows),
        "nonworse_or_redundant_components": sorted({row["ablation"] for row in nonworse}),
        "positive_mean_removal_components": sorted({row["ablation"] for row in positive_removal}),
        "contradicts_all_components_necessary": bool(nonworse),
    }


def _eta_zero_evidence(stability_rows: list[dict[str, str]]) -> dict[str, Any]:
    eta_rows = [row for row in stability_rows if row.get("control") == "eta"]
    zero_best_domains = [
        row.get("domain", "")
        for row in eta_rows
        if str(row.get("test_best_value", "")).strip() in {"0", "0.0"}
    ]
    stable_domains = [
        row.get("domain", "")
        for row in eta_rows
        if _as_bool(row.get("stable_within_tolerance"))
    ]
    return {
        "eta_domain_count": len(eta_rows),
        "eta_test_best_zero_domains": sorted(zero_best_domains),
        "eta_stable_domains": sorted(stable_domains),
        "contradicts_positive_eta_necessity": len(zero_best_domains) == len(EXPECTED_DOMAINS),
    }


def _citation_audit(manuscript_state: dict[str, Any]) -> dict[str, Any]:
    must_add_categories = [
        "LLM-based recommendation and sequential recommendation foundations",
        "uncertainty estimation, calibration, and selective/risk-aware prediction",
        "all eight official baselines used in the main same-candidate table",
        "same-candidate/reranking evaluation protocol and paired significance testing",
        "recent 2025-2026 LLM recommender and uncertainty-aware recommender work",
    ]
    if not manuscript_state["has_manuscript"] or not manuscript_state["has_bibliography"]:
        status = "BLOCKED_NO_MANUSCRIPT_OR_BIBLIOGRAPHY"
        verdict = "NEEDS_WRITING_BEFORE_FINAL_CITATION_AUDIT"
    else:
        status = "READY_FOR_MANUSCRIPT_AND_BIBTEX_AUDIT"
        verdict = "NEEDS_MANUAL_REFERENCE_COMPLETENESS_CHECK"
    return {
        "status": status,
        "verdict": verdict,
        "manuscript_state": manuscript_state,
        "must_add_categories": must_add_categories,
        "notes": [
            "This audit does not fabricate BibTeX entries.",
            "Run a full citation audit after a manuscript and .bib file exist.",
        ],
    }


def build_final_claim_audit(
    *,
    root: str | Path = ".",
    certificate_json: str | Path = DEFAULT_CERTIFICATE_JSON,
    ledger_json: str | Path = DEFAULT_LEDGER_JSON,
    observation_json: str | Path = DEFAULT_OBSERVATION_JSON,
    component_json: str | Path = DEFAULT_COMPONENT_JSON,
    component_summary_csv: str | Path = DEFAULT_COMPONENT_SUMMARY_CSV,
    hyperparameter_json: str | Path = DEFAULT_HYPERPARAMETER_JSON,
    hyperparameter_stability_csv: str | Path = DEFAULT_HYPERPARAMETER_STABILITY_CSV,
    framework_json: str | Path = DEFAULT_FRAMEWORK_JSON,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    paths = {
        "certificate": repo / certificate_json,
        "ledger": repo / ledger_json,
        "observation": repo / observation_json,
        "component": repo / component_json,
        "component_summary": repo / component_summary_csv,
        "hyperparameter": repo / hyperparameter_json,
        "hyperparameter_stability": repo / hyperparameter_stability_csv,
        "framework": repo / framework_json,
    }
    missing = [label for label, path in paths.items() if not path.exists()]
    failures = [f"missing_input:{label}:{paths[label]}" for label in missing]

    certificate = _read_json(paths["certificate"]) if "certificate" not in missing else {}
    ledger = _read_json(paths["ledger"]) if "ledger" not in missing else {}
    observation = _read_json(paths["observation"]) if "observation" not in missing else {}
    component = _read_json(paths["component"]) if "component" not in missing else {}
    component_rows = _read_csv(paths["component_summary"]) if "component_summary" not in missing else []
    hyperparameter = _read_json(paths["hyperparameter"]) if "hyperparameter" not in missing else {}
    hyperparameter_rows = _read_csv(paths["hyperparameter_stability"]) if "hyperparameter_stability" not in missing else []
    framework = _read_json(paths["framework"]) if "framework" not in missing else {}

    comparison_ready = (
        certificate.get("ok") is True
        and certificate.get("comparison_certificate_ready") is True
        and ledger.get("ok") is True
        and ledger.get("comparison_ledger_ready") is True
        and int(ledger.get("official_row_count") or 0) == EXPECTED_OFFICIAL_ROWS
        and int(ledger.get("row_count") or 0) == EXPECTED_METHOD_ROWS
    )
    observation_ready = (
        observation.get("ok") is True
        and observation.get("paper_claim_ready") is True
        and observation.get("claim_gate_pass") is True
        and observation.get("table_eligibility") == "motivation_only_not_main_table_sota"
    )
    component_ready = (
        component.get("ok") is True
        and component.get("paper_claim_ready") is True
        and component.get("table_eligibility") == "supplementary_diagnostic_only"
    )
    hyperparameter_ready = (
        hyperparameter.get("ok") is True
        and hyperparameter.get("paper_claim_ready") is True
        and hyperparameter.get("all_controls_stable") is True
        and hyperparameter.get("table_eligibility") == "supplementary_hyperparameter_stability_only"
    )
    framework_ready = (
        framework.get("paper_claim_ready") is True
        and framework.get("claim_boundary") == "controlled_same_candidate_ranking_not_full_catalog"
    )

    component_negative = _component_negative_evidence(component_rows)
    eta_zero = _eta_zero_evidence(hyperparameter_rows)
    manuscript_state = _discover_manuscript_files(repo)
    citation = _citation_audit(manuscript_state)

    claim_rows = [
        _claim(
            "C1",
            "C-CRP v3 ranks first against eight official-code-level baselines on Sports, Toys, Home, and Tools under 10k-user/101-candidate same-candidate evaluation, with all per-domain paired Holm tests positive and significant.",
            category="performance",
            status=_status(comparison_ready),
            evidence_location=str(paths["certificate"].relative_to(repo)),
            allowed_wording="C-CRP v3 ranks first in the four tested same-candidate new-domain certificates against eight official-code-level baselines.",
            forbidden_wording="full-catalog SOTA; universal recommender SOTA; paper-ready SOTA without scoped modules and manuscript review",
            notes=str(certificate.get("claim_supported") or ""),
        ),
        _claim(
            "C2",
            "The main comparison ledger contains 32 official baseline rows and 4 C-CRP rows with complete full-metric, row-count, score-coverage, provenance, and local/server evidence checks.",
            category="protocol",
            status=_status(comparison_ready),
            evidence_location=str(paths["ledger"].relative_to(repo)),
            allowed_wording="The paper-facing ledger is complete for the declared four-domain same-candidate comparison.",
            forbidden_wording="treating the ledger alone as a whole-paper readiness verdict",
            notes=f"row_count={ledger.get('row_count')}; official_row_count={ledger.get('official_row_count')}; ccrp_row_count={ledger.get('ccrp_row_count')}",
        ),
        _claim(
            "C3",
            "C-CRP event-level uncertainty stratifies ranking reliability: high-uncertainty bins have lower NDCG@10, MRR, and HR@10 than low-uncertainty bins in all four domains.",
            category="qualitative",
            status=_status(observation_ready),
            evidence_location=str(paths["observation"].relative_to(repo)),
            allowed_wording="Use as descriptive motivation evidence that C-CRP uncertainty stratifies same-candidate ranking reliability.",
            forbidden_wording="causal uncertainty effect; statistically significant bin effect; exhaustive baseline calibration evidence",
            notes=f"claim_status={observation.get('claim_status')}; table_eligibility={observation.get('table_eligibility')}",
        ),
        _claim(
            "C4",
            "The four-domain component ablation is package-ready and supports a diagnostic discussion of weak, redundant, or mixed C-CRP components.",
            category="ablation",
            status=_status(component_ready),
            evidence_location=str(paths["component"].relative_to(repo)),
            allowed_wording="Report leave-one-component-out diagnostics; several components are neutral, weak, redundant, or mixed.",
            forbidden_wording="every component is necessary; every uncertainty term improves performance; component effects are statistically significant",
            notes="nonworse_or_redundant_components="
            + ",".join(component_negative["nonworse_or_redundant_components"]),
        ),
        _claim(
            "C5",
            "Every nontrivial C-CRP component is necessary for the observed gains.",
            category="ablation",
            status="CONTRADICTED" if component_negative["contradicts_all_components_necessary"] else "UNSUPPORTED",
            evidence_location=str(paths["component_summary"].relative_to(repo)),
            allowed_wording="Do not make this claim.",
            forbidden_wording="necessary; essential; each component contributes; removing any component hurts",
            notes="Positive or neutral removal deltas directly contradict a strong necessity claim.",
        ),
        _claim(
            "C6",
            "The hyperparameter module supports NDCG@10 stability/sensitivity for eta and the uncertainty weight grid across four domains under validation-only selection and reporting-only test sweeps.",
            category="hyperparameter",
            status=_status(hyperparameter_ready),
            evidence_location=str(paths["hyperparameter"].relative_to(repo)),
            allowed_wording="Eta and weight-grid choices are stable within the preregistered tolerance on NDCG@10 across the four domains.",
            forbidden_wording="all-metric robustness; universal optimum; test-selected tuning; main-table SOTA evidence",
            notes=f"controls={','.join(hyperparameter.get('controls') or [])}; metric={hyperparameter.get('metric')}",
        ),
        _claim(
            "C7",
            "A positive risk exponent or risk penalty is necessary or uniformly beneficial.",
            category="hyperparameter",
            status="CONTRADICTED" if eta_zero["contradicts_positive_eta_necessity"] else "OVERCLAIMED",
            evidence_location=str(paths["hyperparameter_stability"].relative_to(repo)),
            allowed_wording="Do not claim risk-penalty necessity; discuss eta as stable/sensitivity-only.",
            forbidden_wording="eta must be positive; risk penalty is necessary; risk penalty uniformly improves",
            notes="eta_test_best_zero_domains=" + ",".join(eta_zero["eta_test_best_zero_domains"]),
        ),
        _claim(
            "C8",
            "The framework overview figure is ready to describe the controlled same-candidate pipeline and evidence gates.",
            category="method",
            status=_status(framework_ready),
            evidence_location=str(paths["framework"].relative_to(repo)),
            allowed_wording="Use the figure to explain the pipeline and where uncertainty/calibration/risk-adjusted ranking enter.",
            forbidden_wording="using the figure as evidence that observation, ablation, or hyperparameter modules are complete",
            notes=f"claim_boundary={framework.get('claim_boundary')}",
        ),
        _claim(
            "C9",
            "The project is submission-ready now.",
            category="submission",
            status="UNSUPPORTED",
            evidence_location="manuscript/bibliography discovery",
            allowed_wording="Evidence is ready for strict paper drafting and subsequent manuscript-level claim/citation review.",
            forbidden_wording="submission-ready; final READY; cleared for camera-ready",
            notes=f"citation_status={citation['status']}",
        ),
        _claim(
            "C10",
            "The method is a full-catalog or universal recommender SOTA system.",
            category="generality",
            status="UNSUPPORTED",
            evidence_location=str(paths["certificate"].relative_to(repo)),
            allowed_wording="Controlled same-candidate ranking/reranking reliability under the tested protocol.",
            forbidden_wording="full-catalog SOTA; universal cross-domain winner beyond tested same-candidate domains",
            notes="The certificate explicitly excludes full-catalog and universal SOTA claims.",
        ),
    ]

    status_counts = Counter(row["status"] for row in claim_rows)
    evidence_ready_for_drafting = all(
        [comparison_ready, observation_ready, component_ready, hyperparameter_ready, framework_ready]
    )
    final_submission_ready = False
    if not evidence_ready_for_drafting:
        failures.append("paper_facing_evidence_not_all_ready")

    verdict = "NEEDS_EVIDENCE_REPAIR_BEFORE_WRITING"
    if evidence_ready_for_drafting:
        if citation["status"] == "BLOCKED_NO_MANUSCRIPT_OR_BIBLIOGRAPHY":
            verdict = "READY_FOR_WRITING_NOT_FINAL_SUBMISSION_AUDIT"
        else:
            verdict = "READY_FOR_MANUSCRIPT_LEVEL_CLAIM_AND_CITATION_AUDIT"

    if citation["status"] == "BLOCKED_NO_MANUSCRIPT_OR_BIBLIOGRAPHY":
        manuscript_next_action = (
            "Create manuscript and bibliography files, then run ARIS paper-claim-audit "
            "and aris-citation-audit on the actual text."
        )
    else:
        manuscript_next_action = (
            "Run ARIS paper-claim-audit and aris-citation-audit on the existing manuscript "
            "and bibliography, then revise text to the allowed wording in this audit."
        )

    return {
        "schema_version": "2026-06-12.final_paper_claim_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "local_final_paper_facing_claim_audit",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "ok": evidence_ready_for_drafting and not failures,
        "paper_evidence_ready_for_drafting": evidence_ready_for_drafting,
        "final_submission_ready": final_submission_ready,
        "verdict": verdict,
        "root": str(repo),
        "input_paths": {label: _path_state(path) for label, path in paths.items()},
        "evidence_gates": {
            "comparison_ready": comparison_ready,
            "observation_ready": observation_ready,
            "component_ready": component_ready,
            "hyperparameter_ready": hyperparameter_ready,
            "framework_ready": framework_ready,
        },
        "claim_status_counts": dict(sorted(status_counts.items())),
        "claims": claim_rows,
        "component_negative_evidence": component_negative,
        "hyperparameter_eta_zero_evidence": eta_zero,
        "citation_audit": citation,
        "red_flag_terms": {
            "state-of-the-art": "Allowed only with controlled same-candidate scope; forbidden for full-catalog or universal claims.",
            "significant": "Allowed for the paired official comparison only where Holm-significant tests are cited.",
            "necessary": "Forbidden for all C-CRP components and positive eta/risk penalty.",
            "robust": "Use stability/sensitivity for NDCG@10 controls, not broad robustness.",
            "causal": "Unsupported by the observation module.",
            "first": "Blocked until a literature-backed novelty and citation audit exists.",
        },
        "failures": failures,
        "next_actions": [
            "Draft the paper using only SUPPORTED allowed wording and the explicit forbidden-wording guardrails.",
            manuscript_next_action,
            "Run GPT-5.5/Codex review on each written section before calling the paper submission-ready.",
        ],
    }


CLAIM_FIELDS = [
    "id",
    "claim",
    "category",
    "status",
    "evidence_location",
    "allowed_wording",
    "forbidden_wording",
    "notes",
]


def write_markdown(path: str | Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Final Paper-Facing Claim Audit",
        "",
        f"- Generated UTC: `{audit['created_at_utc']}`",
        f"- Verdict: `{audit['verdict']}`",
        f"- Evidence ready for drafting: `{audit['paper_evidence_ready_for_drafting']}`",
        f"- Final submission ready: `{audit['final_submission_ready']}`",
        f"- Will start experiment: `{audit['will_start_experiment']}`",
        "",
        "## Evidence Gates",
        "",
    ]
    for key, value in audit["evidence_gates"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Claim Status Counts", ""])
    for key, value in audit["claim_status_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Supported Or Allowed Claims", ""])
    for claim in audit["claims"]:
        if claim["status"] == "SUPPORTED":
            lines.append(f"- `{claim['id']}` {claim['allowed_wording']}")
    lines.extend(["", "## Overclaim Guards", ""])
    for claim in audit["claims"]:
        if claim["status"] in {"CONTRADICTED", "OVERCLAIMED", "UNSUPPORTED"}:
            lines.append(f"- `{claim['id']}` `{claim['status']}`: {claim['forbidden_wording']}")
    lines.extend(["", "## Citation Audit", ""])
    citation = audit["citation_audit"]
    lines.append(f"- Status: `{citation['status']}`")
    lines.append(f"- Verdict: `{citation['verdict']}`")
    lines.append(f"- Manuscript files: `{len(citation['manuscript_state']['tex_files'])}`")
    lines.append(f"- BibTeX files: `{len(citation['manuscript_state']['bib_files'])}`")
    lines.extend(["", "Must-add citation categories before final submission:"])
    for category in citation["must_add_categories"]:
        lines.append(f"- {category}")
    lines.extend(["", "## Failures", ""])
    if audit["failures"]:
        lines.extend(f"- {failure}" for failure in audit["failures"])
    else:
        lines.append("- none")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {action}" for action in audit["next_actions"])
    lines.append("")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local final paper-facing claim audit.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--certificate_json", default=str(DEFAULT_CERTIFICATE_JSON))
    parser.add_argument("--ledger_json", default=str(DEFAULT_LEDGER_JSON))
    parser.add_argument("--observation_json", default=str(DEFAULT_OBSERVATION_JSON))
    parser.add_argument("--component_json", default=str(DEFAULT_COMPONENT_JSON))
    parser.add_argument("--component_summary_csv", default=str(DEFAULT_COMPONENT_SUMMARY_CSV))
    parser.add_argument("--hyperparameter_json", default=str(DEFAULT_HYPERPARAMETER_JSON))
    parser.add_argument("--hyperparameter_stability_csv", default=str(DEFAULT_HYPERPARAMETER_STABILITY_CSV))
    parser.add_argument("--framework_json", default=str(DEFAULT_FRAMEWORK_JSON))
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_claims_csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_final_claim_audit(
        root=args.root,
        certificate_json=args.certificate_json,
        ledger_json=args.ledger_json,
        observation_json=args.observation_json,
        component_json=args.component_json,
        component_summary_csv=args.component_summary_csv,
        hyperparameter_json=args.hyperparameter_json,
        hyperparameter_stability_csv=args.hyperparameter_stability_csv,
        framework_json=args.framework_json,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.output_md, audit)
    _write_csv(args.output_claims_csv, audit["claims"], CLAIM_FIELDS)
    print(
        json.dumps(
            {
                "ok": audit["ok"],
                "paper_evidence_ready_for_drafting": audit["paper_evidence_ready_for_drafting"],
                "final_submission_ready": audit["final_submission_ready"],
                "verdict": audit["verdict"],
                "claim_status_counts": audit["claim_status_counts"],
                "citation_status": audit["citation_audit"]["status"],
                "failures": audit["failures"],
            },
            indent=2,
        )
    )
    if not audit["paper_evidence_ready_for_drafting"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
