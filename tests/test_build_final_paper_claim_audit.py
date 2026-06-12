import csv
import json
from pathlib import Path

from scripts.audit.main_build_final_paper_claim_audit import build_final_claim_audit


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _seed_inputs(tmp_path: Path) -> dict[str, Path]:
    certificate = _write_json(
        tmp_path / "certificate.json",
        {
            "ok": True,
            "comparison_certificate_ready": True,
            "claim_supported": "C-CRP ranks first under the declared same-candidate protocol.",
        },
    )
    ledger = _write_json(
        tmp_path / "ledger.json",
        {
            "ok": True,
            "comparison_ledger_ready": True,
            "row_count": 36,
            "official_row_count": 32,
            "ccrp_row_count": 4,
        },
    )
    observation = _write_json(
        tmp_path / "observation.json",
        {
            "ok": True,
            "paper_claim_ready": True,
            "claim_gate_pass": True,
            "claim_status": "uncertainty_stratifies_reliability",
            "table_eligibility": "motivation_only_not_main_table_sota",
        },
    )
    component = _write_json(
        tmp_path / "component.json",
        {
            "ok": True,
            "paper_claim_ready": True,
            "table_eligibility": "supplementary_diagnostic_only",
        },
    )
    component_summary = _write_csv(
        tmp_path / "component_summary.csv",
        [
            {
                "ablation": "without_boundary_uncertainty",
                "metric": "NDCG@10",
                "mean_delta_removal_minus_full": "0.0",
                "nonworse_domain_count": "4",
                "classification": "removal_nonworse_in_3plus_domains_harmful_or_redundant",
            },
            {
                "ablation": "without_evidence_support",
                "metric": "NDCG@10",
                "mean_delta_removal_minus_full": "-0.001",
                "nonworse_domain_count": "1",
                "classification": "directionally_supportive_not_significant",
            },
        ],
    )
    hyperparameter = _write_json(
        tmp_path / "hyperparameter.json",
        {
            "ok": True,
            "paper_claim_ready": True,
            "all_controls_stable": True,
            "table_eligibility": "supplementary_hyperparameter_stability_only",
            "controls": ["eta", "weight_grid_label"],
            "metric": "NDCG@10",
        },
    )
    hyperparameter_stability = _write_csv(
        tmp_path / "hyperparameter_stability.csv",
        [
            {
                "control": "eta",
                "domain": domain,
                "test_best_value": "0",
                "stable_within_tolerance": "True",
            }
            for domain in ("sports", "toys", "home", "tools")
        ],
    )
    framework = _write_json(
        tmp_path / "framework.json",
        {
            "paper_claim_ready": True,
            "claim_boundary": "controlled_same_candidate_ranking_not_full_catalog",
        },
    )
    return {
        "certificate_json": certificate,
        "ledger_json": ledger,
        "observation_json": observation,
        "component_json": component,
        "component_summary_csv": component_summary,
        "hyperparameter_json": hyperparameter,
        "hyperparameter_stability_csv": hyperparameter_stability,
        "framework_json": framework,
    }


def test_final_claim_audit_marks_evidence_ready_but_not_submission_ready(tmp_path):
    paths = _seed_inputs(tmp_path)

    audit = build_final_claim_audit(root=tmp_path, **paths)

    assert audit["paper_evidence_ready_for_drafting"] is True
    assert audit["final_submission_ready"] is False
    assert audit["verdict"] == "READY_FOR_WRITING_NOT_FINAL_SUBMISSION_AUDIT"
    assert audit["citation_audit"]["status"] == "BLOCKED_NO_MANUSCRIPT_OR_BIBLIOGRAPHY"
    assert audit["claim_status_counts"]["SUPPORTED"] == 6


def test_final_claim_audit_contradicts_component_and_eta_necessity_claims(tmp_path):
    paths = _seed_inputs(tmp_path)

    audit = build_final_claim_audit(root=tmp_path, **paths)
    claims = {row["id"]: row for row in audit["claims"]}

    assert claims["C5"]["status"] == "CONTRADICTED"
    assert claims["C7"]["status"] == "CONTRADICTED"
    assert "without_boundary_uncertainty" in audit["component_negative_evidence"]["nonworse_or_redundant_components"]
    assert audit["hyperparameter_eta_zero_evidence"]["eta_test_best_zero_domains"] == [
        "home",
        "sports",
        "tools",
        "toys",
    ]


def test_final_claim_audit_fails_closed_on_missing_evidence(tmp_path):
    paths = _seed_inputs(tmp_path)
    paths["ledger_json"].unlink()

    audit = build_final_claim_audit(root=tmp_path, **paths)

    assert audit["paper_evidence_ready_for_drafting"] is False
    assert any(failure.startswith("missing_input:ledger") for failure in audit["failures"])
    assert audit["verdict"] == "NEEDS_EVIDENCE_REPAIR_BEFORE_WRITING"


def test_final_claim_audit_detects_existing_manuscript_and_bibliography(tmp_path):
    paths = _seed_inputs(tmp_path)
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "main.tex").write_text("\\section{Intro}\n", encoding="utf-8")
    (paper_dir / "references.bib").write_text("@article{x, title={x}, year={2026}}\n", encoding="utf-8")

    audit = build_final_claim_audit(root=tmp_path, **paths)

    assert audit["citation_audit"]["status"] == "READY_FOR_MANUSCRIPT_AND_BIBTEX_AUDIT"
    assert audit["verdict"] == "READY_FOR_MANUSCRIPT_LEVEL_CLAIM_AND_CITATION_AUDIT"
    assert "existing manuscript" in audit["next_actions"][1]
