import json

from scripts.analysis.main_aggregate_ccrp_component_ablation import aggregate_component_ablation


METRICS = ("MRR", "HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20")
ABLATIONS = (
    "full",
    "without_boundary_uncertainty",
    "without_calibration_gap",
    "without_evidence_support",
    "without_counterevidence",
    "without_risk_penalty",
)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _json(path, payload):
    return _write(path, json.dumps(payload, indent=2) + "\n")


def _seed_package(root, domain, *, audit_ok=True, counter_delta=0.01):
    package = root / f"ccrp_ablation_{domain}"
    _json(package / "phase2_5_component_ablation_package_audit.json", {"ok": audit_ok, "paper_claim_ready": audit_ok})
    _json(package / "component_ablation_provenance.json", {"ok": True})
    header = [
        "domain",
        "split",
        "ablation",
        "n_events",
        "expected_candidates_per_event",
        "audit_ok",
        "degeneracy_audit_ok",
        "score_coverage_rate",
        *METRICS,
    ]
    rows = []
    for ablation in ABLATIONS:
        value = 0.2
        if ablation == "without_counterevidence":
            value += counter_delta
        if ablation == "without_calibration_gap":
            value -= 0.01
        row = [
            domain,
            "test",
            ablation,
            "10000",
            "101",
            "true",
            "true",
            "1.0",
            *[str(value) for _ in METRICS],
        ]
        rows.append(",".join(row))
    _write(package / "component_ablation_summary.csv", ",".join(header) + "\n" + "\n".join(rows) + "\n")
    return package


def test_aggregate_component_ablation_classifies_nonworse_removals(tmp_path):
    root = tmp_path / "packages"
    for domain in ("sports", "toys", "home", "tools"):
        _seed_package(root, domain, counter_delta=0.01)

    provenance = aggregate_component_ablation(package_root=root, output_dir=tmp_path / "out", skip_plot=True)

    assert provenance["ok"] is True
    summary = (tmp_path / "out" / "component_ablation_four_domain_component_summary.csv").read_text(encoding="utf-8")
    assert "removal_minus_full" in summary
    assert "without_counterevidence,NDCG@10" in summary
    assert "removal_nonworse_in_3plus_domains_harmful_or_redundant" in summary
    assert provenance["table_eligibility"] == "supplementary_diagnostic_only"


def test_aggregate_component_ablation_fails_closed_on_unready_package(tmp_path):
    root = tmp_path / "packages"
    for domain in ("sports", "toys", "home", "tools"):
        _seed_package(root, domain, audit_ok=(domain != "toys"))

    provenance = aggregate_component_ablation(package_root=root, output_dir=tmp_path / "out", skip_plot=True)

    assert provenance["ok"] is False
    assert "package_audit_not_ready:toys" in provenance["failures"]


def test_aggregate_component_ablation_requires_all_ablation_rows(tmp_path):
    root = tmp_path / "packages"
    for domain in ("sports", "toys", "home", "tools"):
        _seed_package(root, domain)
    path = root / "ccrp_ablation_home" / "component_ablation_summary.csv"
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if "without_risk_penalty" not in line]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    provenance = aggregate_component_ablation(package_root=root, output_dir=tmp_path / "out", skip_plot=True)

    assert provenance["ok"] is False
    assert "missing_ablation:home:without_risk_penalty" in provenance["failures"]
