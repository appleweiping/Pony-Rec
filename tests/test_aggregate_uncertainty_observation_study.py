import json
from pathlib import Path

from scripts.analysis.main_aggregate_uncertainty_observation_study import aggregate_uncertainty_observation


METRICS = ("MRR", "HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20")


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _json(path, payload):
    return _write(path, json.dumps(payload, indent=2) + "\n")


def _seed_observation_package(root, domain, *, audit_ok=True, degraded=True):
    package = root / f"observation_{domain}"
    _json(package / "phase2_5_observation_motivation_package_audit.json", {"ok": audit_ok, "paper_claim_ready": audit_ok})
    _json(
        package / "observation_provenance.json",
        {
            "ok": True,
            "uncertainty_summary": {"event_count": 10000},
        },
    )
    header = [
        "domain",
        "method",
        "uncertainty_bin_index",
        "uncertainty_bin",
        "n_events",
        "event_uncertainty_mean",
        "avg_positive_rank",
        *METRICS,
    ]
    rows = []
    for method in ("ccrp_v3", "llmemb"):
        low = 0.3
        high = 0.2 if degraded else 0.31
        for idx, label, value in ((0, "Q01", low), (4, "Q05", high), (-1, "ALL", (low + high) / 2)):
            rows.append([domain, method, idx, label, 10000 if idx == -1 else 2000, 0.1 + idx, 5, *([value] * len(METRICS))])
    _write(package / "observation_summary.csv", ",".join(header) + "\n" + "\n".join(",".join(map(str, row)) for row in rows) + "\n")
    return package


def test_observation_aggregate_passes_stratification_gate(tmp_path):
    root = tmp_path / "packages"
    for domain in ("sports", "toys", "home", "tools"):
        _seed_observation_package(root, domain, degraded=True)

    provenance = aggregate_uncertainty_observation(package_root=root, output_dir=tmp_path / "out", skip_plot=True)

    assert provenance["ok"] is True
    assert provenance["claim_gate_pass"] is True
    assert provenance["claim_status"] == "uncertainty_stratifies_reliability"
    assert provenance["primary_gate"]["degraded_domain_count"] == 4
    assert provenance["table_eligibility"] == "motivation_only_not_main_table_sota"


def test_observation_aggregate_writes_delta_and_trend_figures(tmp_path):
    root = tmp_path / "packages"
    for domain in ("sports", "toys", "home", "tools"):
        _seed_observation_package(root, domain, degraded=True)

    provenance = aggregate_uncertainty_observation(package_root=root, output_dir=tmp_path / "out")

    figure_paths = provenance["outputs"]["figure_paths"]
    assert len(figure_paths) == 4
    assert any(path.endswith("fig_uncertainty_observation_four_domain.png") for path in figure_paths)
    assert any(path.endswith("fig_uncertainty_observation_four_domain_trend.png") for path in figure_paths)
    for path in figure_paths:
        assert (tmp_path / "out" / Path(path).name).exists()


def test_observation_aggregate_downgrades_mixed_pattern_without_failing_package(tmp_path):
    root = tmp_path / "packages"
    for domain in ("sports", "toys", "home", "tools"):
        _seed_observation_package(root, domain, degraded=(domain in {"sports", "toys"}))

    provenance = aggregate_uncertainty_observation(package_root=root, output_dir=tmp_path / "out", skip_plot=True)

    assert provenance["ok"] is True
    assert provenance["claim_gate_pass"] is False
    assert provenance["claim_status"] == "mixed_diagnostic_pattern"


def test_observation_aggregate_downgrades_subset_domains_even_when_pattern_passes(tmp_path):
    root = tmp_path / "packages"
    for domain in ("sports", "toys", "home"):
        _seed_observation_package(root, domain, degraded=True)

    provenance = aggregate_uncertainty_observation(
        package_root=root,
        output_dir=tmp_path / "out",
        domains=("sports", "toys", "home"),
        min_domains_for_stratification=3,
        skip_plot=True,
    )

    assert provenance["ok"] is True
    assert provenance["paper_claim_ready"] is False
    assert provenance["claim_gate_pass"] is False
    assert provenance["status_label"] == "diagnostic_subset_only"
    assert provenance["exact_four_domain_set"] is False


def test_observation_aggregate_fails_closed_on_unready_domain_package(tmp_path):
    root = tmp_path / "packages"
    for domain in ("sports", "toys", "home", "tools"):
        _seed_observation_package(root, domain, audit_ok=(domain != "home"))

    provenance = aggregate_uncertainty_observation(package_root=root, output_dir=tmp_path / "out", skip_plot=True)

    assert provenance["ok"] is False
    assert "package_audit_not_ready:home" in provenance["failures"]
