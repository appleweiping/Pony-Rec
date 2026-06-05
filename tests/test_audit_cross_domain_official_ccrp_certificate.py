import csv
import json
from pathlib import Path

from scripts.audit.main_audit_cross_domain_official_ccrp_certificate import (
    CCRP_METHOD,
    OFFICIAL_METHODS,
    REQUIRED_DOMAINS,
    REQUIRED_METRICS,
    build_audit,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n", encoding="utf-8")
    return str(path)


def _write_ccrp_local_evidence(root: Path) -> None:
    for domain in REQUIRED_DOMAINS:
        raw = root / "outputs" / f"{domain}_large10000_100neg_ccrp_v3"
        tables = (
            root
            / "outputs"
            / f"{domain}_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate"
            / "tables"
        )
        raw.mkdir(parents=True, exist_ok=True)
        tables.mkdir(parents=True, exist_ok=True)
        (raw / "report.json").write_text("{}\n", encoding="utf-8")
        (raw / "user_ranks.jsonl").write_text("".join("{}\n" for _ in range(10_000)), encoding="utf-8")
        (tables / "ranking_eval_records.csv").write_text(
            "source_event_id,positive_rank\n" + "".join(f"e{i},1\n" for i in range(10_000)),
            encoding="utf-8",
        )
        (tables / "ranking_metrics.csv").write_text("sample_count,avg_candidates\n10000,101\n", encoding="utf-8")
        (tables / "same_candidate_external_baseline_summary.csv").write_text(
            "status_label,artifact_class\nsame_schema_internal_method,completed_result\n",
            encoding="utf-8",
        )
        (tables / "external_score_coverage.csv").write_text("score_coverage_rate\n1.0\n", encoding="utf-8")


def _domain_rows(tmp_path: Path) -> list[dict]:
    rows = []
    for domain in REQUIRED_DOMAINS:
        row = {
            "domain": domain,
            "primary_gate_json": _touch(tmp_path / domain / "gate.json"),
            "post_cleanup_gate_json": "",
            "comparison_csv": _touch(tmp_path / domain / "comparison.csv"),
            "paired_tests_csv": _touch(tmp_path / domain / "paired_tests.csv"),
            "paired_summary_json": _touch(tmp_path / domain / "paired_summary.json"),
            "gate_ok": "True",
            "post_cleanup_gate_ok": "",
            "official_ok_count": "8",
            "comparison_row_count": "9",
            "official_baseline_row_count": "8",
            "ccrp_row_count": "1",
            "ccrp_rank_by_NDCG@10": "1",
            "ccrp_rank1_all_metrics": "True",
            "paired_test_count": "56",
            "paired_summary_count": "56",
            "paired_positive_holm_significant_count": "56",
            "paired_all_positive_holm_significant": "True",
            "min_delta": "0.01",
            "min_ci_low": "0.001",
            "max_holm_p_value": "0.001",
            "min_effect_cohen_dz": "0.05",
            "min_abs_effect_cohen_dz": "0.05",
            "sample_count": "10000",
            "avg_candidates": "101.0",
            "score_coverage_rate": "1.0",
            "claim_gate": f"{domain}_domain_pass",
            "claim_scope_warning": "domain-only",
        }
        for metric in REQUIRED_METRICS:
            row[f"ccrp_{metric}"] = "0.2"
            row[f"delta_vs_best_official_{metric}"] = "0.01"
        rows.append(row)
    return rows


def _method_rows(tmp_path: Path) -> list[dict]:
    rows = []
    for domain in REQUIRED_DOMAINS:
        methods = [CCRP_METHOD, *OFFICIAL_METHODS]
        for index, method in enumerate(methods, start=1):
            is_ccrp = method == CCRP_METHOD
            evidence_path = ""
            if not is_ccrp:
                evidence_path = str(tmp_path / "evidence" / f"{domain}_{method}")
                Path(evidence_path).mkdir(parents=True, exist_ok=True)
            row = {
                "domain": domain,
                "rank_by_NDCG@10": str(index),
                "method": method,
                "kind": "internal_method" if is_ccrp else "official_baseline",
                "status_label": "same_schema_internal_method" if is_ccrp else "same_schema_external_baseline",
                "artifact_class": "completed_result",
                "sample_count": "10000",
                "avg_candidates": "101.0",
                "score_coverage_rate": "1.0",
                "ranking_events": "10000",
                "total_candidates": "1010000",
                "matched_candidates": "1010000",
                "scores_csv_lines": "1010001",
                "predictions_jsonl_lines": "10000",
                "predictions_jsonl_line_source": "server_final_evidence_audit",
                "ranking_eval_records_csv_lines": "10001",
                "summary_path": "/server/summary.csv",
                "eval_path": "/server/eval.csv",
                "scores_path": "/server/scores.csv",
                "evidence_path": evidence_path,
                "official_light_package_ok": "" if is_ccrp else "True",
                "ccrp_local_event_restat_ready": "False" if is_ccrp else "",
            }
            for metric in REQUIRED_METRICS:
                row[metric] = "0.2" if is_ccrp else "0.1"
            rows.append(row)
    return rows


def _write_valid_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    _write_ccrp_local_evidence(tmp_path)
    summary = {
        "all_blocking_checks_ok": True,
        "aris_verdict": "CONDITIONAL_PASS_FOR_COMPACT_FOUR_NEW_DOMAIN_GATE_SUMMARY",
        "domain_order": list(REQUIRED_DOMAINS),
        "domains": [{"domain": domain} for domain in REQUIRED_DOMAINS],
        "claim_supported": "C-CRP ranks first under same-candidate evaluation.",
        "claim_not_supported": ["paper-ready SOTA", "full-catalog recommender SOTA"],
    }
    evidence = {
        "ok": True,
        "domains": list(REQUIRED_DOMAINS),
        "row_count": 32,
        "ok_count": 32,
        "failure_count": 0,
    }
    paper_critical = {
        "paper_ready": False,
        "summary": {
            "four_domain_evidence_consistent": True,
            "signal_rows_available": False,
            "phase2_5_storage_launch_allowed": False,
        },
    }
    return (
        _write_json(tmp_path / "summary.json", summary),
        _write_csv(tmp_path / "domain_summary.csv", _domain_rows(tmp_path)),
        _write_csv(tmp_path / "method_rows.csv", _method_rows(tmp_path)),
        _write_json(tmp_path / "evidence.json", evidence),
        _write_json(tmp_path / "paper_critical.json", paper_critical),
    )


def test_cross_domain_certificate_audit_accepts_valid_compact_certificate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary_json, domain_csv, method_csv, evidence_json, paper_json = _write_valid_inputs(tmp_path)

    audit = build_audit(
        summary_json=summary_json,
        domain_summary_csv=domain_csv,
        method_rows_csv=method_csv,
        evidence_consistency_json=evidence_json,
        paper_critical_json=paper_json,
    )

    assert audit["ok"] is True
    assert audit["comparison_certificate_ready"] is True
    assert audit["paper_ready"] is False
    assert audit["method_rows_audit"]["row_count"] == 36
    assert audit["evidence_consistency_audit"]["ok_count"] == 32
    assert audit["warnings"] == []


def test_cross_domain_certificate_audit_rejects_missing_official_row(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary_json, domain_csv, method_csv, evidence_json, paper_json = _write_valid_inputs(tmp_path)
    rows = _method_rows(tmp_path)
    rows = [row for row in rows if not (row["domain"] == "sports" and row["method"] == "llmemb")]
    _write_csv(method_csv, rows)

    audit = build_audit(
        summary_json=summary_json,
        domain_summary_csv=domain_csv,
        method_rows_csv=method_csv,
        evidence_consistency_json=evidence_json,
        paper_critical_json=paper_json,
    )

    assert audit["ok"] is False
    assert "sports:method_row_count_not_9" in audit["failures"]
    assert "method_rows_total_not_36" in audit["failures"]


def test_cross_domain_certificate_audit_rejects_non_significant_domain(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary_json, domain_csv, method_csv, evidence_json, paper_json = _write_valid_inputs(tmp_path)
    rows = _domain_rows(tmp_path)
    rows[0]["paired_positive_holm_significant_count"] = "55"
    rows[0]["paired_all_positive_holm_significant"] = "False"
    _write_csv(domain_csv, rows)

    audit = build_audit(
        summary_json=summary_json,
        domain_summary_csv=domain_csv,
        method_rows_csv=method_csv,
        evidence_consistency_json=evidence_json,
        paper_critical_json=paper_json,
    )

    assert audit["ok"] is False
    assert "sports:paired_positive_count_not_56" in audit["failures"]
    assert "sports:paired_not_all_positive_holm" in audit["failures"]
