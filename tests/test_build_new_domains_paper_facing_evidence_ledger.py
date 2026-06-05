import csv
import json
from pathlib import Path

from scripts.audit.main_build_new_domains_paper_facing_evidence_ledger import (
    CCRP_METHOD,
    DOMAINS,
    LEDGER_FIELDS,
    METRICS,
    OFFICIAL_METHODS,
    build_ledger,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_official_package(path: Path, *, blockers: list[str] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "fairness_provenance.json").write_text(
        json.dumps(
            {
                "implementation_status": "official_completed",
                "blockers": blockers or [],
                "score_coverage_rate": 1.0,
                "comparison_variant": "official_code_qwen3base_default_hparams_declared_adaptation",
                "official_repo": "https://example.test/repo",
                "pinned_commit": "abc123",
                "baseline_hyperparameter_source": "official_default_or_recommended",
                "test_set_model_selection_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "server_final_evidence_audit.json",
        "server_large_artifact_manifest.json",
        "light_evidence_sync_manifest.json",
        "method_score_audit.json",
        "method_same_candidate_score_audit.txt",
        "method_run_summary.json",
    ):
        (path / name).write_text("{}\n", encoding="utf-8")


def _domain_rows(tmp_path: Path) -> list[dict]:
    rows = []
    for domain in DOMAINS:
        row = {
            "domain": domain,
            "primary_gate_json": str(tmp_path / domain / "gate.json"),
            "post_cleanup_gate_json": "",
            "comparison_csv": str(tmp_path / domain / "comparison.csv"),
            "paired_tests_csv": str(tmp_path / domain / "paired_tests.csv"),
            "paired_summary_json": str(tmp_path / domain / "paired_summary.json"),
            "gate_ok": "True",
            "claim_gate": f"{domain}_domain_pass",
            "ccrp_rank1_all_metrics": "True",
            "paired_positive_holm_significant_count": "56",
            "paired_all_positive_holm_significant": "True",
            "min_delta": "0.01",
            "min_ci_low": "0.001",
            "max_holm_p_value": "0.001",
        }
        for field in ("primary_gate_json", "comparison_csv", "paired_tests_csv", "paired_summary_json"):
            Path(row[field]).parent.mkdir(parents=True, exist_ok=True)
            Path(row[field]).write_text("ok\n", encoding="utf-8")
        rows.append(row)
    return rows


def _method_rows(tmp_path: Path) -> list[dict]:
    rows = []
    for domain in DOMAINS:
        ccrp_base = Path("outputs") / f"{domain}_large10000_100neg_ccrp_v3"
        ccrp_base.mkdir(parents=True, exist_ok=True)
        (ccrp_base / "report.json").write_text("{}\n", encoding="utf-8")
        for method in (CCRP_METHOD, *OFFICIAL_METHODS):
            is_ccrp = method == CCRP_METHOD
            evidence_path = ""
            if not is_ccrp:
                evidence_path = str(tmp_path / "evidence" / f"{domain}_{method}")
                _write_official_package(Path(evidence_path))
            row = {
                "domain": domain,
                "rank_by_NDCG@10": "1" if is_ccrp else "2",
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
                "predictions_jsonl_line_source": "file",
                "ranking_eval_records_csv_lines": "10001",
                "summary_path": "/server/summary.csv",
                "eval_path": "/server/eval.csv",
                "scores_path": "/server/scores.csv",
                "evidence_path": evidence_path,
                "official_light_package_ok": "" if is_ccrp else "True",
                "ccrp_local_event_restat_ready": "False" if is_ccrp else "",
            }
            for metric in METRICS:
                row[metric] = "0.2" if is_ccrp else "0.1"
            rows.append(row)
    return rows


def _inputs(tmp_path: Path):
    method_rows = _method_rows(tmp_path)
    evidence_rows = [
        {"domain": row["domain"], "method": row["method"], "ok": True, "local_dir": row["evidence_path"]}
        for row in method_rows
        if row["method"] != CCRP_METHOD
    ]
    return (
        _write_csv(tmp_path / "methods.csv", method_rows),
        _write_csv(tmp_path / "domains.csv", _domain_rows(tmp_path)),
        _write_json(tmp_path / "evidence.json", {"ok": True, "rows": evidence_rows}),
        _write_json(
            tmp_path / "certificate.json",
            {
                "ok": True,
                "claim_supported": "C-CRP ranks first.",
                "claim_not_supported": ["paper-ready SOTA"],
            },
        ),
    )


def test_build_ledger_emits_paper_facing_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    method_csv, domain_csv, evidence_json, certificate_json = _inputs(tmp_path)

    ledger = build_ledger(
        method_rows_csv=method_csv,
        domain_summary_csv=domain_csv,
        evidence_consistency_json=evidence_json,
        certificate_audit_json=certificate_json,
    )

    assert ledger["ok"] is True
    assert ledger["row_count"] == 36
    assert ledger["official_row_count"] == 32
    assert ledger["ccrp_row_count"] == 4
    assert "implementation_status" in LEDGER_FIELDS
    official = next(row for row in ledger["rows"] if row["method"] == "llmemb")
    assert official["implementation_status"] == "official_completed"
    assert official["table_eligibility"] == "main_official_comparison_eligible"
    ccrp = next(row for row in ledger["rows"] if row["method"] == CCRP_METHOD)
    assert ccrp["table_eligibility"] == "main_internal_comparison_compact_certificate"
    assert "compact_certificate_not_self_contained_for_event_restat" in ccrp["row_warnings"]


def test_build_ledger_fails_when_official_has_blocker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    method_csv, domain_csv, evidence_json, certificate_json = _inputs(tmp_path)
    blocked_dir = tmp_path / "evidence" / "sports_llmemb"
    _write_official_package(blocked_dir, blockers=["blocked"])

    ledger = build_ledger(
        method_rows_csv=method_csv,
        domain_summary_csv=domain_csv,
        evidence_consistency_json=evidence_json,
        certificate_audit_json=certificate_json,
    )

    assert ledger["ok"] is False
    assert "sports/llmemb:official_not_main_eligible" in ledger["failures"]
