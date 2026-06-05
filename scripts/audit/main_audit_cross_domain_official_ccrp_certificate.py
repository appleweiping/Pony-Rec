from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_DOMAINS = ("sports", "toys", "home", "tools")
REQUIRED_METRICS = ("HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR")
OFFICIAL_METHODS = (
    "llmemb",
    "proex_profile",
    "promax_profile",
    "elmrec_graph",
    "irllrec_intent",
    "rlmrec_graphcl",
    "llm2rec_sasrec",
    "llmesr_sasrec",
)
CCRP_METHOD = "ccrp_v3_qwen3base_pointwise"
EXPECTED_USERS = 10_000
EXPECTED_CANDIDATES_PER_USER = 101
EXPECTED_SCORE_CSV_LINES = EXPECTED_USERS * EXPECTED_CANDIDATES_PER_USER + 1
EXPECTED_EVAL_CSV_LINES = EXPECTED_USERS + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the compact four-new-domain official+C-CRP certificate without "
            "SSH, deletion, copying, or experiment execution."
        )
    )
    parser.add_argument(
        "--summary_json",
        default="outputs/summary/new_domains_official_ccrp_cross_domain_20260605_summary.json",
    )
    parser.add_argument(
        "--domain_summary_csv",
        default="outputs/summary/new_domains_official_ccrp_cross_domain_20260605_domain_summary.csv",
    )
    parser.add_argument(
        "--method_rows_csv",
        default="outputs/summary/new_domains_official_ccrp_cross_domain_20260605_method_rows.csv",
    )
    parser.add_argument(
        "--evidence_consistency_json",
        default=(
            "outputs/summary/paper_critical/"
            "local_server_evidence_consistency_new_domains_post_backfill_20260606.json"
        ),
    )
    parser.add_argument(
        "--paper_critical_json",
        default=(
            "outputs/summary/paper_critical/"
            "paper_critical_module_audit_post_evidence_backfill_20260606_0155.json"
        ),
    )
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_md", default="")
    return parser.parse_args()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    if text in {"", "none", "null"}:
        return None
    return None


def _as_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(str(value)))
    except Exception:
        return None


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _is_finite_number(value: Any) -> bool:
    return math.isfinite(_as_float(value))


def _path_exists(path_text: str) -> bool:
    return bool(path_text) and Path(path_text).exists()


def _require(condition: bool, failures: list[str], label: str) -> None:
    if not condition:
        failures.append(label)


def _domain_summary_by_domain(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("domain", "")).strip(): row for row in rows}


def _method_rows_by_domain(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("domain", "")).strip(), []).append(row)
    return grouped


def _audit_domain_row(domain: str, row: dict[str, str]) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    _require(_as_bool(row.get("gate_ok")) is True, failures, f"{domain}:gate_not_ok")
    _require(_as_int(row.get("official_ok_count")) == len(OFFICIAL_METHODS), failures, f"{domain}:official_ok_count_not_8")
    _require(_as_int(row.get("comparison_row_count")) == len(OFFICIAL_METHODS) + 1, failures, f"{domain}:comparison_row_count_not_9")
    _require(_as_int(row.get("official_baseline_row_count")) == len(OFFICIAL_METHODS), failures, f"{domain}:official_baseline_row_count_not_8")
    _require(_as_int(row.get("ccrp_row_count")) == 1, failures, f"{domain}:ccrp_row_count_not_1")
    _require(_as_int(row.get("ccrp_rank_by_NDCG@10")) == 1, failures, f"{domain}:ccrp_rank_by_ndcg10_not_1")
    _require(_as_bool(row.get("ccrp_rank1_all_metrics")) is True, failures, f"{domain}:ccrp_not_rank1_all_metrics")
    _require(_as_int(row.get("paired_test_count")) == len(OFFICIAL_METHODS) * len(REQUIRED_METRICS), failures, f"{domain}:paired_test_count_not_56")
    _require(_as_int(row.get("paired_summary_count")) == len(OFFICIAL_METHODS) * len(REQUIRED_METRICS), failures, f"{domain}:paired_summary_count_not_56")
    _require(_as_int(row.get("paired_positive_holm_significant_count")) == len(OFFICIAL_METHODS) * len(REQUIRED_METRICS), failures, f"{domain}:paired_positive_count_not_56")
    _require(_as_bool(row.get("paired_all_positive_holm_significant")) is True, failures, f"{domain}:paired_not_all_positive_holm")
    _require(_as_int(row.get("sample_count")) == EXPECTED_USERS, failures, f"{domain}:sample_count_not_10000")
    _require(abs(_as_float(row.get("avg_candidates")) - EXPECTED_CANDIDATES_PER_USER) < 1e-9, failures, f"{domain}:avg_candidates_not_101")
    _require(abs(_as_float(row.get("score_coverage_rate")) - 1.0) < 1e-12, failures, f"{domain}:score_coverage_not_1")
    _require(_as_float(row.get("min_delta")) > 0.0, failures, f"{domain}:min_delta_not_positive")
    _require(_as_float(row.get("min_ci_low")) > 0.0, failures, f"{domain}:min_ci_low_not_positive")
    _require(0.0 <= _as_float(row.get("max_holm_p_value")) < 0.05, failures, f"{domain}:max_holm_p_not_significant")
    for metric in REQUIRED_METRICS:
        _require(_is_finite_number(row.get(f"ccrp_{metric}")), failures, f"{domain}:missing_ccrp_{metric}")
        _require(_as_float(row.get(f"delta_vs_best_official_{metric}")) > 0.0, failures, f"{domain}:delta_{metric}_not_positive")
    for field in ("primary_gate_json", "comparison_csv", "paired_tests_csv", "paired_summary_json"):
        if not _path_exists(str(row.get(field, ""))):
            failures.append(f"{domain}:missing_local_source_path:{field}")
    if row.get("post_cleanup_gate_json") and not _path_exists(str(row.get("post_cleanup_gate_json"))):
        warnings.append(f"{domain}:post_cleanup_gate_path_not_local:{row.get('post_cleanup_gate_json')}")
    return {
        "domain": domain,
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "official_ok_count": _as_int(row.get("official_ok_count")),
        "paired_positive_holm_significant_count": _as_int(row.get("paired_positive_holm_significant_count")),
        "ccrp_rank1_all_metrics": _as_bool(row.get("ccrp_rank1_all_metrics")),
        "min_delta": _as_float(row.get("min_delta")),
        "max_holm_p_value": _as_float(row.get("max_holm_p_value")),
    }


def _audit_method_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    grouped = _method_rows_by_domain(rows)
    expected_methods = set(OFFICIAL_METHODS) | {CCRP_METHOD}
    official_count = 0
    ccrp_count = 0
    for domain in REQUIRED_DOMAINS:
        domain_rows = grouped.get(domain, [])
        _require(len(domain_rows) == len(expected_methods), failures, f"{domain}:method_row_count_not_9")
        observed_methods = {row.get("method", "") for row in domain_rows}
        if observed_methods != expected_methods:
            failures.append(f"{domain}:method_set_mismatch:{sorted(observed_methods)}")
        for row in domain_rows:
            method = str(row.get("method", ""))
            kind = str(row.get("kind", ""))
            artifact_class = str(row.get("artifact_class", ""))
            status_label = str(row.get("status_label", ""))
            if artifact_class != "completed_result":
                failures.append(f"{domain}/{method}:artifact_class_not_completed_result")
            if not status_label:
                failures.append(f"{domain}/{method}:missing_status_label")
            for metric in REQUIRED_METRICS:
                if not _is_finite_number(row.get(metric)):
                    failures.append(f"{domain}/{method}:missing_metric:{metric}")
            if _as_int(row.get("sample_count")) != EXPECTED_USERS:
                failures.append(f"{domain}/{method}:sample_count_not_10000")
            if abs(_as_float(row.get("avg_candidates")) - EXPECTED_CANDIDATES_PER_USER) >= 1e-9:
                failures.append(f"{domain}/{method}:avg_candidates_not_101")
            if abs(_as_float(row.get("score_coverage_rate")) - 1.0) >= 1e-12:
                failures.append(f"{domain}/{method}:score_coverage_not_1")
            if _as_int(row.get("scores_csv_lines")) != EXPECTED_SCORE_CSV_LINES:
                failures.append(f"{domain}/{method}:scores_csv_lines_not_1010001")
            if _as_int(row.get("predictions_jsonl_lines")) != EXPECTED_USERS:
                failures.append(f"{domain}/{method}:predictions_jsonl_lines_not_10000")
            if _as_int(row.get("ranking_eval_records_csv_lines")) != EXPECTED_EVAL_CSV_LINES:
                failures.append(f"{domain}/{method}:ranking_eval_records_lines_not_10001")
            if method == CCRP_METHOD:
                ccrp_count += 1
                if kind != "internal_method":
                    failures.append(f"{domain}/{method}:kind_not_internal_method")
                if status_label != "same_schema_internal_method":
                    failures.append(f"{domain}/{method}:status_not_same_schema_internal_method")
                if _as_bool(row.get("ccrp_local_event_restat_ready")) is not True:
                    warnings.append(f"{domain}/{method}:local_event_restat_not_self_contained")
            else:
                official_count += 1
                if method not in OFFICIAL_METHODS:
                    failures.append(f"{domain}/{method}:unexpected_official_method")
                if kind != "official_baseline":
                    failures.append(f"{domain}/{method}:kind_not_official_baseline")
                if status_label != "same_schema_external_baseline":
                    failures.append(f"{domain}/{method}:status_not_same_schema_external_baseline")
                if _as_bool(row.get("official_light_package_ok")) is not True:
                    failures.append(f"{domain}/{method}:official_light_package_not_ok")
                evidence_path = str(row.get("evidence_path", ""))
                if not evidence_path or not Path(evidence_path).exists():
                    failures.append(f"{domain}/{method}:missing_local_evidence_path")
    _require(len(rows) == len(REQUIRED_DOMAINS) * len(expected_methods), failures, "method_rows_total_not_36")
    _require(official_count == len(REQUIRED_DOMAINS) * len(OFFICIAL_METHODS), failures, "official_method_rows_total_not_32")
    _require(ccrp_count == len(REQUIRED_DOMAINS), failures, "ccrp_method_rows_total_not_4")
    return {
        "ok": not failures,
        "row_count": len(rows),
        "official_row_count": official_count,
        "ccrp_row_count": ccrp_count,
        "failures": failures,
        "warnings": warnings,
    }


def _audit_summary(summary: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    _require(summary.get("all_blocking_checks_ok") is True, failures, "summary_all_blocking_checks_not_ok")
    if list(summary.get("domain_order") or []) != list(REQUIRED_DOMAINS):
        failures.append("summary_domain_order_mismatch")
    if summary.get("aris_verdict") != "CONDITIONAL_PASS_FOR_COMPACT_FOUR_NEW_DOMAIN_GATE_SUMMARY":
        warnings.append(f"unexpected_aris_verdict:{summary.get('aris_verdict')}")
    domains = summary.get("domains") if isinstance(summary.get("domains"), list) else []
    _require(len(domains) == len(REQUIRED_DOMAINS), failures, "summary_domain_count_not_4")
    claim_not_supported = set(summary.get("claim_not_supported") or [])
    if "paper-ready SOTA" not in claim_not_supported:
        failures.append("summary_missing_paper_ready_sota_disclaimer")
    return {
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "aris_verdict": summary.get("aris_verdict"),
        "claim_supported": summary.get("claim_supported", ""),
        "claim_not_supported": sorted(claim_not_supported),
    }


def _audit_evidence_consistency(payload: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    _require(payload.get("ok") is True, failures, "evidence_consistency_not_ok")
    _require(_as_int(payload.get("row_count")) == len(REQUIRED_DOMAINS) * len(OFFICIAL_METHODS), failures, "evidence_row_count_not_32")
    _require(_as_int(payload.get("ok_count")) == len(REQUIRED_DOMAINS) * len(OFFICIAL_METHODS), failures, "evidence_ok_count_not_32")
    _require(_as_int(payload.get("failure_count")) == 0, failures, "evidence_failure_count_not_0")
    if list(payload.get("domains") or []) != list(REQUIRED_DOMAINS):
        failures.append("evidence_domains_mismatch")
    return {
        "ok": not failures,
        "failures": failures,
        "row_count": payload.get("row_count"),
        "ok_count": payload.get("ok_count"),
        "failure_count": payload.get("failure_count"),
    }


def _paper_critical_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"present": False, "paper_ready": None, "warnings": ["paper_critical_audit_not_supplied"]}
    summary = payload.get("summary") or {}
    warnings: list[str] = []
    if payload.get("paper_ready") is not False:
        warnings.append("paper_critical_paper_ready_not_false_or_missing")
    if summary.get("four_domain_evidence_consistent") is not True:
        warnings.append("paper_critical_four_domain_evidence_not_consistent")
    if summary.get("signal_rows_available") is not False:
        warnings.append("paper_critical_signal_row_status_unexpected")
    if summary.get("phase2_5_storage_launch_allowed") is not False:
        warnings.append("paper_critical_storage_gate_status_unexpected")
    return {
        "present": True,
        "paper_ready": payload.get("paper_ready"),
        "four_domain_evidence_consistent": summary.get("four_domain_evidence_consistent"),
        "signal_rows_available": summary.get("signal_rows_available"),
        "phase2_5_storage_launch_allowed": summary.get("phase2_5_storage_launch_allowed"),
        "warnings": warnings,
    }


def build_audit(
    *,
    summary_json: str | Path,
    domain_summary_csv: str | Path,
    method_rows_csv: str | Path,
    evidence_consistency_json: str | Path,
    paper_critical_json: str | Path | None = None,
) -> dict[str, Any]:
    summary = _read_json(summary_json)
    domain_rows = _read_csv(domain_summary_csv)
    method_rows = _read_csv(method_rows_csv)
    evidence = _read_json(evidence_consistency_json)
    paper_critical = _read_json(paper_critical_json) if paper_critical_json and Path(paper_critical_json).exists() else None

    summary_audit = _audit_summary(summary)
    domain_by_name = _domain_summary_by_domain(domain_rows)
    domain_audits = []
    failures: list[str] = []
    warnings: list[str] = []
    for domain in REQUIRED_DOMAINS:
        row = domain_by_name.get(domain)
        if not row:
            domain_audits.append({"domain": domain, "ok": False, "failures": [f"{domain}:missing_domain_summary_row"], "warnings": []})
            continue
        domain_audits.append(_audit_domain_row(domain, row))
    method_audit = _audit_method_rows(method_rows)
    evidence_audit = _audit_evidence_consistency(evidence)
    paper_summary = _paper_critical_summary(paper_critical)

    for section in [summary_audit, method_audit, evidence_audit, *domain_audits]:
        failures.extend(section.get("failures", []))
        warnings.extend(section.get("warnings", []))
    warnings.extend(paper_summary.get("warnings", []))

    comparison_certificate_ready = not failures
    return {
        "schema_version": "2026-06-06.cross_domain_official_ccrp_certificate_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "local_cross_domain_official_ccrp_certificate_audit",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "ok": comparison_certificate_ready,
        "comparison_certificate_ready": comparison_certificate_ready,
        "paper_ready": False,
        "summary_json": str(summary_json),
        "domain_summary_csv": str(domain_summary_csv),
        "method_rows_csv": str(method_rows_csv),
        "evidence_consistency_json": str(evidence_consistency_json),
        "paper_critical_json": str(paper_critical_json) if paper_critical_json else "",
        "domain_order": list(REQUIRED_DOMAINS),
        "required_metrics": list(REQUIRED_METRICS),
        "expected": {
            "domains": len(REQUIRED_DOMAINS),
            "official_methods_per_domain": len(OFFICIAL_METHODS),
            "total_official_rows": len(REQUIRED_DOMAINS) * len(OFFICIAL_METHODS),
            "methods_per_domain_including_ccrp": len(OFFICIAL_METHODS) + 1,
            "total_method_rows": len(REQUIRED_DOMAINS) * (len(OFFICIAL_METHODS) + 1),
            "paired_tests_per_domain": len(OFFICIAL_METHODS) * len(REQUIRED_METRICS),
            "users": EXPECTED_USERS,
            "candidates_per_user": EXPECTED_CANDIDATES_PER_USER,
        },
        "claim_supported": summary_audit.get("claim_supported"),
        "claim_not_supported": summary_audit.get("claim_not_supported"),
        "claim_scope": (
            "compact four-new-domain same-candidate official comparison certificate; "
            "not a paper-readiness verdict and not full-catalog SOTA"
        ),
        "summary_audit": summary_audit,
        "domain_audits": domain_audits,
        "method_rows_audit": method_audit,
        "evidence_consistency_audit": evidence_audit,
        "paper_critical_summary": paper_summary,
        "failures": failures,
        "warnings": warnings,
    }


def write_markdown(path: str | Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Cross-Domain Official+C-CRP Certificate Audit",
        "",
        f"- Generated UTC: `{audit['created_at_utc']}`",
        f"- OK: `{audit['ok']}`",
        f"- Comparison certificate ready: `{audit['comparison_certificate_ready']}`",
        f"- Paper ready: `{audit['paper_ready']}`",
        f"- Read only: `{audit['read_only']}`",
        f"- Will start experiment: `{audit['will_start_experiment']}`",
        "",
        "## Supported Claim",
        "",
        str(audit.get("claim_supported") or ""),
        "",
        "## Domain Gates",
        "",
        "| Domain | OK | Official rows | C-CRP rank-all | Positive Holm tests | Min delta | Max Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit["domain_audits"]:
        lines.append(
            f"| {row['domain']} | `{row['ok']}` | {row.get('official_ok_count')} | "
            f"`{row.get('ccrp_rank1_all_metrics')}` | {row.get('paired_positive_holm_significant_count')}/56 | "
            f"{float(row.get('min_delta') or 0):.6f} | {float(row.get('max_holm_p_value') or 0):.6g} |"
        )
    lines.extend(
        [
            "",
            "## Evidence Consistency",
            "",
            f"- Local/server evidence consistency OK: `{audit['evidence_consistency_audit']['ok']}`",
            f"- Official local-light rows OK: `{audit['evidence_consistency_audit'].get('ok_count')}/32`",
            f"- Method rows audited: `{audit['method_rows_audit']['row_count']}/36`",
            "",
            "## Paper Boundary",
            "",
            f"- Paper-critical audit paper_ready: `{audit['paper_critical_summary'].get('paper_ready')}`",
            f"- Signal rows available: `{audit['paper_critical_summary'].get('signal_rows_available')}`",
            f"- Phase 2.5 storage launch allowed: `{audit['paper_critical_summary'].get('phase2_5_storage_launch_allowed')}`",
            "",
            "## Failures",
            "",
        ]
    )
    lines.extend(f"- {failure}" for failure in audit["failures"])
    if not audit["failures"]:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in audit["warnings"])
    if not audit["warnings"]:
        lines.append("- none")
    lines.append("")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit = build_audit(
        summary_json=args.summary_json,
        domain_summary_csv=args.domain_summary_csv,
        method_rows_csv=args.method_rows_csv,
        evidence_consistency_json=args.evidence_consistency_json,
        paper_critical_json=args.paper_critical_json or None,
    )
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(args.output_md, audit)
    print(
        json.dumps(
            {
                "ok": audit["ok"],
                "comparison_certificate_ready": audit["comparison_certificate_ready"],
                "paper_ready": audit["paper_ready"],
                "failures": audit["failures"],
                "warning_count": len(audit["warnings"]),
            },
            indent=2,
        )
    )
    if not audit["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
