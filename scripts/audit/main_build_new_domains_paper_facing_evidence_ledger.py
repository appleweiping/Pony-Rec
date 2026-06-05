from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DOMAINS = ("sports", "toys", "home", "tools")
METRICS = ("HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a paper-facing full-metric evidence ledger for the four new-domain "
            "official+C-CRP same-candidate comparison. This is local-only and does not "
            "run experiments."
        )
    )
    parser.add_argument(
        "--method_rows_csv",
        default="outputs/summary/new_domains_official_ccrp_cross_domain_20260605_method_rows.csv",
    )
    parser.add_argument(
        "--domain_summary_csv",
        default="outputs/summary/new_domains_official_ccrp_cross_domain_20260605_domain_summary.csv",
    )
    parser.add_argument(
        "--evidence_consistency_json",
        default=(
            "outputs/summary/paper_critical/"
            "local_server_evidence_consistency_new_domains_post_backfill_20260606.json"
        ),
    )
    parser.add_argument(
        "--certificate_audit_json",
        default="outputs/summary/paper_critical/cross_domain_official_ccrp_certificate_audit_20260606_0235.json",
    )
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    return parser.parse_args()


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


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _domain_map(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["domain"]: row for row in rows}


def _evidence_map(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload.get("rows") or []:
        if isinstance(row, dict):
            result[(str(row.get("domain")), str(row.get("method")))] = row
    return result


def _first_existing(base: Path, patterns: tuple[str, ...]) -> str:
    for pattern in patterns:
        matches = sorted(base.glob(pattern))
        for match in matches:
            if match.is_file():
                return str(match)
    return ""


def _load_provenance(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _read_json(path)
    except Exception:
        return {}


def _official_paths(local_dir: str) -> dict[str, str]:
    base = Path(local_dir)
    return {
        "fairness_provenance_path": str(base / "fairness_provenance.json") if (base / "fairness_provenance.json").exists() else "",
        "server_final_evidence_audit_path": str(base / "server_final_evidence_audit.json") if (base / "server_final_evidence_audit.json").exists() else "",
        "server_large_artifact_manifest_path": str(base / "server_large_artifact_manifest.json") if (base / "server_large_artifact_manifest.json").exists() else "",
        "light_evidence_sync_manifest_path": str(base / "light_evidence_sync_manifest.json") if (base / "light_evidence_sync_manifest.json").exists() else "",
        "local_light_evidence_audit_path": str(base / "local_light_evidence_audit.json") if (base / "local_light_evidence_audit.json").exists() else "",
        "score_audit_json_path": _first_existing(base, ("*score_audit.json",)),
        "score_audit_txt_path": _first_existing(base, ("*score_audit.txt", "*same_candidate_score_audit.txt")),
        "run_summary_json_path": _first_existing(base, ("*run_summary.json",)),
    }


def _line_count(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _ccrp_paths(domain: str) -> dict[str, str]:
    base = Path("outputs") / f"{domain}_large10000_100neg_ccrp_v3"
    imported = Path("outputs") / f"{domain}_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate"
    return {
        "ccrp_report_path": str(base / "report.json") if (base / "report.json").exists() else "",
        "ccrp_user_ranks_path": str(base / "user_ranks.jsonl") if (base / "user_ranks.jsonl").exists() else "",
        "local_ccrp_summary_path": str(imported / "tables" / "same_candidate_external_baseline_summary.csv")
        if (imported / "tables" / "same_candidate_external_baseline_summary.csv").exists()
        else "",
        "local_ccrp_eval_path": str(imported / "tables" / "ranking_eval_records.csv")
        if (imported / "tables" / "ranking_eval_records.csv").exists()
        else "",
        "local_ccrp_metrics_path": str(imported / "tables" / "ranking_metrics.csv")
        if (imported / "tables" / "ranking_metrics.csv").exists()
        else "",
        "local_ccrp_coverage_path": str(imported / "tables" / "external_score_coverage.csv")
        if (imported / "tables" / "external_score_coverage.csv").exists()
        else "",
    }


def _row_warning_text(warnings: list[str]) -> str:
    return ";".join(sorted(set(warnings)))


def _build_row(
    method_row: dict[str, str],
    *,
    domain_summary: dict[str, str],
    evidence_row: dict[str, Any] | None,
) -> dict[str, Any]:
    domain = method_row["domain"]
    method = method_row["method"]
    kind = method_row["kind"]
    warnings: list[str] = []
    output: dict[str, Any] = {
        "domain": domain,
        "method": method,
        "kind": kind,
        "rank_by_NDCG@10": method_row.get("rank_by_NDCG@10", ""),
        "status_label": method_row.get("status_label", ""),
        "artifact_class": method_row.get("artifact_class", ""),
        "sample_count": method_row.get("sample_count", ""),
        "avg_candidates": method_row.get("avg_candidates", ""),
        "score_coverage_rate": method_row.get("score_coverage_rate", ""),
        "ranking_events": method_row.get("ranking_events", ""),
        "total_candidates": method_row.get("total_candidates", ""),
        "matched_candidates": method_row.get("matched_candidates", ""),
        "scores_csv_lines": method_row.get("scores_csv_lines", ""),
        "predictions_jsonl_lines": method_row.get("predictions_jsonl_lines", ""),
        "predictions_jsonl_line_source": method_row.get("predictions_jsonl_line_source", ""),
        "ranking_eval_records_csv_lines": method_row.get("ranking_eval_records_csv_lines", ""),
        "source_summary_path": method_row.get("summary_path", ""),
        "source_eval_path": method_row.get("eval_path", ""),
        "source_scores_path": method_row.get("scores_path", ""),
        "primary_gate_json": domain_summary.get("primary_gate_json", ""),
        "post_cleanup_gate_json": domain_summary.get("post_cleanup_gate_json", ""),
        "comparison_csv": domain_summary.get("comparison_csv", ""),
        "paired_tests_csv": domain_summary.get("paired_tests_csv", ""),
        "paired_summary_json": domain_summary.get("paired_summary_json", ""),
        "gate_ok": domain_summary.get("gate_ok", ""),
        "claim_gate": domain_summary.get("claim_gate", ""),
        "ccrp_rank1_all_metrics": domain_summary.get("ccrp_rank1_all_metrics", ""),
        "paired_positive_holm_significant_count": domain_summary.get("paired_positive_holm_significant_count", ""),
        "paired_all_positive_holm_significant": domain_summary.get("paired_all_positive_holm_significant", ""),
        "min_delta": domain_summary.get("min_delta", ""),
        "min_ci_low": domain_summary.get("min_ci_low", ""),
        "max_holm_p_value": domain_summary.get("max_holm_p_value", ""),
        "local_evidence_dir": "",
        "implementation_status": "",
        "blocker_count": "",
        "blockers_json": "",
        "comparison_variant": "",
        "official_repo": "",
        "pinned_commit": "",
        "baseline_hyperparameter_source": "",
        "test_set_model_selection_allowed": "",
        "fairness_provenance_path": "",
        "server_final_evidence_audit_path": "",
        "server_large_artifact_manifest_path": "",
        "light_evidence_sync_manifest_path": "",
        "local_light_evidence_audit_path": "",
        "score_audit_json_path": "",
        "score_audit_txt_path": "",
        "run_summary_json_path": "",
        "official_light_package_ok": "",
        "local_server_evidence_consistency_ok": "",
        "ccrp_report_path": "",
        "ccrp_user_ranks_path": "",
        "local_ccrp_summary_path": "",
        "local_ccrp_eval_path": "",
        "local_ccrp_metrics_path": "",
        "local_ccrp_coverage_path": "",
        "ccrp_local_event_restat_ready": method_row.get("ccrp_local_event_restat_ready", ""),
        "table_eligibility": "",
        "row_warnings": "",
    }
    for metric in METRICS:
        output[metric] = method_row.get(metric, "")

    if method == CCRP_METHOD:
        paths = _ccrp_paths(domain)
        output.update(paths)
        user_rank_lines = _line_count(Path(paths["ccrp_user_ranks_path"])) if paths["ccrp_user_ranks_path"] else None
        eval_lines = _line_count(Path(paths["local_ccrp_eval_path"])) if paths["local_ccrp_eval_path"] else None
        local_event_ready = (
            bool(paths["ccrp_report_path"])
            and bool(paths["ccrp_user_ranks_path"])
            and bool(paths["local_ccrp_summary_path"])
            and bool(paths["local_ccrp_eval_path"])
            and bool(paths["local_ccrp_metrics_path"])
            and bool(paths["local_ccrp_coverage_path"])
            and user_rank_lines == 10_000
            and eval_lines == 10_001
        )
        output["ccrp_local_event_restat_ready"] = str(local_event_ready)
        output["implementation_status"] = "internal_completed"
        output["blocker_count"] = "0" if _as_bool(domain_summary.get("gate_ok")) is True else "1"
        output["blockers_json"] = "[]"
        output["comparison_variant"] = "same_schema_internal_method"
        output["table_eligibility"] = "main_internal_comparison_compact_certificate"
        if not paths["ccrp_report_path"]:
            warnings.append("missing_local_ccrp_report")
        if not paths["ccrp_user_ranks_path"]:
            warnings.append("missing_local_ccrp_user_ranks")
        if paths["ccrp_user_ranks_path"] and user_rank_lines != 10_000:
            warnings.append("ccrp_user_ranks_lines_not_10000")
        if paths["local_ccrp_eval_path"] and eval_lines != 10_001:
            warnings.append("ccrp_ranking_eval_records_lines_not_10001")
        for key in ("local_ccrp_summary_path", "local_ccrp_eval_path", "local_ccrp_metrics_path", "local_ccrp_coverage_path"):
            if not paths[key]:
                warnings.append(f"missing_{key}")
        if not local_event_ready:
            warnings.append("compact_certificate_not_self_contained_for_event_restat")
    else:
        evidence_dir = str(method_row.get("evidence_path") or (evidence_row or {}).get("local_dir") or "")
        output["local_evidence_dir"] = evidence_dir
        output["official_light_package_ok"] = method_row.get("official_light_package_ok", "")
        output["local_server_evidence_consistency_ok"] = str(bool(evidence_row and evidence_row.get("ok") is True))
        if evidence_dir:
            paths = _official_paths(evidence_dir)
            output.update(paths)
            provenance = _load_provenance(Path(paths["fairness_provenance_path"])) if paths["fairness_provenance_path"] else {}
            blockers = provenance.get("blockers") if isinstance(provenance.get("blockers"), list) else []
            output.update(
                {
                    "implementation_status": provenance.get("implementation_status", ""),
                    "blocker_count": len(blockers),
                    "blockers_json": _json_compact(blockers),
                    "comparison_variant": provenance.get("comparison_variant", ""),
                    "official_repo": provenance.get("official_repo", ""),
                    "pinned_commit": provenance.get("pinned_commit") or provenance.get("official_repo_commit", ""),
                    "baseline_hyperparameter_source": provenance.get("baseline_hyperparameter_source", ""),
                    "test_set_model_selection_allowed": provenance.get("test_set_model_selection_allowed", ""),
                }
            )
        if output["implementation_status"] != "official_completed":
            warnings.append("implementation_status_not_official_completed")
        if str(output["blocker_count"]) != "0":
            warnings.append("official_blockers_present")
        if output["local_server_evidence_consistency_ok"] != "True":
            warnings.append("local_server_evidence_consistency_not_ok")
        required_paths = (
            "fairness_provenance_path",
            "server_final_evidence_audit_path",
            "server_large_artifact_manifest_path",
            "light_evidence_sync_manifest_path",
            "score_audit_json_path",
            "score_audit_txt_path",
            "run_summary_json_path",
        )
        for key in required_paths:
            if not output.get(key):
                warnings.append(f"missing_{key}")
        output["table_eligibility"] = (
            "main_official_comparison_eligible"
            if not warnings
            or all(
                warning
                not in {
                    "implementation_status_not_official_completed",
                    "official_blockers_present",
                    "local_server_evidence_consistency_not_ok",
                }
                for warning in warnings
            )
            else "not_main_eligible"
        )

    output["row_warnings"] = _row_warning_text(warnings)
    return output


def build_ledger(
    *,
    method_rows_csv: str | Path,
    domain_summary_csv: str | Path,
    evidence_consistency_json: str | Path,
    certificate_audit_json: str | Path,
) -> dict[str, Any]:
    method_rows = _read_csv(method_rows_csv)
    domain_rows = _domain_map(_read_csv(domain_summary_csv))
    evidence_payload = _read_json(evidence_consistency_json)
    certificate = _read_json(certificate_audit_json)
    evidence_rows = _evidence_map(evidence_payload)
    rows = [
        _build_row(
            row,
            domain_summary=domain_rows.get(row["domain"], {}),
            evidence_row=evidence_rows.get((row["domain"], row["method"])),
        )
        for row in method_rows
    ]
    failures: list[str] = []
    warnings: list[str] = []
    if len(rows) != len(DOMAINS) * (len(OFFICIAL_METHODS) + 1):
        failures.append("ledger_row_count_not_36")
    if certificate.get("ok") is not True:
        failures.append("certificate_audit_not_ok")
    if evidence_payload.get("ok") is not True:
        failures.append("evidence_consistency_not_ok")
    official_rows = [row for row in rows if row["kind"] == "official_baseline"]
    ccrp_rows = [row for row in rows if row["method"] == CCRP_METHOD]
    if len(official_rows) != 32:
        failures.append("official_row_count_not_32")
    if len(ccrp_rows) != 4:
        failures.append("ccrp_row_count_not_4")
    for row in official_rows:
        if row["table_eligibility"] != "main_official_comparison_eligible":
            failures.append(f"{row['domain']}/{row['method']}:official_not_main_eligible")
    for row in ccrp_rows:
        if row["table_eligibility"] != "main_internal_comparison_compact_certificate":
            failures.append(f"{row['domain']}/{row['method']}:ccrp_not_compact_eligible")
        if row["row_warnings"]:
            warnings.append(f"{row['domain']}/{row['method']}:{row['row_warnings']}")
    domain_counts = {
        domain: {
            "rows": sum(1 for row in rows if row["domain"] == domain),
            "official_rows": sum(1 for row in official_rows if row["domain"] == domain),
            "ccrp_rows": sum(1 for row in ccrp_rows if row["domain"] == domain),
            "gate_ok": domain_rows.get(domain, {}).get("gate_ok", ""),
            "paired_positive_holm_significant_count": domain_rows.get(domain, {}).get(
                "paired_positive_holm_significant_count", ""
            ),
        }
        for domain in DOMAINS
    }
    return {
        "schema_version": "2026-06-06.paper_facing_evidence_ledger.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "local_new_domain_paper_facing_evidence_ledger",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "ok": not failures,
        "paper_ready": False,
        "comparison_ledger_ready": not failures,
        "row_count": len(rows),
        "official_row_count": len(official_rows),
        "ccrp_row_count": len(ccrp_rows),
        "domain_counts": domain_counts,
        "certificate_audit_ok": certificate.get("ok"),
        "local_server_evidence_consistency_ok": evidence_payload.get("ok"),
        "claim_supported": certificate.get("claim_supported"),
        "claim_scope": (
            "paper-facing full-metric evidence ledger for the four-domain same-candidate "
            "official comparison; not a paper-readiness verdict and not full-catalog SOTA"
        ),
        "claim_not_supported": certificate.get("claim_not_supported", []),
        "failures": failures,
        "warnings": warnings,
        "rows": rows,
    }


LEDGER_FIELDS = [
    "domain",
    "method",
    "kind",
    "rank_by_NDCG@10",
    *METRICS,
    "sample_count",
    "avg_candidates",
    "score_coverage_rate",
    "ranking_events",
    "total_candidates",
    "matched_candidates",
    "scores_csv_lines",
    "predictions_jsonl_lines",
    "predictions_jsonl_line_source",
    "ranking_eval_records_csv_lines",
    "status_label",
    "artifact_class",
    "implementation_status",
    "blocker_count",
    "blockers_json",
    "comparison_variant",
    "official_repo",
    "pinned_commit",
    "baseline_hyperparameter_source",
    "test_set_model_selection_allowed",
    "primary_gate_json",
    "comparison_csv",
    "paired_tests_csv",
    "paired_summary_json",
    "claim_gate",
    "gate_ok",
    "ccrp_rank1_all_metrics",
    "paired_positive_holm_significant_count",
    "paired_all_positive_holm_significant",
    "min_delta",
    "min_ci_low",
    "max_holm_p_value",
    "local_evidence_dir",
    "fairness_provenance_path",
    "server_final_evidence_audit_path",
    "server_large_artifact_manifest_path",
    "light_evidence_sync_manifest_path",
    "local_light_evidence_audit_path",
    "score_audit_json_path",
    "score_audit_txt_path",
    "run_summary_json_path",
    "official_light_package_ok",
    "local_server_evidence_consistency_ok",
    "ccrp_report_path",
    "ccrp_user_ranks_path",
    "local_ccrp_summary_path",
    "local_ccrp_eval_path",
    "local_ccrp_metrics_path",
    "local_ccrp_coverage_path",
    "ccrp_local_event_restat_ready",
    "table_eligibility",
    "row_warnings",
]


def write_markdown(path: str | Path, ledger: dict[str, Any]) -> None:
    lines = [
        "# New-Domain Paper-Facing Evidence Ledger",
        "",
        f"- Generated UTC: `{ledger['created_at_utc']}`",
        f"- OK: `{ledger['ok']}`",
        f"- Comparison ledger ready: `{ledger['comparison_ledger_ready']}`",
        f"- Paper ready: `{ledger['paper_ready']}`",
        f"- Rows: `{ledger['row_count']}`",
        f"- Official rows: `{ledger['official_row_count']}`",
        f"- C-CRP rows: `{ledger['ccrp_row_count']}`",
        f"- Certificate audit OK: `{ledger['certificate_audit_ok']}`",
        f"- Local/server evidence consistency OK: `{ledger['local_server_evidence_consistency_ok']}`",
        "",
        "## Supported Claim",
        "",
        str(ledger.get("claim_supported") or ""),
        "",
        "## Domain Counts",
        "",
        "| Domain | Rows | Official | C-CRP | Gate OK | Positive Holm Tests |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for domain, row in ledger["domain_counts"].items():
        lines.append(
            f"| {domain} | {row['rows']} | {row['official_rows']} | {row['ccrp_rows']} | "
            f"`{row['gate_ok']}` | {row['paired_positive_holm_significant_count']}/56 |"
        )
    lines.extend(["", "## Failures", ""])
    lines.extend(f"- {failure}" for failure in ledger["failures"])
    if not ledger["failures"]:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in ledger["warnings"])
    if not ledger["warnings"]:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This ledger supports the four new-domain same-candidate official comparison table. "
            "It does not make the paper ready because observation/motivation, C-CRP component "
            "ablation, and hyperparameter-curve modules still need full-scale uncertainty signal rows.",
            "",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ledger = build_ledger(
        method_rows_csv=args.method_rows_csv,
        domain_summary_csv=args.domain_summary_csv,
        evidence_consistency_json=args.evidence_consistency_json,
        certificate_audit_json=args.certificate_audit_json,
    )
    _write_csv(args.output_csv, ledger["rows"], LEDGER_FIELDS)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.output_md, ledger)
    print(
        json.dumps(
            {
                "ok": ledger["ok"],
                "comparison_ledger_ready": ledger["comparison_ledger_ready"],
                "paper_ready": ledger["paper_ready"],
                "row_count": ledger["row_count"],
                "failures": ledger["failures"],
                "warning_count": len(ledger["warnings"]),
            },
            indent=2,
        )
    )
    if not ledger["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
