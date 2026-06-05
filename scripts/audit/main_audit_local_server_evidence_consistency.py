from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_LOCAL_ROOT = "outputs/baselines/official_adapters"
DEFAULT_OUTPUT_JSON = "outputs/summary/paper_critical/local_server_evidence_consistency_audit.json"
DEFAULT_OUTPUT_MD = "outputs/summary/paper_critical/local_server_evidence_consistency_audit.md"
OFFICIAL_METHODS = [
    "proex_profile",
    "promax_profile",
    "elmrec_graph",
    "llmemb",
    "irllrec_intent",
    "rlmrec_graphcl",
    "llm2rec_sasrec",
    "llmesr_sasrec",
]
REQUIRED_LOCAL_FILES = [
    "fairness_provenance.json",
    "server_final_evidence_audit.json",
    "server_large_artifact_manifest.json",
    "server_large_artifact_manifest.sha256",
    "light_evidence_sync_manifest.json",
    "tables/ranking_metrics.csv",
    "tables/same_candidate_external_baseline_summary.csv",
    "tables/external_score_coverage.csv",
    "tables/ranking_exposure_distribution.csv",
    "tables/ranking_eval_records.csv",
]
SERVER_ONLY_REL_PATHS = {
    "scores.csv",
    "predictions/rank_predictions.jsonl",
}
SERVER_ONLY_SUFFIXES = (
    ".bin",
    ".ckpt",
    ".npy",
    ".npz",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
)
REQUIRED_METRICS = ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]
PRIMARY_VARIANT = "official_code_qwen3base_default_hparams_declared_adaptation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit local lightweight official-baseline evidence packages against the "
            "copied server manifest evidence. This is read-only: no SSH, copy, delete, "
            "or experiment launch is performed."
        )
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--local_root", default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output_md", default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--expected_users", type=int, default=10000)
    parser.add_argument("--expected_candidates_per_user", type=int, default=101)
    parser.add_argument("--allow_local_large", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _line_count(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _read_single_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one row in {path}, got {len(rows)}")
    return rows[0]


def _method_dir_name(domain: str, method: str) -> str:
    return f"{domain}_large10000_100neg_{method}_official_qwen3base_same_candidate"


def _is_server_only_large_rel_path(rel_path: str) -> bool:
    rel = rel_path.replace("\\", "/")
    return rel in SERVER_ONLY_REL_PATHS or rel.lower().endswith(SERVER_ONLY_SUFFIXES)


def _audit_checked_files(local_dir: Path, sync_manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if sync_manifest.get("ok") is not True:
        failures.append("light_sync_manifest_not_ok")
    if sync_manifest.get("failures"):
        failures.append("light_sync_manifest_has_failures")

    for row in sync_manifest.get("checked_files", []):
        rel_path = str(row.get("rel_path", "")).replace("\\", "/")
        if not rel_path:
            failures.append("light_sync_manifest_empty_rel_path")
            continue
        path = local_dir / Path(rel_path)
        if not path.exists() or not path.is_file():
            failures.append(f"missing_checked_local_file:{rel_path}")
            continue
        size = path.stat().st_size
        expected_size = int(row.get("size", -1))
        if size != expected_size:
            failures.append(f"checked_local_size_mismatch:{rel_path}:{size}!={expected_size}")
        expected_sha = str(row.get("sha256", ""))
        if expected_sha and sha256_file(path) != expected_sha:
            failures.append(f"checked_local_sha256_mismatch:{rel_path}")
    return failures


def _audit_required_local_files(local_dir: Path, expected_users: int, expected_candidates_per_user: int) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    files: dict[str, Any] = {}
    for rel_path in REQUIRED_LOCAL_FILES:
        path = local_dir / Path(rel_path)
        row: dict[str, Any] = {
            "present": path.exists() and path.is_file() and path.stat().st_size > 0,
            "size": path.stat().st_size if path.exists() else 0,
        }
        if row["present"] and path.suffix in {".csv", ".jsonl"}:
            row["lines"] = _line_count(path)
        files[rel_path] = row
        if not row["present"]:
            failures.append(f"missing_required_local_file:{rel_path}")

    eval_records = files.get("tables/ranking_eval_records.csv", {})
    if eval_records.get("present") and eval_records.get("lines") != expected_users + 1:
        failures.append("ranking_eval_records_line_count_mismatch")

    metrics_path = local_dir / "tables/ranking_metrics.csv"
    metrics: dict[str, Any] = {}
    if metrics_path.exists() and metrics_path.stat().st_size > 0:
        try:
            metrics = _read_single_csv(metrics_path)
            for metric in REQUIRED_METRICS:
                if metrics.get(metric, "") == "":
                    failures.append(f"missing_metric:{metric}")
            if int(_as_float(metrics.get("sample_count"), -1)) != expected_users:
                failures.append("ranking_metrics_sample_count_mismatch")
            if _as_float(metrics.get("avg_candidates"), -1) != float(expected_candidates_per_user):
                failures.append("ranking_metrics_avg_candidates_mismatch")
        except Exception as exc:
            failures.append(f"ranking_metrics_read_error:{exc}")

    files["metrics_summary"] = {key: metrics.get(key, "") for key in [*REQUIRED_METRICS, "sample_count", "avg_candidates"]}
    return failures, files


def _audit_provenance(local_dir: Path) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    summary: dict[str, Any] = {}
    path = local_dir / "fairness_provenance.json"
    if not path.exists() or path.stat().st_size <= 0:
        return ["missing_provenance_for_consistency_audit"], summary
    try:
        data = _load_json(path)
    except Exception as exc:
        return [f"provenance_read_error:{exc}"], summary

    blockers = data.get("blockers")
    summary = {
        "implementation_status": data.get("implementation_status"),
        "blocker_count": len(blockers) if isinstance(blockers, list) else "invalid",
        "score_coverage_rate": data.get("score_coverage_rate"),
        "comparison_variant": data.get("comparison_variant"),
        "score_schema": data.get("score_schema"),
    }
    if data.get("implementation_status") != "official_completed":
        failures.append("provenance_not_official_completed")
    if blockers != []:
        failures.append("provenance_blockers_not_empty")
    if _as_float(data.get("score_coverage_rate"), -1) != 1.0:
        failures.append("provenance_score_coverage_not_1")
    if data.get("comparison_variant") != PRIMARY_VARIANT:
        failures.append("provenance_comparison_variant_mismatch")
    return failures, summary


def _audit_server_manifest(local_dir: Path, *, allow_local_large: bool) -> tuple[list[str], list[str], dict[str, Any]]:
    failures: list[str] = []
    warnings: list[str] = []
    path = local_dir / "server_large_artifact_manifest.json"
    if not path.exists() or path.stat().st_size <= 0:
        return ["missing_server_large_artifact_manifest_json"], warnings, {}
    try:
        manifest = _load_json(path)
    except Exception as exc:
        return [f"server_large_artifact_manifest_read_error:{exc}"], warnings, {}

    if manifest.get("ok") is not True:
        failures.append("server_large_artifact_manifest_not_ok")
    if manifest.get("failures"):
        failures.append("server_large_artifact_manifest_has_failures")
    rel_paths = {str(row.get("rel_path", "")).replace("\\", "/") for row in manifest.get("files", [])}
    for required in SERVER_ONLY_REL_PATHS:
        if required not in rel_paths:
            failures.append(f"server_manifest_missing_required_large_artifact:{required}")
    if int(manifest.get("model_artifact_count") or 0) <= 0 and manifest.get("require_model_artifact") is not False:
        failures.append("server_manifest_missing_model_artifact")

    local_large_present = []
    for rel_path in sorted(rel_paths):
        if not _is_server_only_large_rel_path(rel_path):
            continue
        local_path = local_dir / Path(rel_path)
        if local_path.exists():
            local_large_present.append(rel_path)
    if local_large_present and not allow_local_large:
        failures.extend(f"server_only_large_artifact_present_locally:{rel_path}" for rel_path in local_large_present)
    elif local_large_present:
        warnings.extend(f"server_only_large_artifact_present_locally:{rel_path}" for rel_path in local_large_present)

    summary = {
        "ok": manifest.get("ok"),
        "file_count": manifest.get("file_count"),
        "model_artifact_count": manifest.get("model_artifact_count"),
        "server_large_rel_paths": sorted(rel_paths),
        "local_large_present": local_large_present,
    }
    return failures, warnings, summary


def audit_package(
    *,
    local_dir: str | Path,
    expected_users: int = 10000,
    expected_candidates_per_user: int = 101,
    allow_local_large: bool = False,
) -> dict[str, Any]:
    base = Path(local_dir)
    failures: list[str] = []
    warnings: list[str] = []
    if not base.exists() or not base.is_dir():
        return {
            "local_dir": str(base),
            "ok": False,
            "failures": ["missing_local_evidence_dir"],
            "warnings": warnings,
        }

    required_failures, file_summary = _audit_required_local_files(base, expected_users, expected_candidates_per_user)
    failures.extend(required_failures)

    sync_path = base / "light_evidence_sync_manifest.json"
    sync_summary: dict[str, Any] = {}
    if sync_path.exists() and sync_path.stat().st_size > 0:
        try:
            sync_manifest = _load_json(sync_path)
            sync_summary = {
                "ok": sync_manifest.get("ok"),
                "allowed_file_count": sync_manifest.get("allowed_file_count"),
                "excluded_file_count": sync_manifest.get("excluded_file_count"),
                "remote_evidence_dir": sync_manifest.get("remote_evidence_dir"),
                "checked_file_count": len(sync_manifest.get("checked_files", [])),
            }
            failures.extend(_audit_checked_files(base, sync_manifest))
        except Exception as exc:
            failures.append(f"light_sync_manifest_read_error:{exc}")
    else:
        failures.append("missing_light_evidence_sync_manifest")

    provenance_failures, provenance = _audit_provenance(base)
    failures.extend(provenance_failures)

    server_failures, server_warnings, server_manifest = _audit_server_manifest(
        base,
        allow_local_large=allow_local_large,
    )
    failures.extend(server_failures)
    warnings.extend(server_warnings)

    return {
        "local_dir": str(base),
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "files": file_summary,
        "provenance": provenance,
        "light_sync_manifest": sync_summary,
        "server_large_artifact_manifest": server_manifest,
    }


def build_audit(
    *,
    root: str | Path = ".",
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    domains: list[str] | None = None,
    methods: list[str] | None = None,
    expected_users: int = 10000,
    expected_candidates_per_user: int = 101,
    allow_local_large: bool = False,
) -> dict[str, Any]:
    repo = Path(root)
    local_base = Path(local_root)
    if not local_base.is_absolute():
        local_base = repo / local_base
    selected_domains = [str(domain).strip() for domain in domains or [] if str(domain).strip()]
    selected_methods = [str(method).strip() for method in methods or [] if str(method).strip()] or OFFICIAL_METHODS
    failures: list[str] = []
    rows: list[dict[str, Any]] = []

    for domain in selected_domains:
        for method in selected_methods:
            local_dir = local_base / _method_dir_name(domain, method)
            audit = audit_package(
                local_dir=local_dir,
                expected_users=expected_users,
                expected_candidates_per_user=expected_candidates_per_user,
                allow_local_large=allow_local_large,
            )
            row = {
                "domain": domain,
                "method": method,
                **audit,
            }
            rows.append(row)
            failures.extend(f"{domain}/{method}:{failure}" for failure in audit.get("failures", []))

    if not selected_domains:
        failures.append("no_domains_selected")

    return {
        "schema_version": "2026-06-06.v1",
        "mode": "local_server_evidence_consistency_audit",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "root": str(repo),
        "local_root": str(local_base),
        "domains": selected_domains,
        "methods": selected_methods,
        "expected_users": expected_users,
        "expected_candidates_per_user": expected_candidates_per_user,
        "row_count": len(rows),
        "ok_count": sum(1 for row in rows if row.get("ok")),
        "failure_count": len(failures),
        "ok": not failures,
        "failures": failures,
        "rows": rows,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Local/Server Evidence Consistency Audit",
        "",
        f"- ok: `{str(audit.get('ok')).lower()}`",
        f"- mode: `{audit.get('mode')}`",
        f"- read_only: `{str(audit.get('read_only')).lower()}`",
        f"- domains: `{', '.join(audit.get('domains', []))}`",
        f"- rows: `{audit.get('ok_count')}/{audit.get('row_count')}` ok",
        f"- failures: `{audit.get('failure_count')}`",
        "",
        "| domain | method | ok | failures | server large artifacts | checked local files |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in audit.get("rows", []):
        server = row.get("server_large_artifact_manifest", {})
        sync = row.get("light_sync_manifest", {})
        failures = "<br>".join(row.get("failures", [])[:8])
        lines.append(
            "| {domain} | {method} | `{ok}` | {failures} | `{large}` | `{checked}` |".format(
                domain=row.get("domain", ""),
                method=row.get("method", ""),
                ok=str(row.get("ok")).lower(),
                failures=failures or "",
                large=server.get("file_count", ""),
                checked=sync.get("checked_file_count", ""),
            )
        )
    lines.append("")
    if audit.get("failures"):
        lines.extend(["## Failures", ""])
        lines.extend(f"- `{failure}`" for failure in audit["failures"][:100])
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    audit = build_audit(
        root=args.root,
        local_root=args.local_root,
        domains=args.domain,
        methods=args.method,
        expected_users=args.expected_users,
        expected_candidates_per_user=args.expected_candidates_per_user,
        allow_local_large=args.allow_local_large,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(audit), encoding="utf-8")
    if not args.quiet:
        print(json.dumps({k: audit[k] for k in ["ok", "row_count", "ok_count", "failure_count", "failures"]}, indent=2))
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
