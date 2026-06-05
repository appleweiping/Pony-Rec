from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit.main_audit_ccrp_uncertainty_sources import audit_sources


SIGNAL_SYNC_MANIFEST = "signal_evidence_sync_manifest.json"
SIGNAL_PACKAGE_AUDIT = "signal_evidence_package_audit.json"
SIGNAL_ROW_SUFFIX = "_ccrp_signal_rows.csv"
SIGNAL_PROVENANCE_SUFFIX = "_ccrp_signal_rows_provenance.json"
REQUIRED_SIGNAL_COLUMNS = (
    "source_event_id",
    "user_id",
    "candidate_item_id",
    "item_id",
    "candidate_idx",
    "relevance_probability",
    "calibrated_relevance_probability",
    "evidence_support",
    "counterevidence_strength",
    "parse_success",
    "signal_schema_version",
)


REMOTE_MANIFEST_SCRIPT = r"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Any

SIGNAL_ROW_SUFFIX = "_ccrp_signal_rows.csv"
SIGNAL_PROVENANCE_SUFFIX = "_ccrp_signal_rows_provenance.json"
EXCLUDED_SUFFIXES = (
    ".bin", ".ckpt", ".npy", ".npz", ".parquet", ".pkl", ".pickle",
    ".pt", ".pth", ".safetensors", ".tar", ".tar.gz", ".zip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--max-metadata-bytes", type=int, required=True)
    parser.add_argument("--max-signal-bytes", type=int, required=True)
    parser.add_argument("--exclude-signal-rows", action="store_true")
    parser.add_argument("--extra-file", action="append", default=[])
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def signal_role(rel_path: str, *, include_signal_rows: bool) -> str:
    rel = rel_path.replace("\\", "/")
    name = Path(rel).name
    lower = name.lower()
    if any(lower.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
        return ""
    if lower.endswith(SIGNAL_ROW_SUFFIX):
        return "signal_rows" if include_signal_rows else ""
    if lower.endswith(SIGNAL_PROVENANCE_SUFFIX):
        return "signal_provenance"
    if lower.endswith("_parse_failures.jsonl"):
        return "parse_failures"
    if fnmatch.fnmatch(lower, "*source*audit*.json") or fnmatch.fnmatch(lower, "*source*audit*.csv"):
        return "source_audit"
    if fnmatch.fnmatch(lower, "*manifest*.json") or lower.endswith(".sha256"):
        return "manifest"
    if lower.endswith(".log") or lower in {"log_snippets.md", "log_snippets.txt"}:
        return "log"
    if lower in {"commands.md", "commands.txt", "run_config.json", "config.json"}:
        return "run_metadata"
    return ""


def add_file(payload: dict[str, Any], *, path: Path, rel_path: str, role: str, limit: int) -> None:
    size = path.stat().st_size
    if size > limit:
        payload["too_large"].append({"rel_path": rel_path, "remote_path": str(path), "role": role, "size": size, "limit": limit})
        return
    row: dict[str, Any] = {
        "rel_path": rel_path,
        "remote_path": str(path),
        "role": role,
        "size": size,
        "sha256": sha256_file(path),
    }
    if path.suffix.lower() in {".csv", ".jsonl"}:
        row["lines"] = line_count(path)
    payload["files"].append(row)


def main() -> int:
    args = parse_args()
    project = Path(args.project).expanduser()
    evidence_arg = Path(args.evidence).expanduser()
    evidence = evidence_arg if evidence_arg.is_absolute() else project / evidence_arg
    include_signal_rows = not args.exclude_signal_rows
    payload: dict[str, Any] = {
        "project": str(project),
        "evidence_dir": str(evidence),
        "exists": evidence.exists() and evidence.is_dir(),
        "include_signal_rows": include_signal_rows,
        "files": [],
        "excluded": [],
        "too_large": [],
    }
    if not payload["exists"]:
        print(json.dumps(payload, ensure_ascii=False))
        return 2
    for path in sorted(evidence.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(evidence).as_posix()
        role = signal_role(rel_path, include_signal_rows=include_signal_rows)
        if not role:
            payload["excluded"].append({"rel_path": rel_path, "size": path.stat().st_size})
            continue
        limit = args.max_signal_bytes if role == "signal_rows" else args.max_metadata_bytes
        add_file(payload, path=path, rel_path=rel_path, role=role, limit=limit)
    for extra in args.extra_file:
        extra_arg = Path(extra).expanduser()
        extra_path = extra_arg if extra_arg.is_absolute() else project / extra_arg
        if not extra_path.exists() or not extra_path.is_file():
            payload["excluded"].append({"rel_path": f"extras/{extra_path.name}", "remote_path": str(extra_path), "missing": True})
            continue
        role = signal_role(extra_path.name, include_signal_rows=False) or "extra"
        add_file(
            payload,
            path=extra_path,
            rel_path=f"extras/{extra_path.name}",
            role=role,
            limit=args.max_metadata_bytes,
        )
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if not payload["too_large"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sync and audit a completed Phase 2.5 C-CRP signal-row evidence package. "
            "The package is intended for full-scale valid/test signal rows, not toy runs."
        )
    )
    parser.add_argument("--remote_host", default="pony-rec-gpu")
    parser.add_argument("--remote_python", default="/home/ajifang/miniconda3/envs/qwen_vllm/bin/python")
    parser.add_argument("--remote_project", default="/home/ajifang/projects/pony-rec-rescue-shadow-v6")
    parser.add_argument("--remote_signal_dir", default="")
    parser.add_argument("--remote_extra_file", action="append", default=[])
    parser.add_argument("--local_package_dir", required=True)
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--audit_only", action="store_true")
    parser.add_argument("--exclude_signal_rows", action="store_true")
    parser.add_argument("--max_metadata_mb", type=float, default=80.0)
    parser.add_argument("--max_signal_rows_mb", type=float, default=900.0)
    parser.add_argument("--expected_events", type=int, default=10000)
    parser.add_argument("--expected_candidates_per_event", type=int, default=101)
    parser.add_argument("--candidate_items_path", default="")
    parser.add_argument("--source_audit_json", default="")
    parser.add_argument("--manifest_json", default="")
    parser.add_argument("--audit_json", default="")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def signal_role(rel_path: str, *, include_signal_rows: bool = True) -> str:
    rel = rel_path.replace("\\", "/")
    name = Path(rel).name.lower()
    if name.endswith(SIGNAL_ROW_SUFFIX):
        return "signal_rows" if include_signal_rows else ""
    if name.endswith(SIGNAL_PROVENANCE_SUFFIX):
        return "signal_provenance"
    if name.endswith("_parse_failures.jsonl"):
        return "parse_failures"
    if fnmatch.fnmatch(name, "*source*audit*.json") or fnmatch.fnmatch(name, "*source*audit*.csv"):
        return "source_audit"
    if fnmatch.fnmatch(name, "*manifest*.json") or name.endswith(".sha256"):
        return "manifest"
    if name.endswith(".log") or name in {"log_snippets.md", "log_snippets.txt"}:
        return "log"
    if name in {"commands.md", "commands.txt", "run_config.json", "config.json"}:
        return "run_metadata"
    return ""


def _run_remote_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if not args.remote_signal_dir:
        raise ValueError("--remote_signal_dir is required unless --audit_only is used")
    cmd = [
        "ssh",
        args.remote_host,
        args.remote_python,
        "-",
        "--project",
        args.remote_project,
        "--evidence",
        args.remote_signal_dir,
        "--max-metadata-bytes",
        str(int(args.max_metadata_mb * 1024 * 1024)),
        "--max-signal-bytes",
        str(int(args.max_signal_rows_mb * 1024 * 1024)),
    ]
    if args.exclude_signal_rows:
        cmd.append("--exclude-signal-rows")
    for path in args.remote_extra_file:
        cmd.extend(["--extra-file", path])
    result = subprocess.run(
        cmd,
        input=REMOTE_MANIFEST_SCRIPT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode not in {0, 3}:
        raise RuntimeError(
            "remote_signal_manifest_failed "
            f"returncode={result.returncode} stderr={result.stderr.strip()}"
        )
    payload = json.loads(result.stdout)
    if not payload.get("exists"):
        raise FileNotFoundError(f"remote_signal_dir_missing:{payload.get('evidence_dir')}")
    return payload


def _remote_source(host: str, remote_path: str) -> str:
    if any(token in remote_path for token in ("\n", "\r", "\0")):
        raise ValueError(f"unsafe_remote_path:{remote_path!r}")
    return f"{host}:{remote_path}"


def _copy_file(args: argparse.Namespace, item: dict[str, Any], local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["scp", "-p", _remote_source(args.remote_host, str(item["remote_path"])), str(local_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"scp_failed:{item['rel_path']}:returncode={result.returncode}:stderr={result.stderr.strip()}"
        )


def sync_package(args: argparse.Namespace) -> dict[str, Any]:
    local_dir = Path(args.local_package_dir).expanduser()
    if args.copy:
        local_dir.mkdir(parents=True, exist_ok=True)
    remote_manifest = _run_remote_manifest(args)
    checked_files: list[dict[str, Any]] = []
    failures: list[str] = []
    for item in remote_manifest.get("files", []):
        rel_path = str(item["rel_path"])
        local_path = local_dir / Path(rel_path)
        if args.copy:
            _copy_file(args, item, local_path)
        if not local_path.exists() or not local_path.is_file():
            failures.append(f"missing_local_file:{rel_path}")
            continue
        local_size = local_path.stat().st_size
        local_sha = sha256_file(local_path)
        size_ok = local_size == int(item["size"])
        sha_ok = local_sha == str(item["sha256"])
        if not size_ok:
            failures.append(f"size_mismatch:{rel_path}:{local_size}!={item['size']}")
        if not sha_ok:
            failures.append(f"sha256_mismatch:{rel_path}:{local_sha}!={item['sha256']}")
        checked_files.append(
            {
                "rel_path": rel_path,
                "role": item.get("role", ""),
                "remote_path": item.get("remote_path", ""),
                "local_path": str(local_path),
                "size": local_size,
                "sha256": local_sha,
                "expected_sha256": item["sha256"],
                "size_ok": size_ok,
                "sha256_ok": sha_ok,
                "lines": line_count(local_path) if local_path.suffix.lower() in {".csv", ".jsonl"} else None,
            }
        )
    for row in remote_manifest.get("too_large", []):
        failures.append(f"allowed_file_too_large:{row['rel_path']}:{row['size']}>{row['limit']}")
    manifest = {
        "ok": not failures,
        "copy": bool(args.copy),
        "remote_host": args.remote_host,
        "remote_project": args.remote_project,
        "remote_signal_dir": remote_manifest.get("evidence_dir"),
        "local_package_dir": str(local_dir),
        "include_signal_rows": not args.exclude_signal_rows,
        "allowed_file_count": len(remote_manifest.get("files", [])),
        "excluded_file_count": len(remote_manifest.get("excluded", [])),
        "excluded_files_preview": remote_manifest.get("excluded", [])[:50],
        "too_large": remote_manifest.get("too_large", []),
        "checked_files": checked_files,
        "failures": failures,
    }
    manifest_path = Path(args.manifest_json).expanduser() if args.manifest_json else local_dir / SIGNAL_SYNC_MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _find_one(base: Path, pattern: str, failures: list[str], label: str) -> Path | None:
    matches = [path for path in sorted(base.glob(pattern)) if path.is_file() and path.stat().st_size > 0]
    if len(matches) != 1:
        failures.append(f"expected_one_{label}:{len(matches)}")
        return None
    return matches[0]


def _as_int(value: Any, default: int = -1) -> int:
    try:
        parsed = float(str(value).strip())
    except Exception:
        return default
    return int(parsed) if parsed.is_integer() else default


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _sync_manifest_has_hash(manifest: dict[str, Any], rel_path: str) -> bool:
    for row in manifest.get("checked_files", []):
        if str(row.get("rel_path")) != rel_path:
            continue
        return bool(row.get("size_ok") is True and row.get("sha256_ok") is True and row.get("sha256"))
    return False


def _load_source_audit(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if isinstance(payload.get("sources"), list) and payload["sources"]:
        first = payload["sources"][0]
        return first if isinstance(first, dict) else {}
    return payload


def build_package_audit(
    *,
    package_dir: str | Path,
    expected_events: int = 10000,
    expected_candidates_per_event: int = 101,
    candidate_items_path: str = "",
    source_audit_json: str = "",
) -> dict[str, Any]:
    base = Path(package_dir).expanduser()
    failures: list[str] = []
    warnings: list[str] = []
    expected_rows = int(expected_events) * int(expected_candidates_per_event)

    if not base.exists() or not base.is_dir():
        return {
            "ok": False,
            "package_dir": str(base),
            "failures": ["missing_package_dir"],
            "warnings": [],
        }

    signal_path = _find_one(base, f"*{SIGNAL_ROW_SUFFIX}", failures, "signal_rows_csv")
    provenance_path = _find_one(base, f"*{SIGNAL_PROVENANCE_SUFFIX}", failures, "signal_provenance_json")
    manifest_path = base / SIGNAL_SYNC_MANIFEST
    if not manifest_path.exists() or manifest_path.stat().st_size <= 0:
        failures.append(f"missing_{SIGNAL_SYNC_MANIFEST}")
        sync_manifest: dict[str, Any] = {}
    else:
        try:
            sync_manifest = read_json(manifest_path)
        except Exception as exc:
            failures.append(f"sync_manifest_read_error:{exc}")
            sync_manifest = {}
        if sync_manifest.get("ok") is not True:
            failures.append("sync_manifest_not_ok")

    provenance: dict[str, Any] = {}
    if provenance_path:
        try:
            provenance = read_json(provenance_path)
        except Exception as exc:
            failures.append(f"signal_provenance_read_error:{exc}")
    if signal_path:
        rel_signal = signal_path.relative_to(base).as_posix()
        if not _sync_manifest_has_hash(sync_manifest, rel_signal):
            failures.append(f"sync_manifest_missing_hash_evidence:{rel_signal}")
        lines = line_count(signal_path)
        if lines != expected_rows + 1:
            failures.append(f"signal_rows_line_count:{lines}!={expected_rows + 1}")
        missing_columns = [col for col in REQUIRED_SIGNAL_COLUMNS if col not in csv_header(signal_path)]
        if missing_columns:
            failures.append(f"signal_rows_missing_columns:{','.join(missing_columns)}")
        expected_sha = str(provenance.get("signal_rows_sha256", "")).strip()
        actual_sha = sha256_file(signal_path)
        if expected_sha and actual_sha != expected_sha:
            failures.append(f"signal_rows_sha256_mismatch:{actual_sha}!={expected_sha}")
    if provenance_path:
        rel_prov = provenance_path.relative_to(base).as_posix()
        if not _sync_manifest_has_hash(sync_manifest, rel_prov):
            failures.append(f"sync_manifest_missing_hash_evidence:{rel_prov}")

    if provenance:
        checks = {
            "artifact_class": provenance.get("artifact_class") == "paper_critical_signal_rows",
            "status_label": provenance.get("status_label") == "ccrp_v3_recomputable_signal_rows_generated",
            "n_events": _as_int(provenance.get("n_events")) == expected_events,
            "expected_signal_rows": _as_int(provenance.get("expected_signal_rows")) == expected_rows,
            "n_signal_rows": _as_int(provenance.get("n_signal_rows")) == expected_rows,
            "prompt_count": _as_int(provenance.get("prompt_count")) == expected_rows,
            "meta_row_count": _as_int(provenance.get("meta_row_count")) == expected_rows,
            "raw_result_count": _as_int(provenance.get("raw_result_count")) == expected_rows,
            "expected_candidates_per_event": _as_int(provenance.get("expected_candidates_per_event"))
            == expected_candidates_per_event,
        }
        for key, ok in checks.items():
            if not ok:
                failures.append(f"signal_provenance_check_failed:{key}")
        parse_failure_rate = _as_float(provenance.get("parse_failure_rate"))
        max_parse_failure_rate = _as_float(provenance.get("max_parse_failure_rate"), 0.005)
        if not math.isfinite(parse_failure_rate):
            failures.append("parse_failure_rate_nonfinite")
        elif not math.isfinite(max_parse_failure_rate):
            failures.append("max_parse_failure_rate_nonfinite")
        elif parse_failure_rate > max_parse_failure_rate:
            failures.append(f"parse_failure_rate_exceeds_limit:{parse_failure_rate}>{max_parse_failure_rate}")
        if not str(provenance.get("git_commit", "")).strip():
            failures.append("missing_git_commit")
        if not str(provenance.get("data_sha256", "")).strip():
            failures.append("missing_data_sha256")

    source_audit_path = Path(source_audit_json).expanduser() if source_audit_json else None
    if source_audit_path is None or not source_audit_path.exists():
        source_candidates = [path for path in sorted(base.glob("*source*audit*.json")) if path.is_file()]
        source_audit_path = source_candidates[0] if source_candidates else None
    source_audit_summary: dict[str, Any] = {}
    if source_audit_path and source_audit_path.exists():
        try:
            source_audit_summary = _load_source_audit(source_audit_path)
            status = source_audit_summary.get("status")
            if status != "recomputable_signal_rows":
                failures.append(f"source_audit_status_not_recomputable:{status}")
            coverage = _as_float(source_audit_summary.get("candidate_key_coverage_rate"), -1.0)
            if coverage != 1.0:
                failures.append(f"source_audit_candidate_key_coverage_not_one:{coverage}")
            if _as_int(source_audit_summary.get("matched_candidate_keys")) != expected_rows:
                failures.append("source_audit_matched_candidate_keys_mismatch")
            if _as_int(source_audit_summary.get("extra_source_keys"), 0) != 0:
                failures.append("source_audit_extra_source_keys")
            if _as_int(source_audit_summary.get("missing_candidate_keys"), 0) != 0:
                failures.append("source_audit_missing_candidate_keys")
        except Exception as exc:
            failures.append(f"source_audit_read_error:{exc}")
    elif candidate_items_path and signal_path:
        payload = audit_sources(
            sources=[f"signal={signal_path}"],
            candidate_items_path=candidate_items_path,
            expected_events=expected_events,
            expected_candidates_per_event=expected_candidates_per_event,
        )
        source_audit_summary = payload.get("sources", [{}])[0]
        if source_audit_summary.get("status") != "recomputable_signal_rows":
            failures.append(f"candidate_items_audit_status:{source_audit_summary.get('status')}")
        if _as_float(source_audit_summary.get("candidate_key_coverage_rate"), -1.0) != 1.0:
            failures.append("candidate_items_audit_coverage_not_one")
    else:
        failures.append("missing_source_audit_json_or_candidate_items_path")

    return {
        "ok": not failures,
        "package_dir": str(base),
        "expected_events": expected_events,
        "expected_candidates_per_event": expected_candidates_per_event,
        "expected_signal_rows": expected_rows,
        "signal_rows_path": str(signal_path) if signal_path else "",
        "signal_provenance_path": str(provenance_path) if provenance_path else "",
        "sync_manifest_path": str(manifest_path),
        "source_audit_path": str(source_audit_path) if source_audit_path else "",
        "source_audit": source_audit_summary,
        "failures": failures,
        "warnings": warnings,
    }


def main() -> int:
    args = parse_args()
    manifest: dict[str, Any] | None = None
    if not args.audit_only:
        manifest = sync_package(args)
    audit = build_package_audit(
        package_dir=args.local_package_dir,
        expected_events=args.expected_events,
        expected_candidates_per_event=args.expected_candidates_per_event,
        candidate_items_path=args.candidate_items_path,
        source_audit_json=args.source_audit_json,
    )
    if manifest is not None:
        audit["sync_manifest_ok"] = manifest.get("ok")
    audit_path = Path(args.audit_json).expanduser() if args.audit_json else Path(args.local_package_dir) / SIGNAL_PACKAGE_AUDIT
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.quiet:
        print(json.dumps({"sync_manifest": manifest, "package_audit": audit}, indent=2, ensure_ascii=False))
    return 0 if audit["ok"] and (manifest is None or manifest.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
