from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REMOTE_MANIFEST_SCRIPT = r"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Any


EXCLUDED_EXACT = {
    "scores.csv",
    "predictions/rank_predictions.jsonl",
}
EXCLUDED_PREFIXES = (
    "predictions/",
    "checkpoints/",
    "embeddings/",
    "calibrated/",
    "reranked/",
    "figures/",
)
EXCLUDED_SUFFIXES = (
    ".bin",
    ".ckpt",
    ".npy",
    ".npz",
    ".parquet",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".tar.gz",
    ".zip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--max-bytes", type=int, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_excluded(rel_path: str) -> bool:
    rel = rel_path.replace("\\", "/")
    lower = rel.lower()
    if rel in EXCLUDED_EXACT:
        return True
    if any(rel.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return True
    if any(lower.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
        return True
    return False


def is_allowed(rel_path: str) -> bool:
    rel = rel_path.replace("\\", "/")
    name = Path(rel).name
    if is_excluded(rel):
        return False
    if rel in {"fairness_provenance.json", "inspect_fairness_provenance.json"}:
        return True
    if rel.startswith("tables/"):
        return True
    patterns = [
        "*evidence*audit*.json",
        "*package*audit*.json",
        "*score*audit*.json",
        "*same_candidate_score_audit.txt",
        "*server_final_audit*.json",
        "*local_light_audit*.json",
        "*run_summary.json",
        "*manifest*.json",
        "*metadata*.json",
        "*.sha256",
        "*.log",
    ]
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def main() -> int:
    args = parse_args()
    project = Path(args.project).expanduser()
    evidence_arg = Path(args.evidence).expanduser()
    evidence = evidence_arg if evidence_arg.is_absolute() else project / evidence_arg
    payload: dict[str, Any] = {
        "project": str(project),
        "evidence_dir": str(evidence),
        "exists": evidence.exists() and evidence.is_dir(),
        "max_bytes": args.max_bytes,
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
        rel = path.relative_to(evidence).as_posix()
        size = path.stat().st_size
        if not is_allowed(rel):
            payload["excluded"].append({"rel_path": rel, "size": size})
            continue
        if size > args.max_bytes:
            payload["too_large"].append({"rel_path": rel, "size": size})
            continue
        payload["files"].append(
            {
                "rel_path": rel,
                "size": size,
                "sha256": sha256_file(path),
            }
        )
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if not payload["too_large"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy and verify a lightweight official-baseline evidence package from "
            "pony-rec-gpu. The allowlist keeps final provenance, audits, run summaries, "
            "and imported tables; large scores, predictions, embeddings, and checkpoints "
            "stay server-side."
        )
    )
    parser.add_argument("--remote_host", default="pony-rec-gpu")
    parser.add_argument("--remote_python", default="/home/ajifang/miniconda3/bin/python")
    parser.add_argument("--remote_project", default="/home/ajifang/projects/pony-rec-rescue-shadow-v6")
    parser.add_argument(
        "--remote_evidence_dir",
        required=True,
        help="Remote evidence dir, absolute or relative to --remote_project.",
    )
    parser.add_argument("--local_evidence_dir", required=True)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Actually copy allowed files. Without this flag, verify an existing local copy only.",
    )
    parser.add_argument("--max_file_mb", type=float, default=80.0)
    parser.add_argument(
        "--manifest_json",
        default="",
        help="Optional local JSON manifest path. Defaults to <local_evidence_dir>/light_evidence_sync_manifest.json.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_remote_manifest(args: argparse.Namespace) -> dict[str, Any]:
    max_bytes = int(args.max_file_mb * 1024 * 1024)
    cmd = [
        "ssh",
        args.remote_host,
        args.remote_python,
        "-",
        "--project",
        args.remote_project,
        "--evidence",
        args.remote_evidence_dir,
        "--max-bytes",
        str(max_bytes),
    ]
    result = subprocess.run(
        cmd,
        input=REMOTE_MANIFEST_SCRIPT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode not in {0, 3}:
        raise RuntimeError(
            "remote_manifest_failed "
            f"returncode={result.returncode} stderr={result.stderr.strip()}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"remote_manifest_json_error:{exc}: {result.stdout[:500]}") from exc
    if not payload.get("exists"):
        raise FileNotFoundError(f"remote_evidence_dir_missing:{payload.get('evidence_dir')}")
    return payload


def _remote_source(host: str, evidence_dir: str, rel_path: str) -> str:
    if any(token in rel_path for token in ("\n", "\r", "\0")):
        raise ValueError(f"unsafe_remote_relative_path:{rel_path!r}")
    remote_path = evidence_dir.rstrip("/") + "/" + rel_path
    return f"{host}:{remote_path}"


def _copy_file(args: argparse.Namespace, evidence_dir: str, rel_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "scp",
        "-p",
        _remote_source(args.remote_host, evidence_dir, rel_path),
        str(local_path),
    ]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"scp_failed:{rel_path}:returncode={result.returncode}:stderr={result.stderr.strip()}"
        )


def _verify_local_files(
    *,
    args: argparse.Namespace,
    remote_manifest: dict[str, Any],
    local_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    copied_or_checked: list[dict[str, Any]] = []
    failures: list[str] = []
    evidence_dir = str(remote_manifest["evidence_dir"])
    for item in remote_manifest.get("files", []):
        rel_path = str(item["rel_path"])
        local_path = local_dir / Path(rel_path)
        if args.copy:
            _copy_file(args, evidence_dir, rel_path, local_path)
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
        copied_or_checked.append(
            {
                "rel_path": rel_path,
                "local_path": str(local_path),
                "size": local_size,
                "sha256": local_sha,
                "size_ok": size_ok,
                "sha256_ok": sha_ok,
            }
        )
    return copied_or_checked, failures


def main() -> int:
    args = parse_args()
    local_dir = Path(args.local_evidence_dir).expanduser()
    if args.copy:
        local_dir.mkdir(parents=True, exist_ok=True)
    remote_manifest = _run_remote_manifest(args)
    checked_files, failures = _verify_local_files(
        args=args,
        remote_manifest=remote_manifest,
        local_dir=local_dir,
    )
    if remote_manifest.get("too_large"):
        failures.extend(f"allowed_file_too_large:{row['rel_path']}" for row in remote_manifest["too_large"])

    manifest = {
        "ok": not failures,
        "copy": bool(args.copy),
        "remote_host": args.remote_host,
        "remote_project": args.remote_project,
        "remote_evidence_dir": remote_manifest.get("evidence_dir"),
        "local_evidence_dir": str(local_dir),
        "allowed_file_count": len(remote_manifest.get("files", [])),
        "excluded_file_count": len(remote_manifest.get("excluded", [])),
        "excluded_files_preview": remote_manifest.get("excluded", [])[:50],
        "too_large": remote_manifest.get("too_large", []),
        "checked_files": checked_files,
        "failures": failures,
    }
    manifest_path = (
        Path(args.manifest_json).expanduser()
        if args.manifest_json
        else local_dir / "light_evidence_sync_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.quiet:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
