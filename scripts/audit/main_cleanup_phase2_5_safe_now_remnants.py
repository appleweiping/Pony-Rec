from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAFE_NOW_TARGETS = (
    "outputs/baselines/paper_adapters/tools_large10000_100neg_llm2rec_official_adapter",
    "outputs/baselines/paper_adapters/tools_large10000_100neg_llmesr_official_adapter",
    "tmp_llm2rec_sync",
)
PROCESS_PATTERN = "pony-rec|ccrp|baseline|uncertainty"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manifest and optionally remove only audited low-yield Phase 2.5 "
            "safe-now remnants. The target list is intentionally fixed."
        )
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip_process_check", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _du_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def _target_manifest(root: Path, rel_path: str) -> dict[str, Any]:
    path = (root / rel_path).resolve()
    failures: list[str] = []
    if root != path and root not in path.parents:
        failures.append(f"outside_root:{path}")
    if rel_path not in SAFE_NOW_TARGETS:
        failures.append(f"unexpected_target:{rel_path}")
    if not path.exists():
        return {
            "rel_path": rel_path,
            "resolved_path": str(path),
            "present_before": False,
            "size_bytes": 0,
            "file_count": 0,
            "files": [],
            "failures": failures,
        }
    files = [path] if path.is_file() else [item for item in sorted(path.rglob("*")) if item.is_file()]
    return {
        "rel_path": rel_path,
        "resolved_path": str(path),
        "present_before": True,
        "size_bytes": _du_size(path),
        "file_count": len(files),
        "files": [
            {
                "rel_path": str(file_path.relative_to(root)),
                "size_bytes": file_path.stat().st_size,
                "sha256": _sha256(file_path),
            }
            for file_path in files
        ],
        "failures": failures,
    }


def _active_processes() -> list[str]:
    try:
        output = subprocess.check_output(["ps", "aux"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    rows: list[str] = []
    tokens = ("pony-rec", "ccrp", "baseline", "uncertainty")
    for line in output.splitlines():
        lower = line.lower()
        if "python" in lower and any(token in lower for token in tokens):
            rows.append(line)
    return rows


def cleanup_safe_now(
    *,
    root: str | Path = ".",
    output_json: str | Path,
    execute: bool = False,
    skip_process_check: bool = False,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    failures: list[str] = []
    process_rows = [] if skip_process_check else _active_processes()
    if process_rows:
        failures.append("active_project_python_processes_present")
    targets = [_target_manifest(repo, rel_path) for rel_path in SAFE_NOW_TARGETS]
    for target in targets:
        failures.extend(target["failures"])

    deleted: list[str] = []
    if execute and not failures:
        for target in targets:
            path = Path(target["resolved_path"])
            if not target["present_before"]:
                continue
            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()
            deleted.append(target["rel_path"])
    payload = {
        "schema_version": "2026-06-05.safe_now_cleanup.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classification": "SAFE_NOW_LOW_YIELD_DISPOSABLE_STAGING",
        "target_policy": "fixed_exact_target_list_only",
        "execute_requested": execute,
        "delete_performed": bool(execute and not failures),
        "active_processes": process_rows,
        "targets": targets,
        "total_size_bytes": sum(int(target["size_bytes"]) for target in targets),
        "deleted_targets": deleted,
        "post_delete_presence": {
            rel_path: (repo / rel_path).exists()
            for rel_path in SAFE_NOW_TARGETS
        },
        "failures": failures,
    }
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    payload = cleanup_safe_now(
        root=args.root,
        output_json=args.output_json,
        execute=args.execute,
        skip_process_check=args.skip_process_check,
    )
    print(
        json.dumps(
            {
                "ok": not payload["failures"],
                "delete_performed": payload["delete_performed"],
                "total_size_bytes": payload["total_size_bytes"],
                "deleted_targets": payload["deleted_targets"],
                "failures": payload["failures"],
            },
            indent=2,
        )
    )
    if payload["failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
