from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SUBMISSION_PACKAGE_AUDIT = Path(
    "outputs/summary/paper_critical/submission_package_audit_20260612.json"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/submission_source_package_20260612")
DEFAULT_OUTPUT_JSON = Path("outputs/summary/paper_critical/submission_source_package_20260612.json")
DEFAULT_OUTPUT_MD = Path("outputs/summary/paper_critical/submission_source_package_20260612.md")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_PACKAGE_PATH_FRAGMENTS = (
    "private_confirmation",
    "manual_submission_private",
    "conflict",
    "coi",
    "reviewer_suggestion",
    "reviewer_exclusion",
    "submission_account",
)
ALLOWED_GENERATED_PAPER_ARTIFACTS = {
    "Paper/main.bbl",
    "Paper/main.pdf",
    "Paper/figures/framework_overview.pdf",
    "Paper/figures/framework_overview.svg",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe manifest path: {value}")
    return path


def _safe_remove_tree(path: Path, repo: Path, *, expected: Path) -> None:
    resolved = path.resolve()
    repo_resolved = repo.resolve()
    resolved.relative_to(repo_resolved)
    if resolved != expected.resolve():
        raise ValueError(f"refusing to remove unexpected path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _git_file_state(repo: Path, rel_path: str) -> dict[str, Any]:
    if not (repo / ".git").exists():
        return {
            "git_available": False,
            "tracked": None,
            "ignored": None,
            "allowed_generated_artifact": rel_path in ALLOWED_GENERATED_PAPER_ARTIFACTS,
        }
    tracked = (
        subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", rel_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )
    ignored = (
        subprocess.run(
            ["git", "-C", str(repo), "check-ignore", "-q", "--", rel_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )
    return {
        "git_available": True,
        "tracked": tracked,
        "ignored": ignored,
        "allowed_generated_artifact": rel_path in ALLOWED_GENERATED_PAPER_ARTIFACTS,
    }


def _validate_manifest_entry(entry: dict[str, Any], *, seen: set[str]) -> list[str]:
    failures: list[str] = []
    rel_path_raw = str(entry.get("path") or "")
    if not rel_path_raw:
        failures.append("manifest_entry_missing_path")
    elif rel_path_raw in seen:
        failures.append(f"duplicate_manifest_path:{rel_path_raw}")
    else:
        seen.add(rel_path_raw)
    role = str(entry.get("role") or "")
    if not role:
        failures.append(f"manifest_entry_missing_role:{rel_path_raw}")
    sha = str(entry.get("sha256") or "")
    if not SHA256_RE.match(sha):
        failures.append(f"manifest_entry_invalid_sha256:{rel_path_raw}")
    if entry.get("exists") is not True:
        failures.append(f"manifest_entry_not_marked_existing:{rel_path_raw}")
    try:
        size = int(entry.get("size_bytes"))
    except (TypeError, ValueError):
        failures.append(f"manifest_entry_invalid_size:{rel_path_raw}")
    else:
        if size <= 0:
            failures.append(f"manifest_entry_nonpositive_size:{rel_path_raw}")
    return failures


def _copy_manifest_file(
    *,
    repo: Path,
    package_root: Path,
    entry: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    rel_path_raw = str(entry.get("path") or "")
    if not rel_path_raw:
        return None, "manifest_entry_missing_path"
    normalized_raw = rel_path_raw.replace("\\", "/").lower()
    for fragment in FORBIDDEN_PACKAGE_PATH_FRAGMENTS:
        if fragment in normalized_raw:
            return None, f"forbidden_private_manifest_path:{rel_path_raw}"
    try:
        rel_path = _safe_relative_path(rel_path_raw)
    except ValueError as exc:
        return None, str(exc)

    source = (repo / rel_path).resolve()
    files_root = (package_root / "files").resolve()
    target = (files_root / rel_path).resolve()
    try:
        source.relative_to(repo.resolve())
        target.relative_to(files_root)
    except ValueError:
        return None, f"path_escape:{rel_path_raw}"
    if not source.exists() or not source.is_file():
        return None, f"source_file_missing:{rel_path_raw}"

    expected_sha = str(entry.get("sha256") or "")
    actual_sha = _sha256_file(source)
    if expected_sha and actual_sha != expected_sha:
        return None, f"source_sha256_mismatch:{rel_path_raw}"
    expected_size = int(entry.get("size_bytes") or 0)
    actual_size = source.stat().st_size
    if expected_size and actual_size != expected_size:
        return None, f"source_size_mismatch:{rel_path_raw}"
    if target.exists():
        return None, f"staging_target_exists:{_rel(target, repo)}"

    git_state = _git_file_state(repo, rel_path_raw.replace("\\", "/"))
    if git_state["git_available"] and not (
        git_state["tracked"] or git_state["allowed_generated_artifact"]
    ):
        return None, f"source_not_tracked_or_allowed_generated:{rel_path_raw}"
    if git_state["git_available"] and git_state["ignored"] and not (
        git_state["tracked"] or git_state["allowed_generated_artifact"]
    ):
        return None, f"source_ignored_without_allowlist:{rel_path_raw}"

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    copied_sha = _sha256_file(target)
    if copied_sha != actual_sha:
        return None, f"copied_sha256_mismatch:{rel_path_raw}"

    record = {
        "path": rel_path_raw,
        "role": entry.get("role"),
        "source_size_bytes": actual_size,
        "copied_size_bytes": target.stat().st_size,
        "sha256": copied_sha,
        "source_path": _rel(source, repo),
        "package_path": _rel(target, repo),
        "git_state": git_state,
    }
    return record, None


def _manifest_digest(records: list[dict[str, Any]]) -> str:
    lines = [
        f"{record['sha256']}  {record['path']}  {record.get('role') or ''}  {record['copied_size_bytes']}"
        for record in sorted(records, key=lambda item: item["path"])
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def build_submission_source_package(
    *,
    root: str | Path = ".",
    submission_package_audit_json: str | Path = DEFAULT_SUBMISSION_PACKAGE_AUDIT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
    allow_final_submission_ready: bool = False,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    audit_path = (repo / submission_package_audit_json).resolve()
    package_root = (repo / output_dir).resolve()
    staging_root = package_root.parent / f".{package_root.name}.tmp"

    failures: list[str] = []
    warnings: list[str] = []
    copied_files: list[dict[str, Any]] = []
    audit = _read_json(audit_path)

    try:
        package_root.relative_to(repo)
    except ValueError:
        failures.append(f"output_dir_outside_repo:{package_root}")

    if audit.get("ok") is not True:
        failures.append("submission_package_audit_not_ok")
    if audit.get("submission_package_ready_for_target_formatting") is not True:
        failures.append("submission_package_not_ready_for_target_formatting")
    if audit.get("final_submission_ready") is not False and not allow_final_submission_ready:
        failures.append("submission_package_audit_unexpectedly_final_ready")
    if audit.get("failures") != []:
        failures.append("submission_package_audit_failures_not_empty_or_missing")

    anonymity = audit.get("anonymity") or {}
    if anonymity.get("source_leak_scan_ok") is not True:
        failures.append("anonymous_source_leak_scan_not_ok")
    source_manifest = audit.get("source_package_manifest") or {}
    manifest_entries = source_manifest.get("files") or []
    if not isinstance(manifest_entries, list) or not manifest_entries:
        failures.append("source_package_manifest_missing_files")
        manifest_entries = []
    external_refs = source_manifest.get("external_source_references") or []
    if external_refs != []:
        failures.append("source_package_manifest_has_external_references")
    seen_paths: set[str] = set()
    for entry in manifest_entries:
        if not isinstance(entry, dict):
            failures.append("invalid_manifest_entry")
            continue
        failures.extend(_validate_manifest_entry(entry, seen=seen_paths))

    if package_root.exists() and any(package_root.iterdir()) and not overwrite:
        failures.append(f"output_dir_not_empty:{_rel(package_root, repo)}")

    if not failures:
        try:
            _safe_remove_tree(staging_root, repo, expected=staging_root)
            for entry in manifest_entries:
                record, error = _copy_manifest_file(
                    repo=repo,
                    package_root=staging_root,
                    entry=entry,
                )
                if error:
                    failures.append(error)
                elif record is not None:
                    copied_files.append(record)

            expected_tree = {str(entry["path"]).replace("\\", "/") for entry in manifest_entries}
            copied_tree = {
                str(path.relative_to(staging_root / "files")).replace("\\", "/")
                for path in (staging_root / "files").rglob("*")
                if path.is_file()
            }
            if copied_tree != expected_tree:
                missing = sorted(expected_tree - copied_tree)
                extra = sorted(copied_tree - expected_tree)
                if missing:
                    failures.append(f"staging_tree_missing:{missing}")
                if extra:
                    failures.append(f"staging_tree_extra:{extra}")

            if failures:
                _safe_remove_tree(staging_root, repo, expected=staging_root)
            else:
                if package_root.exists():
                    _safe_remove_tree(package_root, repo, expected=package_root)
                package_root.parent.mkdir(parents=True, exist_ok=True)
                staging_root.rename(package_root)
                for record in copied_files:
                    record["package_path"] = _rel(
                        package_root / "files" / Path(record["path"]), repo
                    )
        except Exception as exc:
            failures.append(f"package_build_exception:{type(exc).__name__}:{exc}")
            try:
                _safe_remove_tree(staging_root, repo, expected=staging_root)
            except Exception:
                pass

    expected_count = int(source_manifest.get("file_count") or 0)
    if not failures and expected_count and len(copied_files) != expected_count:
        failures.append(f"copied_file_count_mismatch:{len(copied_files)} != {expected_count}")

    copied_total_bytes = sum(int(item["copied_size_bytes"]) for item in copied_files)
    copied_manifest_sha = _manifest_digest(copied_files) if copied_files else ""
    expected_manifest_sha = str(source_manifest.get("manifest_sha256") or "")
    if not SHA256_RE.match(expected_manifest_sha):
        failures.append("source_manifest_invalid_sha256")
    elif not failures and copied_manifest_sha != expected_manifest_sha:
        failures.append(f"copied_manifest_sha256_mismatch:{copied_manifest_sha} != {expected_manifest_sha}")

    package_ready = not failures
    return {
        "schema_version": "2026-06-12.submission_source_package.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_copy_from_audited_source_manifest",
        "read_only": False,
        "will_ssh": False,
        "will_copy": package_ready,
        "will_delete": bool(overwrite),
        "will_start_experiment": False,
        "ok": package_ready,
        "submission_source_package_ready": package_ready,
        "final_submission_ready": False,
        "input_paths": {
            "submission_package_audit_json": _rel(audit_path, repo),
        },
        "output": {
            "package_dir": _rel(package_root, repo),
            "files_dir": _rel(package_root / "files", repo),
            "overwrite": overwrite,
        },
        "source_audit_crosscheck": {
            "submission_package_audit_ok": audit.get("ok"),
            "submission_package_ready_for_target_formatting": audit.get(
                "submission_package_ready_for_target_formatting"
            ),
            "audit_final_submission_ready": audit.get("final_submission_ready"),
            "anonymous_source_leak_scan_ok": anonymity.get("source_leak_scan_ok"),
            "source_manifest_file_count": source_manifest.get("file_count"),
            "source_manifest_total_bytes": source_manifest.get("total_bytes"),
            "source_manifest_sha256": expected_manifest_sha,
            "source_manifest_external_source_references": external_refs,
        },
        "copied_manifest": {
            "file_count": len(copied_files),
            "total_bytes": copied_total_bytes,
            "manifest_sha256": copied_manifest_sha,
            "files": copied_files,
        },
        "warnings": warnings,
        "failures": failures,
        "remaining_blockers": list(audit.get("remaining_blockers") or []),
        "next_actions": [
            "Use this package directory as the local anonymous source-package staging area.",
            "Do not add private confirmation files, author metadata, COI data, or submission-account data to this package.",
            "Rebuild this package after any Paper source, BibTeX, figure, or PDF change.",
            "Keep final_submission_ready=false until external proceedings metadata and manual submission-system gates are closed.",
        ],
    }


def _write_md(path: Path, package: dict[str, Any]) -> None:
    cross = package["source_audit_crosscheck"]
    copied = package["copied_manifest"]
    lines = [
        "# Submission Source Package",
        "",
        f"Generated: {package['created_at_utc']}",
        "",
        f"- OK: `{str(package['ok']).lower()}`",
        "- Submission source package ready: "
        f"`{str(package['submission_source_package_ready']).lower()}`",
        f"- Final submission ready: `{str(package['final_submission_ready']).lower()}`",
        f"- Package dir: `{package['output']['package_dir']}`",
        f"- Files dir: `{package['output']['files_dir']}`",
        f"- Source audit OK: `{str(cross['submission_package_audit_ok']).lower()}`",
        "- Anonymous source leak scan OK: "
        f"`{str(cross['anonymous_source_leak_scan_ok']).lower()}`",
        f"- Copied files: `{copied['file_count']}`",
        f"- Copied total bytes: `{copied['total_bytes']}`",
        f"- Copied manifest sha256: `{copied['manifest_sha256']}`",
        f"- Source manifest sha256: `{cross['source_manifest_sha256']}`",
        "",
        "## Copied Files",
        "",
    ]
    if copied["files"]:
        lines.extend(
            f"- `{item['path']}` ({item.get('role')}, {item['copied_size_bytes']} bytes)"
            for item in copied["files"]
        )
    else:
        lines.append("- None")
    lines.extend(["", "## Remaining Blockers", ""])
    blockers = package.get("remaining_blockers") or []
    lines.extend(f"- {blocker}" for blocker in blockers) if blockers else lines.append("- None")
    lines.extend(["", "## Failures", ""])
    failures = package.get("failures") or []
    lines.extend(f"- `{failure}`" for failure in failures) if failures else lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = package.get("warnings") or []
    lines.extend(f"- `{warning}`" for warning in warnings) if warnings else lines.append("- None")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {action}" for action in package.get("next_actions", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--submission-package-audit-json", default=str(DEFAULT_SUBMISSION_PACKAGE_AUDIT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-final-submission-ready", action="store_true")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package = build_submission_source_package(
        root=args.root,
        submission_package_audit_json=args.submission_package_audit_json,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        allow_final_submission_ready=args.allow_final_submission_ready,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(Path(args.output_md), package)
    return 0 if package["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
