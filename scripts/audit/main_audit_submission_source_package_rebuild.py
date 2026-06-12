from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


DEFAULT_SOURCE_PACKAGE_JSON = Path(
    "outputs/summary/paper_critical/submission_source_package_20260612.json"
)
DEFAULT_BUILD_DIR = Path("artifacts/submission_source_package_rebuild_20260612")
DEFAULT_OUTPUT_JSON = Path(
    "outputs/summary/paper_critical/submission_source_package_rebuild_20260612.json"
)
DEFAULT_OUTPUT_MD = Path(
    "outputs/summary/paper_critical/submission_source_package_rebuild_20260612.md"
)
OUTPUT_RE = re.compile(r"Output written on .*?\((\d+) pages?, (\d+) bytes\)")
WARNING_RE = re.compile(r"warning\$\s*--\s*(\d+)")
PDF_PAGE_RE = re.compile(rb"/Type\s*/Page\b")
STALE_BUILD_OUTPUTS = (
    "main.pdf",
    "main.log",
    "main.aux",
    "main.blg",
    "main.bbl",
    "main.out",
)


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


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


def _safe_remove_tree(path: Path, repo: Path, *, expected: Path) -> None:
    resolved = path.resolve()
    resolved.relative_to(repo.resolve())
    if resolved != expected.resolve():
        raise ValueError(f"refusing to remove unexpected path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {value}")
    return path


def _manifest_digest(records: list[dict[str, Any]]) -> str:
    lines = [
        f"{record['sha256']}  {record['path']}  {record.get('role') or ''}  {record['copied_size_bytes']}"
        for record in sorted(records, key=lambda item: item["path"])
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _verify_package_files(package: dict[str, Any], repo: Path) -> tuple[list[str], list[dict[str, Any]]]:
    failures: list[str] = []
    verified: list[dict[str, Any]] = []
    copied = package.get("copied_manifest") or {}
    files = copied.get("files") or []
    if not isinstance(files, list) or not files:
        return ["copied_manifest_missing_files"], []
    files_dir_raw = str((package.get("output") or {}).get("files_dir") or "")
    try:
        files_dir_rel = _safe_relative_path(files_dir_raw)
    except ValueError as exc:
        return [f"source_package_files_dir_unsafe:{exc}"], []
    files_dir = (repo / files_dir_rel).resolve()
    try:
        files_dir.relative_to(repo.resolve())
    except ValueError:
        return [f"source_package_files_dir_outside_repo:{files_dir_raw}"], []
    if files_dir.is_symlink():
        failures.append(f"source_package_files_dir_is_symlink:{files_dir_raw}")
    if not files_dir.exists() or not files_dir.is_dir():
        failures.append(f"source_package_files_dir_missing:{files_dir_raw}")

    for item in files:
        if not isinstance(item, dict):
            failures.append("invalid_copied_manifest_entry")
            continue
        package_path = repo / str(item.get("package_path") or "")
        rel_path = str(item.get("path") or "")
        if not rel_path or not package_path.exists() or not package_path.is_file():
            failures.append(f"package_file_missing:{rel_path}")
            continue
        try:
            package_path.resolve().relative_to(repo.resolve())
        except ValueError:
            failures.append(f"package_file_outside_repo:{rel_path}")
            continue
        sha = _sha256_file(package_path)
        if sha != item.get("sha256"):
            failures.append(f"package_file_sha256_mismatch:{rel_path}")
        size = package_path.stat().st_size
        if size != int(item.get("copied_size_bytes") or -1):
            failures.append(f"package_file_size_mismatch:{rel_path}")
        verified.append(
            {
                "path": rel_path,
                "role": item.get("role"),
                "sha256": sha,
                "copied_size_bytes": size,
                "package_path": _rel(package_path, repo),
            }
        )

    expected_tree = {str(item.get("path") or "").replace("\\", "/") for item in files}
    if files_dir.exists() and files_dir.is_dir():
        actual_tree = {
            str(path.relative_to(files_dir)).replace("\\", "/")
            for path in files_dir.rglob("*")
            if path.is_file()
        }
        if actual_tree != expected_tree:
            missing = sorted(expected_tree - actual_tree)
            extra = sorted(actual_tree - expected_tree)
            if missing:
                failures.append(f"source_package_tree_missing:{missing}")
            if extra:
                failures.append(f"source_package_tree_extra:{extra}")

    copied_sha = _manifest_digest(verified) if verified else ""
    if copied_sha != copied.get("manifest_sha256"):
        failures.append(f"package_manifest_sha256_mismatch:{copied_sha} != {copied.get('manifest_sha256')}")
    source_sha = (package.get("source_audit_crosscheck") or {}).get("source_manifest_sha256")
    if copied_sha != source_sha:
        failures.append(f"package_source_manifest_sha256_mismatch:{copied_sha} != {source_sha}")
    return failures, verified


def _copy_to_worktree(
    *,
    repo: Path,
    package: dict[str, Any],
    build_dir: Path,
    overwrite: bool,
) -> tuple[list[str], Path]:
    failures: list[str] = []
    files_dir_raw = str((package.get("output") or {}).get("files_dir") or "")
    try:
        files_dir = (repo / _safe_relative_path(files_dir_raw)).resolve()
    except ValueError as exc:
        failures.append(f"source_package_files_dir_unsafe:{exc}")
        return failures, work_dir
    work_dir = build_dir / "work"
    staging_dir = build_dir / ".work.tmp"
    try:
        build_dir.resolve().relative_to(repo.resolve())
    except ValueError:
        failures.append(f"build_dir_outside_repo:{build_dir}")
        return failures, work_dir
    if work_dir.exists() and any(work_dir.iterdir()) and not overwrite:
        failures.append(f"work_dir_not_empty:{_rel(work_dir, repo)}")
        return failures, work_dir
    if not files_dir.exists() or not files_dir.is_dir():
        failures.append(f"source_package_files_dir_missing:{_rel(files_dir, repo)}")
        return failures, work_dir
    if files_dir.is_symlink():
        failures.append(f"source_package_files_dir_is_symlink:{_rel(files_dir, repo)}")
        return failures, work_dir

    try:
        _safe_remove_tree(staging_dir, repo, expected=staging_dir)
        build_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(files_dir, staging_dir)
        if failures:
            _safe_remove_tree(staging_dir, repo, expected=staging_dir)
        else:
            if work_dir.exists():
                _safe_remove_tree(work_dir, repo, expected=work_dir)
            build_dir.mkdir(parents=True, exist_ok=True)
            staging_dir.rename(work_dir)
    except Exception as exc:
        failures.append(f"copy_to_worktree_exception:{type(exc).__name__}:{exc}")
        try:
            _safe_remove_tree(staging_dir, repo, expected=staging_dir)
        except Exception:
            pass
    return failures, work_dir


def _run_latex_sequence(
    *,
    paper_dir: Path,
    runner: CommandRunner,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    resolved = {name: shutil.which(name) or "" for name in ["pdflatex", "bibtex"]}
    commands = [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ]
    records: list[dict[str, Any]] = []
    for command in commands:
        started = datetime.now(timezone.utc).isoformat()
        result = runner(
            command,
            cwd=paper_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
        records.append(
            {
                "command": command,
                "resolved_executable": resolved.get(command[0], ""),
                "started_at_utc": started,
                "returncode": result.returncode,
                "stdout_tail": (result.stdout or "")[-2000:],
                "stderr_tail": (result.stderr or "")[-2000:],
            }
        )
        if result.returncode != 0:
            break
    return records


def _parse_build_state(paper_dir: Path, repo: Path) -> dict[str, Any]:
    pdf = paper_dir / "main.pdf"
    log = paper_dir / "main.log"
    blg = paper_dir / "main.blg"
    log_text = _read_text(log)
    blg_text = _read_text(blg)
    output_matches = OUTPUT_RE.findall(log_text)
    page_count = int(output_matches[-1][0]) if output_matches else None
    logged_pdf_bytes = int(output_matches[-1][1]) if output_matches else None
    warning_match = WARNING_RE.search(blg_text)
    bibtex_warning_count = int(warning_match.group(1)) if warning_match else None
    actual_pdf_pages = len(PDF_PAGE_RE.findall(pdf.read_bytes())) if pdf.exists() else None
    return {
        "paper_dir": _rel(paper_dir, repo),
        "pdf": {
            "path": _rel(pdf, repo),
            "exists": pdf.exists(),
            "size_bytes": pdf.stat().st_size if pdf.exists() else 0,
            "sha256": _sha256_file(pdf) if pdf.exists() else "",
        },
        "log": {"path": _rel(log, repo), "exists": log.exists(), "size_bytes": log.stat().st_size if log.exists() else 0},
        "blg": {"path": _rel(blg, repo), "exists": blg.exists(), "size_bytes": blg.stat().st_size if blg.exists() else 0},
        "page_count": page_count,
        "actual_pdf_page_count": actual_pdf_pages,
        "logged_pdf_bytes": logged_pdf_bytes,
        "bibtex_warning_count": bibtex_warning_count,
        "undefined_citation_count": len(re.findall(r"Citation `[^']+' undefined", log_text)),
        "undefined_reference_count": len(re.findall(r"Reference `[^']+' undefined", log_text)),
        "rerun_warning_count": log_text.count("Rerun to get cross-references right"),
        "overfull_hbox_count": log_text.count("Overfull \\hbox"),
        "underfull_hbox_count": log_text.count("Underfull \\hbox"),
        "underfull_vbox_count": log_text.count("Underfull \\vbox"),
    }


def audit_submission_source_package_rebuild(
    *,
    root: str | Path = ".",
    source_package_json: str | Path = DEFAULT_SOURCE_PACKAGE_JSON,
    build_dir: str | Path = DEFAULT_BUILD_DIR,
    overwrite: bool = False,
    max_pages: int = 9,
    max_overfull_hbox: int = 0,
    timeout_seconds: int = 90,
    runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    package_path = (repo / source_package_json).resolve()
    build_root = (repo / build_dir).resolve()
    failures: list[str] = []
    warnings: list[str] = []
    package = _read_json(package_path)

    if package.get("ok") is not True:
        failures.append("source_package_not_ok")
    if package.get("submission_source_package_ready") is not True:
        failures.append("source_package_not_ready")
    if package.get("final_submission_ready") is not False:
        failures.append("source_package_unexpectedly_final_ready")
    if package.get("failures") != []:
        failures.append("source_package_failures_not_empty_or_missing")

    verify_failures, verified_files = _verify_package_files(package, repo)
    failures.extend(verify_failures)
    copy_failures: list[str] = []
    work_dir = build_root / "work"
    commands: list[dict[str, Any]] = []
    build_state: dict[str, Any] = {}
    removed_stale_outputs: list[dict[str, Any]] = []
    if not failures:
        copy_failures, work_dir = _copy_to_worktree(
            repo=repo,
            package=package,
            build_dir=build_root,
            overwrite=overwrite,
        )
        failures.extend(copy_failures)
    if not failures:
        paper_dir = work_dir / "Paper"
        if not (paper_dir / "main.tex").exists():
            failures.append("worktree_missing_paper_main_tex")
        else:
            for name in STALE_BUILD_OUTPUTS:
                path = paper_dir / name
                if path.exists():
                    removed_stale_outputs.append(
                        {"path": _rel(path, repo), "size_bytes": path.stat().st_size}
                    )
                    path.unlink()
            commands = _run_latex_sequence(
                paper_dir=paper_dir,
                runner=runner,
                timeout_seconds=timeout_seconds,
            )
            failed_commands = [record for record in commands if record["returncode"] != 0]
            if failed_commands:
                failures.append(f"latex_command_failed:{failed_commands[0]['command'][0]}")
            build_state = _parse_build_state(paper_dir, repo)

    if build_state:
        if not build_state["pdf"]["exists"] or build_state["pdf"]["size_bytes"] <= 100_000:
            failures.append("rebuilt_pdf_missing_or_too_small")
        if build_state["page_count"] is None:
            failures.append("rebuilt_page_count_not_found")
        elif build_state["page_count"] > max_pages:
            failures.append(f"rebuilt_page_count_exceeds_limit:{build_state['page_count']} > {max_pages}")
        if build_state["actual_pdf_page_count"] is None:
            failures.append("rebuilt_actual_pdf_page_count_not_found")
        elif build_state["actual_pdf_page_count"] != build_state["page_count"]:
            failures.append(
                "rebuilt_actual_pdf_page_count_mismatch:"
                f"{build_state['actual_pdf_page_count']} != {build_state['page_count']}"
            )
        if (
            build_state["logged_pdf_bytes"] is not None
            and build_state["pdf"]["exists"]
            and build_state["logged_pdf_bytes"] != build_state["pdf"]["size_bytes"]
        ):
            failures.append(
                "rebuilt_logged_pdf_bytes_mismatch:"
                f"{build_state['logged_pdf_bytes']} != {build_state['pdf']['size_bytes']}"
            )
        if build_state["bibtex_warning_count"] is None:
            failures.append("rebuilt_bibtex_warning_count_not_found")
        elif build_state["bibtex_warning_count"] != 0:
            failures.append(f"rebuilt_bibtex_warnings:{build_state['bibtex_warning_count']}")
        if build_state["undefined_citation_count"]:
            failures.append(f"rebuilt_undefined_citations:{build_state['undefined_citation_count']}")
        if build_state["undefined_reference_count"]:
            failures.append(f"rebuilt_undefined_references:{build_state['undefined_reference_count']}")
        if build_state["rerun_warning_count"]:
            failures.append(f"rebuilt_latex_rerun_warnings:{build_state['rerun_warning_count']}")
        if build_state["overfull_hbox_count"] > max_overfull_hbox:
            failures.append(
                f"rebuilt_overfull_hbox_count:{build_state['overfull_hbox_count']} > {max_overfull_hbox}"
            )
        if build_state["underfull_hbox_count"] or build_state["underfull_vbox_count"]:
            warnings.append(
                "rebuilt_underfull_layout_warnings:"
                f"hbox={build_state['underfull_hbox_count']},vbox={build_state['underfull_vbox_count']}"
            )

    ok = not failures
    return {
        "schema_version": "2026-06-12.submission_source_package_rebuild.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_rebuild_from_staged_anonymous_source_package",
        "read_only": False,
        "will_ssh": False,
        "will_copy": True,
        "will_delete": bool(overwrite),
        "will_start_experiment": False,
        "ok": ok,
        "submission_source_package_rebuild_ready": ok,
        "final_submission_ready": False,
        "input_paths": {"source_package_json": _rel(package_path, repo)},
        "output": {"build_dir": _rel(build_root, repo), "work_dir": _rel(work_dir, repo), "overwrite": overwrite},
        "source_package_crosscheck": {
            "source_package_ok": package.get("ok"),
            "source_package_ready": package.get("submission_source_package_ready"),
            "source_package_final_submission_ready": package.get("final_submission_ready"),
            "copied_manifest_sha256": (package.get("copied_manifest") or {}).get("manifest_sha256"),
            "source_manifest_sha256": (package.get("source_audit_crosscheck") or {}).get("source_manifest_sha256"),
            "verified_file_count": len(verified_files),
        },
        "commands": commands,
        "removed_stale_outputs_before_rebuild": removed_stale_outputs,
        "build": build_state,
        "warnings": warnings,
        "failures": failures,
        "remaining_blockers": list(package.get("remaining_blockers") or []),
        "next_actions": [
            "Treat this as a local rebuildability gate, not as final submission readiness.",
            "Re-run source package staging and this rebuild audit after any paper/source/BibTeX/figure/PDF change.",
            "Keep final_submission_ready=false until ProMax proceedings metadata and private manual submission-system gates close.",
        ],
    }


def _write_md(path: Path, audit: dict[str, Any]) -> None:
    build = audit.get("build") or {}
    pdf = build.get("pdf") or {}
    lines = [
        "# Submission Source Package Rebuild Audit",
        "",
        f"Generated: {audit['created_at_utc']}",
        "",
        f"- OK: `{str(audit['ok']).lower()}`",
        "- Submission source package rebuild ready: "
        f"`{str(audit['submission_source_package_rebuild_ready']).lower()}`",
        f"- Final submission ready: `{str(audit['final_submission_ready']).lower()}`",
        f"- Work dir: `{audit['output']['work_dir']}`",
        f"- Verified files: `{audit['source_package_crosscheck']['verified_file_count']}`",
        f"- PDF: `{pdf.get('path')}`, `{build.get('page_count')}` pages, `{pdf.get('size_bytes')}` bytes",
        f"- BibTeX warnings: `{build.get('bibtex_warning_count')}`",
        f"- Overfull hbox warnings: `{build.get('overfull_hbox_count')}`",
        f"- Underfull hbox/vbox warnings: `{build.get('underfull_hbox_count')}` / `{build.get('underfull_vbox_count')}`",
        "",
        "## Commands",
        "",
    ]
    for record in audit.get("commands") or []:
        lines.append(f"- `{' '.join(record['command'])}` -> `{record['returncode']}`")
    if not audit.get("commands"):
        lines.append("- None")
    lines.extend(["", "## Remaining Blockers", ""])
    blockers = audit.get("remaining_blockers") or []
    lines.extend(f"- {blocker}" for blocker in blockers) if blockers else lines.append("- None")
    lines.extend(["", "## Failures", ""])
    failures = audit.get("failures") or []
    lines.extend(f"- `{failure}`" for failure in failures) if failures else lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = audit.get("warnings") or []
    lines.extend(f"- `{warning}`" for warning in warnings) if warnings else lines.append("- None")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {action}" for action in audit.get("next_actions", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--source-package-json", default=str(DEFAULT_SOURCE_PACKAGE_JSON))
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-pages", type=int, default=9)
    parser.add_argument("--max-overfull-hbox", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = audit_submission_source_package_rebuild(
        root=args.root,
        source_package_json=args.source_package_json,
        build_dir=args.build_dir,
        overwrite=args.overwrite,
        max_pages=args.max_pages,
        max_overfull_hbox=args.max_overfull_hbox,
        timeout_seconds=args.timeout_seconds,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(Path(args.output_md), audit)
    if not args.output_json:
        print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
