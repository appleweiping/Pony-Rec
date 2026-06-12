import hashlib
import json
import subprocess
from pathlib import Path

from scripts.audit.main_audit_submission_source_package_rebuild import (
    audit_submission_source_package_rebuild,
)


def _write(path: Path, payload: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_sha(entries: list[dict]) -> str:
    lines = [
        f"{entry['sha256']}  {entry['path']}  {entry.get('role') or ''}  {entry['copied_size_bytes']}"
        for entry in sorted(entries, key=lambda item: item["path"])
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _seed_source_package(tmp_path: Path) -> Path:
    files_root = tmp_path / "artifacts" / "submission_source_package" / "files"
    main = _write(files_root / "Paper" / "main.tex", "\\title{X}\\begin{document}X\\end{document}\n")
    refs = _write(files_root / "Paper" / "references.bib", "@inproceedings{x,title={X}}\n")
    pdf = _write(files_root / "Paper" / "main.pdf", (b"/Type /Page\n" * 9) + b"stale")
    fig = _write(files_root / "Paper" / "figures" / "framework_overview.pdf", b"x" * 120_000)
    entries = [
        {
            "path": "Paper/main.tex",
            "role": "tex_source",
            "package_path": "artifacts/submission_source_package/files/Paper/main.tex",
            "copied_size_bytes": main.stat().st_size,
            "sha256": _sha(main),
        },
        {
            "path": "Paper/references.bib",
            "role": "bibliography_source",
            "package_path": "artifacts/submission_source_package/files/Paper/references.bib",
            "copied_size_bytes": refs.stat().st_size,
            "sha256": _sha(refs),
        },
        {
            "path": "Paper/main.pdf",
            "role": "compiled_pdf",
            "package_path": "artifacts/submission_source_package/files/Paper/main.pdf",
            "copied_size_bytes": pdf.stat().st_size,
            "sha256": _sha(pdf),
        },
        {
            "path": "Paper/figures/framework_overview.pdf",
            "role": "figure",
            "package_path": "artifacts/submission_source_package/files/Paper/figures/framework_overview.pdf",
            "copied_size_bytes": fig.stat().st_size,
            "sha256": _sha(fig),
        },
    ]
    manifest = _manifest_sha(entries)
    return _write_json(
        tmp_path / "source_package.json",
        {
            "ok": True,
            "submission_source_package_ready": True,
            "final_submission_ready": False,
            "failures": [],
            "output": {
                "files_dir": "artifacts/submission_source_package/files",
            },
            "copied_manifest": {
                "file_count": len(entries),
                "total_bytes": sum(entry["copied_size_bytes"] for entry in entries),
                "manifest_sha256": manifest,
                "files": entries,
            },
            "source_audit_crosscheck": {
                "source_manifest_sha256": manifest,
            },
            "remaining_blockers": ["external metadata"],
        },
    )


def _successful_runner(command, cwd, **kwargs):
    cwd = Path(cwd)
    if command[0] == "pdflatex":
        _write(
            cwd / "main.log",
            "Output written on main.pdf (9 pages, 120000 bytes).\n",
        )
        _write(cwd / "main.pdf", (b"/Type /Page\n" * 9) + (b"p" * 119_892))
    if command[0] == "bibtex":
        _write(cwd / "main.blg", "warning$ -- 0\n")
    return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")


def test_source_package_rebuild_passes_with_fake_latex_runner(tmp_path: Path) -> None:
    source_json = _seed_source_package(tmp_path)

    audit = audit_submission_source_package_rebuild(
        root=tmp_path,
        source_package_json=source_json.relative_to(tmp_path),
        build_dir="artifacts/rebuild",
        overwrite=True,
        runner=_successful_runner,
    )

    assert audit["ok"] is True
    assert audit["submission_source_package_rebuild_ready"] is True
    assert audit["final_submission_ready"] is False
    assert audit["source_package_crosscheck"]["verified_file_count"] == 4
    assert len(audit["commands"]) == 4
    assert audit["build"]["page_count"] == 9
    assert audit["build"]["actual_pdf_page_count"] == 9
    assert audit["build"]["bibtex_warning_count"] == 0
    assert audit["remaining_blockers"] == ["external metadata"]


def test_source_package_rebuild_rejects_manifest_hash_drift(tmp_path: Path) -> None:
    source_json = _seed_source_package(tmp_path)
    _write(
        tmp_path / "artifacts" / "submission_source_package" / "files" / "Paper" / "main.tex",
        "changed\n",
    )

    audit = audit_submission_source_package_rebuild(
        root=tmp_path,
        source_package_json=source_json.relative_to(tmp_path),
        build_dir="artifacts/rebuild",
        overwrite=True,
        runner=_successful_runner,
    )

    assert audit["ok"] is False
    assert "package_file_sha256_mismatch:Paper/main.tex" in audit["failures"]
    assert audit["commands"] == []


def test_source_package_rebuild_rejects_extra_unmanifested_staged_file(tmp_path: Path) -> None:
    source_json = _seed_source_package(tmp_path)
    _write(
        tmp_path
        / "artifacts"
        / "submission_source_package"
        / "files"
        / "Paper"
        / "private_notes.txt",
        "secret\n",
    )

    audit = audit_submission_source_package_rebuild(
        root=tmp_path,
        source_package_json=source_json.relative_to(tmp_path),
        build_dir="artifacts/rebuild",
        overwrite=True,
        runner=_successful_runner,
    )

    assert audit["ok"] is False
    assert "source_package_tree_extra:['Paper/private_notes.txt']" in audit["failures"]
    assert audit["commands"] == []


def test_source_package_rebuild_rejects_failed_latex_command(tmp_path: Path) -> None:
    source_json = _seed_source_package(tmp_path)

    def failing_runner(command, cwd, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")

    audit = audit_submission_source_package_rebuild(
        root=tmp_path,
        source_package_json=source_json.relative_to(tmp_path),
        build_dir="artifacts/rebuild",
        overwrite=True,
        runner=failing_runner,
    )

    assert audit["ok"] is False
    assert "latex_command_failed:pdflatex" in audit["failures"]
    assert "rebuilt_pdf_missing_or_too_small" in audit["failures"]


def test_source_package_rebuild_removes_stale_outputs_before_rebuild(tmp_path: Path) -> None:
    source_json = _seed_source_package(tmp_path)
    seen = {"pdflatex": 0, "bibtex": 0}

    def runner(command, cwd, **kwargs):
        cwd = Path(cwd)
        if command[0] == "pdflatex":
            seen["pdflatex"] += 1
            if seen["pdflatex"] == 1:
                assert not (cwd / "main.pdf").exists()
            _write(cwd / "main.log", "Output written on main.pdf (9 pages, 120000 bytes).\n")
            _write(cwd / "main.pdf", (b"/Type /Page\n" * 9) + (b"p" * 119_892))
        if command[0] == "bibtex":
            seen["bibtex"] += 1
            assert not (cwd / "main.blg").exists()
            _write(cwd / "main.blg", "warning$ -- 0\n")
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    audit = audit_submission_source_package_rebuild(
        root=tmp_path,
        source_package_json=source_json.relative_to(tmp_path),
        build_dir="artifacts/rebuild",
        overwrite=True,
        runner=runner,
    )

    removed = {Path(item["path"]).name for item in audit["removed_stale_outputs_before_rebuild"]}
    assert audit["ok"] is True
    assert {"main.pdf"}.issubset(removed)


def test_source_package_rebuild_rejects_pdf_page_mismatch(tmp_path: Path) -> None:
    source_json = _seed_source_package(tmp_path)

    def runner(command, cwd, **kwargs):
        cwd = Path(cwd)
        if command[0] == "pdflatex":
            _write(cwd / "main.log", "Output written on main.pdf (9 pages, 120000 bytes).\n")
            _write(cwd / "main.pdf", (b"/Type /Page\n" * 8) + (b"p" * 119_904))
        if command[0] == "bibtex":
            _write(cwd / "main.blg", "warning$ -- 0\n")
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    audit = audit_submission_source_package_rebuild(
        root=tmp_path,
        source_package_json=source_json.relative_to(tmp_path),
        build_dir="artifacts/rebuild",
        overwrite=True,
        runner=runner,
    )

    assert audit["ok"] is False
    assert "rebuilt_actual_pdf_page_count_mismatch:8 != 9" in audit["failures"]


def test_source_package_rebuild_rejects_bibtex_warning_and_overfull(tmp_path: Path) -> None:
    source_json = _seed_source_package(tmp_path)

    def runner(command, cwd, **kwargs):
        cwd = Path(cwd)
        if command[0] == "pdflatex":
            _write(
                cwd / "main.log",
                "Overfull \\hbox (5.0pt too wide)\n"
                "Output written on main.pdf (9 pages, 120000 bytes).\n",
            )
            _write(cwd / "main.pdf", (b"/Type /Page\n" * 9) + (b"p" * 119_892))
        if command[0] == "bibtex":
            _write(cwd / "main.blg", "warning$ -- 1\n")
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    audit = audit_submission_source_package_rebuild(
        root=tmp_path,
        source_package_json=source_json.relative_to(tmp_path),
        build_dir="artifacts/rebuild",
        overwrite=True,
        runner=runner,
    )

    assert audit["ok"] is False
    assert "rebuilt_bibtex_warnings:1" in audit["failures"]
    assert "rebuilt_overfull_hbox_count:1 > 0" in audit["failures"]


def test_source_package_rebuild_refuses_nonempty_work_without_overwrite(tmp_path: Path) -> None:
    source_json = _seed_source_package(tmp_path)
    _write(tmp_path / "artifacts" / "rebuild" / "work" / "old.txt", "old\n")

    audit = audit_submission_source_package_rebuild(
        root=tmp_path,
        source_package_json=source_json.relative_to(tmp_path),
        build_dir="artifacts/rebuild",
        overwrite=False,
        runner=_successful_runner,
    )

    assert audit["ok"] is False
    assert "work_dir_not_empty:artifacts\\rebuild\\work" in audit["failures"]


def test_source_package_rebuild_rejects_source_package_final_ready(tmp_path: Path) -> None:
    source_json = _seed_source_package(tmp_path)
    payload = json.loads(source_json.read_text(encoding="utf-8"))
    payload["final_submission_ready"] = True
    source_json.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_submission_source_package_rebuild(
        root=tmp_path,
        source_package_json=source_json.relative_to(tmp_path),
        build_dir="artifacts/rebuild",
        overwrite=True,
        runner=_successful_runner,
    )

    assert audit["ok"] is False
    assert "source_package_unexpectedly_final_ready" in audit["failures"]
