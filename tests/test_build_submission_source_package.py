import hashlib
import json
from pathlib import Path

from scripts.audit.main_build_submission_source_package import build_submission_source_package


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_sha(entries: list[dict]) -> str:
    lines = [
        f"{entry['sha256']}  {entry['path']}  {entry['role']}  {entry['size_bytes']}"
        for entry in sorted(entries, key=lambda item: item["path"])
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _seed_audit(tmp_path: Path, *, source_leak_scan_ok: bool = True) -> dict[str, Path]:
    main = _write(tmp_path / "Paper" / "main.tex", "\\title{Anonymous Paper}\n")
    abstract = _write(tmp_path / "Paper" / "sections" / "abstract.tex", "Abstract.\n")
    refs = _write(tmp_path / "Paper" / "references.bib", "@inproceedings{x, title={X}}\n")
    entries = [
        {
            "path": "Paper/main.tex",
            "role": "tex_source",
            "exists": True,
            "size_bytes": main.stat().st_size,
            "sha256": _sha(main),
        },
        {
            "path": "Paper/sections/abstract.tex",
            "role": "tex_source",
            "exists": True,
            "size_bytes": abstract.stat().st_size,
            "sha256": _sha(abstract),
        },
        {
            "path": "Paper/references.bib",
            "role": "bibliography_source",
            "exists": True,
            "size_bytes": refs.stat().st_size,
            "sha256": _sha(refs),
        },
    ]
    audit = {
        "ok": True,
        "submission_package_ready_for_target_formatting": True,
        "final_submission_ready": False,
        "failures": [],
        "anonymity": {"source_leak_scan_ok": source_leak_scan_ok},
        "source_package_manifest": {
            "file_count": len(entries),
            "total_bytes": sum(entry["size_bytes"] for entry in entries),
            "manifest_sha256": _manifest_sha(entries),
            "external_source_references": [],
            "files": entries,
        },
        "remaining_blockers": ["external metadata"],
    }
    audit_path = _write_json(tmp_path / "submission_package_audit.json", audit)
    return {"audit": audit_path, "main": main}


def test_build_submission_source_package_copies_exact_audited_manifest(tmp_path: Path) -> None:
    paths = _seed_audit(tmp_path)

    package = build_submission_source_package(
        root=tmp_path,
        submission_package_audit_json=paths["audit"].relative_to(tmp_path),
        output_dir="package",
    )

    assert package["ok"] is True
    assert package["submission_source_package_ready"] is True
    assert package["final_submission_ready"] is False
    assert package["copied_manifest"]["file_count"] == 3
    assert package["copied_manifest"]["manifest_sha256"] == package["source_audit_crosscheck"][
        "source_manifest_sha256"
    ]
    assert (tmp_path / "package" / "files" / "Paper" / "main.tex").read_text(encoding="utf-8") == (
        tmp_path / "Paper" / "main.tex"
    ).read_text(encoding="utf-8")
    assert package["remaining_blockers"] == ["external metadata"]


def test_build_submission_source_package_rejects_anonymity_leak_state(tmp_path: Path) -> None:
    paths = _seed_audit(tmp_path, source_leak_scan_ok=False)

    package = build_submission_source_package(
        root=tmp_path,
        submission_package_audit_json=paths["audit"].relative_to(tmp_path),
        output_dir="package",
    )

    assert package["ok"] is False
    assert "anonymous_source_leak_scan_not_ok" in package["failures"]
    assert not (tmp_path / "package" / "files" / "Paper" / "main.tex").exists()


def test_build_submission_source_package_rejects_external_references_and_path_escape(
    tmp_path: Path,
) -> None:
    paths = _seed_audit(tmp_path)
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    audit["source_package_manifest"]["external_source_references"] = ["outside/main.tex"]
    audit["source_package_manifest"]["files"][0]["path"] = "../outside.tex"
    paths["audit"].write_text(json.dumps(audit), encoding="utf-8")

    package = build_submission_source_package(
        root=tmp_path,
        submission_package_audit_json=paths["audit"].relative_to(tmp_path),
        output_dir="package",
    )

    assert package["ok"] is False
    assert "source_package_manifest_has_external_references" in package["failures"]
    assert not (tmp_path / "package" / "files").exists()


def test_build_submission_source_package_rejects_path_escape_without_external_refs(
    tmp_path: Path,
) -> None:
    paths = _seed_audit(tmp_path)
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    audit["source_package_manifest"]["files"][0]["path"] = "../outside.tex"
    paths["audit"].write_text(json.dumps(audit), encoding="utf-8")

    package = build_submission_source_package(
        root=tmp_path,
        submission_package_audit_json=paths["audit"].relative_to(tmp_path),
        output_dir="package",
    )

    assert package["ok"] is False
    assert "unsafe manifest path: ../outside.tex" in package["failures"]


def test_build_submission_source_package_rejects_private_manifest_path(tmp_path: Path) -> None:
    paths = _seed_audit(tmp_path)
    private_file = _write(tmp_path / "configs" / "paper_manual_submission_private_confirmation.json", "{}\n")
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    entry = {
        "path": "configs/paper_manual_submission_private_confirmation.json",
        "role": "private_confirmation",
        "exists": True,
        "size_bytes": private_file.stat().st_size,
        "sha256": _sha(private_file),
    }
    audit["source_package_manifest"]["files"].append(entry)
    audit["source_package_manifest"]["file_count"] += 1
    audit["source_package_manifest"]["total_bytes"] += entry["size_bytes"]
    audit["source_package_manifest"]["manifest_sha256"] = _manifest_sha(
        audit["source_package_manifest"]["files"]
    )
    paths["audit"].write_text(json.dumps(audit), encoding="utf-8")

    package = build_submission_source_package(
        root=tmp_path,
        submission_package_audit_json=paths["audit"].relative_to(tmp_path),
        output_dir="package",
    )

    assert package["ok"] is False
    assert (
        "forbidden_private_manifest_path:configs/paper_manual_submission_private_confirmation.json"
        in package["failures"]
    )


def test_build_submission_source_package_rejects_missing_hashes(tmp_path: Path) -> None:
    paths = _seed_audit(tmp_path)
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    audit["source_package_manifest"]["files"][0].pop("sha256")
    audit["source_package_manifest"]["manifest_sha256"] = ""
    paths["audit"].write_text(json.dumps(audit), encoding="utf-8")

    package = build_submission_source_package(
        root=tmp_path,
        submission_package_audit_json=paths["audit"].relative_to(tmp_path),
        output_dir="package",
    )

    assert package["ok"] is False
    assert "manifest_entry_invalid_sha256:Paper/main.tex" in package["failures"]
    assert "source_manifest_invalid_sha256" in package["failures"]


def test_build_submission_source_package_rolls_back_after_late_manifest_failure(
    tmp_path: Path,
) -> None:
    paths = _seed_audit(tmp_path)
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    audit["source_package_manifest"]["files"][-1]["sha256"] = "0" * 64
    paths["audit"].write_text(json.dumps(audit), encoding="utf-8")

    package = build_submission_source_package(
        root=tmp_path,
        submission_package_audit_json=paths["audit"].relative_to(tmp_path),
        output_dir="package",
    )

    assert package["ok"] is False
    assert "source_sha256_mismatch:Paper/references.bib" in package["failures"]
    assert not (tmp_path / "package").exists()
    assert not (tmp_path / ".package.tmp").exists()


def test_build_submission_source_package_rejects_changed_source_hash(tmp_path: Path) -> None:
    paths = _seed_audit(tmp_path)
    paths["main"].write_text("changed after audit\n", encoding="utf-8")

    package = build_submission_source_package(
        root=tmp_path,
        submission_package_audit_json=paths["audit"].relative_to(tmp_path),
        output_dir="package",
    )

    assert package["ok"] is False
    assert "source_sha256_mismatch:Paper/main.tex" in package["failures"]


def test_build_submission_source_package_refuses_nonempty_output_without_overwrite(
    tmp_path: Path,
) -> None:
    paths = _seed_audit(tmp_path)
    _write(tmp_path / "package" / "existing.txt", "old\n")

    package = build_submission_source_package(
        root=tmp_path,
        submission_package_audit_json=paths["audit"].relative_to(tmp_path),
        output_dir="package",
    )

    assert package["ok"] is False
    assert "output_dir_not_empty:package" in package["failures"]


def test_build_submission_source_package_overwrite_replaces_stale_tree(tmp_path: Path) -> None:
    paths = _seed_audit(tmp_path)
    stale = _write(tmp_path / "package" / "files" / "private_confirmation.txt", "secret\n")

    package = build_submission_source_package(
        root=tmp_path,
        submission_package_audit_json=paths["audit"].relative_to(tmp_path),
        output_dir="package",
        overwrite=True,
    )

    assert package["ok"] is True
    assert package["copied_manifest"]["file_count"] == 3
    assert not stale.exists()
    copied_paths = {
        str(path.relative_to(tmp_path / "package" / "files")).replace("\\", "/")
        for path in (tmp_path / "package" / "files").rglob("*")
        if path.is_file()
    }
    assert copied_paths == {
        "Paper/main.tex",
        "Paper/sections/abstract.tex",
        "Paper/references.bib",
    }
