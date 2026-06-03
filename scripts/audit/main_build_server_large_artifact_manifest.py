from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_RELATIVE_FILES = (
    "scores.csv",
    "predictions/rank_predictions.jsonl",
)

MODEL_ARTIFACT_SUFFIXES = (
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".safetensors",
)

LARGE_ARTIFACT_SUFFIXES = (
    *MODEL_ARTIFACT_SUFFIXES,
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a sha256 manifest for server-only official-baseline artifacts. "
            "Run this after server_final evidence audit and before local-light sync "
            "or any disk-pressure cleanup decision."
        )
    )
    parser.add_argument("--evidence_dir", required=True)
    parser.add_argument(
        "--output_sha256",
        default="",
        help="Defaults to <evidence_dir>/server_large_artifact_manifest.sha256.",
    )
    parser.add_argument(
        "--output_json",
        default="",
        help="Defaults to <evidence_dir>/server_large_artifact_manifest.json.",
    )
    parser.add_argument(
        "--include_suffix",
        action="append",
        default=[],
        help="Extra suffix to include as a server-only large artifact, for example .faiss.",
    )
    parser.add_argument(
        "--require_model_artifact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require at least one model/checkpoint artifact such as *_official_model.pt.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_suffixes(extra_suffixes: list[str]) -> tuple[str, ...]:
    suffixes = {suffix.lower() for suffix in LARGE_ARTIFACT_SUFFIXES}
    for suffix in extra_suffixes:
        clean = str(suffix).strip().lower()
        if clean:
            suffixes.add(clean if clean.startswith(".") else f".{clean}")
    return tuple(sorted(suffixes))


def _is_model_artifact(path: Path) -> bool:
    return path.name.lower().endswith(MODEL_ARTIFACT_SUFFIXES)


def _is_large_artifact(path: Path, suffixes: tuple[str, ...]) -> bool:
    lower_name = path.name.lower()
    return lower_name.endswith(suffixes)


def _file_row(evidence_dir: Path, path: Path) -> dict[str, Any]:
    rel_path = path.relative_to(evidence_dir).as_posix()
    return {
        "rel_path": rel_path,
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_manifest(
    *,
    evidence_dir: str | Path,
    include_suffixes: list[str] | None = None,
    require_model_artifact: bool = True,
) -> dict[str, Any]:
    base = Path(evidence_dir).expanduser()
    suffixes = _normalized_suffixes(include_suffixes or [])
    failures: list[str] = []
    warnings: list[str] = []
    files_by_rel: dict[str, dict[str, Any]] = {}

    if not base.exists() or not base.is_dir():
        return {
            "ok": False,
            "evidence_dir": str(base),
            "files": [],
            "failures": ["missing_evidence_dir"],
            "warnings": warnings,
            "required_relative_files": list(REQUIRED_RELATIVE_FILES),
            "require_model_artifact": require_model_artifact,
            "include_suffixes": list(suffixes),
        }

    for rel_path in REQUIRED_RELATIVE_FILES:
        path = base / rel_path
        if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
            failures.append(f"missing_required_large_artifact:{rel_path}")
            continue
        files_by_rel[rel_path] = _file_row(base, path)

    model_artifact_count = 0
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        rel_path = path.relative_to(base).as_posix()
        if rel_path in files_by_rel:
            continue
        if _is_large_artifact(path, suffixes):
            files_by_rel[rel_path] = _file_row(base, path)
            model_artifact_count += int(_is_model_artifact(path))

    if require_model_artifact and model_artifact_count == 0:
        failures.append("missing_model_artifact")
    if not files_by_rel:
        failures.append("no_large_artifacts_manifested")
    if any(row["size"] <= 0 for row in files_by_rel.values()):
        failures.append("zero_size_manifested_artifact")
    if model_artifact_count > 1:
        warnings.append("multiple_model_artifacts_manifested")

    files = sorted(files_by_rel.values(), key=lambda row: row["rel_path"])
    return {
        "ok": not failures,
        "evidence_dir": str(base),
        "files": files,
        "file_count": len(files),
        "model_artifact_count": model_artifact_count,
        "failures": failures,
        "warnings": warnings,
        "required_relative_files": list(REQUIRED_RELATIVE_FILES),
        "require_model_artifact": require_model_artifact,
        "include_suffixes": list(suffixes),
    }


def write_manifest(result: dict[str, Any], *, output_sha256: str | Path, output_json: str | Path) -> None:
    sha_path = Path(output_sha256)
    json_path = Path(output_json)
    sha_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with sha_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in result.get("files", []):
            fh.write(f"{row['sha256']}  {row['rel_path']}\n")
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    evidence_dir = Path(args.evidence_dir).expanduser()
    result = build_manifest(
        evidence_dir=evidence_dir,
        include_suffixes=args.include_suffix,
        require_model_artifact=bool(args.require_model_artifact),
    )
    output_sha256 = (
        Path(args.output_sha256).expanduser()
        if args.output_sha256
        else evidence_dir / "server_large_artifact_manifest.sha256"
    )
    output_json = (
        Path(args.output_json).expanduser()
        if args.output_json
        else evidence_dir / "server_large_artifact_manifest.json"
    )
    write_manifest(result, output_sha256=output_sha256, output_json=output_json)
    if not args.quiet:
        print(json.dumps({**result, "output_sha256": str(output_sha256), "output_json": str(output_json)}, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
