from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


PATCH_MARKER = "# PONY_SAME_CANDIDATE_DATASETS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install an LLM2Rec same-candidate adapter package into a cloned "
            "HappyPointer/LLM2Rec checkout and patch the upstream hard-coded "
            "dataset maps so native extraction/evaluation scripts can see it."
        )
    )
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--llm2rec_repo_dir", required=True)
    parser.add_argument("--dataset_alias", default=None)
    parser.add_argument("--link_mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--skip_patch", action="store_true")
    return parser.parse_args()


def _load_metadata(adapter_dir: Path) -> dict[str, Any]:
    metadata_path = adapter_dir / "adapter_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"adapter_metadata.json not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("adapter_name") != "llm2rec_same_candidate":
        raise ValueError(f"Not an LLM2Rec adapter package: adapter_name={metadata.get('adapter_name')!r}")
    return metadata


def _copytree_or_link(source: Path, target: Path, *, link_mode: str) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Adapter LLM2Rec data directory not found: {source}")
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "symlink":
        target.symlink_to(source.resolve(), target_is_directory=True)
    else:
        shutil.copytree(source, target)


def install_adapter_data(
    adapter_dir: Path,
    llm2rec_repo_dir: Path,
    *,
    dataset_alias: str | None = None,
    link_mode: str = "copy",
) -> dict[str, Any]:
    metadata = _load_metadata(adapter_dir)
    alias = dataset_alias or str(metadata.get("dataset_alias") or "").strip()
    if not alias:
        raise ValueError("dataset_alias is missing from adapter metadata and was not passed explicitly.")

    source_dir = adapter_dir / "llm2rec" / "data" / alias
    target_dir = llm2rec_repo_dir / "data" / alias
    _copytree_or_link(source_dir, target_dir, link_mode=link_mode)
    return {
        "dataset_alias": alias,
        "source_data_dir": str(source_dir),
        "installed_data_dir": str(target_dir),
        "relative_dataset_path": f"{alias}/downstream",
        "link_mode": link_mode,
    }


def _patched_dict_assignment(source: str, variable_name: str, alias: str, relative_path: str) -> str:
    marker = f"{PATCH_MARKER}: {alias}"
    lines = source.splitlines()
    updated: list[str] = []
    replaced = False
    for line in lines:
        if marker in line:
            if not replaced:
                indent = line[: len(line) - len(line.lstrip())]
                updated.append(f'{indent}{variable_name}["{alias}"] = "{relative_path}"  {marker}')
                replaced = True
            continue
        updated.append(line)
    if replaced:
        return "\n".join(updated) + ("\n" if source.endswith("\n") else "")

    anchor = f"{variable_name} = {{"
    insertion_index = None
    assignment_indent = ""
    for idx, line in enumerate(updated):
        if anchor in line:
            assignment_indent = line[: len(line) - len(line.lstrip())]
            depth = 0
            for jdx in range(idx, len(updated)):
                depth += updated[jdx].count("{")
                depth -= updated[jdx].count("}")
                if jdx > idx and depth <= 0:
                    insertion_index = jdx + 1
                    break
            break
    if insertion_index is None:
        raise ValueError(f"Could not find dictionary assignment for {variable_name!r}")
    assignment = f'{assignment_indent}{variable_name}["{alias}"] = "{relative_path}"  {marker}'
    updated.insert(insertion_index, assignment)
    return "\n".join(updated) + ("\n" if source.endswith("\n") else "")


def patch_upstream_dataset_maps(llm2rec_repo_dir: Path, *, dataset_alias: str, relative_path: str) -> dict[str, str]:
    recdata_path = llm2rec_repo_dir / "seqrec" / "recdata.py"
    extract_path = llm2rec_repo_dir / "extract_llm_embedding.py"
    if not recdata_path.exists():
        raise FileNotFoundError(f"Upstream seqrec/recdata.py not found: {recdata_path}")
    if not extract_path.exists():
        raise FileNotFoundError(f"Upstream extract_llm_embedding.py not found: {extract_path}")

    recdata_source = recdata_path.read_text(encoding="utf-8")
    recdata_source = _patched_dict_assignment(recdata_source, "source_dict", dataset_alias, relative_path)
    recdata_path.write_text(recdata_source, encoding="utf-8")

    extract_source = extract_path.read_text(encoding="utf-8")
    extract_source = _patched_dict_assignment(extract_source, "dataset_name_mappings", dataset_alias, relative_path)
    extract_path.write_text(extract_source, encoding="utf-8")

    return {
        "recdata_patch_path": str(recdata_path),
        "extract_patch_path": str(extract_path),
    }


def prepare_upstream_adapter(
    adapter_dir: Path,
    llm2rec_repo_dir: Path,
    *,
    dataset_alias: str | None = None,
    link_mode: str = "copy",
    skip_patch: bool = False,
) -> dict[str, Any]:
    if not llm2rec_repo_dir.exists():
        raise FileNotFoundError(f"LLM2Rec repo directory does not exist: {llm2rec_repo_dir}")
    install_summary = install_adapter_data(
        adapter_dir,
        llm2rec_repo_dir,
        dataset_alias=dataset_alias,
        link_mode=link_mode,
    )
    patch_summary: dict[str, Any] = {}
    if not skip_patch:
        patch_summary = patch_upstream_dataset_maps(
            llm2rec_repo_dir,
            dataset_alias=install_summary["dataset_alias"],
            relative_path=install_summary["relative_dataset_path"],
        )
    summary = {
        "adapter_dir": str(adapter_dir),
        "llm2rec_repo_dir": str(llm2rec_repo_dir),
        "status": "llm2rec_upstream_adapter_prepared",
        "upstream_repo": "https://github.com/HappyPointer/LLM2Rec",
        "patch_applied": not skip_patch,
        **install_summary,
        **patch_summary,
    }
    summary_path = adapter_dir / "llm2rec_upstream_prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = prepare_upstream_adapter(
        Path(args.adapter_dir).expanduser(),
        Path(args.llm2rec_repo_dir).expanduser(),
        dataset_alias=args.dataset_alias,
        link_mode=args.link_mode,
        skip_patch=args.skip_patch,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
