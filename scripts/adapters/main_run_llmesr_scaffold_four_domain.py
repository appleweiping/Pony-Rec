from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DOMAIN_DATASETS = {
    "beauty": "amazon_beauty",
    "books": "amazon_books_small",
    "electronics": "amazon_electronics_small",
    "movies": "amazon_movies_small",
}

RAW_METADATA_CANDIDATES = {
    "beauty": [
        "amazon_beauty/meta_Beauty.jsonl",
        "amazon_beauty/meta_All_Beauty.jsonl.gz",
    ],
    "books": [
        "amazon_books/meta_Books.jsonl.gz",
        "amazon_books/meta_Books.jsonl",
    ],
    "electronics": [
        "amazon_electronics/meta_Electronics.jsonl.gz",
        "amazon_electronics/meta_Electronics.jsonl",
    ],
    "movies": [
        "amazon_movies/meta_Movies_and_TV.jsonl.gz",
        "amazon_movies/meta_Movies_and_TV.jsonl",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the LLM-ESR same-candidate scaffold adapter pipeline across domains. "
            "This produces adapter_scaffold_score rows only; it does not create "
            "completed paper-project baseline results."
        )
    )
    parser.add_argument("--processed_root", default="data/processed")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--domains", default="beauty,books,electronics,movies")
    parser.add_argument("--embedding_dim", type=int, default=384)
    parser.add_argument(
        "--embedding_backend",
        choices=["deterministic_text_hash", "sentence_transformers", "hf_mean_pool"],
        default="deterministic_text_hash",
    )
    parser.add_argument("--sentence_model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--sentence_batch_size", type=int, default=64)
    parser.add_argument("--hf_model_name", default="")
    parser.add_argument("--hf_batch_size", type=int, default=2)
    parser.add_argument("--hf_max_length", type=int, default=256)
    parser.add_argument("--hf_trust_remote_code", action="store_true")
    parser.add_argument("--hf_torch_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--hf_device_map", default="")
    parser.add_argument("--similar_user_weight", type=float, default=0.15)
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--raw_metadata_root", default="")
    parser.add_argument(
        "--raw_metadata_path",
        action="append",
        default=[],
        help="Per-domain raw metadata override, e.g. movies=/path/to/meta_Movies_and_TV.jsonl.gz. Can repeat.",
    )
    parser.add_argument("--allow_missing_raw_metadata", action="store_true")
    parser.add_argument("--skip_task_export", action="store_true")
    parser.add_argument("--summary_name", default="llmesr_scaffold_four_domain_summary")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "domain",
        "exp_name",
        "adapter_dir",
        "scores_path",
        "summary_path",
        "status_label",
        "artifact_class",
        "score_coverage_rate",
        "sample_count",
        "NDCG@10",
        "MRR",
        "title_coverage_after",
        "non_id_embedding_text_coverage_after",
        "embedding_backend",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _domain_list(domains_text: str) -> list[str]:
    domains = [item.strip() for item in domains_text.split(",") if item.strip()]
    unknown = [domain for domain in domains if domain not in DOMAIN_DATASETS]
    if unknown:
        raise ValueError(f"Unknown domains: {unknown}. Known domains: {sorted(DOMAIN_DATASETS)}")
    return domains


def _task_dir(output_root: Path, domain: str) -> Path:
    return output_root / "baselines" / "external_tasks" / f"{domain}_week8_same_candidate_external"


def _adapter_dir(output_root: Path, domain: str) -> Path:
    return output_root / "baselines" / "paper_adapters" / f"{domain}_llmesr_same_candidate_adapter"


def _raw_metadata_overrides(raw_metadata_paths: list[str]) -> dict[str, list[Path]]:
    overrides: dict[str, list[Path]] = {}
    for raw_value in raw_metadata_paths:
        if "=" not in raw_value:
            raise ValueError(
                "--raw_metadata_path must be DOMAIN=PATH, for example "
                "movies=/path/to/meta_Movies_and_TV.jsonl.gz"
            )
        domain, path_text = raw_value.split("=", 1)
        domain = domain.strip()
        if domain not in DOMAIN_DATASETS:
            raise ValueError(f"Unknown raw metadata override domain={domain!r}. Known domains: {sorted(DOMAIN_DATASETS)}")
        path = Path(path_text.strip()).expanduser()
        overrides.setdefault(domain, []).append(path)
    return overrides


def _fallback_raw_metadata_paths(raw_metadata_root: Path, domain: str) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for rel_path in RAW_METADATA_CANDIDATES.get(domain, []):
        basename = Path(rel_path).name
        for path in raw_metadata_root.rglob(basename):
            resolved = path.resolve()
            if resolved not in seen and path.exists():
                found.append(path)
                seen.add(resolved)
    return found


def _raw_metadata_args(
    raw_metadata_root: Path | None,
    domain: str,
    *,
    allow_missing: bool,
    raw_metadata_overrides: dict[str, list[Path]] | None = None,
) -> list[str]:
    args: list[str] = []
    override_paths = (raw_metadata_overrides or {}).get(domain, [])
    for path in override_paths:
        if path.exists():
            args.extend(["--raw_metadata_path", str(path)])
        elif allow_missing:
            print(f"WARNING: raw metadata override does not exist for domain={domain}: {path}", flush=True)
        else:
            raise FileNotFoundError(f"raw metadata override does not exist for domain={domain}: {path}")
    if args or raw_metadata_root is None:
        return args
    if not raw_metadata_root.exists():
        message = f"--raw_metadata_root does not exist: {raw_metadata_root}"
        if allow_missing:
            print(f"WARNING: {message}", flush=True)
            return []
        raise FileNotFoundError(message)
    checked_paths = []
    for rel_path in RAW_METADATA_CANDIDATES.get(domain, []):
        path = raw_metadata_root / rel_path
        checked_paths.append(path)
        if path.exists():
            args.extend(["--raw_metadata_path", str(path)])
    if not args:
        fallback_paths = _fallback_raw_metadata_paths(raw_metadata_root, domain)
        for path in fallback_paths:
            args.extend(["--raw_metadata_path", str(path)])
    if args:
        return args
    checked = ", ".join(str(path) for path in checked_paths)
    basenames = ", ".join(Path(path).name for path in RAW_METADATA_CANDIDATES.get(domain, []))
    message = (
        f"--raw_metadata_root was provided, but no raw metadata file was found for domain={domain}. "
        f"Checked: {checked}. Also searched recursively for: {basenames}. "
        f"Pass an explicit override such as --raw_metadata_path {domain}=/path/to/raw_metadata.jsonl.gz."
    )
    if allow_missing:
        print(f"WARNING: {message}", flush=True)
        return []
    raise FileNotFoundError(message)


def _summary_row(output_root: Path, domain: str, adapter_dir: Path) -> dict[str, Any]:
    exp_name = f"{domain}_llmesr_scaffold_same_candidate"
    summary_path = output_root / exp_name / "tables" / "same_candidate_external_baseline_summary.csv"
    rows = _read_csv(summary_path)
    record = rows[0] if rows else {}
    enrichment_path = adapter_dir / "item_text_enrichment_summary.json"
    enrichment = {}
    if enrichment_path.exists():
        enrichment = json.loads(enrichment_path.read_text(encoding="utf-8"))
    embedding_metadata_path = adapter_dir / "llmesr_embedding_metadata.json"
    embedding_metadata = {}
    if embedding_metadata_path.exists():
        embedding_metadata = json.loads(embedding_metadata_path.read_text(encoding="utf-8"))
    return {
        "domain": domain,
        "exp_name": exp_name,
        "adapter_dir": str(adapter_dir),
        "scores_path": str(adapter_dir / "llmesr_scaffold_scores.csv"),
        "summary_path": str(summary_path),
        "status_label": record.get("status_label", ""),
        "artifact_class": record.get("artifact_class", ""),
        "score_coverage_rate": record.get("score_coverage_rate", ""),
        "sample_count": record.get("sample_count", ""),
        "NDCG@10": record.get("NDCG@10", ""),
        "MRR": record.get("MRR", ""),
        "title_coverage_after": enrichment.get("title_coverage_after", ""),
        "non_id_embedding_text_coverage_after": enrichment.get("non_id_embedding_text_coverage_after", ""),
        "embedding_backend": embedding_metadata.get("backend", ""),
    }


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    raw_metadata_root = Path(args.raw_metadata_root).expanduser() if args.raw_metadata_root else None
    raw_metadata_overrides = _raw_metadata_overrides(args.raw_metadata_path)
    domains = _domain_list(args.domains)
    rows: list[dict[str, Any]] = []

    for domain in domains:
        dataset = DOMAIN_DATASETS[domain]
        print(f"\n========== {domain} ==========", flush=True)
        task_dir = _task_dir(output_root, domain)
        adapter_dir = _adapter_dir(output_root, domain)
        ranking_input_path = processed_root / dataset / "ranking_test.jsonl"
        processed_dir = processed_root / dataset

        if not task_dir.exists():
            if args.skip_task_export:
                raise FileNotFoundError(f"Task dir missing and --skip_task_export set: {task_dir}")
            _run(
                [
                    sys.executable,
                    "main_export_same_candidate_baseline_task.py",
                    "--processed_dir",
                    str(processed_dir),
                    "--ranking_input_path",
                    str(ranking_input_path),
                    "--exp_name",
                    f"{domain}_week8_same_candidate_external",
                    "--output_root",
                    str(output_root),
                ]
            )

        _run(
            [
                sys.executable,
                "main_export_llmesr_same_candidate_task.py",
                "--task_dir",
                str(task_dir),
                "--exp_name",
                f"{domain}_llmesr_same_candidate_adapter",
                "--output_root",
                str(output_root),
            ]
        )
        _run([sys.executable, "main_audit_llmesr_adapter_package.py", "--adapter_dir", str(adapter_dir)])
        _run(
            [
                sys.executable,
                "main_enrich_llmesr_item_text_seed.py",
                "--adapter_dir",
                str(adapter_dir),
                "--processed_dir",
                str(processed_dir),
                *_raw_metadata_args(
                    raw_metadata_root,
                    domain,
                    allow_missing=args.allow_missing_raw_metadata,
                    raw_metadata_overrides=raw_metadata_overrides,
                ),
            ]
        )
        if args.embedding_backend == "sentence_transformers":
            _run(
                [
                    sys.executable,
                    "main_generate_llmesr_sentence_embeddings.py",
                    "--adapter_dir",
                    str(adapter_dir),
                    "--backend",
                    "sentence_transformers",
                    "--model_name",
                    args.sentence_model_name,
                    "--batch_size",
                    str(args.sentence_batch_size),
                ]
            )
        elif args.embedding_backend == "hf_mean_pool":
            hf_model_name = args.hf_model_name or args.sentence_model_name
            cmd = [
                sys.executable,
                "main_generate_llmesr_sentence_embeddings.py",
                "--adapter_dir",
                str(adapter_dir),
                "--backend",
                "hf_mean_pool",
                "--model_name",
                hf_model_name,
                "--batch_size",
                str(args.hf_batch_size),
                "--max_length",
                str(args.hf_max_length),
                "--torch_dtype",
                args.hf_torch_dtype,
            ]
            if args.hf_trust_remote_code:
                cmd.append("--trust_remote_code")
            if args.hf_device_map:
                cmd.extend(["--hf_device_map", args.hf_device_map])
            _run(cmd)
        else:
            _run(
                [
                    sys.executable,
                    "main_generate_llmesr_text_embeddings.py",
                    "--adapter_dir",
                    str(adapter_dir),
                    "--embedding_dim",
                    str(args.embedding_dim),
                ]
            )
        _run([sys.executable, "main_audit_llmesr_adapter_package.py", "--adapter_dir", str(adapter_dir)])
        _run(
            [
                sys.executable,
                "main_score_llmesr_same_candidate_adapter.py",
                "--adapter_dir",
                str(adapter_dir),
                "--similar_user_weight",
                str(args.similar_user_weight),
                "--max_seq_len",
                str(args.max_seq_len),
            ]
        )
        _run(
            [
                sys.executable,
                "main_audit_same_candidate_score_file.py",
                "--candidate_items_path",
                str(task_dir / "candidate_items.csv"),
                "--scores_path",
                str(adapter_dir / "llmesr_scaffold_scores.csv"),
            ]
        )
        _run(
            [
                sys.executable,
                "main_import_same_candidate_baseline_scores.py",
                "--baseline_name",
                "llmesr_scaffold",
                "--exp_name",
                f"{domain}_llmesr_scaffold_same_candidate",
                "--domain",
                domain,
                "--ranking_input_path",
                str(ranking_input_path),
                "--scores_path",
                str(adapter_dir / "llmesr_scaffold_scores.csv"),
                "--status_label",
                "llmesr_adapter_scaffold_score",
                "--artifact_class",
                "adapter_scaffold_score",
            ]
        )
        rows.append(_summary_row(output_root, domain, adapter_dir))

    summary_path = output_root / "summary" / f"{args.summary_name}.csv"
    _write_csv(rows, summary_path)
    print(f"\nSaved summary: {summary_path}")
    for row in rows:
        print(
            f"{row['domain']} coverage={row['score_coverage_rate']} "
            f"NDCG@10={row['NDCG@10']} MRR={row['MRR']} "
            f"title_cov={row['title_coverage_after']} "
            f"non_id_text_cov={row['non_id_embedding_text_coverage_after']} "
            f"embedding_backend={row['embedding_backend']} "
            f"artifact_class={row['artifact_class']}"
        )


if __name__ == "__main__":
    main()
