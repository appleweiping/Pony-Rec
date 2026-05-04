from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--similar_user_weight", type=float, default=0.15)
    parser.add_argument("--max_seq_len", type=int, default=200)
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


def _summary_row(output_root: Path, domain: str, adapter_dir: Path) -> dict[str, Any]:
    exp_name = f"{domain}_llmesr_scaffold_same_candidate"
    summary_path = output_root / exp_name / "tables" / "same_candidate_external_baseline_summary.csv"
    rows = _read_csv(summary_path)
    record = rows[0] if rows else {}
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
    }


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed_root).expanduser()
    output_root = Path(args.output_root).expanduser()
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
            f"artifact_class={row['artifact_class']}"
        )


if __name__ == "__main__":
    main()
