from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


BASE_FIELDS = [
    "item_id",
    "llmesr_item_idx",
    "candidate_title",
    "embedding_text",
    "title_source",
    "candidate_text",
    "catalog_categories",
    "catalog_description",
    "text_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich an LLM-ESR adapter package's item_text_seed.csv from processed "
            "catalog files, ranking JSONL text, and optional raw Amazon metadata."
        )
    )
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--raw_metadata_path", action="append", default=[])
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--max_text_chars", type=int, default=1200)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.strip().split())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_text(item) for item in value if _text(item))
    if isinstance(value, dict):
        return " ".join(f"{_text(k)}: {_text(v)}" for k, v in value.items() if _text(v))
    return " ".join(str(value).strip().split())


def _literalish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return ""
    if text[0] not in "[{(":
        return text
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _usable_text(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    if text.lower() == "none":
        return ""
    if text.lower().startswith("item id:"):
        return ""
    return text


def _compose_text(
    *,
    title: str,
    categories: str = "",
    description: str = "",
    features: str = "",
    main_category: str = "",
    details: str = "",
    item_id: str = "",
) -> str:
    parts = []
    if title:
        parts.append(f"Title: {title}")
    elif item_id:
        parts.append(f"Item ID: {item_id}")
    if main_category:
        parts.append(f"Main category: {main_category}")
    if categories:
        parts.append(f"Categories: {categories}")
    if description:
        parts.append(f"Description: {description}")
    if features:
        parts.append(f"Features: {features}")
    if details:
        parts.append(f"Details: {details}")
    return " ".join(parts)


def _better_text(current: str, candidate: str) -> bool:
    current = _usable_text(current)
    candidate = _usable_text(candidate)
    if not candidate:
        return False
    if not current:
        return True
    # Prefer materially richer catalog strings but avoid tiny churn.
    return len(candidate) >= len(current) + 24


def _non_id_embedding_text(row: dict[str, Any]) -> bool:
    text = _usable_text(row.get("embedding_text"))
    item_id = _text(row.get("item_id"))
    if not text:
        return False
    if item_id and text == item_id:
        return False
    if item_id and text.lower() == f"item id: {item_id}".lower():
        return False
    return True


def _update_item(
    item: dict[str, Any],
    *,
    title: str,
    candidate_text: str,
    categories: str,
    description: str,
    source: str,
    max_text_chars: int,
) -> bool:
    changed = False
    title = _usable_text(title)
    candidate_text = _usable_text(candidate_text)
    categories = _usable_text(categories)
    description = _usable_text(description)
    composed = _usable_text(candidate_text) or _compose_text(
        title=title,
        categories=categories,
        description=description,
    )
    composed = composed[:max_text_chars]

    if title and not _usable_text(item.get("candidate_title")):
        item["candidate_title"] = title
        item["title_source"] = source
        changed = True
    if candidate_text and not _usable_text(item.get("candidate_text")):
        item["candidate_text"] = candidate_text[:max_text_chars]
        changed = True
    if categories and not _usable_text(item.get("catalog_categories")):
        item["catalog_categories"] = categories
        changed = True
    if description and not _usable_text(item.get("catalog_description")):
        item["catalog_description"] = description[:max_text_chars]
        changed = True
    if _better_text(str(item.get("embedding_text", "")), composed):
        item["embedding_text"] = composed
        item["text_source"] = source
        changed = True
    return changed


def _load_processed_items(processed_dir: Path, *, max_text_chars: int) -> dict[str, dict[str, str]]:
    path = processed_dir / "items.csv"
    if not path.exists():
        return {}
    sources = {}
    for row in _read_csv(path):
        item_id = _text(row.get("item_id"))
        if not item_id:
            continue
        title = _usable_text(row.get("title"))
        categories = _usable_text(row.get("categories"))
        description = _usable_text(row.get("description"))
        candidate_text = _usable_text(row.get("candidate_text"))[:max_text_chars]
        sources[item_id] = {
            "title": title,
            "categories": categories,
            "description": description,
            "candidate_text": candidate_text,
            "source": "processed_items_csv",
        }
    return sources


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _load_ranking_texts(processed_dir: Path, *, max_text_chars: int) -> dict[str, dict[str, str]]:
    files = [
        "ranking_test.jsonl",
        "ranking_valid.jsonl",
        "test.jsonl",
        "valid.jsonl",
        "train.jsonl",
        "pairwise_test.jsonl",
        "pairwise_valid.jsonl",
    ]
    sources: dict[str, dict[str, str]] = {}
    for name in files:
        path = processed_dir / name
        if not path.exists():
            continue
        for sample in _iter_jsonl(path):
            ids = sample.get("candidate_item_ids") or []
            titles = sample.get("candidate_titles") or []
            texts = sample.get("candidate_texts") or []
            if not ids and sample.get("candidate_item_id"):
                ids = [sample.get("candidate_item_id")]
                titles = [sample.get("candidate_title", "")]
                texts = [sample.get("candidate_text", "")]
            for idx, item_id_raw in enumerate(ids):
                item_id = _text(item_id_raw)
                title = _usable_text(titles[idx] if idx < len(titles) else "")
                candidate_text = _usable_text(texts[idx] if idx < len(texts) else "")
                if item_id and (title or candidate_text):
                    current = sources.get(item_id, {})
                    if _better_text(current.get("candidate_text", ""), candidate_text):
                        sources[item_id] = {
                            "title": title or current.get("title", ""),
                            "categories": "",
                            "description": "",
                            "candidate_text": candidate_text[:max_text_chars],
                            "source": f"processed_{name}",
                        }
    return sources


def _raw_record_source(record: dict[str, Any], *, max_text_chars: int) -> dict[str, str]:
    item_id = _usable_text(record.get("parent_asin") or record.get("asin"))
    title = _usable_text(record.get("title") or record.get("subtitle"))
    main_category = _usable_text(record.get("main_category"))
    categories = _usable_text(_literalish(record.get("categories")))
    description = _usable_text(_literalish(record.get("description")))
    features = _usable_text(_literalish(record.get("features")))
    details = _usable_text(_literalish(record.get("details")))
    candidate_text = _compose_text(
        title=title,
        main_category=main_category,
        categories=categories,
        description=description,
        features=features,
        details=details,
        item_id=item_id,
    )[:max_text_chars]
    return {
        "title": title,
        "categories": categories or main_category,
        "description": description or details,
        "candidate_text": candidate_text,
        "source": "raw_metadata",
    }


def _load_raw_metadata_sources(
    raw_paths: list[Path],
    needed_item_ids: set[str],
    *,
    max_text_chars: int,
) -> dict[str, dict[str, str]]:
    if not needed_item_ids:
        return {}
    sources: dict[str, dict[str, str]] = {}
    remaining = set(needed_item_ids)
    for path in raw_paths:
        if not path.exists() or not remaining:
            continue
        for record in _iter_jsonl(path):
            item_id = _text(record.get("parent_asin") or record.get("asin"))
            if item_id not in remaining:
                continue
            source = _raw_record_source(record, max_text_chars=max_text_chars)
            if source.get("title") or source.get("candidate_text"):
                source["source"] = f"raw_metadata:{path.name}"
                sources[item_id] = source
                remaining.discard(item_id)
        if not remaining:
            break
    return sources


def enrich_item_text_seed(
    *,
    adapter_dir: Path,
    processed_dir: Path,
    raw_metadata_paths: list[Path],
    output_path: Path | None = None,
    summary_path: Path | None = None,
    max_text_chars: int = 1200,
) -> dict[str, Any]:
    item_text_path = adapter_dir / "item_text_seed.csv"
    if not item_text_path.exists():
        raise FileNotFoundError(f"item_text_seed.csv not found: {item_text_path}")
    rows = _read_csv(item_text_path)
    by_item = {row["item_id"]: dict(row) for row in rows}

    before_title = sum(1 for row in by_item.values() if _usable_text(row.get("candidate_title")))
    before_text = sum(1 for row in by_item.values() if _usable_text(row.get("embedding_text")))
    before_non_id_text = sum(1 for row in by_item.values() if _non_id_embedding_text(row))

    source_counts: Counter[str] = Counter()
    for source_map in [
        _load_processed_items(processed_dir, max_text_chars=max_text_chars),
        _load_ranking_texts(processed_dir, max_text_chars=max_text_chars),
    ]:
        for item_id, source in source_map.items():
            if item_id in by_item and _update_item(by_item[item_id], max_text_chars=max_text_chars, **source):
                source_counts[source["source"]] += 1

    missing_title_ids = {
        item_id
        for item_id, row in by_item.items()
        if not _usable_text(row.get("candidate_title")) or not _usable_text(row.get("embedding_text"))
    }
    raw_sources = _load_raw_metadata_sources(raw_metadata_paths, missing_title_ids, max_text_chars=max_text_chars)
    for item_id, source in raw_sources.items():
        if item_id in by_item and _update_item(by_item[item_id], max_text_chars=max_text_chars, **source):
            source_counts[source["source"]] += 1

    sorted_rows = sorted(by_item.values(), key=lambda row: int(float(row["llmesr_item_idx"])))
    fieldnames = list(dict.fromkeys(BASE_FIELDS + [field for row in sorted_rows for field in row.keys()]))
    output_path = output_path or item_text_path
    _write_csv(sorted_rows, output_path, fieldnames)

    after_title = sum(1 for row in sorted_rows if _usable_text(row.get("candidate_title")))
    after_text = sum(1 for row in sorted_rows if _usable_text(row.get("embedding_text")))
    after_non_id_text = sum(1 for row in sorted_rows if _non_id_embedding_text(row))
    summary = {
        "adapter_dir": str(adapter_dir),
        "processed_dir": str(processed_dir),
        "item_rows": len(sorted_rows),
        "title_coverage_before": float(before_title / len(sorted_rows)) if sorted_rows else 0.0,
        "title_coverage_after": float(after_title / len(sorted_rows)) if sorted_rows else 0.0,
        "embedding_text_coverage_before": float(before_text / len(sorted_rows)) if sorted_rows else 0.0,
        "embedding_text_coverage_after": float(after_text / len(sorted_rows)) if sorted_rows else 0.0,
        "non_id_embedding_text_coverage_before": float(before_non_id_text / len(sorted_rows)) if sorted_rows else 0.0,
        "non_id_embedding_text_coverage_after": float(after_non_id_text / len(sorted_rows)) if sorted_rows else 0.0,
        "source_counts": dict(source_counts),
        "output_path": str(output_path),
    }
    summary_path = summary_path or adapter_dir / "item_text_enrichment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = enrich_item_text_seed(
        adapter_dir=Path(args.adapter_dir).expanduser(),
        processed_dir=Path(args.processed_dir).expanduser(),
        raw_metadata_paths=[Path(path).expanduser() for path in args.raw_metadata_path],
        output_path=Path(args.output_path).expanduser() if args.output_path else None,
        summary_path=Path(args.summary_path).expanduser() if args.summary_path else None,
        max_text_chars=args.max_text_chars,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
