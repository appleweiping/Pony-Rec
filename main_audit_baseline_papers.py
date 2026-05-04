from __future__ import annotations

import argparse
import csv
import re
import zipfile
from pathlib import Path
from typing import Any


FIELDS = [
    "collection",
    "source_file",
    "file_name",
    "title",
    "method_name",
    "page_count",
    "audit_status",
    "paper_family",
    "runnable_priority",
    "candidate_set_rerank_fit",
    "sequential_history_fit",
    "same_metric_fit",
    "requires_training",
    "requires_external_code_or_model",
    "confidence_output_fit",
    "code_available_signal",
    "detected_code_urls",
    "likely_protocol_gap",
    "recommended_action",
    "notes",
]


TITLE_OVERRIDES = {
    "11465_SLMRec_Distilling_Large_.pdf": (
        "SLMRec: Distilling Large Language Models into Small Models for Sequential Recommendation"
    ),
    "5845_Towards_Unified_Multi_Mod.pdf": (
        "Towards Unified Multi-Modal Personalization: Large Vision-Language Models for Generative Recommendation and Beyond"
    ),
}


METHOD_HINTS = {
    "slmrec": "SLMRec",
    "llm-esr": "LLM-ESR",
    "llmemb": "LLMEmb",
    "llm4rsr": "LLM4RSR",
    "openp5": "OpenP5",
    "recprefer": "RecPrefer",
    "recexplainer": "RecExplainer",
    "sprec": "SPRec",
    "agrec": "AGRec",
    "elmrec": "ELMRec",
    "cove": "CoVE",
    "iagent": "iAgent",
    "lrd": "LRD",
    "ed2": "ED2",
    "msl": "MSL",
    "pad": "PAD",
    "cikgrec": "CIKGRec",
    "llm2rec": "LLM2Rec",
    "rlmrec": "RLMRec",
    "dealrec": "DEALRec",
    "toolrec": "ToolRec",
    "transrec": "TransRec",
    "lard": "LLaRD",
    "corona": "CORONA",
    "irllrec": "IRLLRec",
    "hyperllm": "HyperLLM",
    "setrec": "SETRec",
    "exp3rt": "EXP3RT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit senior-recommended baseline papers into a protocol-gap matrix."
    )
    parser.add_argument("--baseline_root", default="Paper/BASELINE")
    parser.add_argument("--collections", default="NH,NR")
    parser.add_argument("--output_root", default="outputs/summary")
    parser.add_argument("--output_name", default="baseline_paper_audit_matrix")
    parser.add_argument("--max_pages", type=int, default=6)
    parser.add_argument("--include_archives", action="store_true")
    return parser.parse_args()


def _try_pdf_reader() -> Any | None:
    try:
        from pypdf import PdfReader  # type: ignore

        return PdfReader
    except Exception:
        return None


def _extract_pdf_text(path: Path, *, max_pages: int) -> tuple[str, str, int]:
    reader_cls = _try_pdf_reader()
    if reader_cls is None:
        return "", "filename_only_no_pdf_parser", 0

    try:
        reader = reader_cls(str(path))
        page_count = len(reader.pages)
        chunks = []
        for page in reader.pages[: max(1, max_pages)]:
            try:
                chunks.append(page.extract_text() or "")
            except Exception:
                continue
        text = "\n".join(chunks)
        status = "paper_text_extracted" if text.strip() else "paper_text_empty"
        return text, status, page_count
    except Exception as exc:
        return f"extract_error={type(exc).__name__}", "extract_failed", 0


def _normalize_text(text: str) -> str:
    text = text.replace("\u2217", "*").replace("\u223c", "~")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_url(url: str) -> str:
    return url.rstrip(".,;:)]}")


def _detect_urls(text: str) -> list[str]:
    joined = re.sub(r"https://github\.\s*com", "https://github.com", text, flags=re.IGNORECASE)
    joined = re.sub(r"https://github\.\s+", "https://github.", joined, flags=re.IGNORECASE)
    urls = re.findall(r"https?://[^\s\])>,]+|github\.com/[^\s\])>,]+", joined, flags=re.IGNORECASE)
    cleaned = []
    for url in urls:
        clean = _clean_url(url)
        if clean.lower().startswith("github.com/"):
            clean = f"https://{clean}"
        if clean not in cleaned:
            cleaned.append(clean)
    return cleaned


def _line_is_metadata(line: str) -> bool:
    lower = line.lower()
    return (
        not line
        or "proceedings of" in lower
        or "published as" in lower
        or "association for computational linguistics" in lower
        or lower.startswith("november ")
        or lower.startswith("july ")
        or lower.startswith("august ")
        or "copyright" in lower
    )


def _line_looks_like_author(line: str) -> bool:
    tokens = re.findall(r"[A-Za-z][A-Za-z.-]*", line)
    lower = line.lower()
    method_words = {
        "recommendation",
        "recommender",
        "sequential",
        "language",
        "model",
        "models",
        "large",
        "graph",
        "embedding",
        "representation",
        "uncertainty",
        "aligning",
        "controllable",
        "personalization",
        "framework",
        "identifier",
    }
    if any(word in lower for word in method_words):
        return False
    if "," in line or " and " in lower:
        return True
    return 1 <= len(tokens) <= 4 and all(token[:1].isupper() for token in tokens)


def _title_from_text(path: Path, text: str) -> str:
    if path.name in TITLE_OVERRIDES:
        return TITLE_OVERRIDES[path.name]

    lines = [_normalize_text(line) for line in text.splitlines()]
    lines = [line for line in lines if line and not _line_is_metadata(line)]
    title_lines: list[str] = []
    for line in lines[:30]:
        lower = line.lower()
        if lower == "abstract":
            break
        if "@" in line or "university" in lower or "institute" in lower or "school of" in lower:
            if title_lines:
                break
            continue
        if title_lines and _line_looks_like_author(line):
            break
        title_lines.append(line)
        if len(title_lines) >= 3:
            break
    if title_lines:
        title = " ".join(title_lines)
        title = title.replace(" - ", "-").replace(" :", ":")
        return _normalize_text(title)
    return _title_from_filename(path.name)


def _title_from_filename(file_name: str) -> str:
    stem = Path(file_name).stem
    stem = re.sub(r"^\d+[_-]", "", stem)
    stem = stem.replace("_", " ").replace("-", " ")
    stem = re.sub(r"\s+", " ", stem)
    return stem.strip()


def _method_from_title(title: str, file_name: str, urls: list[str]) -> str:
    haystack = " ".join([title, file_name, " ".join(urls)]).lower()
    for key, value in METHOD_HINTS.items():
        if key in haystack:
            return value
    before_colon = title.split(":", 1)[0].strip()
    if 2 <= len(before_colon) <= 32 and re.search(r"[A-Za-z]", before_colon):
        return before_colon
    return ""


def _paper_family(title: str) -> str:
    lower = title.lower()
    if "uncertainty" in lower:
        return "uncertainty_llm_recommendation"
    if "long-tail" in lower or "long tailed" in lower or "long-tailed" in lower:
        return "long_tail_sequential_recommendation"
    if "controllable" in lower or "shield" in lower or "agent" in lower:
        return "controllable_or_agent_recommendation"
    if "multi-modal" in lower or "vision-language" in lower or "multi mod" in lower:
        return "multimodal_multitask_recommendation"
    if "knowledge" in lower or "graph" in lower or "hyperbolic" in lower:
        return "knowledge_graph_or_graph_recommendation"
    if "embedding" in lower or "representation" in lower or "identifier" in lower:
        return "embedding_or_representation_recommendation"
    if "denois" in lower or "robust" in lower or "data corrector" in lower:
        return "robustness_or_denoising_recommendation"
    if "open-source platform" in lower or "platform" in lower:
        return "open_platform_reference"
    if "explaining" in lower or "evaluating" in lower or "understand" in lower:
        return "evaluation_or_explanation_reference"
    if "sequential" in lower:
        return "llm_sequential_recommendation"
    return "general_llm_recommendation"


def _has_code(urls: list[str], text: str) -> bool:
    lower = text.lower()
    return any("github" in url.lower() for url in urls) or "code is available" in lower or "codes are available" in lower


def _priority_and_action(
    *,
    family: str,
    title: str,
    method: str,
    has_code: bool,
) -> tuple[str, str, str]:
    lower = title.lower()
    method_lower = method.lower()

    if not has_code:
        return (
            "D_related_only",
            "proxy_only",
            "Use as related-work positioning unless code is found and protocol can be matched.",
        )

    if family in {
        "llm_sequential_recommendation",
        "long_tail_sequential_recommendation",
        "embedding_or_representation_recommendation",
        "open_platform_reference",
    }:
        return (
            "B_adapter_candidate",
            "adapter_candidate",
            "Inspect repository and add an adapter only if it can score the exact candidate set.",
        )

    if "uncertainty" in lower:
        return (
            "B_adapter_candidate",
            "diagnostic_reference",
            "Audit uncertainty outputs first; use as a diagnostic baseline unless it emits comparable ranks.",
        )

    if method_lower in {"recexplainer", "iagent"} or family == "evaluation_or_explanation_reference":
        return (
            "C_proxy_only",
            "proxy_or_related_work",
            "Keep out of the main result table unless converted to candidate ranking predictions.",
        )

    return (
        "C_proxy_only",
        "adapter_after_classical_baselines",
        "Defer until SASRec/BERT4Rec/GRU4Rec/LightGCN and higher-fit sequential methods are aligned.",
    )


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _audit_pdf(path: Path, baseline_root: Path, *, max_pages: int) -> dict[str, Any]:
    text, status, page_count = _extract_pdf_text(path, max_pages=max_pages)
    normalized = _normalize_text(text)
    urls = _detect_urls(text)
    title = _title_from_text(path, text)
    method = _method_from_title(title, path.name, urls)
    family = _paper_family(title)
    code_found = _has_code(urls, normalized)
    priority, role, action = _priority_and_action(
        family=family,
        title=title,
        method=method,
        has_code=code_found,
    )
    lower = f"{title} {normalized}".lower()
    sequential_fit = "high" if "sequential" in lower or "history" in lower else "medium"
    candidate_fit = "medium_requires_adapter" if priority.startswith("B") else "low_or_unknown"
    same_metric_fit = "needs_same_candidate_metric_adapter"
    confidence_fit = "uncertainty_related" if "uncertainty" in lower else "no_confidence_output_detected"
    requires_training = any(
        token in lower
        for token in [
            "fine-tuning",
            "fine tuning",
            "train",
            "distill",
            "pre-train",
            "embedding",
            "graph",
            "lora",
        ]
    )
    protocol_gap = [
        "different_candidate_space_until_adapter_exists",
        "same_split_not_verified",
        "same_metric_not_verified",
    ]
    if "amazon" not in lower and "movie" not in lower and "yelp" not in lower:
        protocol_gap.append("dataset_overlap_not_verified")
    if confidence_fit != "uncertainty_related":
        protocol_gap.append("no_relevance_confidence_output")

    try:
        source_file = str(path.relative_to(baseline_root))
        collection = path.relative_to(baseline_root).parts[0]
    except ValueError:
        source_file = str(path)
        collection = path.parent.name

    return {
        "collection": collection,
        "source_file": source_file.replace("\\", "/"),
        "file_name": path.name,
        "title": title,
        "method_name": method,
        "page_count": page_count,
        "audit_status": status,
        "paper_family": family,
        "runnable_priority": priority,
        "candidate_set_rerank_fit": candidate_fit,
        "sequential_history_fit": sequential_fit,
        "same_metric_fit": same_metric_fit,
        "requires_training": _yes_no(requires_training),
        "requires_external_code_or_model": _yes_no(code_found or requires_training),
        "confidence_output_fit": confidence_fit,
        "code_available_signal": "code_url_detected" if code_found else "not_detected",
        "detected_code_urls": ";".join(urls),
        "likely_protocol_gap": ";".join(protocol_gap),
        "recommended_action": action,
        "notes": role,
    }


def _audit_archive(path: Path, baseline_root: Path) -> dict[str, Any]:
    count = 0
    try:
        with zipfile.ZipFile(path) as archive:
            count = len([name for name in archive.namelist() if name.lower().endswith(".pdf")])
        status = "archive_inventory_only"
    except Exception:
        status = "archive_inventory_failed"
    return {
        "collection": path.parent.name,
        "source_file": str(path.relative_to(baseline_root)).replace("\\", "/"),
        "file_name": path.name,
        "title": _title_from_filename(path.name),
        "method_name": "",
        "page_count": "",
        "audit_status": status,
        "paper_family": "baseline_archive",
        "runnable_priority": "D_related_only",
        "candidate_set_rerank_fit": "not_applicable",
        "sequential_history_fit": "not_applicable",
        "same_metric_fit": "not_applicable",
        "requires_training": "unknown",
        "requires_external_code_or_model": "unknown",
        "confidence_output_fit": "unknown",
        "code_available_signal": "archive_not_scanned",
        "detected_code_urls": "",
        "likely_protocol_gap": "archive_needs_extraction_before_paper_audit",
        "recommended_action": f"Archive contains {count} PDF files; audit extracted PDFs instead.",
        "notes": "archive_inventory",
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "\n"
    lines = [
        "| " + " | ".join(FIELDS) + " |",
        "| " + " | ".join(["---"] * len(FIELDS)) + " |",
    ]
    for row in rows:
        values = [str(row.get(field, "")).replace("\n", " ").replace("|", "/") for field in FIELDS]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    baseline_root = Path(args.baseline_root).expanduser()
    collections = [item.strip() for item in args.collections.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    for collection in collections:
        collection_root = baseline_root / collection
        for path in sorted(collection_root.glob("*.pdf")):
            rows.append(_audit_pdf(path, baseline_root, max_pages=args.max_pages))

    if args.include_archives:
        for archive_path in sorted(baseline_root.glob("*.zip")):
            rows.append(_audit_archive(archive_path, baseline_root))

    rows.sort(
        key=lambda row: (
            str(row.get("runnable_priority", "")),
            str(row.get("collection", "")),
            str(row.get("method_name", "")),
            str(row.get("file_name", "")),
        )
    )
    output_root = Path(args.output_root).expanduser()
    csv_path = output_root / f"{args.output_name}.csv"
    md_path = output_root / f"{args.output_name}.md"
    _write_csv(rows, csv_path)
    md_path.write_text(_to_markdown(rows), encoding="utf-8")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
