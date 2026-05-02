from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.protocol import (
    DataProtocolConfig,
    load_raw_interactions,
    load_raw_items,
    protocol_config_from_dict,
    iterative_k_core_filter,
)


@dataclass
class RawValidationReport:
    dataset: str
    domain: str
    ok: bool
    errors: list[str]
    warnings: list[str]
    raw_interactions_path: str
    raw_items_path: str | None
    raw_format: str
    expected_interaction_fields: list[str]
    expected_item_fields: list[str]
    num_interactions: int = 0
    num_items_in_metadata: int = 0
    timestamp_parseable_fraction: float = 0.0
    rating_distribution: dict[str, int] | None = None
    duplicate_interactions: int = 0
    metadata_coverage: float = 0.0
    missing_title_fraction: float = 0.0
    missing_categories_fraction: float = 0.0
    missing_description_fraction: float = 0.0
    positive_interactions: int = 0
    estimated_post_k_core_interactions: int = 0
    estimated_post_k_core_users: int = 0
    estimated_post_k_core_items: int = 0
    large_enough_after_filtering: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _expected_fields(config: DataProtocolConfig) -> tuple[list[str], list[str]]:
    if config.raw_format.lower() in {"movie", "movielens", "csv"}:
        return ["userId/user_id", "movieId/item_id", "rating", "timestamp"], ["movieId/item_id", "title", "genres"]
    return ["user_id/reviewerID", "parent_asin/asin", "rating/overall", "timestamp/unixReviewTime"], [
        "parent_asin/asin",
        "title",
        "categories",
        "description",
    ]


def validate_raw_data_config(config_dict: dict[str, Any], *, min_post_filter_interactions: int = 1000) -> RawValidationReport:
    config = protocol_config_from_dict(config_dict)
    expected_interaction_fields, expected_item_fields = _expected_fields(config)
    errors: list[str] = []
    warnings: list[str] = []
    report = RawValidationReport(
        dataset=config.dataset,
        domain=config.domain,
        ok=False,
        errors=errors,
        warnings=warnings,
        raw_interactions_path=str(config.raw_interactions_path),
        raw_items_path=str(config.raw_items_path) if config.raw_items_path else None,
        raw_format=config.raw_format,
        expected_interaction_fields=expected_interaction_fields,
        expected_item_fields=expected_item_fields,
        rating_distribution={},
    )
    if not config.raw_interactions_path.exists():
        errors.append(
            f"Missing raw interactions file: {config.raw_interactions_path}. "
            f"Expected {config.raw_format} with fields {expected_interaction_fields}."
        )
        return report
    if config.raw_items_path is not None and not config.raw_items_path.exists():
        errors.append(
            f"Missing raw metadata file: {config.raw_items_path}. "
            f"Expected {config.raw_format} metadata with fields {expected_item_fields}."
        )
        return report
    try:
        raw_interactions = load_raw_interactions(config)
    except Exception as exc:
        errors.append(f"Failed to parse raw interactions: {exc}")
        return report
    report.num_interactions = int(len(raw_interactions))
    if raw_interactions.empty:
        errors.append("Raw interactions parsed successfully but contain zero valid rows.")
        return report
    report.timestamp_parseable_fraction = float(raw_interactions["timestamp"].notna().mean())
    report.rating_distribution = {
        str(key): int(value) for key, value in raw_interactions["rating"].round().value_counts().sort_index().items()
    }
    report.duplicate_interactions = int(raw_interactions.duplicated(subset=["user_id", "item_id", "timestamp"]).sum())
    positives = raw_interactions[raw_interactions["rating"] >= config.rating_threshold].copy()
    report.positive_interactions = int(len(positives))
    filtered, _ = iterative_k_core_filter(positives, k_core=config.k_core)
    report.estimated_post_k_core_interactions = int(len(filtered))
    report.estimated_post_k_core_users = int(filtered["user_id"].nunique()) if len(filtered) else 0
    report.estimated_post_k_core_items = int(filtered["item_id"].nunique()) if len(filtered) else 0
    report.large_enough_after_filtering = report.estimated_post_k_core_interactions >= min_post_filter_interactions
    if not report.large_enough_after_filtering:
        warnings.append(
            f"Post-filter interactions below threshold: {report.estimated_post_k_core_interactions} < {min_post_filter_interactions}."
        )
    try:
        items = load_raw_items(config, set(raw_interactions["item_id"].astype(str).unique()))
    except Exception as exc:
        errors.append(f"Failed to parse raw metadata: {exc}")
        return report
    report.num_items_in_metadata = int(len(items))
    observed_items = set(raw_interactions["item_id"].astype(str).unique())
    metadata_items = set(items["item_id"].astype(str).unique())
    report.metadata_coverage = len(observed_items & metadata_items) / max(1, len(observed_items))
    report.missing_title_fraction = float((items["title"].fillna("").astype(str).str.strip() == "").mean())
    report.missing_categories_fraction = float((items["categories"].fillna("").astype(str).str.strip() == "").mean())
    report.missing_description_fraction = float((items["description"].fillna("").astype(str).str.strip() == "").mean())
    if report.metadata_coverage < 0.8:
        warnings.append(f"Low metadata coverage: {report.metadata_coverage:.3f}.")
    report.ok = len(errors) == 0
    return report
