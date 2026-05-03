"""Deterministic item / candidate text cleaning for prompt construction.

Does **not** modify ``data/processed`` on disk. Intended for a prompt-view layer
or in-memory lookup passed to ``history_block`` / ``candidate_block``.

Rules (documented; stable across runs):
- Strip HTML tags (regex), then ``html.unescape``.
- Remove C0 control characters except tab/newline; normalize other whitespace runs.
- Optionally truncate title and category strings to fixed max UTF-8 character lengths.
- Prefer **title + categories**; do **not** append full description by default (opt-in).
- Missing / placeholder titles become a stable short line keyed by ``item_id``.
- ``item_id`` is passed through unchanged (cleaning applies to prose fields only).
- For machine-generated JSON payloads elsewhere, use ``json.dumps`` — this module
  cleans human-readable lines embedded in listwise templates.
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from dataclasses import dataclass

# C0 controls excluding tab (0x09) and newline (0x0a)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HTML_TAG_RE = re.compile(r"<[^>]{0,2000}?>")

_PLACEHOLDER_TITLES = frozenset(
    {
        "",
        "unknown",
        "none",
        "nan",
        "n/a",
        "null",
        "untitled",
        "no title",
        "missing title",
    }
)


def _is_placeholder_title(title: str) -> bool:
    t = str(title).strip().lower()
    return t in _PLACEHOLDER_TITLES


@dataclass(frozen=True)
class CleaningConfig:
    max_title_chars: int = 280
    max_categories_chars: int = 160
    include_description: bool = False
    max_description_chars: int = 400
    ellipsis: str = "…"


def strip_html_tags(text: str) -> str:
    return _HTML_TAG_RE.sub(" ", text)


def remove_control_chars(text: str) -> str:
    return _CONTROL_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    # Unicode normalize for stable comparisons; collapse internal space
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in t.split("\n")]
    return " ".join(p for p in parts if p)


def truncate_chars(text: str, max_chars: int, *, ellipsis: str = "…") -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= len(ellipsis):
        return text[:max_chars]
    return text[: max_chars - len(ellipsis)].rstrip() + ellipsis


def clean_free_text(text: str, *, max_chars: int, config: CleaningConfig) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    raw = html.unescape(raw)
    raw = strip_html_tags(raw)
    raw = remove_control_chars(raw)
    raw = normalize_whitespace(raw)
    return truncate_chars(raw, max_chars, ellipsis=config.ellipsis)


def missing_title_placeholder(item_id: str) -> str:
    return f"Title: (missing metadata) item_id={item_id}"


def build_prompt_line_for_item(
    item_id: str,
    *,
    title: str = "",
    categories: str = "",
    description: str = "",
    candidate_text: str = "",
    config: CleaningConfig | None = None,
) -> str:
    """Single-line style text for one item (history or candidate row), listwise-safe."""
    cfg = config or CleaningConfig()
    base = str(candidate_text or "").strip()
    if base:
        line = clean_free_text(base, max_chars=max(cfg.max_title_chars, cfg.max_description_chars), config=cfg)
        if line:
            return line
    title_clean = clean_free_text(title, max_chars=cfg.max_title_chars, config=cfg)
    cat_clean = clean_free_text(categories, max_chars=cfg.max_categories_chars, config=cfg)
    if _is_placeholder_title(title_clean) or not title_clean:
        title_clean = missing_title_placeholder(item_id)
    else:
        title_clean = f"Title: {title_clean}"
    parts = [title_clean]
    if cat_clean:
        parts.append(f"Categories: {cat_clean}")
    if cfg.include_description and str(description or "").strip():
        desc = clean_free_text(description, max_chars=cfg.max_description_chars, config=cfg)
        if desc:
            parts.append(f"Description: {desc}")
    return " ".join(parts)


def build_item_lookup_for_prompts(
    items_df,
    *,
    config: CleaningConfig | None = None,
) -> dict[str, str]:
    """Map item_id -> cleaned display string (pandas DataFrame rows)."""
    import pandas as pd

    if not isinstance(items_df, pd.DataFrame):
        raise TypeError("items_df must be a pandas DataFrame")
    cfg = config or CleaningConfig()
    col = None
    for c in ("item_id", "parent_asin", "asin", "movieId", "iid"):
        if c in items_df.columns:
            col = c
            break
    if col is None:
        return {}
    out: dict[str, str] = {}
    for _, row in items_df.iterrows():
        iid = str(row[col]).strip()
        if not iid:
            continue
        title = str(row.get("title", "") or "")
        categories = str(row.get("categories", "") or "")
        description = str(row.get("description", "") or "")
        cand = str(row.get("candidate_text", "") or "")
        out[iid] = build_prompt_line_for_item(
            iid,
            title=title,
            categories=categories,
            description=description,
            candidate_text=cand,
            config=cfg,
        )
    return out


def json_escape_for_debug(payload: dict) -> str:
    """Serialize with json.dumps (UTF-8 safe, escaped quotes/newlines)."""
    return json.dumps(payload, ensure_ascii=False)


def item_id_column(items_df) -> str | None:
    """Return the canonical item id column name in a processed items DataFrame."""
    import pandas as pd

    if not isinstance(items_df, pd.DataFrame):
        raise TypeError("items_df must be a pandas DataFrame")
    for c in ("item_id", "parent_asin", "asin", "movieId", "iid"):
        if c in items_df.columns:
            return c
    return None


def build_cleaned_lookup_for_ids(
    items_df,
    needed_ids: set[str],
    *,
    config: CleaningConfig | None = None,
) -> dict[str, str]:
    """Build cleaned lookup strings only for rows whose ids appear in ``needed_ids`` (fast for huge catalogs)."""
    import pandas as pd

    col = item_id_column(items_df)
    if col is None:
        return {}
    mask = items_df[col].astype(str).str.strip().isin({str(x).strip() for x in needed_ids})
    sub = items_df.loc[mask].copy()
    if sub.empty:
        return {}
    if col != "item_id":
        sub = sub.rename(columns={col: "item_id"})
    return build_item_lookup_for_prompts(sub, config=config)
