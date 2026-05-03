from __future__ import annotations

import pandas as pd

from src.data.item_text_cleaning import (
    CleaningConfig,
    build_cleaned_lookup_for_ids,
    build_prompt_line_for_item,
    clean_free_text,
    missing_title_placeholder,
    normalize_whitespace,
    remove_control_chars,
    strip_html_tags,
)


def test_strip_html_tags() -> None:
    assert "bold" in strip_html_tags("<b>bold</b> text")
    assert "<" not in strip_html_tags("<b>bold</b> text")


def test_remove_control_chars_keeps_tab_newline() -> None:
    s = "a\x00b\tc\nd"
    out = remove_control_chars(s)
    assert "\t" in out and "\n" in out
    assert "\x00" not in out


def test_normalize_whitespace() -> None:
    assert normalize_whitespace("  a  \n  b  ") == "a b"


def test_clean_free_text_truncates() -> None:
    cfg = CleaningConfig(max_title_chars=20, ellipsis="…")
    long = "x" * 50
    out = clean_free_text(long, max_chars=10, config=cfg)
    assert len(out) == 10
    assert out.endswith("…")


def test_missing_title_placeholder() -> None:
    assert "B00" in missing_title_placeholder("B00TESTITEM")


def test_build_prompt_line_prefers_title_categories_not_huge_desc() -> None:
    line = build_prompt_line_for_item(
        "id1",
        title="Hello",
        categories="Fiction",
        description="D" * 5000,
        candidate_text="",
        config=CleaningConfig(include_description=False),
    )
    assert "Hello" in line
    assert "Fiction" in line
    assert "D" * 100 not in line


def test_build_cleaned_lookup_for_ids_subset() -> None:
    df = pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "title": ['<p>Hi</p>\x00', ""],
            "categories": ["", ""],
            "description": ["", ""],
            "candidate_text": ["", ""],
        }
    )
    out = build_cleaned_lookup_for_ids(df, {"a", "b"}, config=CleaningConfig())
    assert "a" in out
    assert "Hi" in out["a"] or "hi" in out["a"].lower()
