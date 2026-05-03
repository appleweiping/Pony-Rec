from __future__ import annotations

from src.backends.hf_local_backend import _truncate_by_stop_strings, _truncate_to_first_json_object_text


def test_truncate_to_first_json_object_text_keeps_first_object_only() -> None:
    text = 'prefix {"ranking":["A","B"],"confidence":0.7} trailing prose'
    out = _truncate_to_first_json_object_text(text)
    assert out == '{"ranking":["A","B"],"confidence":0.7}'


def test_truncate_to_first_json_object_text_passthrough_when_no_object() -> None:
    text = "plain text without json"
    assert _truncate_to_first_json_object_text(text) == text


def test_truncate_by_stop_strings_uses_earliest_match() -> None:
    text = '{"ranking":["A"]}\n\nextra\n/ignored'
    out = _truncate_by_stop_strings(text, ["\n/", "\n\n"])
    assert out == '{"ranking":["A"]}'
