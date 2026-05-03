from __future__ import annotations

from src.cli.run_pilot_reprocessed_deepseek import parse_args


def test_parse_args_accepts_optional_topk() -> None:
    ns = parse_args(["--topk", "99"])
    assert ns.topk == 99


def test_parse_args_topk_defaults_to_none() -> None:
    ns = parse_args([])
    assert ns.topk is None
