from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from src.baselines.same_candidate_external import load_score_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit exact-candidate score coverage before importing an external baseline.")
    parser.add_argument("--candidate_items_path", required=True)
    parser.add_argument("--scores_path", required=True)
    parser.add_argument("--user_col", default="user_id")
    parser.add_argument("--item_col", default="item_id")
    parser.add_argument("--score_col", default="score")
    parser.add_argument("--source_event_col", default="source_event_id")
    parser.add_argument(
        "--allow_user_item_fallback",
        action="store_true",
        help="Allow user_id/item_id coverage when source_event_id keys are incomplete.",
    )
    parser.add_argument(
        "--allow_extra_score_keys",
        action="store_true",
        help="Allow score rows not present in candidate_items.csv.",
    )
    parser.add_argument(
        "--no_fail",
        action="store_true",
        help="Print diagnostics without returning a non-zero status.",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def main() -> None:
    args = parse_args()
    candidate_rows = _read_csv(Path(args.candidate_items_path).expanduser())
    score_rows = load_score_rows(Path(args.scores_path).expanduser())

    candidate_event_keys = set()
    candidate_user_item_keys = set()
    for row in candidate_rows:
        source_event_id = _text(row.get("source_event_id"))
        user_id = _text(row.get("user_id"))
        item_id = _text(row.get("item_id"))
        if source_event_id and user_id and item_id:
            candidate_event_keys.add((source_event_id, user_id, item_id))
        if user_id and item_id:
            candidate_user_item_keys.add((user_id, item_id))

    score_event_keys = set()
    score_user_item_keys = set()
    blank_score_keys = 0
    invalid_scores = 0
    duplicate_score_event_keys = 0
    duplicate_score_user_item_keys = 0
    for row in score_rows:
        source_event_id = _text(row.get(args.source_event_col))
        user_id = _text(row.get(args.user_col))
        item_id = _text(row.get(args.item_col))
        if not user_id or not item_id:
            blank_score_keys += 1
            continue
        try:
            score = float(row.get(args.score_col))
        except Exception:
            invalid_scores += 1
            continue
        if not math.isfinite(score):
            invalid_scores += 1
            continue
        user_item_key = (user_id, item_id)
        duplicate_score_user_item_keys += int(user_item_key in score_user_item_keys)
        score_user_item_keys.add(user_item_key)
        if source_event_id:
            event_key = (source_event_id, user_id, item_id)
            duplicate_score_event_keys += int(event_key in score_event_keys)
            score_event_keys.add(event_key)

    exact_matches = candidate_event_keys & score_event_keys
    fallback_matches = candidate_user_item_keys & score_user_item_keys
    missing_event_keys = candidate_event_keys - score_event_keys
    extra_event_keys = score_event_keys - candidate_event_keys
    extra_user_item_keys = score_user_item_keys - candidate_user_item_keys
    exact_rate = len(exact_matches) / len(candidate_event_keys) if candidate_event_keys else 0.0
    fallback_rate = len(fallback_matches) / len(candidate_user_item_keys) if candidate_user_item_keys else 0.0

    print(f"candidate_rows={len(candidate_rows)}")
    print(f"candidate_event_keys={len(candidate_event_keys)}")
    print(f"candidate_user_item_keys={len(candidate_user_item_keys)}")
    print(f"score_rows={len(score_rows)}")
    print(f"score_event_keys={len(score_event_keys)}")
    print(f"score_user_item_keys={len(score_user_item_keys)}")
    print(f"blank_score_keys={blank_score_keys}")
    print(f"invalid_scores={invalid_scores}")
    print(f"duplicate_score_event_keys={duplicate_score_event_keys}")
    print(f"duplicate_score_user_item_keys={duplicate_score_user_item_keys}")
    print(f"exact_event_key_matches={len(exact_matches)}")
    print(f"exact_event_key_match_rate={exact_rate:.6f}")
    print(f"user_item_key_matches={len(fallback_matches)}")
    print(f"user_item_key_match_rate={fallback_rate:.6f}")
    print(f"missing_event_keys={len(missing_event_keys)}")
    print(f"extra_event_keys={len(extra_event_keys)}")
    print(f"extra_user_item_keys={len(extra_user_item_keys)}")

    failures = []
    if len(score_rows) != len(candidate_rows):
        print("diagnosis=score row count does not equal candidate row count; regenerate scores from candidate_items.csv")
        failures.append("row_count_mismatch")
    elif exact_rate < 1.0 and fallback_rate < 1.0:
        print("diagnosis=score rows exist but keys do not align with candidate_items.csv")
        failures.append("coverage_incomplete")
    elif exact_rate == 1.0:
        print("diagnosis=exact source_event_id/user_id/item_id coverage is complete")
    else:
        print("diagnosis=user_id/item_id fallback coverage is complete but exact event IDs are incomplete")
        if not args.allow_user_item_fallback:
            failures.append("exact_event_key_coverage_incomplete")

    if blank_score_keys:
        failures.append("blank_score_keys")
    if invalid_scores:
        failures.append("invalid_scores")
    if duplicate_score_event_keys:
        failures.append("duplicate_score_event_keys")
    if extra_event_keys and not args.allow_extra_score_keys:
        failures.append("extra_event_keys")
    if extra_user_item_keys and not args.allow_extra_score_keys:
        failures.append("extra_user_item_keys")

    if failures:
        print(f"audit_ok=False")
        print(f"failures={','.join(failures)}")
        if not args.no_fail:
            raise SystemExit(1)
    else:
        print("audit_ok=True")


if __name__ == "__main__":
    main()
