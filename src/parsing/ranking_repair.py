from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.prompts.parsers import extract_first_json_object


@dataclass(frozen=True)
class SafeRepairResult:
    ranking: list[str]
    usable_ranking: bool
    repaired_by: str | None
    repair_reason: str | None
    strict_json_valid: bool
    confidence_available_after_repair: bool


def _extract_ranking_like_ids(obj: dict[str, Any] | None) -> list[str]:
    if not isinstance(obj, dict):
        return []
    value = obj.get("ranking")
    if value is None:
        value = obj.get("ranked_item_ids")
    if value is None:
        value = obj.get("topk_item_ids")
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _ids_in_generation_order(text: str, allowed_item_ids: list[str]) -> list[str]:
    clean = str(text or "")
    hits: list[tuple[int, str]] = []
    for item_id in allowed_item_ids:
        for m in re.finditer(rf"\b{re.escape(str(item_id))}\b", clean):
            hits.append((m.start(), str(item_id)))
    return [item_id for _, item_id in sorted(hits)]


def _dedupe_keep_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _complete_with_candidate_order(seed_ids: list[str], allowed_item_ids: list[str]) -> list[str]:
    deduped = _dedupe_keep_order([x for x in seed_ids if x in set(allowed_item_ids)])
    missing = [x for x in allowed_item_ids if x not in set(deduped)]
    return deduped + missing


def safe_repair_ranking(
    *,
    raw_text: str,
    allowed_item_ids: list[str],
    strict_json_valid: bool,
    strict_ranking: list[str],
    strict_confidence_available: bool,
) -> SafeRepairResult:
    allowed = [str(x) for x in allowed_item_ids]
    if strict_json_valid:
        return SafeRepairResult(
            ranking=list(strict_ranking),
            usable_ranking=True,
            repaired_by=None,
            repair_reason=None,
            strict_json_valid=True,
            confidence_available_after_repair=bool(strict_confidence_available),
        )
    blob = extract_first_json_object(raw_text)
    seed_ids: list[str] = []
    if blob:
        import json

        try:
            obj = json.loads(blob)
            seed_ids = _extract_ranking_like_ids(obj if isinstance(obj, dict) else None)
        except Exception:
            seed_ids = []
    if not seed_ids:
        seed_ids = _ids_in_generation_order(raw_text, allowed)
    repaired = _complete_with_candidate_order(seed_ids, allowed)
    usable = bool(repaired) and len(repaired) == len(allowed) and set(repaired) == set(allowed)
    return SafeRepairResult(
        ranking=repaired if usable else [],
        usable_ranking=usable,
        repaired_by="candidate_completion_fallback",
        repair_reason="strict_json_invalid_then_candidate_completion",
        strict_json_valid=False,
        confidence_available_after_repair=False,
    )


def failure_taxonomy_row(
    *,
    user_id: str,
    request_id: str,
    raw_text: str,
    strict_json_valid: bool,
    strict_ranking: list[str],
    allowed_item_ids: list[str],
    missing_confidence: bool,
    malformed_json: bool,
    duplicate_item: bool,
    output_not_in_candidate_set: bool,
    max_new_tokens: int,
) -> dict[str, Any]:
    raw = str(raw_text or "")
    blob = extract_first_json_object(raw)
    stripped = raw.strip()
    prose_around_json = bool(blob and stripped != blob.strip())
    allowed = [str(x) for x in allowed_item_ids]
    no_json_object = blob is None
    incomplete_ranking = bool(blob) and (len(strict_ranking) != len(allowed) or set(strict_ranking) != set(allowed))
    thinking_text_leaked = ("<think" in raw.lower()) or ("</think>" in raw.lower())
    truncation = bool(max_new_tokens > 0 and len(raw.split()) >= int(max_new_tokens) and not raw.rstrip().endswith("}"))
    malformed_quotes_brackets = bool(malformed_json)
    if strict_json_valid:
        primary = "strict_valid"
    else:
        order = [
            ("no_json_object", no_json_object),
            ("prose_before_after_json", prose_around_json),
            ("incomplete_ranking", incomplete_ranking),
            ("duplicate_item_ids", bool(duplicate_item)),
            ("ood_item_ids", bool(output_not_in_candidate_set)),
            ("malformed_quotes_brackets", malformed_quotes_brackets),
            ("confidence_missing", bool(missing_confidence)),
            ("thinking_text_leaked", thinking_text_leaked),
            ("truncation_or_max_new_tokens_issue", truncation),
        ]
        primary = next((name for name, ok in order if ok), "other")
    return {
        "request_id": request_id,
        "user_id": user_id,
        "strict_json_valid": bool(strict_json_valid),
        "primary_failure": primary,
        "no_json_object": int(no_json_object),
        "prose_before_after_json": int(prose_around_json),
        "incomplete_ranking": int(incomplete_ranking),
        "duplicate_item_ids": int(bool(duplicate_item)),
        "ood_item_ids": int(bool(output_not_in_candidate_set)),
        "malformed_quotes_brackets": int(malformed_quotes_brackets),
        "confidence_missing": int(bool(missing_confidence)),
        "thinking_text_leaked": int(thinking_text_leaked),
        "truncation_or_max_new_tokens_issue": int(truncation),
        "other": int(primary == "other"),
    }


def write_failure_taxonomy_csv(rows: list[dict[str, Any]], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def build_repair_summary(pred_rows: list[dict[str, Any]], taxonomy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(1, len(pred_rows))
    strict_valid = sum(1 for r in pred_rows if bool(r.get("strict_json_valid")))
    usable_after = sum(1 for r in pred_rows if bool(r.get("usable_ranking")))
    conf_strict = sum(1 for r in pred_rows if bool(r.get("confidence_available_strict")))
    conf_after = sum(1 for r in pred_rows if bool(r.get("confidence_available")))
    reason_counts: dict[str, int] = {}
    for r in pred_rows:
        rr = str(r.get("repair_reason") or "none")
        reason_counts[rr] = reason_counts.get(rr, 0) + 1
    failure_counts: dict[str, int] = {}
    for row in taxonomy_rows:
        key = str(row.get("primary_failure") or "other")
        failure_counts[key] = failure_counts.get(key, 0) + 1
    return {
        "n_rows": len(pred_rows),
        "strict_json_valid_rate": strict_valid / total,
        "usable_ranking_rate_after_safe_repair": usable_after / total,
        "invalid_output_rate_strict": 1.0 - (strict_valid / total),
        "invalid_output_rate_after_repair": 1.0 - (usable_after / total),
        "confidence_available_rate_strict": conf_strict / total,
        "confidence_available_rate_after_repair": conf_after / total,
        "repair_reason_counts": reason_counts,
        "primary_failure_counts": failure_counts,
        "note": "Safe repair uses candidate set only; no target labels.",
    }
