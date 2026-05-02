from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ParsedLLMOutput:
    task: str
    raw_text: str
    parsed_json: dict[str, Any] | None
    recommend: str | None = None
    ranked_item_ids: list[str] | None = None
    predicted_item_id: str | None = None
    confidence: float | None = None
    is_valid: bool = False
    invalid_output: bool = False
    malformed_json: bool = False
    repaired_json: bool = False
    hallucinated_item: bool = False
    duplicate_item: bool = False
    missing_confidence: bool = False
    output_not_in_candidate_set: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def strip_reasoning(text: str) -> str:
    clean = re.sub(r"<think\b[^>]*>.*?</think>", "", str(text or ""), flags=re.I | re.S)
    clean = re.sub(r"^```[a-zA-Z0-9_]*\s*", "", clean.strip())
    clean = re.sub(r"\s*```$", "", clean)
    return clean.strip()


def _extract_json(text: str) -> tuple[dict[str, Any] | None, bool, bool]:
    clean = strip_reasoning(text)
    candidates = [clean]
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end > start:
        candidates.append(clean[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate), False, False
        except Exception:
            pass
    if start >= 0 and end > start:
        repaired = clean[start : end + 1]
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = repaired.replace("'", '"')
        try:
            return json.loads(repaired), True, True
        except Exception:
            return None, True, False
    return None, False, False


def normalize_confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        val = float(value)
    else:
        text = str(value).strip()
        match = re.search(r"[-+]?\d*\.?\d+", text)
        if not match:
            return None
        val = float(match.group(0))
        if "%" in text:
            val /= 100.0
    if 1.0 < val <= 100.0:
        val /= 100.0
    if val < 0.0 or val > 1.0:
        return None
    return float(val)


def _normalize_item_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict):
                item_id = str(item.get("item_id") or item.get("id") or "").strip()
            else:
                item_id = str(item).strip()
            if item_id:
                out.append(item_id)
        return out
    text = str(value)
    return [token.strip().strip('"').strip("'") for token in re.split(r"[\s,\[\]]+", text) if token.strip()]


def _flags_for_items(items: list[str], allowed_item_ids: list[str] | None) -> tuple[bool, bool]:
    duplicate = len(items) != len(set(items))
    not_allowed = False
    if allowed_item_ids is not None:
        allowed = set(str(x) for x in allowed_item_ids)
        not_allowed = any(item_id not in allowed for item_id in items)
    return duplicate, not_allowed


def parse_pointwise_output(text: str) -> ParsedLLMOutput:
    obj, malformed, repaired = _extract_json(text)
    recommend = None
    confidence = None
    if obj:
        raw_rec = str(obj.get("recommend", obj.get("answer", ""))).strip().lower()
        recommend = raw_rec if raw_rec in {"yes", "no"} else None
        confidence = normalize_confidence(obj.get("confidence"))
    else:
        clean = strip_reasoning(text)
        rec_match = re.search(r"\b(yes|no)\b", clean, re.I)
        conf_match = re.search(r"(?:confidence|confident)\D+([0-9]*\.?[0-9]+%?)", clean, re.I)
        recommend = rec_match.group(1).lower() if rec_match else None
        confidence = normalize_confidence(conf_match.group(1)) if conf_match else None
    valid = recommend in {"yes", "no"} and confidence is not None
    return ParsedLLMOutput(
        task="pointwise",
        raw_text=text,
        parsed_json=obj,
        recommend=recommend,
        confidence=confidence,
        is_valid=valid,
        invalid_output=not valid,
        malformed_json=malformed,
        repaired_json=repaired,
        missing_confidence=confidence is None,
    )


def parse_ranking_output(
    text: str,
    *,
    allowed_item_ids: list[str],
    topk: int | None = None,
) -> ParsedLLMOutput:
    obj, malformed, repaired = _extract_json(text)
    clean = strip_reasoning(text)
    confidence = None
    ranked: list[str] = []
    if obj:
        ranked = _normalize_item_ids(
            obj.get("ranked_item_ids")
            or obj.get("topk_item_ids")
            or obj.get("ranking")
            or obj.get("predicted_item_id")
        )
        confidence = normalize_confidence(obj.get("confidence"))
    if not ranked:
        matches = []
        for item_id in allowed_item_ids:
            for match in re.finditer(rf"\b{re.escape(str(item_id))}\b", clean):
                matches.append((match.start(), str(item_id)))
        ranked = [item_id for _, item_id in sorted(matches)]
    if topk is not None:
        ranked = ranked[: int(topk)]
    duplicate, not_allowed = _flags_for_items(ranked, allowed_item_ids)
    deduped = list(dict.fromkeys(ranked))
    valid = bool(deduped) and not not_allowed and confidence is not None
    return ParsedLLMOutput(
        task="ranking",
        raw_text=text,
        parsed_json=obj,
        ranked_item_ids=deduped,
        predicted_item_id=deduped[0] if deduped else None,
        confidence=confidence,
        is_valid=valid,
        invalid_output=not valid,
        malformed_json=malformed,
        repaired_json=repaired,
        hallucinated_item=not_allowed,
        duplicate_item=duplicate,
        missing_confidence=confidence is None,
        output_not_in_candidate_set=not_allowed,
    )


def parse_pairwise_output(text: str, *, item_a_id: str, item_b_id: str) -> ParsedLLMOutput:
    obj, malformed, repaired = _extract_json(text)
    preferred = None
    confidence = None
    if obj:
        preferred = str(obj.get("preferred_item") or obj.get("winner") or "").strip()
        confidence = normalize_confidence(obj.get("confidence"))
    else:
        match = re.search(r"\b(A|B|" + re.escape(item_a_id) + r"|" + re.escape(item_b_id) + r")\b", strip_reasoning(text))
        preferred = match.group(1) if match else None
    if preferred == "A":
        preferred = item_a_id
    if preferred == "B":
        preferred = item_b_id
    not_allowed = preferred not in {item_a_id, item_b_id}
    valid = bool(preferred) and not not_allowed and confidence is not None
    return ParsedLLMOutput(
        task="pairwise",
        raw_text=text,
        parsed_json=obj,
        predicted_item_id=preferred if not not_allowed else None,
        confidence=confidence,
        is_valid=valid,
        invalid_output=not valid,
        malformed_json=malformed,
        repaired_json=repaired,
        hallucinated_item=not_allowed,
        output_not_in_candidate_set=not_allowed,
        missing_confidence=confidence is None,
    )
