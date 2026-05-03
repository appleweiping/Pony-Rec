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
    """Strip common reasoning wrappers and markdown fences before JSON parse."""
    from src.llm.parser import strip_thinking_blocks

    clean = strip_thinking_blocks(str(text or ""))
    # Qwen3 may emit explicit think tags (avoid embedding raw tags in this file for XML-safe patches).
    _o, _c = "<" + "think" + ">", "<" + "/" + "think" + ">"
    clean = re.sub(re.escape(_o) + r"[\s\S]*?" + re.escape(_c), "", clean, flags=re.I)
    clean = re.sub(r"^```[a-zA-Z0-9_]*\s*", "", clean.strip())
    clean = re.sub(r"\s*```$", "", clean)
    return clean.strip()


def extract_first_json_object(text: str) -> str | None:
    """Return the first brace-balanced substring that yields a JSON object (strict or light repair)."""
    s = str(text or "")
    for i, ch in enumerate(s):
        if ch != "{":
            continue
        depth = 0
        for j in range(i, len(s)):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    blob = s[i : j + 1]
                    obj, _, _ = _try_load_json_blob(blob)
                    if obj is not None:
                        return blob
                    break
    return None


def _try_load_json_blob(blob: str) -> tuple[dict[str, Any] | None, bool, bool]:
    """Try strict then light repairs (trailing commas, single quotes). Returns (obj, malformed, repaired)."""
    try:
        out = json.loads(blob)
        if isinstance(out, dict):
            return out, False, False
    except Exception:
        pass
    repaired = re.sub(r",\s*([}\]])", r"\1", blob)
    repaired = repaired.replace("'", '"')
    try:
        out = json.loads(repaired)
        if isinstance(out, dict):
            return out, True, True
    except Exception:
        pass
    return None, True, False


def _extract_json(text: str) -> tuple[dict[str, Any] | None, bool, bool]:
    clean = strip_reasoning(text)
    candidates: list[str] = []
    if clean:
        candidates.append(clean)
    balanced = extract_first_json_object(clean)
    if balanced:
        candidates.append(balanced)
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end > start:
        naive = clean[start : end + 1]
        if naive not in candidates:
            candidates.append(naive)
    for candidate in candidates:
        obj, malformed, repaired = _try_load_json_blob(candidate)
        if obj is not None:
            return obj, malformed, repaired
    return None, bool(start >= 0), False


def ranking_parse_strict_for_prompt(prompt_id: str) -> bool:
    """LoRA listwise debug: strict single JSON object, no prose fallback."""
    return str(prompt_id) == "listwise_ranking_json_lora"


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


def _parse_ranking_strict(
    text: str,
    *,
    allowed_item_ids: list[str],
    obj: dict[str, Any] | None,
    malformed: bool,
    repaired: bool,
) -> ParsedLLMOutput:
    """Strict listwise: one JSON object, full candidate slate, no prose ID recovery, no target-side repair."""
    allowed = [str(x) for x in allowed_item_ids]
    allowed_set = set(allowed)
    confidence: float | None = None
    ranked: list[str] = []
    if obj is not None:
        ranked = _normalize_item_ids(obj.get("ranking") or obj.get("ranked_item_ids") or obj.get("topk_item_ids"))
        confidence = normalize_confidence(obj.get("confidence"))
    duplicate, not_allowed = _flags_for_items(ranked, allowed)
    deduped = list(dict.fromkeys(ranked))
    complete = bool(allowed) and len(deduped) == len(allowed) and set(deduped) == allowed_set and not duplicate
    valid = complete and not not_allowed and confidence is not None
    return ParsedLLMOutput(
        task="ranking",
        raw_text=text,
        parsed_json=obj,
        ranked_item_ids=deduped if valid else [],
        predicted_item_id=(deduped[0] if deduped and valid else None),
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


def parse_ranking_output(
    text: str,
    *,
    allowed_item_ids: list[str],
    topk: int | None = None,
    strict_json_only: bool = False,
) -> ParsedLLMOutput:
    obj, malformed, repaired = _extract_json(text)
    if strict_json_only:
        if obj is None:
            blob = extract_first_json_object(strip_reasoning(text))
            if blob:
                obj, malformed, repaired = _try_load_json_blob(blob)
        if obj is None:
            malformed = True
        return _parse_ranking_strict(
            text,
            allowed_item_ids=allowed_item_ids,
            obj=obj,
            malformed=malformed,
            repaired=repaired,
        )
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
