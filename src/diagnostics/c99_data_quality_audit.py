"""Offline audit: processed items + c99 reprocess slates + DeepSeek invalid flags."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.cli.run_pilot_reprocessed_deepseek import _build_ranking_prompt, _merge_item_texts
from src.data.item_text_cleaning import CleaningConfig, build_cleaned_lookup_for_ids
from src.data.protocol import read_jsonl
from src.prompts import candidate_block, history_block


def _item_id_column(df: pd.DataFrame) -> str | None:
    for c in ("item_id", "parent_asin", "asin", "movieId", "iid"):
        if c in df.columns:
            return c
    return None


def _catalog_lookup_subset(items_df: pd.DataFrame, id_col: str, needed_ids: set[str]) -> dict[str, str]:
    """Same mapping as pilot ``_load_item_lookup`` but restricted to ids referenced in the pilot slice."""
    text_col = "candidate_text" if "candidate_text" in items_df.columns else "title"
    keys = {str(x).strip() for x in needed_ids if str(x).strip()}
    mask = items_df[id_col].astype(str).str.strip().isin(keys)
    sub = items_df.loc[mask]
    out: dict[str, str] = {}
    for _, row in sub.iterrows():
        iid = str(row[id_col]).strip()
        out[iid] = str(row.get(text_col, "") or "").strip()
    return out


def _pctiles(values: list[float], ps: tuple[float, ...]) -> dict[str, float]:
    arr = np.array([float(x) for x in values if not math.isnan(float(x))], dtype=np.float64)
    if arr.size == 0:
        return {f"p{int(p*100)}": float("nan") for p in ps}
    out: dict[str, float] = {}
    for p in ps:
        out[f"p{int(p*100)}"] = float(np.quantile(arr, p))
    return out


def _pearson(a: list[float], b: list[float]) -> float | None:
    x = np.array(a, dtype=np.float64)
    y = np.array(b, dtype=np.float64)
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return None
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _missing_series(s: pd.Series) -> float:
    s = s.astype(str).str.strip()
    empty = (s == "") | (s.str.lower() == "nan") | (s.str.lower() == "none")
    return float(empty.mean())


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def audit_items_csv(items_path: Path | None = None, *, items_df: pd.DataFrame | None = None) -> dict[str, Any]:
    if items_df is not None:
        df = items_df.fillna("")
        path_str = str(items_path) if items_path is not None else ""
    else:
        if items_path is None:
            raise ValueError("items_path or items_df required")
        df = pd.read_csv(items_path, low_memory=False).fillna("")
        path_str = str(items_path)
    col = _item_id_column(df)
    if col is None:
        return {"error": "no item id column"}
    n = len(df)
    ids = df[col].astype(str).str.strip()
    dup_id = int(ids.duplicated().sum())
    title = df["title"] if "title" in df.columns else pd.Series([""] * n)
    categories = df["categories"] if "categories" in df.columns else pd.Series([""] * n)
    description = df["description"] if "description" in df.columns else pd.Series([""] * n)
    brand = df["brand"] if "brand" in df.columns else None

    tlen = title.astype(str).str.len()
    dlen = description.astype(str).str.len()

    def weird_title_mask(s: pd.Series) -> pd.Series:
        u = s.astype(str).str.strip().str.lower()
        return u.isin(["", "unknown", "none", "nan", "n/a", "null", "untitled"])

    dup_title = int(title.astype(str).str.strip().str.lower().duplicated().sum())

    html_in_title = int(title.astype(str).str.contains(r"<[a-zA-Z!/?]", regex=True, na=False).sum())
    html_in_desc = int(description.astype(str).str.contains(r"<[a-zA-Z!/?]", regex=True, na=False).sum())
    ctl_in_title = int(title.astype(str).str.contains(_CONTROL_RE, regex=True, na=False).sum())
    ctl_in_desc = int(description.astype(str).str.contains(_CONTROL_RE, regex=True, na=False).sum())

    out: dict[str, Any] = {
        "path": path_str,
        "rows": n,
        "duplicate_item_id_rows": dup_id,
        "title_missing_rate": _missing_series(title),
        "categories_missing_rate": _missing_series(categories),
        "description_missing_rate": _missing_series(description),
        "weird_placeholder_title_rate": float(weird_title_mask(title).mean()),
        "duplicate_title_rows_lower_stripped": dup_title,
        "title_len_mean": float(tlen.mean()) if n else float("nan"),
        "title_len_p95": float(np.quantile(tlen, 0.95)) if n else float("nan"),
        "title_len_max": int(tlen.max()) if n else 0,
        "description_len_p95": float(np.quantile(dlen, 0.95)) if n else float("nan"),
        "description_len_max": int(dlen.max()) if n else 0,
        "html_tag_hits_in_title": html_in_title,
        "html_tag_hits_in_description": html_in_desc,
        "control_char_hits_in_title": ctl_in_title,
        "control_char_hits_in_description": ctl_in_desc,
        "item_id_len_min": int(ids.str.len().min()) if n else 0,
        "item_id_len_max": int(ids.str.len().max()) if n else 0,
        "item_id_numeric_fraction": float(ids.str.match(r"^\d+$", na=False).mean()) if n else float("nan"),
    }
    if brand is not None:
        out["brand_missing_rate"] = _missing_series(brand)
    return out


def _slate_features(sample: dict[str, Any], items_index: set[str]) -> dict[str, Any]:
    cids = [str(x) for x in sample.get("candidate_item_ids") or []]
    target = str(sample.get("target_item_id", "")).strip()
    dup_cand = len(cids) - len(set(cids))
    missing_items = sum(1 for x in cids if x not in items_index)
    titles: list[str] = []
    for i, cid in enumerate(cids):
        ts = sample.get("candidate_titles") or []
        if i < len(ts) and str(ts[i]).strip():
            titles.append(str(ts[i]).strip().lower())
        else:
            titles.append("")
    nonempty = [t for t in titles if t]
    tc = Counter(nonempty)
    # Count surplus occurrences beyond first (duplicate title strings in slate)
    title_collisions = sum(max(0, c - 1) for c in tc.values())
    tgt_in = target in set(cids) if target else False
    buckets = sample.get("candidate_popularity_buckets") or []
    bc = Counter(str(b) for b in buckets)
    return {
        "n_candidates": len(cids),
        "target_in_candidates": bool(tgt_in),
        "duplicate_candidate_ids": int(dup_cand),
        "candidates_missing_from_items_csv": int(missing_items),
        "duplicate_title_count_in_slate": int(title_collisions),
        "bucket_head": float(bc.get("head", 0)) / max(len(buckets), 1),
        "bucket_mid": float(bc.get("mid", 0)) / max(len(buckets), 1),
        "bucket_tail": float(bc.get("tail", 0)) / max(len(buckets), 1),
    }


def audit_domain_split(
    *,
    domain: str,
    split: str,
    processed_dir: Path,
    reprocess_dir: Path,
    pilot_root: Path,
    prompt_id: str,
    topk: int,
    items_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    items_path = processed_dir / "items.csv"
    if items_df is None:
        items_df = pd.read_csv(items_path, low_memory=False).fillna("")
    else:
        items_df = items_df.fillna("")
    id_col = _item_id_column(items_df)
    if id_col is None:
        raise ValueError(f"No item id column in {items_path}")
    items_index = set(str(x).strip() for x in items_df[id_col].tolist() if str(x).strip())

    cand_path = reprocess_dir / domain / f"{split}_candidates.jsonl"
    parsed_path = pilot_root / domain / split / "predictions" / "parsed_responses.jsonl"
    samples = read_jsonl(cand_path)
    parsed = read_jsonl(parsed_path)
    parsed_by_user = {str(r.get("user_id")): r for r in parsed}

    lengths: list[int] = []
    cand_lens: list[int] = []
    hist_lens: list[int] = []
    miss_titles: list[int] = []
    long_titles: list[int] = []
    inv: list[int] = []
    dup_titles: list[int] = []
    miss_meta: list[int] = []
    tgt_bucket: list[str] = []
    p95_cand_title_len: list[int] = []

    needed_ids: set[str] = set()
    for s in samples:
        needed_ids.update(str(x) for x in s.get("candidate_item_ids", []) or [])
        needed_ids.update(str(x) for x in s.get("history_item_ids", []) or [])
    base_lookup = _catalog_lookup_subset(items_df, id_col, needed_ids)
    clean_partial = build_cleaned_lookup_for_ids(items_df, needed_ids, config=CleaningConfig())
    clean_base = dict(base_lookup)
    clean_base.update(clean_partial)

    for s in samples:
        uid = str(s.get("user_id"))
        texts = _merge_item_texts(s, base_lookup)
        prompt = _build_ranking_prompt(s, prompt_id, texts, topk)
        lengths.append(len(prompt))
        # candidate-only mass: approximate by summing per-candidate line lengths
        hb = history_block([str(x) for x in s.get("history_item_ids", [])], texts)
        cb = candidate_block([str(x) for x in s.get("candidate_item_ids", [])], texts)
        hist_lens.append(len(hb))
        cand_lens.append(len(cb))
        cids = [str(x) for x in s.get("candidate_item_ids", [])]
        ts = s.get("candidate_titles") or []
        miss = 0
        lens: list[int] = []
        for i, cid in enumerate(cids):
            t = str(ts[i]).strip() if i < len(ts) else ""
            merged = str(texts.get(cid, "")).strip()
            if not t and not merged:
                miss += 1
            lens.append(len(t) if t else len(merged))
        miss_titles.append(miss)
        long_titles.append(sum(1 for L in lens if L > 200))
        p95_cand_title_len.append(int(np.quantile(lens, 0.95)) if lens else 0)

        slate = _slate_features(s, items_index)
        dup_titles.append(slate["duplicate_title_count_in_slate"])
        mm = 0
        if not slate["target_in_candidates"]:
            mm += 1
        mm += slate["candidates_missing_from_items_csv"]
        mm += slate["duplicate_candidate_ids"]
        miss_meta.append(mm)

        pr = parsed_by_user.get(uid, {})
        inv.append(1 if pr.get("invalid_output") else 0)
        tgt_bucket.append(str(s.get("target_popularity_bucket", "unknown")))

    ps = (0.5, 0.9, 0.95, 1.0)
    pct = _pctiles([float(x) for x in lengths], ps)

    inv_arr = np.array(inv, dtype=np.float64)
    len_arr = np.array([float(x) for x in lengths], dtype=np.float64)
    corr_len = _pearson(len_arr.tolist(), inv_arr.tolist())
    corr_miss = _pearson([float(x) for x in miss_titles], inv_arr.tolist())
    corr_dup = _pearson([float(x) for x in dup_titles], inv_arr.tolist())
    corr_meta = _pearson([float(x) for x in miss_meta], inv_arr.tolist())
    corr_p95 = _pearson([float(x) for x in p95_cand_title_len], inv_arr.tolist())

    invalid_rate = float(inv_arr.mean()) if inv_arr.size else float("nan")

    by_bucket: dict[str, dict[str, float]] = {}
    for b in sorted(set(tgt_bucket)):
        mask = np.array([t == b for t in tgt_bucket], dtype=bool)
        if mask.any():
            by_bucket[b] = {
                "n": int(mask.sum()),
                "invalid_rate": float(inv_arr[mask].mean()),
                "mean_prompt_len": float(len_arr[mask].mean()),
            }

    slate_agg = {
        "target_in_candidates_rate": float(
            np.mean([_slate_features(s, items_index)["target_in_candidates"] for s in samples])
        ),
        "mean_duplicate_candidate_ids": float(
            np.mean([_slate_features(s, items_index)["duplicate_candidate_ids"] for s in samples])
        ),
        "mean_candidates_missing_from_items": float(
            np.mean([_slate_features(s, items_index)["candidates_missing_from_items_csv"] for s in samples])
        ),
        "mean_duplicate_title_count_in_slate": float(
            np.mean([_slate_features(s, items_index)["duplicate_title_count_in_slate"] for s in samples])
        ),
    }

    approx_tokens_mean = float(np.mean(lengths) / 4.0) if lengths else float("nan")

    clean_lengths: list[int] = []
    for s in samples:
        texts = _merge_item_texts(s, clean_base)
        clean_lengths.append(len(_build_ranking_prompt(s, prompt_id, texts, topk)))

    return {
        "n_rows": len(samples),
        "invalid_output_rate": invalid_rate,
        "prompt_char_len": {"mean": float(np.mean(lengths)), **pct, "max": float(max(lengths)) if lengths else float("nan")},
        "approx_token_len_mean_chars_over_4": approx_tokens_mean,
        "history_block_char_mean": float(np.mean(hist_lens)) if hist_lens else float("nan"),
        "candidate_block_char_mean": float(np.mean(cand_lens)) if cand_lens else float("nan"),
        "candidates_missing_title_count_mean": float(np.mean(miss_titles)) if miss_titles else float("nan"),
        "candidates_title_len_gt200_mean": float(np.mean(long_titles)) if long_titles else float("nan"),
        "candidate_title_len_p95_mean": float(np.mean(p95_cand_title_len)) if p95_cand_title_len else float("nan"),
        "slate_quality": slate_agg,
        "correlations_invalid_vs": {
            "prompt_char_len_pearson": corr_len,
            "missing_title_count_pearson": corr_miss,
            "duplicate_title_in_slate_pearson": corr_dup,
            "missing_meta_score_pearson": corr_meta,
            "candidate_title_p95_len_pearson": corr_p95,
        },
        "invalid_by_target_popularity_bucket": by_bucket,
        "prompt_char_len_mean_if_cleaned_lookup": float(np.mean(clean_lengths)) if clean_lengths else float("nan"),
    }


def run_full_audit(
    *,
    domains: list[str],
    splits: list[str],
    data_root: Path,
    reprocess_dir: Path,
    pilot_root: Path,
    prompt_id: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "reprocess_dir": str(reprocess_dir),
        "pilot_root": str(pilot_root),
        "prompt_id": prompt_id,
        "domains": {},
    }
    topk = 99
    for domain in domains:
        for sp in splits:
            p = reprocess_dir / domain / f"{sp}_candidates.jsonl"
            if p.exists():
                rows = read_jsonl(p)
                if rows:
                    topk = len(rows[0].get("candidate_item_ids") or []) or topk
                    break
        else:
            continue
        break

    for domain in domains:
        processed_dir = data_root / domain
        items_path = processed_dir / "items.csv"
        items_df = pd.read_csv(items_path, low_memory=False)
        items_stats = audit_items_csv(items_path, items_df=items_df)
        out["domains"][domain] = {"items_csv": items_stats, "splits": {}, "topk": topk}
        for split in splits:
            out["domains"][domain]["splits"][split] = audit_domain_split(
                domain=domain,
                split=split,
                processed_dir=processed_dir,
                reprocess_dir=reprocess_dir,
                pilot_root=pilot_root,
                prompt_id=prompt_id,
                topk=topk,
                items_df=items_df,
            )
    # attach reference invalid rates from pilot summary if present
    summary_path = pilot_root / "pilot_run_summary.json"
    if summary_path.exists():
        out["pilot_run_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    return out


def write_cleaning_exemplars(
    *,
    domain: str,
    split: str,
    reprocess_dir: Path,
    data_root: Path,
    out_path: Path,
    n: int = 5,
) -> None:
    processed_dir = data_root / domain
    items_df = pd.read_csv(processed_dir / "items.csv", low_memory=False).fillna("")
    samples = read_jsonl(reprocess_dir / domain / f"{split}_candidates.jsonl")[:n]
    needed: set[str] = set()
    for s in samples:
        needed.update(str(x) for x in s.get("candidate_item_ids", []) or [])
        needed.update(str(x) for x in s.get("history_item_ids", []) or [])
    id_col = _item_id_column(items_df)
    if id_col is None:
        raise ValueError("items.csv missing id column")
    base = _catalog_lookup_subset(items_df, id_col, needed)
    clean = build_cleaned_lookup_for_ids(items_df, needed, config=CleaningConfig())
    rows = []
    for s in samples:
        uid = s.get("user_id")
        cids = [str(x) for x in s.get("candidate_item_ids", [])][:3]
        before = {cid: base.get(cid, "")[:400] for cid in cids}
        after = {cid: clean.get(cid, base.get(cid, ""))[:400] for cid in cids}
        rows.append({"user_id": uid, "sample_item_ids": cids, "before": before, "after": after})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
