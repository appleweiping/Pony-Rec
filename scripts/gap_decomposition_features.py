#!/usr/bin/env python3
"""
Gap-decomposition: per-domain DOMAIN FEATURES for explaining why the
Candidate-Conditioned Relevance Posterior (C-CRP v3 pointwise) wins 6/8 domains
but is not first in Beauty / Movies.

CPU-ONLY. Reads existing result/dataset files. Does NOT touch GPU or the running
TGL LoRA job. Emits a compact JSON of per-domain features.

Run on server (pony-rec-rescue-shadow-v6 root):
    python3 scripts/gap_decomposition_features.py --out /tmp/gap_features.json

Features per domain:
  (a) catalog_size            : #unique items (popularity_stats.csv row count)
  (b) gini                    : Gini coeff of item interaction frequency
      head_top1pct_share      : share of interactions held by top-1% items
      head_group_share        : share of items labelled 'head' (dataset's own split)
  (c) cand_tfidf_cos_mean     : mean pairwise TF-IDF cosine sim of the 101
                                candidate TITLES within a panel (avg over sampled users)
      cand_jaccard_mean       : mean pairwise token-Jaccard of candidate titles
      cand_title_len_tokens   : mean candidate title length (tokens)
      cand_vocab_size         : mean #unique tokens across a panel's 101 titles
  (d) mean_user_hist_len      : mean history length of the evaluated test users
  (e) n_test_users            : #evaluated users

Sources (per domain):
  popularity_stats.csv  -> catalog size + skew
  ranking_eval_records.csv (candidate_item_ids) -> panel candidate sets (uniform, survives cleanup)
  items.csv             -> item_id -> title
  ranking_test.jsonl (panel, if present) -> per-user history length
"""
import argparse, ast, csv, json, math, os, random, re, sys
from collections import Counter

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
random.seed(20260613)

DOMAINS = ["beauty", "books", "electronics", "movies", "sports", "toys", "home", "tools"]

# eval-records run dir per domain (same-candidate panels identical across methods;
# use the LLMEmb official run which exists for all 8)
EVAL_RUN = {
    "beauty": "beauty_supplementary_smallerN_100neg_llmemb_official_qwen3base_same_candidate",
    "books": "books_large10000_100neg_llmemb_official_qwen3base_same_candidate",
    "electronics": "electronics_large10000_100neg_llmemb_official_qwen3base_same_candidate",
    "movies": "movies_large10000_100neg_llmemb_official_qwen3base_same_candidate",
    "sports": "sports_large10000_100neg_llmemb_official_qwen3base_same_candidate",
    "toys": "toys_large10000_100neg_llmemb_official_qwen3base_same_candidate",
    "home": "home_large10000_100neg_llmemb_official_qwen3base_same_candidate",
    "tools": "tools_large10000_100neg_llmemb_official_qwen3base_same_candidate",
}
# panel dir holding ranking_test.jsonl (for history length); movies@10k cleaned -> None
PANEL_DIR = {
    "beauty": "beauty_supplementary_smallerN_100neg_test_same_candidate",
    "books": "books_large10000_100neg_test_same_candidate",
    "electronics": "electronics_large10000_100neg_test_same_candidate",
    "movies": None,  # cleaned from server; use interactions.csv proxy
    "sports": "sports_large10000_100neg_test_same_candidate",
    "toys": "toys_large10000_100neg_test_same_candidate",
    "home": "home_large10000_100neg_test_same_candidate",
    "tools": "tools_large10000_100neg_test_same_candidate",
}
PROCESSED = {d: f"data/processed/amazon_{d}" for d in DOMAINS}

N_PANEL_SAMPLE = 400      # users sampled for candidate-text homogeneity
_token_re = re.compile(r"[a-z0-9]+")


def tokens(s):
    return _token_re.findall(s.lower())


# ---------- (a)+(b) catalog size + popularity skew ----------
def catalog_and_skew(domain):
    path = os.path.join(PROCESSED[domain], "popularity_stats.csv")
    counts = []
    head_items = 0
    n_items = 0
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            n_items += 1
            try:
                c = float(row["interaction_count"])
            except (KeyError, ValueError):
                c = 0.0
            counts.append(c)
            if row.get("popularity_group", "").strip().lower() == "head":
                head_items += 1
    counts.sort()
    total = sum(counts)
    n = len(counts)
    # Gini (sorted ascending)
    if total > 0 and n > 0:
        cum = 0.0
        for i, c in enumerate(counts, start=1):
            cum += i * c
        gini = (2.0 * cum) / (n * total) - (n + 1.0) / n
    else:
        gini = float("nan")
    # top-1% interaction share (largest items)
    k = max(1, int(round(0.01 * n)))
    top1pct_share = sum(counts[-k:]) / total if total > 0 else float("nan")
    return {
        "catalog_size": n_items,
        "total_interactions_in_stats": int(total),
        "gini": gini,
        "head_top1pct_share": top1pct_share,
        "head_group_item_share": head_items / n_items if n_items else float("nan"),
    }


# ---------- item_id -> title ----------
def load_titles(domain, needed_ids):
    """Stream items.csv once; keep titles only for needed_ids (memory-safe on huge catalogs)."""
    path = os.path.join(PROCESSED[domain], "items.csv")
    titles = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            iid = row.get("item_id")
            if iid in needed_ids:
                titles[iid] = row.get("title") or ""
                if len(titles) == len(needed_ids):
                    break
    return titles


# ---------- (c) candidate-text homogeneity ----------
def candidate_homogeneity(domain):
    eval_path = os.path.join("outputs", EVAL_RUN[domain], "tables", "ranking_eval_records.csv")
    panels = []  # list of candidate-id lists
    with open(eval_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                cids = ast.literal_eval(row["candidate_item_ids"])
            except Exception:
                continue
            if isinstance(cids, (list, tuple)) and len(cids) >= 2:
                panels.append(list(cids))
    if not panels:
        return {"error": "no_panels"}, []
    n_total = len(panels)
    sample = panels if len(panels) <= N_PANEL_SAMPLE else random.sample(panels, N_PANEL_SAMPLE)
    needed = set()
    for p in sample:
        needed.update(p)
    titles = load_titles(domain, needed)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    cos_means, jac_means, title_lens, vocab_sizes, completeness = [], [], [], [], []
    panels_used = 0
    for p in sample:
        docs_all = [titles.get(i, "") for i in p]
        # metadata completeness = fraction of candidates with a non-empty title
        nonempty_docs = [d for d in docs_all if tokens(d)]
        completeness.append(len(nonempty_docs) / len(docs_all) if docs_all else float("nan"))
        toks = [set(tokens(d)) for d in nonempty_docs]
        if len(toks) < 2:
            continue
        # title length / vocab (over candidates that HAVE a title, so movies'
        # missing-metadata sparsity doesn't masquerade as short titles)
        title_lens.append(sum(len(tokens(d)) for d in nonempty_docs) / len(nonempty_docs))
        vocab = set().union(*toks)
        vocab_sizes.append(len(vocab))
        # TF-IDF cosine (mean off-diagonal) over NON-EMPTY titles only
        try:
            vec = TfidfVectorizer(token_pattern=r"[a-z0-9]+", lowercase=True)
            X = vec.fit_transform(nonempty_docs)
            if X.shape[1] == 0:
                raise ValueError("empty vocab")
            S = cosine_similarity(X)
            m = S.shape[0]
            off = (S.sum() - np.trace(S)) / (m * (m - 1))
            cos_means.append(float(off))
        except Exception:
            pass
        # token Jaccard (mean off-diagonal) over non-empty titles
        pair_j = []
        for a in range(len(toks)):
            ta = toks[a]
            for b in range(a + 1, len(toks)):
                tb = toks[b]
                u = len(ta | tb)
                pair_j.append(len(ta & tb) / u if u else 0.0)
        if pair_j:
            jac_means.append(sum(pair_j) / len(pair_j))
        panels_used += 1

    def mean(x):
        return sum(x) / len(x) if x else float("nan")

    title_cov = len([1 for i in needed if titles.get(i)]) / len(needed) if needed else float("nan")
    return {
        "n_panels_total": n_total,
        "n_panels_sampled": len(sample),
        "n_panels_used": panels_used,
        "cand_title_coverage": title_cov,
        "cand_metadata_completeness": mean(completeness),
        "cand_tfidf_cos_mean": mean(cos_means),
        "cand_jaccard_mean": mean(jac_means),
        "cand_title_len_tokens": mean(title_lens),
        "cand_vocab_size": mean(vocab_sizes),
    }, sample


# ---------- (d) mean user history length ----------
def mean_history_len(domain):
    pdir = PANEL_DIR[domain]
    if pdir:
        rt = os.path.join("outputs", "baselines", "external_tasks", pdir, "ranking_test.jsonl")
        if os.path.exists(rt):
            lens = []
            with open(rt) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    h = ev.get("history_item_ids")
                    if h is None:
                        h = ev.get("history")
                    if isinstance(h, list):
                        lens.append(len(h))
            if lens:
                return {
                    "mean_user_hist_len": sum(lens) / len(lens),
                    "n_users_hist": len(lens),
                    "hist_source": "panel_ranking_test_jsonl",
                }
    # proxy: per-user sequence length from interactions.csv (catalog-level, all users)
    inter = os.path.join(PROCESSED[domain], "interactions.csv")
    per_user = Counter()
    with open(inter, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            u = row.get("user_id")
            if u:
                per_user[u] += 1
    if per_user:
        vals = list(per_user.values())
        return {
            "mean_user_hist_len": sum(vals) / len(vals),
            "n_users_hist": len(vals),
            "hist_source": "interactions_csv_proxy_all_users",
        }
    return {"mean_user_hist_len": float("nan"), "hist_source": "unavailable"}


def n_test_users(domain):
    eval_path = os.path.join("outputs", EVAL_RUN[domain], "tables", "ranking_eval_records.csv")
    n = 0
    with open(eval_path, newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for _ in r:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/gap_features.json")
    ap.add_argument("--domains", nargs="*", default=DOMAINS)
    args = ap.parse_args()

    out = {}
    for d in args.domains:
        sys.stderr.write(f"[gap] {d} ...\n"); sys.stderr.flush()
        rec = {"domain": d, "n_test_users": n_test_users(d)}
        rec.update(catalog_and_skew(d))
        homog, _ = candidate_homogeneity(d)
        rec.update(homog)
        rec.update(mean_history_len(d))
        out[d] = rec
        sys.stderr.write(f"[gap] {d} done: catalog={rec.get('catalog_size')} "
                         f"gini={rec.get('gini'):.4f} cos={rec.get('cand_tfidf_cos_mean')}\n")
        sys.stderr.flush()

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    sys.stderr.write(f"[gap] wrote {args.out}\n")
    print(json.dumps(out, indent=2, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x))


if __name__ == "__main__":
    main()
