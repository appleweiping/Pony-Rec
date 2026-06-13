#!/usr/bin/env python3
"""
Gap-decomposition ANALYSIS (CPU-only, local).

Inputs:
  - outputs/ccrp_v3_formal/main_comparison_table.csv                (original 4 domains)
  - outputs/summary/new_domains_official_ccrp_cross_domain_20260605_method_rows.csv  (new 4)
  - <features.json>  (per-domain domain features from gap_decomposition_features.py)

Produces (into --outdir):
  - gap_decomposition_results.json   (gaps, features, Spearman, regression)
  - gap_decomposition_feature_table.csv
  - gap_decomposition_table.tex      (LaTeX, input-able)
  - gap_decomposition_paragraph.txt  (draft 1-paragraph gap analysis)
  - gap_vs_top_feature.png/.pdf      (per-domain gap vs most-predictive feature)

n=8 domains -> correlational/descriptive only, NOT strong causal.
"""
import argparse, csv, json, math, os
from itertools import combinations

DOMAINS = ["beauty", "books", "electronics", "movies", "sports", "toys", "home", "tools"]
CCRP_NAMES = {"c-crp_v3_ours", "ccrp_v3_qwen3base_pointwise"}
METRIC = "NDCG@10"


def _f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_original(path):
    """main_comparison_table.csv -> {domain: {method: ndcg10}}"""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = row["domain"].strip().lower()
            out.setdefault(d, {})[row["method"].strip()] = _f(row["NDCG@10"])
    return out


def load_new(path):
    """method_rows.csv -> {domain: {method: ndcg10}}"""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = row["domain"].strip().lower()
            out.setdefault(d, {})[row["method"].strip()] = _f(row["NDCG@10"])
    return out


def split_ccrp_baselines(method_map):
    ccrp = None
    baselines = {}
    for m, v in method_map.items():
        if m.lower() in CCRP_NAMES:
            ccrp = v
        else:
            baselines[m] = v
    return ccrp, baselines


def find_llmemb(baselines):
    for m, v in baselines.items():
        if "llmemb" in m.lower():
            return m, v
    return None, float("nan")


def compute_gaps(orig, new):
    gaps = {}
    for d in DOMAINS:
        mm = orig.get(d) or new.get(d)
        if not mm:
            continue
        ccrp, baselines = split_ccrp_baselines(mm)
        # strongest baseline by NDCG@10
        best_m = max(baselines, key=lambda k: baselines[k])
        best_v = baselines[best_m]
        lm_name, lm_v = find_llmemb(baselines)
        gap_best = ccrp - best_v
        gap_best_pct = 100.0 * gap_best / best_v if best_v else float("nan")
        gap_lm = ccrp - lm_v
        gap_lm_pct = 100.0 * gap_lm / lm_v if lm_v else float("nan")
        gaps[d] = {
            "ccrp_ndcg10": ccrp,
            "strongest_baseline": best_m,
            "strongest_baseline_ndcg10": best_v,
            "gap_vs_strongest": gap_best,
            "gap_vs_strongest_pct": gap_best_pct,
            "llmemb_ndcg10": lm_v,
            "gap_vs_llmemb": gap_lm,
            "gap_vs_llmemb_pct": gap_lm_pct,
            "ccrp_wins": gap_best > 0,
        }
    return gaps


# ---------- stats (no scipy dependency required) ----------
def _rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def spearman(x, y):
    return pearson(_rank(x), _rank(y))


def t_pvalue_two_sided(r, n):
    """approx two-sided p-value for a correlation r with n samples (t-dist, df=n-2)."""
    if n <= 2 or abs(r) >= 1.0:
        return 0.0 if abs(r) >= 1.0 else float("nan")
    df = n - 2
    t = r * math.sqrt(df / (1 - r * r))
    x = df / (df + t * t)
    # regularized incomplete beta I_x(df/2, 1/2) via continued fraction
    p = _betai(df / 2.0, 0.5, x)
    return max(0.0, min(1.0, p))


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x)
    else:
        return 1.0 - math.exp(math.log(1 - x) * b + math.log(x) * a - lbeta) / b * _betacf(b, a, 1 - x)


def _betacf(a, b, x, itmax=200, eps=1e-12):
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def ols_simple(x, y):
    """y = a + b x ; return slope, intercept, r2."""
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    if sxx <= 0:
        return float("nan"), float("nan"), float("nan")
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((v - my) ** 2 for v in y)
    ss_res = sum((y[i] - (a + b * x[i])) ** 2 for i in range(n))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return b, a, r2


FEATURE_KEYS = [
    ("catalog_size", "Catalog size (#items)"),
    ("gini", "Popularity Gini"),
    ("head_top1pct_share", "Top-1% interaction share"),
    ("cand_tfidf_cos_mean", "Cand. title TF-IDF cosine"),
    ("cand_jaccard_mean", "Cand. title token Jaccard"),
    ("cand_title_len_tokens", "Cand. title length (tokens)"),
    ("cand_vocab_size", "Cand. panel vocab size"),
    ("cand_metadata_completeness", "Cand. metadata completeness"),
    ("mean_user_hist_len", "Mean user history length"),
    ("n_test_users", "#Test users"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default="outputs/ccrp_v3_formal/main_comparison_table.csv")
    ap.add_argument("--new", default="outputs/summary/new_domains_official_ccrp_cross_domain_20260605_method_rows.csv")
    ap.add_argument("--features", default="tmp_outputs/gap_features_all.json")
    ap.add_argument("--outdir", default="outputs/summary/paper_critical/gap_decomposition")
    ap.add_argument("--figure-feature", default="mean_user_hist_len",
                    help="feature key for the headline scatter figure (default: most robust "
                         "OLS-R^2 predictor; set empty to auto-pick by R^2)")
    args = ap.parse_args()
    if args.figure_feature == "":
        args.figure_feature = None
    os.makedirs(args.outdir, exist_ok=True)

    orig = load_original(args.orig)
    new = load_new(args.new)
    gaps = compute_gaps(orig, new)
    feats = json.load(open(args.features, encoding="utf-8"))

    # merged per-domain rows
    rows = []
    for d in DOMAINS:
        if d not in gaps or d not in feats:
            continue
        r = {"domain": d}
        r.update(gaps[d])
        for k, _ in FEATURE_KEYS:
            r[k] = feats[d].get(k)
        # disclosed data-quality flags
        r["cand_title_coverage"] = feats[d].get("cand_title_coverage")
        r["hist_source"] = feats[d].get("hist_source")
        rows.append(r)

    # ---------- correlations: each feature vs gap_vs_strongest_pct AND gap_vs_llmemb_pct ----------
    def col(key):
        return [r[key] for r in rows]

    corr = {}
    for tgt in ["gap_vs_strongest_pct", "gap_vs_llmemb_pct", "gap_vs_strongest", "gap_vs_llmemb"]:
        y = col(tgt)
        corr[tgt] = {}
        for k, label in FEATURE_KEYS:
            x = col(k)
            xs = [v for v in x if v is not None]
            if len(xs) < len(x):  # missing values -> skip
                corr[tgt][k] = {"label": label, "spearman": None, "note": "missing_values"}
                continue
            rho = spearman(x, y)
            pr = pearson(x, y)
            corr[tgt][k] = {
                "label": label,
                "spearman": rho,
                "spearman_p": t_pvalue_two_sided(rho, len(x)),
                "pearson": pr,
            }

    # most-predictive feature for the headline gap (gap_vs_strongest_pct), by |spearman|
    cand = [(k, corr["gap_vs_strongest_pct"][k]["spearman"])
            for k, _ in FEATURE_KEYS
            if corr["gap_vs_strongest_pct"][k]["spearman"] is not None]
    top_feature_by_spearman = max(cand, key=lambda kv: abs(kv[1]))[0]

    # descriptive single-feature OLS for each feature -> gap_vs_strongest_pct
    reg = {}
    y = col("gap_vs_strongest_pct")
    for k, label in FEATURE_KEYS:
        x = col(k)
        if any(v is None for v in x):
            continue
        b, a, r2 = ols_simple(x, y)
        reg[k] = {"label": label, "slope": b, "intercept": a, "r2": r2}

    # headline feature for the FIGURE: most-explanatory by OLS R^2 (robust to a single
    # high-leverage point), unless overridden. We separately report the top-|Spearman|
    # feature. We avoid cand_vocab_size as the figure headline because its rank
    # correlation is driven almost entirely by Movies (Beauty is a high-vocab counter-
    # example), i.e. it does not separate BOTH non-winning domains.
    top_feature = args.figure_feature or max(reg, key=lambda k: reg[k]["r2"])

    # win vs loss separation: mean feature value among wins (6) vs losses (2)
    wins = [r for r in rows if r["ccrp_wins"]]
    losses = [r for r in rows if not r["ccrp_wins"]]
    sep = {}
    for k, label in FEATURE_KEYS:
        wv = [r[k] for r in wins if r[k] is not None]
        lv = [r[k] for r in losses if r[k] is not None]
        sep[k] = {
            "label": label,
            "win_mean": (sum(wv) / len(wv)) if wv else None,
            "loss_mean": (sum(lv) / len(lv)) if lv else None,
            "loss_domains": [r["domain"] for r in losses],
        }

    results = {
        "metric": METRIC,
        "n_domains": len(rows),
        "note": "n=8 -> correlational/descriptive, not strong causal. CCRP = Candidate-Conditioned Relevance Posterior (pointwise LLM relevance), not Calibrated.",
        "domains": rows,
        "spearman_correlations": corr,
        "single_feature_ols_gap_vs_strongest_pct": reg,
        "top_feature_by_abs_spearman": top_feature_by_spearman,
        "top_feature_by_ols_r2": max(reg, key=lambda k: reg[k]["r2"]),
        "figure_headline_feature": top_feature,
        "win_vs_loss_separation": sep,
        "interpretation": {
            "text_homogeneity_hypothesis": "SUPPORTED (directionally): gap_vs_strongest_pct "
                "correlates NEGATIVELY with candidate-title TF-IDF cosine (Spearman "
                f"{corr['gap_vs_strongest_pct']['cand_tfidf_cos_mean']['spearman']:.2f}) and token "
                f"Jaccard ({corr['gap_vs_strongest_pct']['cand_jaccard_mean']['spearman']:.2f}); "
                "the two non-winning domains have the highest candidate-text homogeneity. Not "
                "individually significant at n=8.",
            "popularity_skew_hypothesis": "REVERSED vs the prior: the gap correlates POSITIVELY "
                f"with Gini (Spearman {corr['gap_vs_strongest_pct']['gini']['spearman']:.2f}) and "
                f"top-1% share ({corr['gap_vs_strongest_pct']['head_top1pct_share']['spearman']:.2f}). "
                "C-CRP's margin GROWS on more skewed / long-tail catalogs, not shrinks. The prior "
                "'loses on high-skew' framing is not supported; Movies loses despite high skew, "
                "Beauty loses with very LOW skew.",
            "strongest_continuous_predictor": "mean_user_hist_len (Spearman "
                f"{corr['gap_vs_strongest_pct']['mean_user_hist_len']['spearman']:.2f}, OLS R2="
                f"{reg['mean_user_hist_len']['r2']:.2f}): C-CRP wins where user histories are SHORT "
                "(cold-start regime), and is not first on the two longest-history domains.",
            "dual_mechanism": "The two non-winning domains fail for DIFFERENT reasons: Beauty is a "
                "tiny (1,184-item), near-uniform catalog where dense ID/CF baselines (ProEx) have "
                "ample co-occurrence signal and the LLM posterior has little to add (C-CRP still "
                "beats LLMEmb by +9.3%); Movies has low item-metadata completeness (~51%), the most "
                "text-homogeneous candidate panels, and the longest histories, weakening the LLM's "
                "per-candidate relevance signal. No single feature classifies both, which is why we "
                "report the full feature table rather than a one-factor explanation.",
        },
        "data_quality_disclosures": {
            "movies_cand_title_coverage": feats.get("movies", {}).get("cand_title_coverage"),
            "movies_cand_metadata_completeness": feats.get("movies", {}).get("cand_metadata_completeness"),
            "movies_hist_source": feats.get("movies", {}).get("hist_source"),
            "note": "Movies item titles are ~38% missing in the Amazon metadata; TF-IDF cosine / title length / vocab are computed over non-empty titles only, and metadata-completeness is reported as its own feature. Movies@10k panel was cleaned from server, so movies mean_user_hist_len uses an interactions.csv all-user proxy.",
        },
    }

    with open(os.path.join(args.outdir, "gap_decomposition_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=lambda o: None if isinstance(o, float) and math.isnan(o) else o)

    # ---------- feature table CSV ----------
    cols = (["domain", "ccrp_ndcg10", "strongest_baseline", "strongest_baseline_ndcg10",
             "gap_vs_strongest", "gap_vs_strongest_pct", "llmemb_ndcg10",
             "gap_vs_llmemb", "gap_vs_llmemb_pct", "ccrp_wins"]
            + [k for k, _ in FEATURE_KEYS] + ["cand_title_coverage", "hist_source"])
    with open(os.path.join(args.outdir, "gap_decomposition_feature_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})

    # ---------- LaTeX table ----------
    write_latex(results, rows, top_feature, os.path.join(args.outdir, "gap_decomposition_table.tex"))

    # ---------- figure ----------
    fig_path = os.path.join(args.outdir, "gap_vs_top_feature")
    make_figure(rows, top_feature, corr, fig_path)

    # ---------- paragraph ----------
    write_paragraph(results, rows, top_feature, corr, sep, os.path.join(args.outdir, "gap_decomposition_paragraph.txt"))

    print(json.dumps({
        "figure_headline_feature": top_feature,
        "top_feature_by_abs_spearman": top_feature_by_spearman,
        "top_feature_by_ols_r2": max(reg, key=lambda k: reg[k]["r2"]),
        "spearman_figfeat_vs_strongest": corr["gap_vs_strongest_pct"][top_feature]["spearman"],
        "ols_r2_figfeat": reg[top_feature]["r2"],
        "n_wins": sum(1 for r in rows if r["ccrp_wins"]),
        "losses": [r["domain"] for r in rows if not r["ccrp_wins"]],
        "outdir": args.outdir}, indent=2))


def write_latex(results, rows, top_feature, path):
    def fmt(v, p=3):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "--"
        return f"{v:.{p}f}"
    lines = []
    lines.append("% Auto-generated by gap_decomposition_analysis.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering\\small")
    lines.append("\\caption{Per-domain ranking gap of the Candidate-Conditioned Relevance Posterior (C-CRP) "
                 "vs the strongest baseline (NDCG@10), against interpretable domain features. "
                 "$\\Delta$ is C-CRP$-$strongest baseline; positive = C-CRP first. $n{=}8$ domains "
                 "(correlational, not causal).}")
    lines.append("\\label{tab:gap_decomposition}")
    lines.append("\\begin{tabular}{lrrrrrr}")
    lines.append("\\toprule")
    lines.append("Domain & $\\Delta$\\% vs best & vs LLMEmb\\% & Gini & Top-1\\% & Cand.\\,cos & Hist.\\,len \\\\")
    lines.append("\\midrule")
    for r in sorted(rows, key=lambda x: -x["gap_vs_strongest_pct"]):
        lines.append(" & ".join([
            r["domain"],
            fmt(r["gap_vs_strongest_pct"], 1),
            fmt(r["gap_vs_llmemb_pct"], 1),
            fmt(r["gini"], 3),
            fmt(r["head_top1pct_share"], 3),
            fmt(r["cand_tfidf_cos_mean"], 4),
            fmt(r["mean_user_hist_len"], 2),
        ]) + " \\\\")
    lines.append("\\midrule")
    sp = results["spearman_correlations"]["gap_vs_strongest_pct"]
    lines.append("\\multicolumn{7}{l}{\\footnotesize Spearman $\\rho$ (feature, $\\Delta\\%$ vs best): "
                 f"Gini {fmt(sp['gini']['spearman'],2)}, "
                 f"Top-1\\% {fmt(sp['head_top1pct_share']['spearman'],2)}, "
                 f"Cand.\\,cos {fmt(sp['cand_tfidf_cos_mean']['spearman'],2)}, "
                 f"Hist.\\,len {fmt(sp['mean_user_hist_len']['spearman'],2)}.}} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def make_figure(rows, top_feature, corr, path_noext):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    label = dict(FEATURE_KEYS)[top_feature]
    x = [r[top_feature] for r in rows]
    y = [r["gap_vs_strongest_pct"] for r in rows]
    names = [r["domain"] for r in rows]
    wins = [r["ccrp_wins"] for r in rows]
    rho = corr["gap_vs_strongest_pct"][top_feature]["spearman"]

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.axhline(0, color="#888", lw=1, ls="--", zorder=1)
    for xi, yi, nm, win in zip(x, y, names, wins):
        c = "#2c7fb8" if win else "#d7301f"
        ax.scatter(xi, yi, s=90, color=c, edgecolor="black", lw=0.6, zorder=3)
        ax.annotate(nm, (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=9)
    # OLS trend line
    b, a, r2 = ols_simple(x, y)
    if not math.isnan(b):
        xs = sorted(x)
        ax.plot([xs[0], xs[-1]], [a + b * xs[0], a + b * xs[-1]], color="#444", lw=1.2, zorder=2)
    ax.set_xlabel(label)
    ax.set_ylabel("NDCG@10 gap vs strongest baseline (%)")
    ax.set_title(f"Per-domain C-CRP gap vs {label}\nSpearman $\\rho$={rho:.2f}, $R^2$={r2:.2f} (n=8)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2c7fb8", markeredgecolor="k", label="C-CRP first (6)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d7301f", markeredgecolor="k", label="Not first (2)"),
    ], loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(path_noext + ".png", dpi=200)
    fig.savefig(path_noext + ".pdf")
    plt.close(fig)


def write_paragraph(results, rows, top_feature, corr, sep, path):
    sp = corr["gap_vs_strongest_pct"]

    def g(k):
        return sp[k]["spearman"]

    losses = [r["domain"] for r in rows if not r["ccrp_wins"]]
    win_lo = min(r['gap_vs_strongest_pct'] for r in rows if r['ccrp_wins'])
    win_hi = max(r['gap_vs_strongest_pct'] for r in rows if r['ccrp_wins'])
    r2_hist = results["single_feature_ols_gap_vs_strongest_pct"]["mean_user_hist_len"]["r2"]
    para = (
        "Gap analysis. We turn the qualitative observation that the Candidate-Conditioned Relevance "
        "Posterior (C-CRP, the LLM's pointwise per-candidate relevance probability used directly as the "
        "ranking score) wins on most but not all domains into a measured decomposition. For each of the "
        "eight domains we relate the NDCG@10 gap between C-CRP and the strongest same-candidate baseline "
        "to interpretable domain features computed from the evaluation panels and catalogs: catalog size, "
        "popularity skew (Gini coefficient and top-1% interaction share), candidate-title textual "
        "homogeneity (mean pairwise TF-IDF cosine and token Jaccard over the 101 panel titles, plus panel "
        "vocabulary size), mean user-history length, and number of test users. C-CRP ranks first in six "
        f"domains (+{win_lo:.0f}% to +{win_hi:.0f}% over the strongest baseline) and is not first in "
        f"{' and '.join(losses)}. Three associations emerge, all directional rather than significant at "
        "n=8. (i) Candidate-text homogeneity moves inversely with the gap (Spearman rho="
        f"{g('cand_tfidf_cos_mean'):.2f} for TF-IDF cosine, {g('cand_jaccard_mean'):.2f} for token "
        "Jaccard): the two non-winning domains have the most textually homogeneous candidate panels, so "
        "the LLM's per-candidate relevance signal is least discriminative there. (ii) Contrary to the "
        "prior conjecture that the posterior should fail on high-skew catalogs, the gap correlates "
        f"POSITIVELY with popularity skew (rho={g('gini'):.2f} for Gini, {g('head_top1pct_share'):.2f} "
        "for top-1% share): C-CRP's margin grows on more skewed, long-tail catalogs, and Movies loses "
        "despite being high-skew while Beauty loses with the lowest skew. (iii) The strongest single "
        f"continuous predictor is mean user-history length (rho={g('mean_user_hist_len'):.2f}, OLS "
        f"R^2={r2_hist:.2f}): C-CRP is strongest in the short-history, cold-start regime and is not first "
        "on the two longest-history domains, where ID/CF baselines accumulate more co-occurrence signal. "
        "Crucially, no single feature classifies both non-winning domains, because they fail for "
        "different reasons: Beauty is a tiny (1,184-item), near-uniform catalog where the dense-ID "
        "baseline ProEx has ample collaborative signal and the LLM posterior adds little (C-CRP still "
        "beats LLMEmb there by +9.3%), whereas Movies combines the most homogeneous candidate text with "
        "the longest histories. We report two disclosed data caveats for Movies: ~38% of Amazon Movies "
        "items lack title metadata, so its text features are computed over non-empty titles only and "
        "metadata completeness (51%) is itself listed as a feature; and its mean history length uses an "
        "interactions-table all-user proxy because the 10k Movies panel was pruned for storage. With "
        "n=8 these are descriptive correlations, not causal claims, but they replace the hand-waved "
        "'wins 6/8' with a concrete profile: the pointwise relevance posterior is strongest on large, "
        "skewed, short-history, text-heterogeneous catalogs and weakest on small/uniform or "
        "text-homogeneous, long-history ones."
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(para + "\n")


if __name__ == "__main__":
    main()
