#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_paper_figures.py
=====================
Generate journal-quality (publication / top-conference) statistical figures for
the C-CRP "pony" paper from EXISTING local data only (no new experiments, CPU-only).

Produces four figures, each saved as BOTH a vector PDF (for includegraphics) and
a 300-dpi PNG, into Paper/figures/:

  1. fig_improvement_bars  - horizontal bars: per-domain NDCG@10 %% improvement of
                             C-CRP over the STRONGEST official baseline (8 domains,
                             sorted descending). Headline figure.
  2. fig_main_heatmap      - heatmap of NDCG@10 for C-CRP vs the 8 official
                             baselines across all 8 domains (C-CRP row highlighted).
  3. fig_significance      - horizontal stacked bars of paired-test outcomes /56 per
                             domain (ahead-Holm-sig / behind-Holm-sig / n.s.).
  4. fig_gap_decomp        - two-panel: (left) gap%% vs mean user-history length
                             (rho=-0.69); (right) gap%% vs popularity Gini (rho=+0.57).

It additionally produces three CSV-DRIVEN diagnostic figures (numbers read live
from the source CSVs at run time, so they always match the data exactly):

  5. fig_hyperparam_sensitivity - single-col line plot: mean NDCG@10 across the 4
                             diagnostic domains vs the risk exponent eta in
                             {0,0.25,0.5,1,2,4}, two lines (valid, test). Flat for
                             eta<=2 then drops at eta=4 (the STABILITY result;
                             eta=0 is test-best -> consistent with the negative
                             result, positive eta does not help).
  6. fig_component_ablation - single-col bar chart: removal-minus-full NDCG@10
                             delta per C-CRP component, line at 0. Bars >=0 mean
                             removing the component does NOT hurt / helps. The
                             characterized negative result (components inert or
                             redundant), NOT "every component necessary".
  7. fig_observation        - single-col line plot: mean NDCG@10 across the 4
                             diagnostic domains vs uncertainty-bin index Q01..Q05,
                             two lines (C-CRP and LLMEmb). Both DECLINE as
                             uncertainty rises -> the uncertainty signal stratifies
                             ranking reliability for the baseline too (descriptive /
                             partly-circular for C-CRP's own uncertainty; the
                             cross-method decline is the motivation).

ALL numbers below are transcribed from (and verified against) the paper source:
  - Paper/tables/improvement_over_strongest.tex           (fig 1)
  - Paper/tables/full_official_ndcg10_ranking.tex
    + outputs/ccrp_v3_formal/main_comparison_table.csv    (fig 2)
  - Paper/tables/significance_summary.tex
    + outputs/summary/paper_critical/significance_all8/
        all8_domains_significance_summary.csv             (fig 3)
  - outputs/summary/paper_critical/gap_decomposition/
        gap_decomposition_results.json                    (fig 4)
and the three diagnostic figures read live from (under
  outputs/summary/paper_critical/
    ccrp_signal_generation_plan_post_performance_gate_20260606/):
  - ccrp_hyperparameter_four_domain/
        ccrp_hyperparameter_four_domain_curve_rows.csv     (fig 5)
  - ccrp_component_ablation_four_domain/
        component_ablation_four_domain_component_summary.csv (fig 6)
  - observation_four_domain/
        observation_four_domain_summary_rows.csv           (fig 7)

Run (local, CPU):
    C:\\Python314\\python.exe scripts\\make_paper_figures.py
"""

import csv
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[1]
FIGDIR = REPO / "Paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Source data for the CSV-driven diagnostic figures (numbers read at run time).
DATA = (REPO / "outputs" / "summary" / "paper_critical" /
        "ccrp_signal_generation_plan_post_performance_gate_20260606")
HYPERPARAM_CSV = (DATA / "ccrp_hyperparameter_four_domain" /
                  "ccrp_hyperparameter_four_domain_curve_rows.csv")
COMPONENT_CSV = (DATA / "ccrp_component_ablation_four_domain" /
                 "component_ablation_four_domain_component_summary.csv")
OBSERVATION_CSV = (DATA / "observation_four_domain" /
                   "observation_four_domain_summary_rows.csv")


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

# --------------------------------------------------------------------------- #
# Global publication style (consistent across every figure)
# --------------------------------------------------------------------------- #
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,   # embed TrueType (editable text in vector PDF)
    "ps.fonttype": 42,
})

# Colorblind-safe palette (Wong / seaborn "colorblind")
CB = sns.color_palette("colorblind")
GREEN = CB[2]   # bluish green  -> wins
RED = CB[3]     # red/vermillion -> significant losses
GRAY = "#9a9a9a"
BLUE = CB[0]

# acmart sigconf column / text widths (inches)
COL_W = 3.3
FULL_W = 7.0


def _save(fig, name):
    """Save a figure as both PDF (vector) and PNG (300 dpi)."""
    pdf = FIGDIR / f"{name}.pdf"
    png = FIGDIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"  wrote {pdf.name} and {png.name}")


# --------------------------------------------------------------------------- #
# Figure 1: per-domain NDCG@10 %% improvement over strongest baseline (HEADLINE)
#   Source: Paper/tables/improvement_over_strongest.tex (NDCG@10 column)
# --------------------------------------------------------------------------- #
def fig_improvement_bars():
    # (domain, NDCG@10 %% improvement over strongest baseline, wins?)
    data = [
        ("Electronics", 53.2, True),
        ("Tools",       43.3, True),
        ("Home",        41.0, True),
        ("Toys",        32.2, True),
        ("Sports",      29.7, True),
        ("Books",       21.6, True),
        ("Beauty",     -11.0, False),
        ("Movies",     -24.2, False),
    ]
    data.sort(key=lambda r: r[1], reverse=True)  # descending
    domains = [d for d, _, _ in data]
    vals = [v for _, v, _ in data]
    wins = [w for _, _, w in data]
    colors = [GREEN if w else RED for w in wins]

    fig, ax = plt.subplots(figsize=(COL_W, 2.9))
    ypos = np.arange(len(domains))[::-1]  # best at top
    ax.barh(ypos, vals, color=colors, edgecolor="black", linewidth=0.5, height=0.66)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(domains)
    ax.set_xlabel("NDCG@10 gain over strongest baseline (%)")
    ax.set_xlim(-34, 72)

    # annotate each bar with its value
    for y, v in zip(ypos, vals):
        if v >= 0:
            ax.text(v + 1.2, y, f"+{v:.1f}", va="center", ha="left",
                    fontsize=7.5, fontweight="bold", color="black")
        else:
            ax.text(v - 1.2, y, f"{v:.1f}", va="center", ha="right",
                    fontsize=7.5, fontweight="bold", color="black")

    ax.set_title("C-CRP ranks first in 6 of 8 domains", fontsize=10, pad=6)

    legend = [
        Patch(facecolor=GREEN, edgecolor="black", label="Ranks first (6 domains)"),
        Patch(facecolor=RED, edgecolor="black", label="Not first (Beauty, Movies)"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=True, framealpha=0.9,
              borderpad=0.4, handlelength=1.2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "fig_improvement_bars")


# --------------------------------------------------------------------------- #
# Figure 2: NDCG@10 heatmap, methods x domains (C-CRP row highlighted)
#   Source: Paper/tables/full_official_ndcg10_ranking.tex
# --------------------------------------------------------------------------- #
def fig_main_heatmap():
    domains = ["Beauty", "Books", "Elec.", "Movies",
               "Sports", "Toys", "Home", "Tools"]
    # rows: C-CRP first, then the 8 official baselines
    methods = ["C-CRP", "LLMEmb", "ProEx", "ProMax", "IRLLRec",
               "RLMRec", "LLM2Rec", "LLM-ESR", "ELMRec"]

    # NDCG@10 values keyed [method][domain] (from full_official_ndcg10_ranking.tex)
    ndcg = {
        "C-CRP":   [0.1341, 0.3328, 0.1833, 0.1281, 0.2329, 0.2708, 0.1324, 0.1661],
        "LLMEmb":  [0.1226, 0.2737, 0.1196, 0.1690, 0.1795, 0.2049, 0.0939, 0.1159],
        "ProEx":   [0.1506, 0.1301, 0.0631, 0.1254, 0.0742, 0.0810, 0.0549, 0.0557],
        "ProMax":  [0.1270, 0.0980, 0.0619, 0.1179, 0.0722, 0.0794, 0.0469, 0.0503],
        "IRLLRec": [0.1288, 0.1665, 0.0997, 0.1397, 0.1269, 0.1338, 0.0709, 0.0853],
        "RLMRec":  [0.1246, 0.1302, 0.0818, 0.1283, 0.1000, 0.1065, 0.0599, 0.0684],
        "LLM2Rec": [0.0516, 0.2234, 0.0532, 0.1464, 0.0957, 0.1789, 0.0509, 0.0815],
        "LLM-ESR": [0.1043, 0.0723, 0.0678, 0.0745, 0.0758, 0.0546, 0.0554, 0.0606],
        "ELMRec":  [0.0789, 0.0523, 0.0489, 0.0667, 0.0484, 0.0486, 0.0460, 0.0459],
    }
    mat = np.array([ndcg[m] for m in methods])

    fig, ax = plt.subplots(figsize=(FULL_W, 3.7))
    sns.heatmap(
        mat, ax=ax, cmap="YlGnBu", annot=True, fmt=".3f",
        annot_kws={"size": 7.0},
        xticklabels=domains, yticklabels=methods,
        cbar_kws={"label": "NDCG@10", "shrink": 0.85, "pad": 0.02},
        linewidths=0.4, linecolor="white",
    )

    # Highlight the C-CRP row (row index 0) with a bold border + mark the best
    # method per domain with a small triangle marker at the cell's top-left.
    ax.add_patch(plt.Rectangle((0, 0), len(domains), 1, fill=False,
                               edgecolor="black", lw=2.2, clip_on=False))
    best_row = mat.argmax(axis=0)
    for j, bi in enumerate(best_row):
        ax.scatter(j + 0.16, bi + 0.16, marker="*", s=42,
                   color="white", edgecolors="black", linewidths=0.4, zorder=5)

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    # emphasise the C-CRP tick label
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_title("Same-candidate NDCG@10 across 8 domains "
                 "($\\bigstar$ = best per domain; C-CRP leads in 6/8)",
                 fontsize=9.5, pad=8)
    ax.set_xlabel("Domain")
    ax.set_ylabel("Method")
    fig.tight_layout()
    _save(fig, "fig_main_heatmap")


# --------------------------------------------------------------------------- #
# Figure 3: paired-test outcomes /56 per domain (stacked horizontal bars)
#   Source: Paper/tables/significance_summary.tex
#         + all8_domains_significance_summary.csv
# --------------------------------------------------------------------------- #
def fig_significance():
    # (domain, ahead-Holm-sig, behind-Holm-sig, n.s.); each row sums to 56.
    # n.s. = significant_ambiguous_sign + not_significant (per CSV); Movies
    # 5 ambiguous + 13 not-sig = 18 n.s.
    data = [
        ("Sports",      56,  0,  0),
        ("Toys",        56,  0,  0),
        ("Home",        56,  0,  0),
        ("Tools",       56,  0,  0),
        ("Electronics", 56,  0,  0),
        ("Books",       54,  1,  1),
        ("Beauty",      14,  0, 42),
        ("Movies",      16, 22, 18),
    ]
    domains = [d for d, *_ in data]
    ahead = np.array([a for _, a, _, _ in data])
    behind = np.array([b for _, _, b, _ in data])
    ns = np.array([n for _, _, _, n in data])

    fig, ax = plt.subplots(figsize=(COL_W, 3.1))
    ypos = np.arange(len(domains))[::-1]
    ax.barh(ypos, ahead, color=GREEN, edgecolor="black", linewidth=0.4,
            label="C-CRP ahead (Holm-sig.)", height=0.68)
    ax.barh(ypos, ns, left=ahead, color=GRAY, edgecolor="black", linewidth=0.4,
            label="n.s.", height=0.68)
    ax.barh(ypos, behind, left=ahead + ns, color=RED, edgecolor="black",
            linewidth=0.4, label="C-CRP behind (Holm-sig.)", height=0.68)

    ax.set_yticks(ypos)
    ax.set_yticklabels(domains)
    ax.set_xlim(0, 56)
    ax.set_xlabel("Paired tests (of 56: 8 baselines $\\times$ 7 metrics)")
    ax.set_title("Signed paired-test outcomes per domain", fontsize=10, pad=6)

    # annotate non-zero "ahead" and "behind" counts
    for y, a, b, n in zip(ypos, ahead, behind, ns):
        if a >= 6:
            ax.text(a / 2, y, f"{a}", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold")
        if b > 0:
            ax.text(a + n + b / 2, y, f"{b}", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1,
              frameon=True, framealpha=0.9, borderpad=0.4, handlelength=1.2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "fig_significance")


# --------------------------------------------------------------------------- #
# Figure 4: two-panel gap decomposition
#   Source: gap_decomposition_results.json
#     left : gap%% vs mean_user_hist_len  (Spearman rho = -0.69)
#     right: gap%% vs popularity Gini      (Spearman rho = +0.57)
# --------------------------------------------------------------------------- #
def fig_gap_decomp():
    # domain : (gap_vs_strongest_pct, mean_user_hist_len, gini, wins)
    rows = {
        "Beauty":      (-10.98, 5.59, 0.258, False),
        "Books":       ( 21.61, 5.80, 0.602, True),
        "Electronics": ( 53.20, 4.77, 0.764, True),
        "Movies":      (-24.17, 6.72, 0.732, False),
        "Sports":      ( 29.73, 4.07, 0.661, True),
        "Toys":        ( 32.19, 4.39, 0.652, True),
        "Home":        ( 40.99, 5.09, 0.744, True),
        "Tools":       ( 43.27, 3.99, 0.691, True),
    }
    names = list(rows)
    gap = np.array([rows[n][0] for n in names])
    hist = np.array([rows[n][1] for n in names])
    gini = np.array([rows[n][2] for n in names])
    wins = [rows[n][3] for n in names]
    pt_colors = [GREEN if w else RED for w in wins]

    # small label offsets to reduce overlap (data units), per panel
    off_hist = {
        "Beauty": (0.04, -4.5), "Books": (0.05, 2.5), "Electronics": (0.05, 0),
        "Movies": (-0.05, 3.0), "Sports": (0.05, -2.5), "Toys": (0.05, 2.5),
        "Home": (0.05, 0), "Tools": (0.05, 2.5),
    }
    off_gini = {
        "Beauty": (0.012, 0), "Books": (0.012, -3.5), "Electronics": (-0.012, 3.5),
        "Movies": (0.012, 0), "Sports": (0.012, -3.0), "Toys": (0.012, 2.5),
        "Home": (0.012, 2.5), "Tools": (0.012, -2.5),
    }

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.1))

    def _panel(ax, x, xlabel, rho, offsets, xpad):
        ax.axhline(0, color=GRAY, linewidth=0.8, linestyle="--", zorder=0)
        # OLS regression line over all 8 points
        coef = np.polyfit(x, gap, 1)
        xs = np.linspace(x.min() - xpad, x.max() + xpad, 50)
        ax.plot(xs, np.polyval(coef, xs), color=BLUE, linewidth=1.4,
                zorder=1, label=f"OLS fit ($\\rho={rho}$)")
        ax.scatter(x, gap, c=pt_colors, edgecolors="black", linewidths=0.6,
                   s=46, zorder=3)
        for n, xi, yi in zip(names, x, gap):
            dx, dy = offsets[n]
            ax.annotate(n, (xi, yi), (xi + dx, yi + dy), fontsize=6.8,
                        ha="left" if dx >= 0 else "right", va="center",
                        zorder=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("NDCG@10 gap vs. strongest baseline (%)")
        ax.legend(loc="upper right", frameon=True, framealpha=0.9,
                  borderpad=0.3, handlelength=1.3, fontsize=7.5)

    _panel(axes[0], hist, "Mean user-history length (items)", "-0.69",
           off_hist, xpad=0.25)
    _panel(axes[1], gini, "Popularity Gini coefficient", "+0.57",
           off_gini, xpad=0.04)

    axes[0].set_title("(a) Shorter histories favor C-CRP", fontsize=9)
    axes[1].set_title("(b) Higher skew favors C-CRP (reversed)", fontsize=9)

    # color legend for win/loss markers, placed on left panel
    marker_legend = [
        Patch(facecolor=GREEN, edgecolor="black", label="C-CRP ranks first"),
        Patch(facecolor=RED, edgecolor="black", label="C-CRP not first"),
    ]
    axes[0].legend(handles=[
        plt.Line2D([], [], color=BLUE, lw=1.4, label=r"OLS fit ($\rho=-0.69$)"),
    ] + marker_legend, loc="lower left", frameon=True, framealpha=0.9,
        borderpad=0.3, handlelength=1.3, fontsize=6.8)

    fig.suptitle(r"Exploratory gap decomposition ($n=8$ domains, "
                 r"descriptive Spearman correlations)", fontsize=9.5, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_gap_decomp")


# --------------------------------------------------------------------------- #
# Figure 5: hyperparameter sensitivity - mean NDCG@10 (4 domains) vs eta
#   Source: ccrp_hyperparameter_four_domain_curve_rows.csv (read live)
#   Stability result: flat for eta<=2, drops at eta=4; eta=0 is test-best.
# --------------------------------------------------------------------------- #
def fig_hyperparam_sensitivity():
    rows = _read_csv(HYPERPARAM_CSV)
    # eta sweep is the main_control rows with control == "eta".
    eta_order = ["0", "0.25", "0.5", "1", "2", "4"]
    eta_vals = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

    def mean_curve(split):
        out = []
        for e in eta_order:
            vals = [float(r["NDCG@10"]) for r in rows
                    if r["row_kind"] == "main_control" and r["control"] == "eta"
                    and r["control_value"] == e and r["split"] == split]
            assert len(vals) == 4, f"expected 4 domains for eta={e}/{split}, got {len(vals)}"
            out.append(sum(vals) / len(vals))
        return out

    valid = mean_curve("valid")
    test = mean_curve("test")
    x = np.arange(len(eta_vals))

    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    ax.plot(x, valid, marker="o", markersize=4.5, linewidth=1.5, color=BLUE,
            label="Validation")
    ax.plot(x, test, marker="s", markersize=4.0, linewidth=1.5, color=GREEN,
            label="Test (reporting only)")

    # mark the 5% pre-registered relative-drop tolerance off the test peak
    test_peak = max(test)
    ax.axhline(test_peak * 0.95, color=GRAY, linewidth=0.8, linestyle=":",
               zorder=0, label=r"$-5\%$ tolerance (test peak)")

    # annotate the test-best (eta=0) point (placed below-right to avoid the title)
    ax.scatter([0], [test[0]], s=58, facecolors="none", edgecolors="black",
               linewidths=1.0, zorder=5)
    span = max(max(valid), max(test)) - min(min(valid), min(test))
    ax.annotate(r"$\eta{=}0$ test-best", (0, test[0]),
                (0.30, test[0] - 0.12 * span), fontsize=7.0, ha="left",
                va="top",
                arrowprops=dict(arrowstyle="-", color="black", lw=0.6))

    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in eta_vals])
    ax.set_xlabel(r"Risk exponent $\eta$")
    ax.set_ylabel("Mean NDCG@10 (4 domains)")
    ax.set_title(r"Stable for $\eta\leq 2$, degrades at $\eta{=}4$",
                 fontsize=9.5, pad=6)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, borderpad=0.4,
              handlelength=1.6, fontsize=7.0)
    fig.tight_layout()
    _save(fig, "fig_hyperparam_sensitivity")


# --------------------------------------------------------------------------- #
# Figure 6: component ablation - removal-minus-full NDCG@10 delta per component
#   Source: component_ablation_four_domain_component_summary.csv (read live)
#   Bars >= 0 -> removing the component does NOT hurt / helps. Negative result.
# --------------------------------------------------------------------------- #
def fig_component_ablation():
    rows = _read_csv(COMPONENT_CSV)
    # (csv ablation key, display label) in the order we want them plotted.
    comps = [
        ("without_boundary_uncertainty", "Boundary\nuncertainty"),
        ("without_counterevidence",      "Counter-\nevidence"),
        ("without_risk_penalty",         "Risk\npenalty"),
        ("without_calibration_gap",      "Calibration\ngap"),
        ("without_evidence_support",     "Evidence\nsupport"),
    ]
    lut = {(r["ablation"], r["metric"]): r for r in rows}
    deltas, nonworse = [], []
    for key, _ in comps:
        rec = lut[(key, "NDCG@10")]
        deltas.append(float(rec["mean_delta_removal_minus_full"]))
        nonworse.append(int(rec["nonworse_domain_count"]))
    deltas = np.array(deltas)

    # green where removing is nonworse / helpful (delta >= 0), gray where mixed,
    # red where removing the component is harmful on the mean (delta < 0).
    colors = [GREEN if d >= 0 else RED for d in deltas]

    fig, ax = plt.subplots(figsize=(COL_W, 2.75))
    xpos = np.arange(len(comps))
    bars = ax.bar(xpos, deltas * 1e3, color=colors, edgecolor="black",
                  linewidth=0.5, width=0.66)
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xticks(xpos)
    ax.set_xticklabels([lbl for _, lbl in comps], fontsize=6.8)
    ax.set_ylabel(r"$\Delta$NDCG@10 ($\times 10^{-3}$)" "\n(remove $-$ full)")
    ax.set_title("Removing a component does not hurt ranking",
                 fontsize=9.0, pad=6)

    # annotate each bar with its raw delta and the nonworse-domain count
    ymax = max(deltas.max(), 0) * 1e3
    ymin = min(deltas.min(), 0) * 1e3
    pad = (ymax - ymin) * 0.06 + 0.05
    for x, d, nw in zip(xpos, deltas, nonworse):
        dv = d * 1e3
        va = "bottom" if dv >= 0 else "top"
        off = pad if dv >= 0 else -pad
        ax.text(x, dv + off, f"{d:+.5f}\n{nw}/4 nonworse", ha="center", va=va,
                fontsize=6.0, fontweight="bold")
    ax.set_ylim(ymin - 6 * pad, ymax + 6 * pad)
    ax.grid(axis="x", visible=False)

    legend = [
        Patch(facecolor=GREEN, edgecolor="black",
              label=r"Removal helps / inert ($\Delta\geq 0$)"),
        Patch(facecolor=RED, edgecolor="black",
              label=r"Removal mildly hurts ($\Delta<0$)"),
    ]
    ax.legend(handles=legend, loc="upper right", frameon=True, framealpha=0.9,
              borderpad=0.3, handlelength=1.1, fontsize=6.4)
    fig.tight_layout()
    _save(fig, "fig_component_ablation")


# --------------------------------------------------------------------------- #
# Figure 7: uncertainty-stratified reliability (baseline-inclusive)
#   Source: observation_four_domain_summary_rows.csv (read live)
#   Mean NDCG@10 (4 domains) vs uncertainty bin Q01..Q05 for C-CRP and LLMEmb;
#   both DECLINE as uncertainty rises (cross-method motivation).
# --------------------------------------------------------------------------- #
def fig_observation():
    rows = _read_csv(OBSERVATION_CSV)
    bins = ["0", "1", "2", "3", "4"]
    labels = ["Q01\n(low)", "Q02", "Q03", "Q04", "Q05\n(high)"]

    def mean_curve(method):
        out = []
        for b in bins:
            vals = [float(r["NDCG@10"]) for r in rows
                    if r["method"] == method and r["uncertainty_bin_index"] == b]
            assert len(vals) == 4, f"expected 4 domains for {method}/bin{b}, got {len(vals)}"
            out.append(sum(vals) / len(vals))
        return out

    ccrp = mean_curve("ccrp_v3")
    llmemb = mean_curve("llmemb")
    x = np.arange(len(bins))

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.plot(x, ccrp, marker="o", markersize=4.5, linewidth=1.5, color=BLUE,
            label="C-CRP")
    ax.plot(x, llmemb, marker="s", markersize=4.0, linewidth=1.5, color=GREEN,
            label="LLMEmb (baseline)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.0)
    ax.set_xlabel("C-CRP uncertainty quantile bin")
    ax.set_ylabel("Mean NDCG@10 (4 domains)")
    ax.set_title("Ranking quality declines as uncertainty rises",
                 fontsize=9.0, pad=6)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, borderpad=0.4,
              handlelength=1.6, fontsize=7.5)
    fig.tight_layout()
    _save(fig, "fig_observation")


def main():
    print(f"Output dir: {FIGDIR}")
    print("Generating figures ...")
    fig_improvement_bars()
    fig_main_heatmap()
    fig_significance()
    fig_gap_decomp()
    fig_hyperparam_sensitivity()
    fig_component_ablation()
    fig_observation()
    print("Done. 7 figures x (pdf + png) = 14 files.")


if __name__ == "__main__":
    main()
