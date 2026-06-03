from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


_REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the paper framework overview figure for Actionable Uncertainty."
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--title", default="Actionable Uncertainty for LLM-based Recommendation")
    parser.add_argument("--subtitle", default="Controlled same-candidate ranking and evidence gates")
    return parser.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_trailing_whitespace(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    stripped = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    if stripped != text:
        path.write_text(stripped, encoding="utf-8")


def _box(
    ax: Any,
    *,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    body: str,
    face: str,
    edge: str,
) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.052, title, ha="center", va="top", fontsize=9.2, fontweight="bold", color="#15202b")
    ax.text(x + 0.027, y + h - 0.115, body, ha="left", va="top", fontsize=7.3, color="#27313c", linespacing=1.18)


def _arrow(ax: Any, start: tuple[float, float], end: tuple[float, float], *, color: str = "#4d6478") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.3,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def build_framework_figure(output_dir: str | Path, *, title: str, subtitle: str) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965, title, ha="center", va="top", fontsize=17, fontweight="bold", color="#111827")
    ax.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=10.5, color="#475569")

    face_a = "#eef6f5"
    face_b = "#f3f6fb"
    face_c = "#fff7ed"
    face_d = "#f5f3ff"
    edge = "#667085"

    main_y = 0.55
    w = 0.155
    h = 0.235
    xs = [0.035, 0.225, 0.415, 0.605, 0.795]
    boxes = [
        (
            "Same-candidate task",
            "User history + fixed\n101-item candidate set\nTrain/valid/test discipline\nNo full-catalog claim",
            face_a,
        ),
        (
            "LLM signal extraction",
            "Task-grounded\nrelevance evidence\nRaw probability/confidence\nEvidence/counterevidence",
            face_b,
        ),
        (
            "Calibration layer",
            "Validation-only calibration\nCalibrated relevance\nposterior\nCalibration gap retained",
            face_b,
        ),
        (
            "C-CRP uncertainty",
            "Boundary ambiguity\nCalibration gap\nEvidence insufficiency\nCounterevidence",
            face_c,
        ),
        (
            "Risk-adjusted ranking",
            "risk_score = posterior\n- eta * uncertainty\nExact score export\nRanked list",
            face_d,
        ),
    ]

    for x, (box_title, body, face) in zip(xs, boxes):
        _box(ax, xy=(x, main_y), wh=(w, h), title=box_title, body=body, face=face, edge=edge)
    for x in xs[:-1]:
        _arrow(ax, (x + w, main_y + h / 2), (x + 0.19, main_y + h / 2))

    _box(
        ax,
        xy=(0.09, 0.265),
        wh=(0.25, 0.205),
        title="Official baseline block",
        body="Pinned official-code-level adapters\nUnified Qwen3-8B representation bridge\nSame candidate rows and metrics",
        face="#f8fafc",
        edge=edge,
    )
    _box(
        ax,
        xy=(0.375, 0.265),
        wh=(0.25, 0.205),
        title="Paper-critical method evidence",
        body="Motivation bins over uncertainty\nLeave-one-component-out ablations\nHyperparameter curves and stability checks",
        face="#f8fafc",
        edge=edge,
    )
    _box(
        ax,
        xy=(0.66, 0.265),
        wh=(0.25, 0.205),
        title="Shared evidence gates",
        body="Exact score coverage = 1.0\nHR/NDCG @5/@10/@20 + MRR\nPaired tests, provenance, local-light package",
        face="#f8fafc",
        edge=edge,
    )

    _arrow(ax, (0.72, main_y), (0.755, 0.47), color="#6b7280")
    _arrow(ax, (0.87, main_y), (0.79, 0.47), color="#6b7280")
    _arrow(ax, (0.215, 0.47), (0.215, main_y), color="#6b7280")
    _arrow(ax, (0.5, 0.47), (0.49, main_y), color="#6b7280")

    ax.text(
        0.5,
        0.18,
        "Claim boundary: C-CRP tests whether task-grounded calibrated uncertainty improves controlled candidate ranking reliability.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#1f2937",
    )
    ax.text(
        0.5,
        0.13,
        "The pipeline does not assert full-catalog recommendation, generative title recommendation, or backbone novelty.",
        ha="center",
        va="center",
        fontsize=9.8,
        color="#64748b",
    )

    caption = (
        "Figure: Framework overview for Actionable Uncertainty. The method builds a controlled same-candidate "
        "ranking task, extracts task-grounded LLM relevance and evidence signals, calibrates the posterior on "
        "validation data, decomposes uncertainty into boundary ambiguity, calibration gap, and evidence "
        "insufficiency/counterevidence, and applies risk-adjusted ranking. Official baselines and C-CRP share "
        "the same candidate rows, metric importer, provenance, and evidence gates."
    )

    paths: dict[str, str] = {}
    for suffix in ("svg", "pdf", "png"):
        path = out / f"framework_overview.{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        if suffix == "svg":
            _strip_trailing_whitespace(path)
        paths[suffix] = str(path)
    plt.close(fig)

    caption_path = out / "framework_overview_caption.md"
    caption_path.write_text(caption + "\n", encoding="utf-8")
    paths["caption"] = str(caption_path)

    provenance_path = out / "framework_overview_provenance.json"
    paths["provenance"] = str(provenance_path)
    manifest_path = out / "framework_overview_manifest.sha256"
    paths["manifest"] = str(manifest_path)
    provenance = {
        "status_label": "paper_critical_framework_overview_draft",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "command": "python scripts/analysis/main_build_framework_overview_figure.py --output_dir "
        + str(out),
        "outputs": paths,
        "caption": caption,
        "claim_boundary": "controlled_same_candidate_ranking_not_full_catalog",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_inputs = [Path(paths[key]) for key in ("svg", "pdf", "png", "caption", "provenance")]
    manifest_path.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in manifest_inputs),
        encoding="utf-8",
    )
    return provenance


def main() -> None:
    args = parse_args()
    provenance = build_framework_figure(args.output_dir, title=args.title, subtitle=args.subtitle)
    print(json.dumps({"ok": True, "outputs": provenance["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
