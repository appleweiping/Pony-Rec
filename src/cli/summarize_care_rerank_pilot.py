"""Print CARE rerank pilot audit tables and verify care_manifest.json files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

EXPECTED_VARIANTS = frozenset(
    {
        "original_deepseek",
        "confidence_only",
        "popularity_penalty_only",
        "uncertainty_only",
        "care_full",
    }
)

VARIANT_PRINT_ORDER = (
    "original_deepseek",
    "confidence_only",
    "popularity_penalty_only",
    "uncertainty_only",
    "care_full",
)

METRIC_KEYS = (
    "HR@1",
    "HR@5",
    "HR@10",
    "Recall@5",
    "Recall@10",
    "NDCG@5",
    "NDCG@10",
    "MRR@5",
    "MRR@10",
    "high_confidence_wrong_rate_after",
    "head_prediction_rate_after",
    "tail_target_hit_at_1_rate_after",
)

# Substrings that must not appear anywhere in a serialized manifest (case-insensitive).
MANIFEST_BANNED_SUBSTRINGS = (
    "root_main",
    "srpd",
    "old_predictions",
    "old predictions",
)

# Rank predictions must live under this DeepSeek 20-user pilot tree.
DEEPSEEK_PILOT_MARK = "deepseek_v4_flash_processed_20u_c19_seed42"


def _f(x: str) -> float:
    return float(x) if x not in ("", None) else float("nan")


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _index_aggregate(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    return {(r["domain"], r["split"], r["variant"]): r for r in rows}


def print_per_domain_variant_metrics(rows: list[dict[str, str]], variants: Iterable[str] | None = None) -> None:
    variants = tuple(variants or VARIANT_PRINT_ORDER)
    ix = _index_aggregate(rows)
    domains = sorted({r["domain"] for r in rows})
    splits = ("valid", "test")
    print("\n=== Per domain / split / variant (key metrics) ===\n")
    header = ("domain", "split", "variant", "HR@1", "HR@5", "NDCG@5", "NDCG@10", "MRR@5", "HC_wrong_after", "head@1_after")
    print("\t".join(header))
    for d in domains:
        for s in splits:
            for v in variants:
                key = (d, s, v)
                r = ix.get(key)
                if not r:
                    continue
                print(
                    "\t".join(
                        [
                            d,
                            s,
                            v,
                            r.get("HR@1", ""),
                            r.get("HR@5", ""),
                            f'{_f(r.get("NDCG@5", "0")):.4f}',
                            f'{_f(r.get("NDCG@10", "0")):.4f}',
                            f'{_f(r.get("MRR@5", "0")):.4f}',
                            f'{_f(r.get("high_confidence_wrong_rate_after", "0")):.4f}',
                            f'{_f(r.get("head_prediction_rate_after", "0")):.4f}',
                        ]
                    )
                )


def _delta_row(
    base: dict[str, str] | None, other: dict[str, str] | None, keys: tuple[str, ...]
) -> dict[str, float]:
    if not base or not other:
        return {}
    out: dict[str, float] = {}
    for k in keys:
        if k not in base or k not in other:
            continue
        dv = _f(other[k]) - _f(base[k])
        if abs(dv) > 1e-9:
            out[k] = dv
    return out


def print_delta_block(
    title: str,
    agg: dict[tuple[str, str, str], dict[str, str]],
    ref_variant: str,
    cmp_variant: str,
    keys: tuple[str, ...] = METRIC_KEYS,
) -> None:
    print(f"\n=== {title} ({cmp_variant} minus {ref_variant}) ===\n")
    printed = False
    domains = sorted({k[0] for k in agg})
    for d in domains:
        for s in ("valid", "test"):
            ref = agg.get((d, s, ref_variant))
            cmp = agg.get((d, s, cmp_variant))
            delta = _delta_row(ref, cmp, keys)
            if not delta:
                continue
            printed = True
            print(f"{d}\t{s}\t" + " ".join(f"{k}={delta[k]:+.6g}" for k in sorted(delta)))
    if not printed:
        print("(no metric deltas above float noise threshold)")


def print_exposure_table(path: Path) -> None:
    rows = _load_csv_rows(path)
    print("\n=== Head / mid / tail top-1 exposure (before → after) ===\n")
    print(
        "\t".join(
            [
                "variant",
                "domain",
                "split",
                "head_b",
                "mid_b",
                "tail_b",
                "head_a",
                "mid_a",
                "tail_a",
            ]
        )
    )
    for r in rows:
        print(
            "\t".join(
                [
                    r["variant"],
                    r["domain"],
                    r["split"],
                    r["before_head_top1_share"],
                    r["before_mid_top1_share"],
                    r["before_tail_top1_share"],
                    r["after_head_top1_share"],
                    r["after_mid_top1_share"],
                    r["after_tail_top1_share"],
                ]
            )
        )


def hc_wrong_summary(path: Path) -> None:
    rows = _load_csv_rows(path)
    by: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by[(r["domain"], r["split"], r["variant"])].append(r)

    print("\n=== High-confidence wrong: fixed / hurt / top1_changed (all users) ===\n")
    print("domain\tsplit\tvariant\tfixed\thurt\ttop1_changed")
    for d in sorted({k[0] for k in by}):
        for s in ("valid", "test"):
            for v in VARIANT_PRINT_ORDER:
                lst = by.get((d, s, v))
                if not lst:
                    continue
                fixed = hurt = changed = 0
                for x in lst:
                    b = x["hc_wrong_before"].lower() == "true"
                    a = x["hc_wrong_after"].lower() == "true"
                    if b and not a:
                        fixed += 1
                    if (not b) and a:
                        hurt += 1
                    if x["top1_changed"].lower() == "true":
                        changed += 1
                print(f"{d}\t{s}\t{v}\t{fixed}\t{hurt}\t{changed}")


def verify_manifest(path: Path, *, repo_root: Path | None = None) -> list[str]:
    """Return list of error strings; empty if OK."""
    errs: list[str] = []
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"{path}: invalid JSON: {e}"]

    rel = str(path)
    if data.get("run_type") != "pilot":
        errs.append(f"{rel}: run_type={data.get('run_type')!r} expected pilot")
    if data.get("backend_type") != "rerank":
        errs.append(f"{rel}: backend_type={data.get('backend_type')!r} expected rerank")
    if data.get("is_paper_result") is not False:
        errs.append(f"{rel}: is_paper_result={data.get('is_paper_result')!r} expected False")
    if int(data.get("candidate_size", -1)) != 19:
        errs.append(f"{rel}: candidate_size={data.get('candidate_size')!r} expected 19")
    if int(data.get("seed", -1)) != 42:
        errs.append(f"{rel}: seed={data.get('seed')!r} expected 42")

    method = str(data.get("method", ""))
    if not method.startswith("care_rerank_"):
        errs.append(f"{rel}: method={method!r} expected care_rerank_<variant>")
    else:
        suf = method.removeprefix("care_rerank_")
        if suf not in EXPECTED_VARIANTS:
            errs.append(f"{rel}: variant suffix {suf!r} not in expected set")

    blob = json.dumps(data, ensure_ascii=False).lower()
    for banned in MANIFEST_BANNED_SUBSTRINGS:
        if banned.lower() in blob:
            errs.append(f"{rel}: forbidden substring {banned!r} in manifest")

    paths: list[str] = []
    for p in data.get("processed_data_paths") or []:
        paths.append(str(p))
    rank_paths = [p for p in paths if "rank_predictions.jsonl" in p.replace("\\", "/")]
    if not rank_paths:
        errs.append(f"{rel}: no rank_predictions.jsonl in processed_data_paths")
    else:
        for rp in rank_paths:
            norm = rp.replace("\\", "/")
            if DEEPSEEK_PILOT_MARK not in norm:
                errs.append(f"{rel}: rank_predictions path must include {DEEPSEEK_PILOT_MARK!r}, got {rp!r}")

    if repo_root is not None:
        # Optional: ensure rank_predictions resolves under repo when absolute.
        for rp in rank_paths:
            try:
                p = Path(rp)
                if p.is_absolute():
                    p.resolve().relative_to(repo_root.resolve())
            except ValueError:
                errs.append(f"{rel}: rank_predictions path outside repo: {rp!r}")

    return errs


def discover_manifests(output_root: Path) -> list[Path]:
    # variant / domain / split / care_manifest.json
    return sorted(output_root.glob("*/*/*/care_manifest.json"))


def print_pilot_meta(path: Path) -> None:
    if not path.is_file():
        print(f"\n(no {path})\n")
        return
    meta = json.loads(path.read_text(encoding="utf-8"))
    print("\n=== pilot_run_meta.json ===\n")
    for k in ("created_at", "git_commit", "pilot_root", "output_root", "config_hash", "note"):
        if k in meta:
            print(f"{k}: {meta[k]}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output_root",
        type=Path,
        default=Path("outputs/pilots/care_rerank_deepseek_v4_flash_processed_20u_c19_seed42"),
        help="CARE rerank pilot output directory",
    )
    p.add_argument(
        "--repo_root",
        type=Path,
        default=None,
        help="Repository root for optional path containment checks (default: cwd)",
    )
    p.add_argument(
        "--skip_manifest_verify",
        action="store_true",
        help="Do not scan care_manifest.json files (for unit tests without outputs)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.output_root.resolve()
    repo_root = (args.repo_root or Path.cwd()).resolve()

    agg_path = root / "care_rerank_aggregate.csv"
    exp_path = root / "exposure_shift.csv"
    hc_path = root / "high_confidence_wrong_changes.csv"
    meta_path = root / "pilot_run_meta.json"

    if not agg_path.is_file():
        print(f"Missing {agg_path}", file=sys.stderr)
        return 2

    agg_rows = _load_csv_rows(agg_path)
    agg_ix = _index_aggregate(agg_rows)
    print_per_domain_variant_metrics(agg_rows)
    print_delta_block("CARE_full vs original_deepseek", agg_ix, "original_deepseek", "care_full")
    print_delta_block("CARE_full vs confidence_only", agg_ix, "confidence_only", "care_full")
    print_delta_block("CARE_full vs popularity_penalty_only", agg_ix, "popularity_penalty_only", "care_full")
    print_delta_block("CARE_full vs uncertainty_only", agg_ix, "uncertainty_only", "care_full")

    if exp_path.is_file():
        print_exposure_table(exp_path)
    else:
        print(f"\n(missing {exp_path})\n", file=sys.stderr)

    if hc_path.is_file():
        hc_wrong_summary(hc_path)
    else:
        print(f"\n(missing {hc_path})\n", file=sys.stderr)

    print_pilot_meta(meta_path)

    if args.skip_manifest_verify:
        return 0

    manifests = discover_manifests(root)
    if len(manifests) != 40:
        print(f"\nWARN: expected 40 manifests, found {len(manifests)}\n", file=sys.stderr)

    all_errs: list[str] = []
    for m in manifests:
        all_errs.extend(verify_manifest(m, repo_root=repo_root))

    print("\n=== Manifest verification ===\n")
    if all_errs:
        for e in all_errs:
            print(e, file=sys.stderr)
        return 1
    print(f"OK: {len(manifests)} care_manifest.json files passed checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
