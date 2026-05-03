"""CARE rerank pilot over DeepSeek listwise rank_predictions (pilot-scale)."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.data.protocol import read_jsonl, write_jsonl
from src.eval.paper_metrics import exposure_metrics, ranking_metrics
from src.methods.care_rerank import (
    VARIANT_ORDER,
    build_reranked_prediction_row,
    high_confidence_wrong_top1,
    load_care_rerank_config,
    rerank_candidates_for_user,
    risk_row_changed_top1,
    sanitize_listwise_ranking,
    top1_bucket,
)
from src.utils.manifest import build_manifest, write_manifest
from src.utils.research_artifacts import config_hash, git_commit_or_unknown, utc_timestamp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CARE rerank pilot on DeepSeek rank_predictions.jsonl.")
    p.add_argument(
        "--pilot_root",
        default="outputs/pilots/deepseek_v4_flash_processed_20u_c19_seed42",
        help="Root containing <domain>/<split>/predictions/rank_predictions.jsonl",
    )
    p.add_argument(
        "--output_root",
        default="outputs/pilots/care_rerank_deepseek_v4_flash_processed_20u_c19_seed42",
    )
    p.add_argument("--config", default="configs/methods/care_rerank_pilot.yaml")
    p.add_argument("--reprocess_dir", default="outputs/reprocessed_processed_source")
    return p.parse_args()


def _load_features_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(str(path)):
        uid = str(row.get("user_id", ""))
        if uid:
            out[uid] = row
    return out


def _target_bucket(row: dict[str, Any]) -> str:
    cands = [str(x) for x in row.get("candidate_item_ids", [])]
    bks = [str(x) for x in row.get("candidate_popularity_buckets", [])]
    tgt = str(row.get("target_item_id", ""))
    try:
        idx = cands.index(tgt)
        return str(bks[idx]).lower() if idx < len(bks) else "unknown"
    except ValueError:
        return "unknown"


def _auxiliary_metrics(orig_rows: list[dict[str, Any]], reranked_rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(reranked_rows)
    if not n:
        return {}
    inv = 0
    conf_ok = 0
    hc_before = 0
    hc_after = 0
    risk_changed = 0
    tail_tgt = 0
    tail_tgt_hit_after = 0
    for o, r in zip(orig_rows, reranked_rows):
        if o.get("is_valid") is False:
            inv += 1
        try:
            x = float(r.get("raw_confidence") or float("nan"))
            if np.isfinite(x):
                conf_ok += 1
        except (TypeError, ValueError):
            pass
        if high_confidence_wrong_top1(o):
            hc_before += 1
        if high_confidence_wrong_top1(r):
            hc_after += 1
        if risk_row_changed_top1(o, r):
            risk_changed += 1
        if _target_bucket(o) == "tail":
            tail_tgt += 1
            if bool(r.get("correctness")):
                tail_tgt_hit_after += 1
    return {
        "invalid_output_rate": float(inv / n),
        "confidence_available_rate": float(conf_ok / n),
        "high_confidence_wrong_rate_before": float(hc_before / n),
        "high_confidence_wrong_rate_after": float(hc_after / n),
        "high_risk_top1_changed_rate": float(risk_changed / n),
        "tail_target_hit_at_1_rate_after": float(tail_tgt_hit_after / tail_tgt) if tail_tgt else float("nan"),
    }


def _exposure_top1_shares(rows: list[dict[str, Any]]) -> dict[str, float]:
    head = mid = tail = unk = 0
    for r in rows:
        b = top1_bucket(r)
        if b == "head":
            head += 1
        elif b == "mid":
            mid += 1
        elif b == "tail":
            tail += 1
        else:
            unk += 1
    n = max(1, len(rows))
    return {
        "head_top1_share": head / n,
        "mid_top1_share": mid / n,
        "tail_top1_share": tail / n,
        "unknown_top1_share": unk / n,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _merge_eval_metrics(eval_path: Path, extras: dict[str, Any]) -> None:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    data["care_rerank_pilot"] = extras
    eval_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _process_one_variant_domain_split(
    *,
    variant: str,
    domain: str,
    split: str,
    preds: list[dict[str, Any]],
    preds_path: Path,
    features_path: Path,
    candidates_path: Path,
    output_base: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    feats = _load_features_map(features_path)
    out_dir = output_base / variant / domain / split
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    score_rows: list[dict[str, Any]] = []
    reranked: list[dict[str, Any]] = []
    hc_changes: list[dict[str, Any]] = []
    for row in preds:
        uid = str(row.get("user_id", ""))
        feat = feats.get(uid, {})
        row_s = sanitize_listwise_ranking(row)
        new_rank, srows = rerank_candidates_for_user(row_s, feat, variant, cfg)
        for s in srows:
            s["variant"] = variant
            s["domain"] = domain
            s["split"] = split
        score_rows.extend(srows)
        new_row = build_reranked_prediction_row(row_s, new_rank, variant=variant)
        pr0 = str(row_s.get("predicted_ranking", [""])[0]) if row_s.get("predicted_ranking") else ""
        hc_changes.append(
            {
                "user_id": uid,
                "domain": domain,
                "split": split,
                "variant": variant,
                "hc_wrong_before": high_confidence_wrong_top1(row_s),
                "hc_wrong_after": high_confidence_wrong_top1(new_row),
                "top1_changed": bool(pr0 != str(new_rank[0])) if new_rank else False,
            }
        )
        reranked.append(new_row)

    write_jsonl(score_rows, str(pred_dir / "care_scores.jsonl"))
    write_jsonl(reranked, str(pred_dir / "reranked_rank_predictions.jsonl"))

    pilot_cfg = {
        "run_type": "pilot",
        "seed": 42,
        "method": f"care_rerank_{variant}",
        "variant": variant,
        "domain": domain,
        "split": split,
        "candidate_size": len(preds[0]["candidate_item_ids"]) if preds else 19,
        "care_config_path": str(Path("configs/methods/care_rerank_pilot.yaml").resolve()),
    }
    write_manifest(
        out_dir / "care_manifest.json",
        build_manifest(
            config=pilot_cfg,
            dataset=domain,
            domain=domain,
            raw_data_paths=[],
            processed_data_paths=[str(preds_path), str(features_path), str(candidates_path)],
            method=f"care_rerank_{variant}",
            backend="rerank",
            model="care-rerank-pilot",
            prompt_template="listwise_ranking_v1",
            seed=42,
            candidate_size=len(preds[0]["candidate_item_ids"]) if preds else None,
            calibration_source="diagnostic_only",
            command=sys.argv,
            mock_data_used=False,
        ),
    )

    eval_dir = out_dir / "eval"
    reranked_path = pred_dir / "reranked_rank_predictions.jsonl"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli.evaluate",
            "--predictions_path",
            str(reranked_path),
            "--output_dir",
            str(eval_dir),
            "--candidates_source_path",
            str(candidates_path),
        ],
        check=True,
        cwd=str(Path.cwd()),
    )
    preds_s = [sanitize_listwise_ranking(r) for r in preds]
    exp_before = _exposure_top1_shares(preds_s)
    exp_after = _exposure_top1_shares(reranked)
    aux = _auxiliary_metrics(preds_s, reranked)
    ranking = ranking_metrics(reranked, ks=(1, 5, 10))
    extras = {
        "auxiliary": aux,
        "exposure_top1_before": exp_before,
        "exposure_top1_after": exp_after,
        "exposure_shift": {k: float(exp_after[k] - exp_before[k]) for k in exp_before},
        "ranking": ranking,
        "exposure_metrics_after": exposure_metrics(reranked),
    }
    _merge_eval_metrics(eval_dir / "metrics.json", extras)
    return {
        "variant": variant,
        "domain": domain,
        "split": split,
        "ranking": ranking,
        "auxiliary": aux,
        "exposure_before": exp_before,
        "exposure_after": exp_after,
        "hc_changes": hc_changes,
    }


def _parse_domain_split(pred_path: Path, pilot_root: Path) -> tuple[str, str]:
    rel = pred_path.resolve().relative_to(pilot_root.resolve())
    parts = rel.parts
    if len(parts) >= 3 and parts[1] in ("valid", "test", "train"):
        return parts[0], parts[1]
    raise ValueError(f"Cannot parse domain/split from {pred_path}")


def main() -> None:
    args = parse_args()
    pilot_root = Path(args.pilot_root).resolve()
    output_root = Path(args.output_root).resolve()
    cfg = load_care_rerank_config(Path(args.config))
    reprocess = Path(args.reprocess_dir)

    paths = sorted(Path(p) for p in glob.glob(os.path.join(str(pilot_root), "**", "rank_predictions.jsonl"), recursive=True))
    if not paths:
        print("[care_rerank_pilot] no rank_predictions.jsonl found")
        return

    aggregate_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    hc_rows: list[dict[str, Any]] = []

    for pred_path in paths:
        preds = read_jsonl(str(pred_path))
        domain, split = _parse_domain_split(pred_path, pilot_root)
        features_path = pred_path.parent / "uncertainty_features.jsonl"
        if not features_path.is_file():
            raise FileNotFoundError(f"missing uncertainty_features: {features_path}")
        candidates_path = reprocess / domain / f"{split}_candidates.jsonl"
        if not candidates_path.is_file():
            raise FileNotFoundError(f"missing candidates: {candidates_path}")
        n_rows = len(preds)
        calib_sum = pilot_root / "calibration_diagnostics" / f"{domain}_{split}" / "calibration_summary.json"
        diag_auroc = float("nan")
        if calib_sum.is_file():
            try:
                diag_auroc = float(json.loads(calib_sum.read_text(encoding="utf-8")).get("auroc", float("nan")))
            except (json.JSONDecodeError, TypeError, ValueError):
                diag_auroc = float("nan")
        for variant in VARIANT_ORDER:
            if variant not in cfg["variants"]:
                raise KeyError(variant)
            meta = _process_one_variant_domain_split(
                variant=variant,
                domain=domain,
                split=split,
                preds=preds,
                preds_path=pred_path,
                features_path=features_path,
                candidates_path=candidates_path,
                output_base=output_root,
                cfg=cfg,
            )
            rk = meta["ranking"]
            aux = meta["auxiliary"]
            ea = meta["exposure_after"]
            aggregate_rows.append(
                {
                    "variant": variant,
                    "domain": domain,
                    "split": split,
                    "rows": n_rows,
                    "HR@1": rk.get("HR@1"),
                    "HR@5": rk.get("HR@5"),
                    "HR@10": rk.get("HR@10"),
                    "Recall@1": rk.get("Recall@1"),
                    "Recall@5": rk.get("Recall@5"),
                    "Recall@10": rk.get("Recall@10"),
                    "NDCG@1": rk.get("NDCG@1"),
                    "NDCG@5": rk.get("NDCG@5"),
                    "NDCG@10": rk.get("NDCG@10"),
                    "MRR@1": rk.get("MRR@1"),
                    "MRR@5": rk.get("MRR@5"),
                    "MRR@10": rk.get("MRR@10"),
                    "confidence_available_rate": aux.get("confidence_available_rate"),
                    "invalid_output_rate": aux.get("invalid_output_rate"),
                    "high_confidence_wrong_rate_before": aux.get("high_confidence_wrong_rate_before"),
                    "high_confidence_wrong_rate_after": aux.get("high_confidence_wrong_rate_after"),
                    "head_prediction_rate_after": ea.get("head_top1_share"),
                    "tail_target_hit_at_1_rate_after": aux.get("tail_target_hit_at_1_rate_after"),
                    "high_risk_top1_changed_rate": aux.get("high_risk_top1_changed_rate"),
                    "confidence_correctness_auc_diag": diag_auroc,
                }
            )
            exposure_rows.append(
                {
                    "variant": variant,
                    "domain": domain,
                    "split": split,
                    **{f"before_{k}": v for k, v in meta["exposure_before"].items()},
                    **{f"after_{k}": v for k, v in meta["exposure_after"].items()},
                }
            )
            hc_rows.extend(meta["hc_changes"])
            print(f"[care_rerank_pilot] {variant} {domain} {split} done")

    out_agg = output_root / "care_rerank_aggregate.csv"
    _write_csv(aggregate_rows, out_agg)
    _write_csv(exposure_rows, output_root / "exposure_shift.csv")
    _write_csv(hc_rows, output_root / "high_confidence_wrong_changes.csv")
    (output_root / "pilot_run_meta.json").write_text(
        json.dumps(
            {
                "created_at": utc_timestamp(),
                "git_commit": git_commit_or_unknown("."),
                "pilot_root": str(pilot_root),
                "output_root": str(output_root),
                "config_hash": config_hash({"care": str(Path(args.config).resolve())}),
                "note": "Pilot only; is_paper_result=false on all rows.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[care_rerank_pilot] aggregate {out_agg}")


if __name__ == "__main__":
    main()
