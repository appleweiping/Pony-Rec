"""Build CARE-aware LoRA training JSONL + summaries (amazon_beauty pilot, n=20)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from src.cli.run_pilot_reprocessed_deepseek import _build_ranking_prompt, _load_item_lookup
from src.data.protocol import read_jsonl, write_jsonl
from src.methods.care_rerank import sanitize_listwise_ranking
from src.methods.care_lora_data import (
    POLICIES,
    build_training_sample,
    evaluate_policies,
    load_care_yaml,
    summarize_policy_run,
)
from src.utils.manifest import build_manifest, write_manifest
from src.utils.research_artifacts import git_commit_or_unknown, utc_timestamp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", default="amazon_beauty")
    p.add_argument("--split", default="valid", choices=("valid", "test"), help="Pilot split to align with DeepSeek (20 users).")
    p.add_argument("--reprocess_dir", default="outputs/reprocessed_processed_source")
    p.add_argument("--deepseek_root", default="outputs/pilots/deepseek_v4_flash_processed_20u_c19_seed42")
    p.add_argument("--processed_dir", default="data/processed/amazon_beauty")
    p.add_argument("--care_rerank_config", default="configs/methods/care_rerank_pilot.yaml")
    p.add_argument("--output_root", default="outputs/pilots/care_lora_qwen3_8b_beauty_20u_c19_seed42_debug")
    p.add_argument("--prompt_id", default="listwise_ranking_v1")
    p.add_argument("--topk", type=int, default=19)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _index_by_user(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r.get("user_id", "")): r for r in rows if r.get("user_id")}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    domain = str(args.domain)
    split = str(args.split)
    rep = Path(args.reprocess_dir) / domain
    pilot = Path(args.deepseek_root) / domain / split / "predictions"
    pred_path = pilot / "rank_predictions.jsonl"
    feat_path = pilot / "uncertainty_features.jsonl"
    cand_path = rep / f"{split}_candidates.jsonl"
    data_dir = Path(args.output_root) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    preds = read_jsonl(pred_path)
    feats = read_jsonl(feat_path)
    cands = read_jsonl(cand_path)
    cand_by_user = _index_by_user(cands)
    feat_by_user = _index_by_user(feats)

    if len(preds) != 20:
        print(f"WARN: expected 20 rank_predictions rows, got {len(preds)}", file=sys.stderr)

    care_cfg = load_care_yaml(args.care_rerank_config)
    item_lookup = _load_item_lookup(Path(args.processed_dir))

    source_paths = {
        "candidates_jsonl": str(cand_path.resolve()),
        "rank_predictions_jsonl": str(pred_path.resolve()),
        "uncertainty_features_jsonl": str(feat_path.resolve()),
        "care_rerank_config": str(Path(args.care_rerank_config).resolve()),
    }

    aligned_rows: list[dict[str, Any]] = []
    pred_list: list[dict[str, Any]] = []
    feat_list: list[dict[str, Any] | None] = []
    prompts: list[str] = []

    for pred in preds:
        uid = str(pred.get("user_id", ""))
        row = cand_by_user.get(uid)
        if not row:
            raise ValueError(f"Missing candidate row for user_id={uid}")
        row = dict(row)
        row["_split_source"] = split
        pr = sanitize_listwise_ranking(dict(pred))
        pr["user_id"] = uid
        aligned_rows.append(row)
        pred_list.append(pr)
        feat_list.append(feat_by_user.get(uid))
        prompts.append(_build_ranking_prompt(row, args.prompt_id, item_lookup, int(args.topk)))

    outcomes_per: dict[str, list] = {pol: [] for pol in POLICIES}
    weighted_records: list[dict[str, Any]] = []
    pruned_records: list[dict[str, Any]] = []
    all_ev = [evaluate_policies(row, pr, ft, care_yaml=care_cfg) for row, pr, ft in zip(aligned_rows, pred_list, feat_list)]

    for row, pr, ft, prompt, ev in zip(aligned_rows, pred_list, feat_list, prompts, all_ev):
        for pol in POLICIES:
            outcomes_per[pol].append(ev[pol])
        wrow = {
            "user_id": row.get("user_id"),
            "policies": {k: {"keep": v.keep, "weight": v.sample_weight, "reason": v.reason} for k, v in ev.items()},
        }
        weighted_records.append(wrow)
        for pol, o in ev.items():
            if pol == "prune_high_uncertainty" and not o.keep:
                pruned_records.append(
                    {
                        "user_id": row.get("user_id"),
                        "policy": pol,
                        "reason": o.reason,
                        "care_risk_features": o.care_risk_features,
                    }
                )
        cf = ev["CARE_full_training"]
        if not cf.keep:
            pruned_records.append(
                {
                    "user_id": row.get("user_id"),
                    "policy": "CARE_full_training",
                    "reason": cf.reason,
                    "care_risk_features": cf.care_risk_features,
                }
            )

    vanilla_samples = [
        build_training_sample(
            row=row,
            prompt=prompt,
            pred=pr,
            feat=ft,
            policy="vanilla_lora_baseline",
            outcome=ev["vanilla_lora_baseline"],
            source_paths=source_paths,
        )
        for row, pr, ft, prompt, ev in zip(aligned_rows, pred_list, feat_list, prompts, all_ev)
    ]

    care_full_samples = []
    for row, pr, ft, prompt, ev in zip(aligned_rows, pred_list, feat_list, prompts, all_ev):
        o = ev["CARE_full_training"]
        if not o.keep:
            continue
        care_full_samples.append(
            build_training_sample(
                row=row,
                prompt=prompt,
                pred=pr,
                feat=ft,
                policy="CARE_full_training",
                outcome=o,
                source_paths=source_paths,
            )
        )

    write_jsonl(vanilla_samples, str(data_dir / "vanilla_lora_baseline_train.jsonl"))
    write_jsonl(care_full_samples, str(data_dir / "care_full_train.jsonl"))
    write_jsonl(pruned_records, str(data_dir / "pruned_samples.jsonl"))
    write_jsonl(weighted_records, str(data_dir / "weighted_samples.jsonl"))

    summary = summarize_policy_run(aligned_rows, outcomes_per)
    summary["policies"] = list(POLICIES)
    summary["domain"] = domain
    summary["split_source"] = split
    summary["seed"] = int(args.seed)
    summary["created_at"] = utc_timestamp()
    summary["git_commit"] = git_commit_or_unknown(".")
    (data_dir / "policy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    bucket_csv = data_dir / "bucket_distribution_before_after.csv"
    with bucket_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "policy", "head_share", "mid_share", "tail_share", "unknown_share"])
        b0 = summary["bucket_before"]
        w.writerow(["before", "all", b0["head"], b0["mid"], b0["tail"], b0["unknown"]])
        for pol, b1 in summary["bucket_after"].items():
            w.writerow(["after_kept", pol, b1["head"], b1["mid"], b1["tail"], b1["unknown"]])

    pilot_cfg = {
        "run_type": "pilot",
        "seed": int(args.seed),
        "method": "care_lora_data_build",
        "domain": domain,
        "split": split,
        "output_root": str(Path(args.output_root).resolve()),
    }
    write_manifest(
        data_dir / "data_manifest.json",
        build_manifest(
            config=pilot_cfg,
            dataset=domain,
            domain=domain,
            raw_data_paths=[],
            processed_data_paths=[str(cand_path.resolve()), str(pred_path.resolve()), str(feat_path.resolve())],
            method="care_lora_data_build",
            backend="lora",
            model="care-lora-data",
            prompt_template=args.prompt_id,
            seed=int(args.seed),
            candidate_size=len(aligned_rows[0]["candidate_item_ids"]) if aligned_rows else 19,
            calibration_source="diagnostic_bundle",
            command=sys.argv,
            mock_data_used=False,
        ),
    )
    print(f"[build_care_lora_data] wrote {data_dir} ({len(vanilla_samples)} vanilla, {len(care_full_samples)} care_full)")


if __name__ == "__main__":
    main(None)
