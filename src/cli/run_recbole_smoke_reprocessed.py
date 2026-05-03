"""RecBole smoke baselines on the same 20-user cohort as reprocess_processed_source (pilot, not paper)."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.baselines.recbole_adapter import (
    copy_baseline_config_to_run,
    export_recbole_atomic_for_reprocess_users,
    run_recbole_baseline,
)
from src.utils.research_artifacts import config_hash, git_commit_or_unknown, utc_timestamp

DEFAULT_DOMAINS = ["amazon_beauty", "amazon_books", "amazon_electronics", "amazon_movies"]
BASELINE_YAMLS = {
    "Pop": "configs/baselines/pop.yaml",
    "BPR": "configs/baselines/bprmf.yaml",
    "LightGCN": "configs/baselines/lightgcn.yaml",
    "SASRec": "configs/baselines/sasrec.yaml",
    "BERT4Rec": "configs/baselines/bert4rec.yaml",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RecBole smoke on reprocess cohort (exports atomic + runs baselines).")
    p.add_argument("--reprocess_root", default="outputs/reprocessed_processed_source")
    p.add_argument("--processed_root", default="data/processed")
    p.add_argument("--output_root", default="outputs/pilots/recbole_smoke_processed_20u_seed42")
    p.add_argument("--domains", nargs="*", default=DEFAULT_DOMAINS)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _smoke_baseline_config(template: dict[str, Any], *, model_key: str, domain: str, seed: int) -> dict[str, Any]:
    cfg = deepcopy(template)
    cfg["dataset_name"] = domain
    cfg["seed"] = int(seed)
    recbole = dict(cfg.get("recbole") or {})
    recbole.setdefault("use_gpu", False)
    recbole.setdefault("MAX_ITEM_LIST_LENGTH", 50)
    if model_key == "Pop":
        cfg["epochs"] = 1
        cfg.setdefault("train_batch_size", 512)
        cfg.setdefault("eval_batch_size", 1024)
    else:
        cfg["epochs"] = min(int(cfg.get("epochs", 100)), 3)
        cfg["train_batch_size"] = min(int(cfg.get("train_batch_size", 2048)), 256)
        cfg["eval_batch_size"] = min(int(cfg.get("eval_batch_size", 4096)), 512)
        if model_key in {"BPR", "LightGCN"}:
            recbole.setdefault("embedding_size", 32)
        if model_key in {"SASRec", "BERT4Rec"}:
            recbole.setdefault("hidden_size", 64)
            recbole.setdefault("inner_size", 128)
            recbole.setdefault("n_layers", 1)
            recbole.setdefault("n_heads", 2)
            recbole.setdefault("dropout_prob", 0.1)
            recbole["train_neg_sample_args"] = None
    cfg["recbole"] = recbole
    return cfg


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    atomic_root = output_root / "atomic"
    runs_root = output_root / "runs"
    atomic_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "created_at": utc_timestamp(),
        "git_commit": git_commit_or_unknown("."),
        "run_type": "pilot",
        "backend_type": "baseline",
        "is_paper_result": False,
        "seed": args.seed,
        "reprocess_root": str(args.reprocess_root),
        "processed_root": str(args.processed_root),
        "domains": {},
    }

    for domain in args.domains:
        rep_dir = Path(args.reprocess_root) / domain
        proc_dir = Path(args.processed_root) / domain
        summary["domains"][domain] = {"export": None, "models": {}}
        export_meta = export_recbole_atomic_for_reprocess_users(
            reprocess_domain_dir=rep_dir,
            processed_dir=proc_dir,
            output_dir=atomic_root,
            dataset_name=domain,
        )
        summary["domains"][domain]["export"] = export_meta

        for model_key, yaml_rel in BASELINE_YAMLS.items():
            tmpl_path = Path(yaml_rel)
            if not tmpl_path.exists():
                summary["domains"][domain]["models"][model_key] = {"ok": False, "error": f"missing {yaml_rel}"}
                continue
            template = _load_yaml(tmpl_path)
            cfg = _smoke_baseline_config(template, model_key=model_key, domain=domain, seed=args.seed)
            run_dir = runs_root / domain / model_key.lower()
            run_dir.mkdir(parents=True, exist_ok=True)
            tmp_cfg = run_dir / "baseline_smoke.yaml"
            tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            copy_baseline_config_to_run(tmpl_path, run_dir)
            meta = {
                "run_type": "pilot",
                "backend_type": "baseline",
                "is_paper_result": False,
                "model": cfg.get("model"),
                "dataset_name": domain,
                "seed": args.seed,
                "config_hash": config_hash(cfg),
            }
            try:
                result = run_recbole_baseline(baseline_config=cfg, dataset_dir=atomic_root, output_dir=run_dir)
                out = {
                    "ok": True,
                    "meta": meta,
                    "config_path": result.get("config_path"),
                    "best_valid_score": None,
                    "test_result": None,
                }
                res = result.get("result")
                if isinstance(res, dict):
                    out["best_valid_score"] = res.get("best_valid_score")
                    out["best_valid_result"] = res.get("best_valid_result")
                    out["test_result"] = res.get("test_result")
                else:
                    out["raw_result_tail"] = str(res)[-500:]
            except Exception as exc:
                out = {"ok": False, "meta": meta, "error": repr(exc)}
            (run_dir / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            summary["domains"][domain]["models"][model_key] = out
            print(f"[recbole_smoke] domain={domain} model={model_key} ok={out.get('ok')}", file=sys.stderr)

    summary_path = output_root / "smoke_run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[recbole_smoke] wrote {summary_path}")


if __name__ == "__main__":
    main()
