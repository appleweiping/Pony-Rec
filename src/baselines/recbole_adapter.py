from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_INSTALL = "pip install recbole"


def ensure_recbole_installed() -> None:
    if importlib.util.find_spec("recbole") is None:
        raise ImportError(f"RecBole is not installed. Install it with `{REQUIRED_INSTALL}` before running this baseline.")


def export_recbole_atomic(
    *,
    processed_dir: str | Path,
    output_dir: str | Path,
    dataset_name: str,
) -> dict[str, str]:
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    atomic_dir = output_dir / dataset_name
    atomic_dir.mkdir(parents=True, exist_ok=True)
    interactions_path = processed_dir / "interactions.csv"
    if not interactions_path.exists():
        raise FileNotFoundError(f"Processed interactions not found: {interactions_path}")
    interactions = pd.read_csv(interactions_path)
    required = {"user_id", "item_id", "timestamp"}
    missing = sorted(required - set(interactions.columns))
    if missing:
        raise ValueError(f"Cannot export RecBole data; missing columns: {missing}")
    if "rating" not in interactions.columns:
        interactions["rating"] = 1.0
    atomic = interactions[["user_id", "item_id", "rating", "timestamp"]].copy()
    atomic.columns = ["user_id:token", "item_id:token", "rating:float", "timestamp:float"]
    inter_path = atomic_dir / f"{dataset_name}.inter"
    atomic.to_csv(inter_path, sep="\t", index=False)
    item_src = processed_dir / "items.csv"
    item_path = atomic_dir / f"{dataset_name}.item"
    if item_src.exists():
        items = pd.read_csv(item_src).fillna("")
        export_cols = ["item_id"]
        if "title" in items.columns:
            export_cols.append("title")
        if "categories" in items.columns:
            export_cols.append("categories")
        item_atomic = items[export_cols].copy()
        item_atomic.columns = [
            "item_id:token" if col == "item_id" else f"{col}:token_seq"
            for col in item_atomic.columns
        ]
        item_atomic.to_csv(item_path, sep="\t", index=False)
    return {"dataset_dir": str(output_dir), "atomic_dir": str(atomic_dir), "inter_file": str(inter_path), "item_file": str(item_path)}


def build_recbole_config(
    *,
    baseline_config: dict[str, Any],
    dataset_name: str,
    dataset_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = baseline_config["model"]
    recbole_cfg = {
        "model": model,
        "dataset": dataset_name,
        "data_path": str(dataset_dir),
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "RATING_FIELD": "rating",
        "TIME_FIELD": "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
        "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO", "mode": "full"},
        "metrics": ["Recall", "MRR", "NDCG", "Hit"],
        "topk": baseline_config.get("topk", [10]),
        "valid_metric": baseline_config.get("valid_metric", "NDCG@10"),
        "epochs": int(baseline_config.get("epochs", 50)),
        "train_batch_size": int(baseline_config.get("train_batch_size", 2048)),
        "eval_batch_size": int(baseline_config.get("eval_batch_size", 4096)),
        "seed": int(baseline_config.get("seed", 42)),
        **(baseline_config.get("recbole", {}) or {}),
    }
    path = output_dir / f"{model.lower()}_recbole.yaml"
    path.write_text(yaml.safe_dump(recbole_cfg, sort_keys=False), encoding="utf-8")
    return path


def run_recbole_baseline(
    *,
    baseline_config: dict[str, Any],
    dataset_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    ensure_recbole_installed()
    from recbole.quick_start import run_recbole

    dataset_name = str(baseline_config["dataset_name"])
    config_path = build_recbole_config(
        baseline_config=baseline_config,
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
    )
    result = run_recbole(model=str(baseline_config["model"]), dataset=dataset_name, config_file_list=[str(config_path)])
    return {"config_path": str(config_path), "result": result}


def copy_baseline_config_to_run(baseline_config_path: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(baseline_config_path, output_dir / Path(baseline_config_path).name)
