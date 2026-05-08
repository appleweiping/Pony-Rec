from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from src.utils.exp_io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit official external baseline repo checkouts and pinned commits.")
    parser.add_argument("--config", default="configs/official_external_baselines.yaml")
    parser.add_argument("--output_dir", default="outputs/summary/official_external_baseline_audit")
    return parser.parse_args()


def _run_git(repo_dir: Path, args: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True, result.stdout.strip()
    except Exception as exc:
        return False, str(exc)


def _repo_status(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    env_name = str(cfg.get("local_repo_env", "")).strip()
    repo_dir = Path(os.environ.get(env_name, str(cfg.get("local_repo_default", "")))).expanduser()
    pinned = str(cfg.get("pinned_commit", "")).strip()
    fairness_contract = cfg.get("fairness_contract", {}) or {}
    row: dict[str, Any] = {
        "method": name,
        "target_baseline_name": cfg.get("target_baseline_name", ""),
        "official_repo": cfg.get("official_repo", ""),
        "comparison_tier": fairness_contract.get("comparison_tier", ""),
        "backbone_family": fairness_contract.get("backbone_family", ""),
        "hparam_policy": fairness_contract.get("hparam_policy", ""),
        "extra_baseline_tuning_allowed": fairness_contract.get("extra_baseline_tuning_allowed", ""),
        "fairness_contract_present": bool(fairness_contract),
        "fairness_contract_ok": bool(
            fairness_contract
            and fairness_contract.get("backbone_family") == "Qwen3-8B"
            and fairness_contract.get("hparam_policy") == "official_default_or_recommended"
            and fairness_contract.get("extra_baseline_tuning_allowed") is False
        ),
        "local_repo_env": env_name,
        "local_repo_dir": str(repo_dir),
        "pinned_commit": pinned,
        "exists": repo_dir.exists(),
        "is_git_repo": False,
        "current_commit": "",
        "remote_origin": "",
        "commit_matches_pin": False,
        "status": "missing_repo",
        "notes": "",
    }
    if not repo_dir.exists():
        row["notes"] = f"Clone {row['official_repo']} into {repo_dir} or set {env_name}."
        return row
    ok, git_dir = _run_git(repo_dir, ["rev-parse", "--git-dir"])
    if not ok:
        row["status"] = "not_git_repo"
        row["notes"] = git_dir
        return row
    row["is_git_repo"] = True

    ok, commit = _run_git(repo_dir, ["rev-parse", "HEAD"])
    if ok:
        row["current_commit"] = commit
    ok, remote = _run_git(repo_dir, ["remote", "get-url", "origin"])
    if ok:
        row["remote_origin"] = remote

    row["commit_matches_pin"] = bool(pinned and row["current_commit"] == pinned)
    if row["commit_matches_pin"]:
        row["status"] = "pinned_ready"
    else:
        row["status"] = "commit_mismatch"
        row["notes"] = f"Run: cd {repo_dir} && git fetch origin && git checkout {pinned}"
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "target_baseline_name",
        "official_repo",
        "comparison_tier",
        "backbone_family",
        "hparam_policy",
        "extra_baseline_tuning_allowed",
        "fairness_contract_present",
        "fairness_contract_ok",
        "local_repo_env",
        "local_repo_dir",
        "pinned_commit",
        "exists",
        "is_git_repo",
        "current_commit",
        "remote_origin",
        "commit_matches_pin",
        "status",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    baselines = cfg.get("official_baselines", {}) or {}
    rows = [_repo_status(name, dict(method_cfg or {})) for name, method_cfg in baselines.items()]
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "repo_pin_audit.csv"
    json_path = output_dir / "repo_pin_audit.json"
    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ready = sum(1 for row in rows if row["status"] == "pinned_ready")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"pinned_ready={ready}/{len(rows)}")
    for row in rows:
        print(f"{row['method']}: {row['status']} {row['local_repo_dir']}")


if __name__ == "__main__":
    main()
