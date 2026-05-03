"""Tests for CARE rerank pilot summarize / manifest audit CLI."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.cli.summarize_care_rerank_pilot import main, verify_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _good_manifest_paths(repo: Path) -> list[str]:
    rank = (
        repo
        / "outputs"
        / "pilots"
        / "deepseek_v4_flash_processed_20u_c19_seed42"
        / "amazon_beauty"
        / "test"
        / "predictions"
        / "rank_predictions.jsonl"
    )
    rank.parent.mkdir(parents=True, exist_ok=True)
    rank.write_text("", encoding="utf-8")
    return [str(rank)]


def test_verify_manifest_passes(tmp_path: Path) -> None:
    repo = tmp_path
    p = tmp_path / "care_manifest.json"
    body = {
        "run_type": "pilot",
        "backend_type": "rerank",
        "is_paper_result": False,
        "candidate_size": 19,
        "seed": 42,
        "method": "care_rerank_care_full",
        "processed_data_paths": _good_manifest_paths(repo),
    }
    p.write_text(json.dumps(body), encoding="utf-8")
    assert verify_manifest(p, repo_root=repo) == []


def test_verify_manifest_rejects_banned_substring(tmp_path: Path) -> None:
    repo = tmp_path
    p = tmp_path / "care_manifest.json"
    body = {
        "run_type": "pilot",
        "backend_type": "rerank",
        "is_paper_result": False,
        "candidate_size": 19,
        "seed": 42,
        "method": "care_rerank_care_full",
        "model": "x_srpd_y",
        "processed_data_paths": _good_manifest_paths(repo),
    }
    p.write_text(json.dumps(body), encoding="utf-8")
    errs = verify_manifest(p, repo_root=repo)
    assert errs and any("srpd" in e.lower() for e in errs)


def test_verify_manifest_rejects_wrong_run_type(tmp_path: Path) -> None:
    repo = tmp_path
    p = tmp_path / "care_manifest.json"
    body = {
        "run_type": "full",
        "backend_type": "rerank",
        "is_paper_result": False,
        "candidate_size": 19,
        "seed": 42,
        "method": "care_rerank_care_full",
        "processed_data_paths": _good_manifest_paths(repo),
    }
    p.write_text(json.dumps(body), encoding="utf-8")
    assert any("run_type" in e for e in verify_manifest(p, repo_root=repo))


def test_verify_manifest_rank_path_must_include_pilot_mark(tmp_path: Path) -> None:
    repo = tmp_path
    p = tmp_path / "care_manifest.json"
    bad_rank = repo / "other" / "rank_predictions.jsonl"
    bad_rank.parent.mkdir(parents=True, exist_ok=True)
    bad_rank.write_text("", encoding="utf-8")
    body = {
        "run_type": "pilot",
        "backend_type": "rerank",
        "is_paper_result": False,
        "candidate_size": 19,
        "seed": 42,
        "method": "care_rerank_care_full",
        "processed_data_paths": [str(bad_rank)],
    }
    p.write_text(json.dumps(body), encoding="utf-8")
    errs = verify_manifest(p, repo_root=repo)
    assert any("deepseek_v4_flash_processed_20u_c19_seed42" in e for e in errs)


def test_main_smoke_skip_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "out"
    root.mkdir()
    agg = root / "care_rerank_aggregate.csv"
    with agg.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "domain",
                "split",
                "rows",
                "HR@1",
                "HR@5",
                "HR@10",
                "Recall@1",
                "Recall@5",
                "Recall@10",
                "NDCG@1",
                "NDCG@5",
                "NDCG@10",
                "MRR@1",
                "MRR@5",
                "MRR@10",
                "confidence_available_rate",
                "invalid_output_rate",
                "high_confidence_wrong_rate_before",
                "high_confidence_wrong_rate_after",
                "head_prediction_rate_after",
                "tail_target_hit_at_1_rate_after",
                "high_risk_top1_changed_rate",
                "confidence_correctness_auc_diag",
            ]
        )
        w.writerow(
            [
                "care_full",
                "amazon_beauty",
                "test",
                "1",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
            ]
        )
    monkeypatch.chdir(tmp_path)
    code = main(["--output_root", str(root), "--skip_manifest_verify"])
    assert code == 0


def test_cli_module_invocation_zero_exit_skip_manifest(tmp_path: Path) -> None:
    root = tmp_path / "out2"
    root.mkdir()
    (root / "care_rerank_aggregate.csv").write_text(
        "variant,domain,split,rows,HR@1,HR@5,HR@10,Recall@1,Recall@5,Recall@10,"
        "NDCG@1,NDCG@5,NDCG@10,MRR@1,MRR@5,MRR@10,confidence_available_rate,"
        "invalid_output_rate,high_confidence_wrong_rate_before,high_confidence_wrong_rate_after,"
        "head_prediction_rate_after,tail_target_hit_at_1_rate_after,high_risk_top1_changed_rate,"
        "confidence_correctness_auc_diag\n"
        "original_deepseek,amazon_beauty,test,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli.summarize_care_rerank_pilot",
            "--output_root",
            str(root),
            "--skip_manifest_verify",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
