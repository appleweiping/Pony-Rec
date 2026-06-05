import argparse
import subprocess

from scripts.audit.main_remote_server_large_artifact_manifest import (
    build_remote_command,
    remote_evidence_path,
    run_remote_manifest,
)


def test_remote_evidence_path_resolves_relative_to_project():
    assert remote_evidence_path("/repo", "outputs/run") == "/repo/outputs/run"
    assert remote_evidence_path("/repo/", "outputs/run") == "/repo/outputs/run"
    assert remote_evidence_path("/repo", "/abs/run") == "/abs/run"
    assert remote_evidence_path("/repo", "~/run") == "~/run"


def test_build_remote_command_passes_helper_args():
    args = argparse.Namespace(
        remote_host="pony-rec-gpu",
        remote_python="/python",
        remote_project="/repo",
        remote_evidence_dir="outputs/run",
        include_suffix=["faiss"],
        require_model_artifact=False,
        allow_certified_missing_prediction_jsonl=True,
        expected_prediction_lines=5000,
        quiet=True,
    )

    assert build_remote_command(args) == [
        "ssh",
        "pony-rec-gpu",
        "/python",
        "-",
        "--evidence_dir",
        "/repo/outputs/run",
        "--include_suffix",
        "faiss",
        "--no-require_model_artifact",
        "--allow_certified_missing_prediction_jsonl",
        "--expected_prediction_lines",
        "5000",
        "--quiet",
    ]


def test_run_remote_manifest_sends_local_helper_source(monkeypatch, tmp_path):
    helper = tmp_path / "helper.py"
    helper.write_text("print('helper')\n", encoding="utf-8")
    captured = {}

    def fake_run(cmd, *, input, text, stdout, stderr):
        captured["cmd"] = cmd
        captured["input"] = input
        captured["text"] = text
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = argparse.Namespace(
        remote_host="host",
        remote_python="/python",
        remote_project="/repo",
        remote_evidence_dir="outputs/run",
        include_suffix=[],
        require_model_artifact=True,
        allow_certified_missing_prediction_jsonl=False,
        expected_prediction_lines=10000,
        quiet=False,
    )

    result = run_remote_manifest(args, helper_path=helper)

    assert result.returncode == 0
    assert captured["cmd"] == [
        "ssh",
        "host",
        "/python",
        "-",
        "--evidence_dir",
        "/repo/outputs/run",
        "--require_model_artifact",
        "--expected_prediction_lines",
        "10000",
    ]
    assert captured["input"] == "print('helper')\n"
    assert captured["text"] is True
    assert captured["stdout"] == subprocess.PIPE
    assert captured["stderr"] == subprocess.PIPE
