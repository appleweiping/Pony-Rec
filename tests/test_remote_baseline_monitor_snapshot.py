import argparse
import subprocess

from scripts.audit.main_remote_baseline_monitor_snapshot import (
    _matching_python_processes,
    _remote_path,
    analyze_log_lines,
    build_remote_command,
    run_remote_monitor,
)


def test_remote_path_resolves_relative_to_project():
    assert _remote_path("/repo", "run.log") == "/repo/run.log"
    assert _remote_path("/repo/", "outputs/run") == "/repo/outputs/run"
    assert _remote_path("/repo", "/abs/run.log") == "/abs/run.log"
    assert _remote_path("/repo", "~/run.log") == "~/run.log"


def test_remote_path_rejects_empty_path():
    try:
        _remote_path("/repo", " ")
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("empty remote path should fail")


def test_analyze_log_lines_detects_latest_progress_completion_and_errors():
    lines = [
        "[hf_mean_pool] encoded 10/100",
        "[hf_mean_pool] encoded 75/100",
        "Traceback (most recent call last):",
        "implementation_status=official_completed",
    ]

    analysis = analyze_log_lines(lines)

    assert analysis["progress"]["current"] == 75
    assert analysis["progress"]["total"] == 100
    assert analysis["progress"]["fraction"] == 0.75
    assert analysis["completion_detected"] is True
    assert analysis["failure_detected"] is True
    assert analysis["error_hits"] == ["Traceback (most recent call last):"]


def test_analyze_log_lines_ignores_clean_active_progress():
    analysis = analyze_log_lines(["[hf_mean_pool] encoded 138592/568891"])

    assert analysis["progress"]["current"] == 138592
    assert analysis["completion_detected"] is False
    assert analysis["failure_detected"] is False


def test_build_remote_command_passes_all_monitor_args():
    args = argparse.Namespace(
        remote_host="pony-rec-gpu",
        remote_python="/python",
        remote_project="/repo",
        log_path="baseline.log",
        tail_lines=40,
        error_window_lines=800,
        disk_path="/",
        disk_free_danger_gb=10.0,
        disk_used_danger_pct=97.0,
        expected_matching_python_processes=1,
        pid=[123, 456],
        process_token=["llm2rec", "home"],
        size_path=["outputs/adapter", "/abs/final"],
    )

    assert build_remote_command(args) == [
        "ssh",
        "pony-rec-gpu",
        "/python",
        "-",
        "--_remote_helper",
        "--remote_project",
        "/repo",
        "--log_path",
        "baseline.log",
        "--tail_lines",
        "40",
        "--error_window_lines",
        "800",
        "--disk_path",
        "/",
        "--disk_free_danger_gb",
        "10.0",
        "--disk_used_danger_pct",
        "97.0",
        "--expected_matching_python_processes",
        "1",
        "--pid",
        "123",
        "--pid",
        "456",
        "--process_token",
        "llm2rec",
        "--process_token",
        "home",
        "--size_path",
        "outputs/adapter",
        "--size_path",
        "/abs/final",
    ]


def test_run_remote_monitor_sends_local_helper_source(monkeypatch, tmp_path):
    helper = tmp_path / "helper.py"
    helper.write_text("print('helper')\n", encoding="utf-8")
    captured = {}

    def fake_run(cmd, *, input, text, encoding, errors, stdout, stderr):
        captured["cmd"] = cmd
        captured["input"] = input
        captured["text"] = text
        captured["encoding"] = encoding
        captured["errors"] = errors
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = argparse.Namespace(
        remote_host="host",
        remote_python="/python",
        remote_project="/repo",
        log_path="baseline.log",
        tail_lines=80,
        error_window_lines=5000,
        disk_path="/",
        disk_free_danger_gb=10.0,
        disk_used_danger_pct=97.0,
        expected_matching_python_processes=1,
        pid=[],
        process_token=[],
        size_path=[],
    )

    result = run_remote_monitor(args, helper_path=helper)

    assert result.returncode == 0
    assert captured["cmd"][:5] == ["ssh", "host", "/python", "-", "--_remote_helper"]
    assert captured["input"] == "print('helper')\n"
    assert captured["text"] is True
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"
    assert captured["stdout"] == subprocess.PIPE
    assert captured["stderr"] == subprocess.PIPE


def test_matching_python_processes_ignores_monitor_helper(monkeypatch):
    ps_output = "\n".join(
        [
            "u 100 0.0 python scripts/adapters/main_run_llm2rec.py --domain home",
            "u 101 0.0 python - --_remote_helper --process_token llm2rec --process_token home",
            "u 102 0.0 python scripts/audit/main_remote_baseline_monitor_snapshot.py llm2rec home",
        ]
    )

    def fake_run_command(parts):
        assert parts == ["ps", "aux"]
        return {"returncode": 0, "stdout": ps_output, "stderr": ""}

    monkeypatch.setattr(
        "scripts.audit.main_remote_baseline_monitor_snapshot._run_command",
        fake_run_command,
    )

    assert _matching_python_processes(["llm2rec", "home"]) == [
        "u 100 0.0 python scripts/adapters/main_run_llm2rec.py --domain home"
    ]
