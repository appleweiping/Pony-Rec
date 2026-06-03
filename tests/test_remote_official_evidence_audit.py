import argparse
import subprocess

from scripts.audit.main_remote_official_evidence_audit import (
    build_remote_command,
    default_remote_output_json,
    remote_evidence_path,
    run_remote_audit,
)


def test_remote_evidence_audit_path_and_default_output():
    assert remote_evidence_path("/repo", "outputs/run") == "/repo/outputs/run"
    assert remote_evidence_path("/repo", "/abs/run") == "/abs/run"
    assert default_remote_output_json("/repo/outputs/run", "server_final") == (
        "/repo/outputs/run/server_final_evidence_audit.json"
    )
    assert default_remote_output_json("/repo/outputs/run", "local_light") == (
        "/repo/outputs/run/local_light_evidence_audit.json"
    )


def test_build_remote_evidence_audit_command_defaults_to_server_final_json():
    args = argparse.Namespace(
        remote_host="pony-rec-gpu",
        remote_python="/python",
        remote_project="/repo",
        remote_evidence_dir="outputs/run",
        mode="server_final",
        expected_users=10000,
        expected_candidates_per_user=101,
        output_json="",
        quiet=True,
    )

    assert build_remote_command(args) == [
        "ssh",
        "pony-rec-gpu",
        "/python",
        "-",
        "--evidence_dir",
        "/repo/outputs/run",
        "--mode",
        "server_final",
        "--expected_users",
        "10000",
        "--expected_candidates_per_user",
        "101",
        "--output_json",
        "/repo/outputs/run/server_final_evidence_audit.json",
        "--quiet",
    ]


def test_build_remote_evidence_audit_command_respects_explicit_output_json():
    args = argparse.Namespace(
        remote_host="host",
        remote_python="/python",
        remote_project="/repo",
        remote_evidence_dir="/abs/run",
        mode="local_light",
        expected_users=5,
        expected_candidates_per_user=7,
        output_json="/tmp/audit.json",
        quiet=False,
    )

    assert build_remote_command(args) == [
        "ssh",
        "host",
        "/python",
        "-",
        "--evidence_dir",
        "/abs/run",
        "--mode",
        "local_light",
        "--expected_users",
        "5",
        "--expected_candidates_per_user",
        "7",
        "--output_json",
        "/tmp/audit.json",
    ]


def test_run_remote_evidence_audit_sends_local_helper_source(monkeypatch, tmp_path):
    helper = tmp_path / "helper.py"
    helper.write_text("print('audit')\n", encoding="utf-8")
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
        mode="server_final",
        expected_users=10000,
        expected_candidates_per_user=101,
        output_json="",
        quiet=False,
    )

    result = run_remote_audit(args, helper_path=helper)

    assert result.returncode == 0
    assert captured["cmd"] == [
        "ssh",
        "host",
        "/python",
        "-",
        "--evidence_dir",
        "/repo/outputs/run",
        "--mode",
        "server_final",
        "--expected_users",
        "10000",
        "--expected_candidates_per_user",
        "101",
        "--output_json",
        "/repo/outputs/run/server_final_evidence_audit.json",
    ]
    assert captured["input"] == "print('audit')\n"
    assert captured["text"] is True
    assert captured["stdout"] == subprocess.PIPE
    assert captured["stderr"] == subprocess.PIPE
