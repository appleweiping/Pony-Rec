import argparse
import subprocess

from scripts.audit.main_remote_discover_ccrp_uncertainty_sources import (
    build_bootstrap_source,
    build_remote_command,
    remote_project_path,
    run_remote_discovery,
)


def test_remote_project_path_resolves_relative_paths():
    assert remote_project_path("/repo", "outputs") == "/repo/outputs"
    assert remote_project_path("/repo/", "outputs/a.json") == "/repo/outputs/a.json"
    assert remote_project_path("/repo", "/abs/a.json") == "/abs/a.json"
    assert remote_project_path("/repo", "~/a.json") == "~/a.json"
    assert remote_project_path("/repo", "") == ""


def test_build_remote_discovery_command_defaults_to_outputs_root():
    args = argparse.Namespace(
        remote_host="pony-rec-gpu",
        remote_python="/python",
        remote_project="/repo",
        root=[],
        domain=["sports", "toys"],
        name_token=["signal"],
        candidate_items_path="outputs/tasks/candidate_items.csv",
        expected_events=10000,
        expected_candidates_per_event=101,
        max_file_mb=700.0,
        full_audit=True,
        output_json="outputs/summary/discovery.json",
        output_csv="outputs/summary/discovery.csv",
        quiet=True,
    )

    assert build_remote_command(args) == [
        "ssh",
        "pony-rec-gpu",
        "/python",
        "-",
        "--root",
        "/repo/outputs",
        "--domain",
        "sports",
        "--domain",
        "toys",
        "--name_token",
        "signal",
        "--candidate_items_path",
        "/repo/outputs/tasks/candidate_items.csv",
        "--expected_events",
        "10000",
        "--expected_candidates_per_event",
        "101",
        "--max_file_mb",
        "700.0",
        "--full_audit",
        "--output_json",
        "/repo/outputs/summary/discovery.json",
        "--output_csv",
        "/repo/outputs/summary/discovery.csv",
        "--quiet",
    ]


def test_build_bootstrap_source_injects_audit_module_before_discovery():
    bootstrap = build_bootstrap_source(
        audit_source="VALUE = 1\n",
        discovery_source="from scripts.audit.main_audit_ccrp_uncertainty_sources import VALUE\nprint(VALUE)\n",
    )

    assert "sys.modules['scripts.audit.main_audit_ccrp_uncertainty_sources'] = audit_mod" in bootstrap
    assert "remote_stdin/main_discover_ccrp_uncertainty_sources.py" in bootstrap
    assert "VALUE = 1" in bootstrap


def test_run_remote_discovery_sends_bootstrap(monkeypatch, tmp_path):
    audit = tmp_path / "audit.py"
    discovery = tmp_path / "discover.py"
    audit.write_text("VALUE = 1\n", encoding="utf-8")
    discovery.write_text("print('discover')\n", encoding="utf-8")
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
        root=["outputs"],
        domain=[],
        name_token=[],
        candidate_items_path="",
        expected_events=10,
        expected_candidates_per_event=101,
        max_file_mb=1.0,
        full_audit=False,
        output_json="",
        output_csv="",
        quiet=False,
    )

    result = run_remote_discovery(args, audit_helper_path=audit, discovery_helper_path=discovery)

    assert result.returncode == 0
    assert captured["cmd"] == [
        "ssh",
        "host",
        "/python",
        "-",
        "--root",
        "/repo/outputs",
        "--expected_events",
        "10",
        "--expected_candidates_per_event",
        "101",
        "--max_file_mb",
        "1.0",
    ]
    assert "VALUE = 1" in captured["input"]
    assert "print('discover')" in captured["input"]
    assert captured["text"] is True
    assert captured["stdout"] == subprocess.PIPE
    assert captured["stderr"] == subprocess.PIPE
