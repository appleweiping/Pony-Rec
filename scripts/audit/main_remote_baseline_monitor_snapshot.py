from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = Path(__file__).resolve()
ERROR_PATTERNS = (
    "traceback",
    "exception",
    "error",
    "oom",
    "out of memory",
    "no space",
    "errno 28",
    "killed",
    "failed",
)


def _remote_path(remote_project: str, path: str) -> str:
    value = str(path).strip()
    if not value:
        raise ValueError("path must not be empty")
    if value.startswith("/") or value.startswith("~"):
        return value
    return f"{remote_project.rstrip('/')}/{value}"


def _tail_lines(path: Path, count: int) -> list[str]:
    if count <= 0 or not path.exists():
        return []
    window: deque[str] = deque(maxlen=count)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            window.append(line.rstrip("\n"))
    return list(window)


def analyze_log_lines(lines: list[str]) -> dict[str, Any]:
    progress: dict[str, Any] = {}
    progress_re = re.compile(r"encoded\s+(\d+)/(\d+)")
    for line in lines:
        match = progress_re.search(line)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            progress = {
                "current": current,
                "total": total,
                "fraction": current / total if total else None,
                "line": line,
            }
    lowered = [line.lower() for line in lines]
    error_hits = [
        line
        for line, low in zip(lines, lowered)
        if any(pattern in low for pattern in ERROR_PATTERNS)
    ]
    completion_lines = [
        line
        for line in lines
        if "DONE " in line
        or "=== All baseline runs complete ===" in line
        or "implementation_status=official_completed" in line
    ]
    return {
        "progress": progress,
        "completion_detected": bool(completion_lines),
        "completion_lines": completion_lines[-20:],
        "failure_detected": bool(error_hits),
        "error_hits": error_hits[-20:],
    }


def _run_command(parts: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        parts,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "cmd": parts,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _process_snapshot(pids: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pid in pids:
        result = _run_command(["ps", "-p", str(pid), "-o", "pid=,ppid=,stat=,etime=,%cpu=,%mem=,cmd="])
        rows.append(
            {
                "pid": pid,
                "alive": result["returncode"] == 0 and bool(result["stdout"]),
                "ps": result["stdout"],
                "stderr": result["stderr"],
            }
        )
    return rows


def _matching_python_processes(tokens: list[str]) -> list[str]:
    result = _run_command(["ps", "aux"])
    if result["returncode"] != 0:
        return []
    lowered_tokens = [token.lower() for token in tokens if token]
    matches = []
    for line in result["stdout"].splitlines():
        low = line.lower()
        if "--_remote_helper" in low or "main_remote_baseline_monitor_snapshot.py" in low:
            continue
        if "python" in low and all(token in low for token in lowered_tokens):
            matches.append(line)
    return matches


def _disk_snapshot(path: str, *, free_danger_gb: float, used_danger_pct: float) -> dict[str, Any]:
    result = _run_command(["df", "-Pk", path])
    if result["returncode"] != 0:
        return {"path": path, "ok": False, "error": result["stderr"] or result["stdout"]}
    lines = result["stdout"].splitlines()
    if len(lines) < 2:
        return {"path": path, "ok": False, "error": "df output missing data row", "raw": result["stdout"]}
    parts = lines[-1].split()
    available_kb = int(parts[3])
    used_pct = float(parts[4].rstrip("%"))
    free_gb = available_kb * 1024 / (1024**3)
    return {
        "path": path,
        "ok": True,
        "free_gb": free_gb,
        "used_pct": used_pct,
        "danger": free_gb < free_danger_gb or used_pct >= used_danger_pct,
        "raw": result["stdout"],
    }


def _gpu_snapshot() -> dict[str, Any]:
    result = _run_command(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"]
    )
    return {
        "ok": result["returncode"] == 0,
        "raw": result["stdout"],
        "stderr": result["stderr"],
    }


def _size_snapshot(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        result = _run_command(["du", "-sh", path])
        rows.append({"path": path, "ok": result["returncode"] == 0, "raw": result["stdout"], "stderr": result["stderr"]})
    return rows


def build_remote_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    project = str(Path(args.remote_project).expanduser())
    log_path = Path(_remote_path(project, args.log_path))
    tail = _tail_lines(log_path, args.tail_lines)
    error_window = _tail_lines(log_path, args.error_window_lines)
    log_analysis = analyze_log_lines(error_window)
    disk = _disk_snapshot(args.disk_path, free_danger_gb=args.disk_free_danger_gb, used_danger_pct=args.disk_used_danger_pct)
    processes = _process_snapshot([int(pid) for pid in args.pid])
    matching = _matching_python_processes(args.process_token)
    completion = log_analysis["completion_detected"]
    failure = log_analysis["failure_detected"]
    all_requested_dead = bool(processes) and all(not row["alive"] for row in processes)
    notify_reasons: list[str] = []
    if completion:
        notify_reasons.append("completion")
    if failure:
        notify_reasons.append("failure")
    if disk.get("danger"):
        notify_reasons.append("disk_danger")
    if args.process_token and len(matching) > args.expected_matching_python_processes:
        notify_reasons.append("duplicate_run_risk")
    if all_requested_dead and not completion:
        notify_reasons.append("tracked_pids_dead_without_completion")
    return {
        "ok": True,
        "remote_project": project,
        "log_path": str(log_path),
        "tail_lines": tail,
        "log_analysis": log_analysis,
        "processes": processes,
        "matching_python_processes": matching,
        "disk": disk,
        "gpu": _gpu_snapshot(),
        "sizes": _size_snapshot([_remote_path(project, path) for path in args.size_path]),
        "should_notify": bool(notify_reasons),
        "notify_reasons": notify_reasons,
    }


def build_remote_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        "ssh",
        args.remote_host,
        args.remote_python,
        "-",
        "--_remote_helper",
        "--remote_project",
        args.remote_project,
        "--log_path",
        args.log_path,
        "--tail_lines",
        str(args.tail_lines),
        "--error_window_lines",
        str(args.error_window_lines),
        "--disk_path",
        args.disk_path,
        "--disk_free_danger_gb",
        str(args.disk_free_danger_gb),
        "--disk_used_danger_pct",
        str(args.disk_used_danger_pct),
        "--expected_matching_python_processes",
        str(args.expected_matching_python_processes),
    ]
    for pid in args.pid:
        cmd.extend(["--pid", str(pid)])
    for token in args.process_token:
        cmd.extend(["--process_token", token])
    for path in args.size_path:
        cmd.extend(["--size_path", path])
    return cmd


def run_remote_monitor(args: argparse.Namespace, *, helper_path: Path = HELPER_PATH) -> subprocess.CompletedProcess[str]:
    helper_source = helper_path.read_text(encoding="utf-8")
    return subprocess.run(
        build_remote_command(args),
        input=helper_source,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def should_write_output_json(stdout: str, *, on_notify_only: bool) -> bool:
    if not on_notify_only:
        return True
    try:
        snapshot = json.loads(stdout)
    except json.JSONDecodeError:
        return True
    return bool(snapshot.get("should_notify"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a robust remote baseline monitor snapshot via SSH stdin.")
    parser.add_argument("--_remote_helper", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--remote_host", default="pony-rec-gpu")
    parser.add_argument("--remote_python", default="/home/ajifang/miniconda3/bin/python")
    parser.add_argument("--remote_project", default="/home/ajifang/projects/pony-rec-rescue-shadow-v6")
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--pid", action="append", default=[])
    parser.add_argument("--process_token", action="append", default=[])
    parser.add_argument("--size_path", action="append", default=[])
    parser.add_argument("--tail_lines", type=int, default=80)
    parser.add_argument("--error_window_lines", type=int, default=5000)
    parser.add_argument("--disk_path", default="/")
    parser.add_argument("--disk_free_danger_gb", type=float, default=10.0)
    parser.add_argument("--disk_used_danger_pct", type=float, default=97.0)
    parser.add_argument("--expected_matching_python_processes", type=int, default=1)
    parser.add_argument("--output_json", default="")
    parser.add_argument(
        "--output_json_on_notify_only",
        action="store_true",
        help="When set, write --output_json only if the snapshot asks for notification.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args._remote_helper:
        snapshot = build_remote_snapshot(args)
        print(json.dumps(snapshot, indent=2, ensure_ascii=False))
        return 0
    result = run_remote_monitor(args)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if (
        result.returncode == 0
        and result.stdout
        and args.output_json
        and should_write_output_json(result.stdout, on_notify_only=args.output_json_on_notify_only)
    ):
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.stdout, encoding="utf-8", newline="\n")
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
