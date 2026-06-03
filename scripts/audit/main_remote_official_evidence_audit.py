from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_HELPER_PATH = REPO_ROOT / "scripts" / "audit" / "main_audit_official_evidence_package.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local official evidence-package audit helper on a remote "
            "evidence directory via SSH stdin. Use this when the server checkout "
            "is behind the local/GitHub commit but the completed row needs a "
            "server_final audit."
        )
    )
    parser.add_argument("--remote_host", default="pony-rec-gpu")
    parser.add_argument("--remote_python", default="/home/ajifang/miniconda3/bin/python")
    parser.add_argument("--remote_project", default="/home/ajifang/projects/pony-rec-rescue-shadow-v6")
    parser.add_argument(
        "--remote_evidence_dir",
        required=True,
        help="Remote evidence dir, absolute or relative to --remote_project.",
    )
    parser.add_argument("--mode", choices=["local_light", "server_final"], default="server_final")
    parser.add_argument("--expected_users", type=int, default=10000)
    parser.add_argument("--expected_candidates_per_user", type=int, default=101)
    parser.add_argument(
        "--output_json",
        default="",
        help="Remote JSON output path. Defaults to <remote_evidence_dir>/<mode>_evidence_audit.json.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def remote_evidence_path(remote_project: str, remote_evidence_dir: str) -> str:
    evidence = str(remote_evidence_dir).strip()
    if not evidence:
        raise ValueError("--remote_evidence_dir must not be empty")
    if evidence.startswith("/") or evidence.startswith("~"):
        return evidence
    return f"{remote_project.rstrip('/')}/{evidence}"


def default_remote_output_json(evidence_dir: str, mode: str) -> str:
    filename = "server_final_evidence_audit.json" if mode == "server_final" else "local_light_evidence_audit.json"
    return f"{evidence_dir.rstrip('/')}/{filename}"


def build_remote_command(args: argparse.Namespace) -> list[str]:
    evidence_dir = remote_evidence_path(args.remote_project, args.remote_evidence_dir)
    output_json = str(args.output_json).strip() or default_remote_output_json(evidence_dir, args.mode)
    cmd = [
        "ssh",
        args.remote_host,
        args.remote_python,
        "-",
        "--evidence_dir",
        evidence_dir,
        "--mode",
        args.mode,
        "--expected_users",
        str(args.expected_users),
        "--expected_candidates_per_user",
        str(args.expected_candidates_per_user),
        "--output_json",
        output_json,
    ]
    if args.quiet:
        cmd.append("--quiet")
    return cmd


def run_remote_audit(args: argparse.Namespace, *, helper_path: Path = LOCAL_HELPER_PATH) -> subprocess.CompletedProcess[str]:
    if not helper_path.exists():
        raise FileNotFoundError(helper_path)
    helper_source = helper_path.read_text(encoding="utf-8")
    return subprocess.run(
        build_remote_command(args),
        input=helper_source,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def main() -> int:
    args = parse_args()
    result = run_remote_audit(args)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
