from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_HELPER_PATH = REPO_ROOT / "scripts" / "audit" / "main_build_server_large_artifact_manifest.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local server-large-artifact manifest helper on a remote "
            "official-baseline evidence directory via SSH stdin. This is useful "
            "when the active server checkout is behind the local/GitHub commit."
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
    parser.add_argument(
        "--include_suffix",
        action="append",
        default=[],
        help="Extra suffix to include as a server-only large artifact, for example .faiss.",
    )
    parser.add_argument(
        "--require_model_artifact",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--allow_certified_missing_prediction_jsonl", action="store_true")
    parser.add_argument("--expected_prediction_lines", type=int, default=10000)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def remote_evidence_path(remote_project: str, remote_evidence_dir: str) -> str:
    evidence = str(remote_evidence_dir).strip()
    if not evidence:
        raise ValueError("--remote_evidence_dir must not be empty")
    if evidence.startswith("/") or evidence.startswith("~"):
        return evidence
    return f"{remote_project.rstrip('/')}/{evidence}"


def build_remote_command(args: argparse.Namespace) -> list[str]:
    evidence_dir = remote_evidence_path(args.remote_project, args.remote_evidence_dir)
    cmd = [
        "ssh",
        args.remote_host,
        args.remote_python,
        "-",
        "--evidence_dir",
        evidence_dir,
    ]
    for suffix in args.include_suffix:
        cmd.extend(["--include_suffix", str(suffix)])
    if args.require_model_artifact:
        cmd.append("--require_model_artifact")
    else:
        cmd.append("--no-require_model_artifact")
    if args.allow_certified_missing_prediction_jsonl:
        cmd.append("--allow_certified_missing_prediction_jsonl")
    cmd.extend(["--expected_prediction_lines", str(int(args.expected_prediction_lines))])
    if args.quiet:
        cmd.append("--quiet")
    return cmd


def run_remote_manifest(args: argparse.Namespace, *, helper_path: Path = LOCAL_HELPER_PATH) -> subprocess.CompletedProcess[str]:
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
    result = run_remote_manifest(args)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
