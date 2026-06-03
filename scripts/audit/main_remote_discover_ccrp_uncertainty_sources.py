from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_AUDIT_HELPER_PATH = REPO_ROOT / "scripts" / "audit" / "main_audit_ccrp_uncertainty_sources.py"
LOCAL_DISCOVERY_HELPER_PATH = REPO_ROOT / "scripts" / "audit" / "main_discover_ccrp_uncertainty_sources.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local C-CRP uncertainty-source discovery helper on the remote "
            "server via SSH stdin. Use this when the server checkout is behind the "
            "local/GitHub commit but paper-critical source discovery must proceed."
        )
    )
    parser.add_argument("--remote_host", default="pony-rec-gpu")
    parser.add_argument("--remote_python", default="/home/ajifang/miniconda3/bin/python")
    parser.add_argument("--remote_project", default="/home/ajifang/projects/pony-rec-rescue-shadow-v6")
    parser.add_argument("--root", action="append", default=[], help="Remote root to scan; relative paths use --remote_project.")
    parser.add_argument("--domain", action="append", default=[], help="Domain token to require in path.")
    parser.add_argument("--name_token", action="append", default=[], help="Filename/path token filter.")
    parser.add_argument("--candidate_items_path", default="", help="Remote candidate_items.csv for --full_audit.")
    parser.add_argument("--expected_events", type=int, default=10000)
    parser.add_argument("--expected_candidates_per_event", type=int, default=101)
    parser.add_argument("--max_file_mb", type=float, default=600.0)
    parser.add_argument("--full_audit", action="store_true")
    parser.add_argument("--output_json", default="", help="Remote JSON output path.")
    parser.add_argument("--output_csv", default="", help="Remote CSV output path.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def remote_project_path(remote_project: str, remote_path: str) -> str:
    value = str(remote_path).strip()
    if not value:
        return ""
    if value.startswith("/") or value.startswith("~"):
        return value
    return f"{remote_project.rstrip('/')}/{value}"


def build_remote_command(args: argparse.Namespace) -> list[str]:
    cmd = ["ssh", args.remote_host, args.remote_python, "-"]
    for root in args.root or ["outputs"]:
        cmd.extend(["--root", remote_project_path(args.remote_project, root)])
    for domain in args.domain:
        cmd.extend(["--domain", str(domain)])
    for token in args.name_token:
        cmd.extend(["--name_token", str(token)])
    if args.candidate_items_path:
        cmd.extend(["--candidate_items_path", remote_project_path(args.remote_project, args.candidate_items_path)])
    cmd.extend(["--expected_events", str(args.expected_events)])
    cmd.extend(["--expected_candidates_per_event", str(args.expected_candidates_per_event)])
    cmd.extend(["--max_file_mb", str(args.max_file_mb)])
    if args.full_audit:
        cmd.append("--full_audit")
    if args.output_json:
        cmd.extend(["--output_json", remote_project_path(args.remote_project, args.output_json)])
    if args.output_csv:
        cmd.extend(["--output_csv", remote_project_path(args.remote_project, args.output_csv)])
    if args.quiet:
        cmd.append("--quiet")
    return cmd


def build_bootstrap_source(*, audit_source: str, discovery_source: str) -> str:
    return "\n".join(
        [
            "from __future__ import annotations",
            "import sys",
            "import types",
            "",
            "audit_source = " + repr(audit_source),
            "discovery_source = " + repr(discovery_source),
            "",
            "scripts_pkg = types.ModuleType('scripts')",
            "scripts_pkg.__path__ = []",
            "sys.modules.setdefault('scripts', scripts_pkg)",
            "audit_pkg = types.ModuleType('scripts.audit')",
            "audit_pkg.__path__ = []",
            "sys.modules.setdefault('scripts.audit', audit_pkg)",
            "audit_mod = types.ModuleType('scripts.audit.main_audit_ccrp_uncertainty_sources')",
            "audit_mod.__file__ = 'remote_stdin/main_audit_ccrp_uncertainty_sources.py'",
            "exec(compile(audit_source, audit_mod.__file__, 'exec'), audit_mod.__dict__)",
            "sys.modules['scripts.audit.main_audit_ccrp_uncertainty_sources'] = audit_mod",
            "setattr(audit_pkg, 'main_audit_ccrp_uncertainty_sources', audit_mod)",
            "discover_globals = {'__name__': '__main__', '__file__': 'remote_stdin/main_discover_ccrp_uncertainty_sources.py'}",
            "exec(compile(discovery_source, discover_globals['__file__'], 'exec'), discover_globals)",
            "",
        ]
    )


def run_remote_discovery(
    args: argparse.Namespace,
    *,
    audit_helper_path: Path = LOCAL_AUDIT_HELPER_PATH,
    discovery_helper_path: Path = LOCAL_DISCOVERY_HELPER_PATH,
) -> subprocess.CompletedProcess[str]:
    if not audit_helper_path.exists():
        raise FileNotFoundError(audit_helper_path)
    if not discovery_helper_path.exists():
        raise FileNotFoundError(discovery_helper_path)
    bootstrap = build_bootstrap_source(
        audit_source=audit_helper_path.read_text(encoding="utf-8"),
        discovery_source=discovery_helper_path.read_text(encoding="utf-8"),
    )
    return subprocess.run(
        build_remote_command(args),
        input=bootstrap,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def main() -> int:
    args = parse_args()
    result = run_remote_discovery(args)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
