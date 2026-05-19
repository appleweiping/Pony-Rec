#!/usr/bin/env python
"""Canonical bootstrap for server-side project work."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> int:
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    readiness = run([sys.executable, "main_project_readiness_check.py"])
    if readiness != 0:
        return readiness
    print("bootstrap_ok=true")
    print("next_read=docs/server_runbook.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

