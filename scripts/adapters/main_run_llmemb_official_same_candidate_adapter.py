from __future__ import annotations

import sys

from main_run_official_same_candidate_adapter import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--method", "llmemb", *sys.argv[1:]]
    raise SystemExit(main())
