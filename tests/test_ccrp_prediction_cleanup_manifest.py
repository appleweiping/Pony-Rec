from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_manifest(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base / "prediction_deletion_manifest.json").write_text(
        """
{
  "mode": "post_domain_gate_prediction_cleanup",
  "ok": true,
  "failures": [],
  "files": {
    "predictions/rank_predictions.jsonl": {
      "deleted": true,
      "lines": 10000,
      "size": 744081868,
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  }
}
""".strip(),
        encoding="utf-8",
    )


def test_domain_gate_accepts_ccrp_prediction_deletion_manifest(tmp_path: Path) -> None:
    module = _load_module(
        "scripts/audit/main_audit_domain_official_gate.py",
        "domain_gate_module",
    )
    _write_manifest(tmp_path)

    row = module._certified_deleted_prediction_row(tmp_path, "predictions/rank_predictions.jsonl")

    assert row["certified_missing"] is True
    assert row["lines"] == 10000
    assert row["certified_original_size"] == 744081868
    assert row["certified_sha256"].startswith("012345")


def test_comparison_builder_allows_cleanup_manifest_only_when_requested(tmp_path: Path) -> None:
    module = _load_module(
        "scripts/experiments/main_build_domain_official_comparison.py",
        "comparison_module",
    )
    _write_manifest(tmp_path)

    assert module._certified_prediction_line_count(tmp_path) == (None, "")
    assert module._certified_prediction_line_count(
        tmp_path,
        allow_deletion_manifest=True,
    ) == (10000, "prediction_deletion_manifest")
