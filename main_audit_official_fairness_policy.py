from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any

from src.utils.exp_io import load_yaml


REQUIRED_SCORE_SCHEMA = ["source_event_id", "user_id", "item_id", "score"]
PRIMARY_VARIANT = "official_code_qwen3base_default_hparams_declared_adaptation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the official external-baseline fairness policy contract.")
    parser.add_argument("--config", default="configs/official_external_baselines.yaml")
    parser.add_argument(
        "--summary_glob",
        default="",
        help="Optional glob of same_candidate_external_baseline_summary.csv files to audit.",
    )
    parser.add_argument("--output_dir", default="outputs/summary/official_external_baseline_audit")
    return parser.parse_args()


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else str(value).strip().lower() == "true"


def _as_schema(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    return []


def _audit_config(cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policy = cfg.get("fairness_policy", {}) or {}
    top_row = {
        "policy_id": policy.get("policy_id", ""),
        "policy_version": policy.get("policy_version", ""),
        "primary_table_variant": policy.get("primary_table_variant", ""),
        "official_code_required": policy.get("official_code_required", ""),
        "unified_backbone_required": policy.get("unified_backbone_required", ""),
        "unified_backbone_family": policy.get("unified_backbone_family", ""),
        "baseline_hyperparameter_policy": policy.get("baseline_hyperparameter_policy", ""),
        "our_hyperparameter_policy": policy.get("our_hyperparameter_policy", ""),
        "baseline_extra_tuning_allowed_in_primary_table": policy.get(
            "baseline_extra_tuning_allowed_in_primary_table", ""
        ),
        "test_set_model_selection_allowed": policy.get("test_set_model_selection_allowed", ""),
        "import_full_catalog_metrics_allowed": policy.get("import_full_catalog_metrics_allowed", ""),
        "score_schema": ",".join(_as_schema(policy.get("score_schema"))),
    }
    top_failures: list[str] = []
    if not top_row["policy_id"]:
        top_failures.append("missing_policy_id")
    if top_row["primary_table_variant"] != PRIMARY_VARIANT:
        top_failures.append("primary_variant_mismatch")
    if not _as_bool(top_row["official_code_required"]):
        top_failures.append("official_code_not_required")
    if not _as_bool(top_row["unified_backbone_required"]):
        top_failures.append("unified_backbone_not_required")
    if top_row["unified_backbone_family"] != "Qwen3-8B":
        top_failures.append("backbone_family_not_qwen3_8b")
    if top_row["baseline_hyperparameter_policy"] != "official_default_or_recommended":
        top_failures.append("baseline_hparam_policy_not_default_or_recommended")
    if _as_bool(top_row["baseline_extra_tuning_allowed_in_primary_table"]):
        top_failures.append("baseline_extra_tuning_allowed_in_primary")
    if _as_bool(top_row["test_set_model_selection_allowed"]):
        top_failures.append("test_set_selection_allowed")
    if _as_bool(top_row["import_full_catalog_metrics_allowed"]):
        top_failures.append("full_catalog_import_allowed")
    if _as_schema(policy.get("score_schema")) != REQUIRED_SCORE_SCHEMA:
        top_failures.append("score_schema_mismatch")
    top_row["ok"] = not top_failures
    top_row["failures"] = ";".join(top_failures)

    method_rows: list[dict[str, Any]] = []
    for method, method_cfg in (cfg.get("official_baselines") or {}).items():
        contract = (method_cfg or {}).get("fairness_contract", {}) or {}
        row = {
            "method": method,
            "target_baseline_name": method_cfg.get("target_baseline_name", ""),
            "implementation_status": method_cfg.get("current_status", ""),
            "comparison_tier": contract.get("comparison_tier", ""),
            "official_code_required": contract.get("official_code_required", ""),
            "backbone_replacement_required": contract.get("backbone_replacement_required", ""),
            "backbone_family": contract.get("backbone_family", ""),
            "hparam_policy": contract.get("hparam_policy", ""),
            "extra_baseline_tuning_allowed": contract.get("extra_baseline_tuning_allowed", ""),
            "accepted_llm_adaptation_modes": ",".join(_as_schema(contract.get("accepted_llm_adaptation_modes"))),
        }
        failures: list[str] = []
        if not contract:
            failures.append("missing_fairness_contract")
        if row["comparison_tier"] != PRIMARY_VARIANT:
            failures.append("comparison_tier_mismatch")
        if not _as_bool(row["official_code_required"]):
            failures.append("official_code_not_required")
        if not _as_bool(row["backbone_replacement_required"]):
            failures.append("backbone_replacement_not_required")
        if row["backbone_family"] != "Qwen3-8B":
            failures.append("backbone_family_not_qwen3_8b")
        if row["hparam_policy"] != "official_default_or_recommended":
            failures.append("hparam_policy_mismatch")
        if _as_bool(row["extra_baseline_tuning_allowed"]):
            failures.append("baseline_extra_tuning_allowed")
        if not row["accepted_llm_adaptation_modes"]:
            failures.append("missing_accepted_llm_adaptation_modes")
        row["ok"] = not failures
        row["failures"] = ";".join(failures)
        method_rows.append(row)
    return top_row, method_rows


def _audit_summary_rows(summary_glob: str) -> list[dict[str, Any]]:
    if not summary_glob:
        return []
    rows: list[dict[str, Any]] = []
    for path_text in sorted(glob.glob(summary_glob, recursive=True)):
        path = Path(path_text)
        try:
            records = _read_csv(path)
        except Exception as exc:
            rows.append({"summary_path": str(path), "ok": False, "failures": f"read_error:{exc}"})
            continue
        for record in records:
            baseline_name = _text(record.get("baseline_name"))
            status_label = _text(record.get("status_label"))
            artifact_class = _text(record.get("artifact_class"))
            official_main = "_official_" in baseline_name and status_label == "same_schema_external_baseline"
            failures: list[str] = []
            if official_main and artifact_class != "completed_result":
                failures.append("official_main_not_completed_result")
            if official_main and _text(record.get("implementation_status")) != "official_completed":
                failures.append("official_main_not_official_completed")
            if official_main and _text(record.get("comparison_variant")) != PRIMARY_VARIANT:
                failures.append("official_main_variant_mismatch")
            if official_main and not _text(record.get("fairness_policy_id")):
                failures.append("missing_fairness_policy_id")
            if official_main and not _text(record.get("provenance_path")):
                failures.append("missing_provenance_path")
            try:
                coverage = float(record.get("score_coverage_rate", 0.0))
            except Exception:
                coverage = 0.0
            if official_main and coverage < 1.0:
                failures.append("score_coverage_below_1")
            if "beauty" in _text(record.get("domain")).lower() and official_main:
                failures.append("beauty_official_main_requires_supplementary_label")
            rows.append(
                {
                    "summary_path": str(path),
                    "baseline_name": baseline_name,
                    "domain": record.get("domain", ""),
                    "status_label": status_label,
                    "artifact_class": artifact_class,
                    "comparison_variant": record.get("comparison_variant", ""),
                    "implementation_status": record.get("implementation_status", ""),
                    "fairness_policy_id": record.get("fairness_policy_id", ""),
                    "score_coverage_rate": coverage,
                    "official_main": official_main,
                    "ok": not failures,
                    "failures": ";".join(failures),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    top_row, method_rows = _audit_config(cfg)
    summary_rows = _audit_summary_rows(args.summary_glob)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "fairness_policy_top_level.csv", [top_row])
    _write_csv(output_dir / "fairness_policy_methods.csv", method_rows)
    _write_csv(output_dir / "fairness_policy_imported_rows.csv", summary_rows)
    (output_dir / "fairness_policy_audit.json").write_text(
        json.dumps(
            {
                "top_level": top_row,
                "methods": method_rows,
                "imported_rows": summary_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    ok = bool(top_row["ok"]) and all(bool(row["ok"]) for row in method_rows) and all(
        bool(row["ok"]) for row in summary_rows
    )
    print(f"Saved audit outputs: {output_dir}")
    print(f"fairness_policy_ok={ok}")
    print(f"method_contracts_ok={sum(1 for row in method_rows if row['ok'])}/{len(method_rows)}")
    if summary_rows:
        print(f"imported_rows_ok={sum(1 for row in summary_rows if row['ok'])}/{len(summary_rows)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
