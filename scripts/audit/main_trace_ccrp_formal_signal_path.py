from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORMAL_RUNNER = REPO_ROOT / "experiments" / "rsc" / "run_ccrp_v3_domain.py"
DEFAULT_SELECTOR = REPO_ROOT / "scripts" / "misc" / "main_select_ccrp_variant_on_valid.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "week8_large_scale_future_framework.yaml"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _contains(text: str, needle: str) -> bool:
    return needle in text


def trace_ccrp_formal_signal_path(
    *,
    formal_runner_path: str | Path = DEFAULT_FORMAL_RUNNER,
    selector_path: str | Path = DEFAULT_SELECTOR,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    runner = Path(formal_runner_path)
    selector = Path(selector_path)
    config = Path(config_path)
    runner_text = _read(runner)
    selector_text = _read(selector)
    config_text = _read(config) if config.exists() else ""

    prompt_fields = {
        "relevance_probability": _contains(runner_text, "relevance_probability"),
        "evidence_support": _contains(runner_text, "evidence_support"),
        "counterevidence": _contains(runner_text, "counterevidence"),
        "uncertainty": _contains(runner_text, "uncertainty"),
    }
    formal_outputs = {
        "scores_csv": _contains(runner_text, "scores.csv"),
        "report_json": _contains(runner_text, "report.json"),
        "user_ranks_jsonl": _contains(runner_text, "user_ranks.jsonl"),
        "selected_scored_rows_csv": _contains(runner_text, "ccrp_selected_test_scored_rows.csv"),
        "internal_provenance_json": _contains(runner_text, "ccrp_internal_provenance.json"),
    }
    formal_score_schema = {
        "source_event_id": _contains(runner_text, "source_event_id"),
        "user_id": _contains(runner_text, "user_id"),
        "item_id": _contains(runner_text, "item_id"),
        "score": _contains(runner_text, "score"),
        "ccrp_uncertainty": _contains(runner_text, "ccrp_uncertainty"),
    }
    selector_contract = {
        "requires_valid_signal_path": _contains(selector_text, "--valid_signal_path"),
        "requires_test_signal_path": _contains(selector_text, "--test_signal_path"),
        "writes_valid_sweep": _contains(selector_text, "valid_ccrp_sweep.csv"),
        "writes_selected_scores": _contains(selector_text, "ccrp_selected_test_scores.csv"),
        "writes_selected_scored_rows": _contains(selector_text, "ccrp_selected_test_scored_rows.csv"),
        "writes_internal_provenance": _contains(selector_text, "ccrp_internal_provenance.json"),
    }
    configured_shadow_paths = {
        "pointwise_output_dir": "pointwise_output_dir:" in config_text,
        "ccrp_formal_output_dir": "ccrp_formal_output_dir:" in config_text,
        "winner_signal_variant": "winner_signal_variant:" in config_text,
    }

    has_recomputable_formal_signal = (
        prompt_fields["relevance_probability"]
        and prompt_fields["evidence_support"]
        and prompt_fields["counterevidence"]
    )
    can_rebuild_uncertainty_from_formal_scores = bool(formal_score_schema["ccrp_uncertainty"])
    blockers: list[str] = []
    if not formal_outputs["selected_scored_rows_csv"]:
        blockers.append("formal_runner_does_not_write_scored_rows_with_uncertainty_components")
    if not has_recomputable_formal_signal:
        blockers.append("formal_runner_prompt_does_not_request_evidence_or_counterevidence_fields")
    if not formal_score_schema["ccrp_uncertainty"]:
        blockers.append("formal_scores_schema_has_no_uncertainty_column")

    return {
        "formal_runner_path": str(runner),
        "selector_path": str(selector),
        "config_path": str(config),
        "prompt_fields_requested_by_formal_runner": prompt_fields,
        "formal_runner_outputs_detected": formal_outputs,
        "formal_score_schema_markers": formal_score_schema,
        "selector_contract_detected": selector_contract,
        "configured_shadow_paths_detected": configured_shadow_paths,
        "formal_prompt_could_generate_recomputable_signal_if_preserved": has_recomputable_formal_signal,
        "can_rebuild_paper_ready_uncertainty_rows_from_formal_scores_only": can_rebuild_uncertainty_from_formal_scores,
        "required_inputs_for_selector_route": [
            "valid_signal_path with source_event_id,user_id,item_id/candidate_item_id and uncertainty/recomputable signal columns",
            "test_signal_path with source_event_id,user_id,item_id/candidate_item_id and uncertainty/recomputable signal columns",
            "valid/test candidate_items.csv for exact key coverage",
            "valid/test ranking_task.jsonl for metric evaluation",
        ],
        "blockers": blockers,
        "recommended_next_action": (
            "Do not try to infer uncertainty from formal scores.csv. Locate saved full-scale valid/test signal rows, "
            "or schedule a new signal-generation run that preserves raw/calibrated signal fields after the active "
            "Home RLMRec baseline completes and gates pass."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace whether formal C-CRP v3 score outputs can regenerate paper-ready uncertainty rows."
    )
    parser.add_argument("--formal_runner_path", default=str(DEFAULT_FORMAL_RUNNER))
    parser.add_argument("--selector_path", default=str(DEFAULT_SELECTOR))
    parser.add_argument("--config_path", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output_json", default="")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = trace_ccrp_formal_signal_path(
        formal_runner_path=args.formal_runner_path,
        selector_path=args.selector_path,
        config_path=args.config_path,
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.quiet:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
