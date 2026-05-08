from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.baselines.official_runner.contract import (
    build_base_provenance,
    inspect_task_package,
    resolve_repo_dir,
    resolve_repo_dir_text,
    text,
)
from src.baselines.official_runner.llm2rec import run_llm2rec_official


METHOD_BLOCKERS = {
    "llm2rec": [],
    "llmesr": [
        "need_official_entrypoint_audit_for_pinned_llmesr_repo",
        "need_default_hparam_source_from_pinned_llmesr_repo",
        "need_llmesr_upstream_wrapper_relabelled_from_style_to_official_after_audit",
    ],
    "llmemb": [
        "need_official_llmemb_data_adapter",
        "need_official_llmemb_training_entrypoint",
        "need_official_llmemb_same_candidate_score_exporter",
    ],
    "rlmrec": [
        "need_official_rlmrec_graph_data_adapter",
        "need_official_rlmrec_training_entrypoint",
        "need_official_rlmrec_same_candidate_score_exporter",
    ],
    "irllrec": [
        "need_official_irllrec_intent_data_adapter",
        "need_official_irllrec_training_entrypoint",
        "need_official_irllrec_same_candidate_score_exporter",
    ],
    "setrec": [
        "need_official_setrec_identifier_data_adapter",
        "need_official_setrec_training_entrypoint",
        "need_official_setrec_same_candidate_score_exporter",
    ],
}


OFFICIAL_ENTRYPOINT_HINTS = {
    "llm2rec": "LLM2Rec seqrec train/evaluate entrypoint plus local exact-score exporter",
    "llmesr": "LLM-ESR model training/evaluation entrypoint plus local exact-score exporter",
    "llmemb": "LLMEmb official training/evaluation entrypoint to be inspected",
    "rlmrec": "RLMRec official graph training/evaluation entrypoint to be inspected",
    "irllrec": "IRLLRec official intent training/evaluation entrypoint to be inspected",
    "setrec": "SETRec official identifier training/evaluation entrypoint to be inspected",
}


def inspect_official_adapter(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    method_cfg: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    repo_dir = resolve_repo_dir(method_cfg, getattr(args, "repo_dir", ""))
    repo_dir_text = resolve_repo_dir_text(method_cfg, getattr(args, "repo_dir", ""))
    blockers = []
    blockers.extend(inspect_task_package(args.task_dir, getattr(args, "valid_task_dir", "")))
    if not repo_dir.exists():
        blockers.append(f"missing_official_repo:{repo_dir_text}")
    if not (Path(args.backbone_path).expanduser().exists() if text(getattr(args, "backbone_path", "")) else True):
        blockers.append(f"missing_backbone_path:{args.backbone_path}")
    blockers.extend(METHOD_BLOCKERS.get(args.method, ["missing_method_official_adapter_implementation"]))

    status = "official_inspection_ready" if not blockers else "official_blocked"
    return build_base_provenance(
        args=args,
        cfg=cfg,
        method_cfg=method_cfg,
        contract=contract,
        implementation_status=status,
        stage=args.stage,
        blockers=blockers,
        official_entrypoint=OFFICIAL_ENTRYPOINT_HINTS.get(args.method, "to_be_inspected"),
        score_coverage_rate=None,
        extra={
            "runner_support_level": "official_llm2rec_qwen3base_sasrec" if args.method == "llm2rec" else "inspect_scaffold",
            "runner_note": (
                "LLM2Rec run support is implemented, but inspect stage never marks a row official_completed."
                if args.method == "llm2rec"
                else "This provenance scaffold intentionally does not mark the row official_completed. "
                "Implement the method adapter against the pinned official repo before importing main-table scores."
            ),
        },
    )


def run_official_adapter(*, args: argparse.Namespace, cfg: dict[str, Any], method_cfg: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    if args.method == "llm2rec":
        return run_llm2rec_official(args=args, cfg=cfg, method_cfg=method_cfg, contract=contract)
    provenance = inspect_official_adapter(args=args, cfg=cfg, method_cfg=method_cfg, contract=contract)
    provenance["stage"] = "run"
    provenance["implementation_status"] = "official_blocked"
    if "run_stage_not_implemented_for_method" not in provenance["blockers"]:
        provenance["blockers"].append("run_stage_not_implemented_for_method")
    return provenance
