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
from src.baselines.official_runner.elmrec import run_elmrec_official
from src.baselines.official_runner.irllrec import run_irllrec_official
from src.baselines.official_runner.llm2rec import run_llm2rec_official
from src.baselines.official_runner.llmemb import run_llmemb_official
from src.baselines.official_runner.llmesr import run_llmesr_official
from src.baselines.official_runner.promax import run_promax_official
from src.baselines.official_runner.proex import run_proex_official
from src.baselines.official_runner.rlmrec import run_rlmrec_official
from src.baselines.official_runner.setrec import run_setrec_official


METHOD_BLOCKERS = {
    "llm2rec": [],
    "llmesr": [],
    "llmemb": [],
    "rlmrec": [],
    "irllrec": [],
    "setrec": [],
    "elmrec": [],
    "proex": [],
    "promax": [],
}


OFFICIAL_ENTRYPOINT_HINTS = {
    "llm2rec": "LLM2Rec seqrec train/evaluate entrypoint plus local exact-score exporter",
    "llmesr": "LLM-ESR model training/evaluation entrypoint plus local exact-score exporter",
    "llmemb": "LLMEmb official training/evaluation entrypoint to be inspected",
    "rlmrec": "RLMRec official graph training/evaluation entrypoint to be inspected",
    "irllrec": "IRLLRec official intent training/evaluation entrypoint to be inspected",
    "setrec": "SETRec official identifier training/evaluation entrypoint to be inspected",
    "elmrec": "ELMRec official high-order graph interaction training/evaluation bridge",
    "proex": "ProEx KDD 2026 official profile-enhanced recommendation entrypoint to be inspected",
    "promax": "ProMax SIGIR 2026 official profile-enhanced recommendation entrypoint through pinned ProRec LightGCN_promax",
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
            "runner_support_level": (
                "official_llm2rec_qwen3base_sasrec"
                if args.method == "llm2rec"
                else "official_llmesr_qwen3base_sasrec"
                if args.method == "llmesr"
                else "official_llmemb_qwen3base_sasrec"
                if args.method == "llmemb"
                else "official_rlmrec_qwen3base_simgcl_plus"
                if args.method == "rlmrec"
                else "official_irllrec_qwen3base_lightgcn_int"
                if args.method == "irllrec"
                else "official_setrec_qwen3base_identifier"
                if args.method == "setrec"
                else "official_elmrec_qwen3base_graph_bridge"
                if args.method == "elmrec"
                else "official_proex_qwen3base_profile"
                if args.method == "proex"
                else "official_promax_qwen3base_profile"
                if args.method == "promax"
                else "inspect_scaffold"
            ),
            "runner_note": (
                "LLM2Rec run support is implemented, but inspect stage never marks a row official_completed."
                if args.method == "llm2rec"
                else "LLM-ESR run support is implemented, but inspect stage never marks a row official_completed."
                if args.method == "llmesr"
                else "LLMEmb run support is implemented, but inspect stage never marks a row official_completed."
                if args.method == "llmemb"
                else "RLMRec run support is implemented, but inspect stage never marks a row official_completed."
                if args.method == "rlmrec"
                else "IRLLRec run support is implemented, but inspect stage never marks a row official_completed."
                if args.method == "irllrec"
                else "SETRec run support is implemented, but inspect stage never marks a row official_completed."
                if args.method == "setrec"
                else "ELMRec run support is implemented, but inspect stage never marks a row official_completed."
                if args.method == "elmrec"
                else "This provenance scaffold intentionally does not mark the row official_completed. "
                "Implement the method adapter against the pinned official repo before importing main-table scores."
            ),
        },
    )


def run_official_adapter(*, args: argparse.Namespace, cfg: dict[str, Any], method_cfg: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    if args.method == "llm2rec":
        return run_llm2rec_official(args=args, cfg=cfg, method_cfg=method_cfg, contract=contract)
    if args.method == "llmesr":
        return run_llmesr_official(args=args, cfg=cfg, method_cfg=method_cfg, contract=contract)
    if args.method == "llmemb":
        return run_llmemb_official(args=args, cfg=cfg, method_cfg=method_cfg, contract=contract)
    if args.method == "rlmrec":
        return run_rlmrec_official(args=args, cfg=cfg, method_cfg=method_cfg, contract=contract)
    if args.method == "irllrec":
        return run_irllrec_official(args=args, cfg=cfg, method_cfg=method_cfg, contract=contract)
    if args.method == "setrec":
        return run_setrec_official(args=args, cfg=cfg, method_cfg=method_cfg, contract=contract)
    if args.method == "elmrec":
        return run_elmrec_official(args=args, cfg=cfg, method_cfg=method_cfg, contract=contract)
    if args.method == "proex":
        return run_proex_official(args=args, cfg=cfg, method_cfg=method_cfg, contract=contract)
    if args.method == "promax":
        return run_promax_official(args=args, cfg=cfg, method_cfg=method_cfg, contract=contract)
    provenance = inspect_official_adapter(args=args, cfg=cfg, method_cfg=method_cfg, contract=contract)
    provenance["stage"] = "run"
    provenance["implementation_status"] = "official_blocked"
    if "run_stage_not_implemented_for_method" not in provenance["blockers"]:
        provenance["blockers"].append("run_stage_not_implemented_for_method")
    return provenance
