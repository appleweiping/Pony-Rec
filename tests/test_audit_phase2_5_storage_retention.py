from scripts.audit.main_audit_phase2_5_storage_retention import build_audit, classify_large_file


def test_classifies_protected_task_split_and_approval_required_checkpoint():
    task = classify_large_file("outputs/baselines/external_tasks/tools_large10000_100neg_test_same_candidate/candidate_items.csv", 840)
    assert task["classification"] == "PROTECTED_TASK_SPLIT"

    model = classify_large_file("outputs/tools_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/llmesr_official_model.pt", 4_798_848_507)
    assert model["classification"] == "NEEDS_APPROVAL_OR_ARCHIVE_DECISION"

    embedding = classify_large_file(
        "/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy",
        5_662_687_360,
    )
    assert embedding["classification"] == "NEEDS_APPROVAL_OR_ARCHIVE_DECISION"


def test_build_audit_keeps_launch_blocked_when_safe_cleanup_is_too_small():
    outputs = {
        "processes": "",
        "gpu": "0 %, 15 MiB, 49140 MiB\n",
        "disk": "/dev/vda5 209709965312 186640080896 12342689792 94% /\n",
        "large_outputs": "\n".join(
            [
                "4798848507 outputs/tools_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/llmesr_official_model.pt",
                "840075550 outputs/baselines/external_tasks/tools_large10000_100neg_test_same_candidate/candidate_items.csv",
                "106879641 outputs/tools_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate/scores.csv",
            ]
        ),
        "llm2rec_item_info": "5662687360 /home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy\n",
        "safe_candidate_dirs": "\n".join(
            [
                "59186235 outputs/baselines/paper_adapters/tools_large10000_100neg_llm2rec_official_adapter",
                "5389043 outputs/baselines/paper_adapters/tools_large10000_100neg_llmesr_official_adapter",
            ]
        ),
        "logs_and_pids": "107099852 ./ccrp_v3_all_domains.log\n",
    }

    def runner(_remote: str, command: str) -> str:
        for key, value in outputs.items():
            if key == "large_outputs" and "find outputs" in command:
                return value
            if key == "llm2rec_item_info" and "LLM2Rec/item_info" in command:
                return value
            if key == "safe_candidate_dirs" and "du -sb" in command:
                return value
            if key == "logs_and_pids" and "*.log" in command:
                return value
            if key == "processes" and "ps aux" in command:
                return value
            if key == "gpu" and "nvidia-smi" in command:
                return value
            if key == "disk" and "df -B1" in command:
                return value
        raise AssertionError(command)

    audit = build_audit(command_runner=runner)

    assert audit["phase2_5_disk_gate"]["experiment_launch_allowed"] is False
    assert audit["safe_now_total_recoverable_bytes"] == 64_575_278
    assert audit["safe_now_sufficient_for_min_free"] is False
    assert len(audit["high_yield_candidates_requiring_approval"]) == 2
    assert audit["high_yield_candidates_requiring_approval"][0]["retention_risk_rank"] == 20
    assert audit["recommended_approval_candidate"]["path"].endswith("pony_qwen3_8b_title_item_embs.npy")
    assert audit["retention_recommendation"]["would_clear_min_free_gate"] is True
    assert audit["audit_verdict"]["delete_performed"] is False
