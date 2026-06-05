import json
from pathlib import Path

from scripts.audit.main_remote_phase2_5_retention_preapproval_audit import build_audit


TARGET = "/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy"
SHA = "306618d974eb4133d9cda87bae3251e17d793aa6f5a8cb38d558b549ed31d56e"
SIZE = 5_662_687_360


def _write_plan(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "remote_project": "/home/ajifang/projects/pony-rec-rescue-shadow-v6",
                "min_free_bytes": 16_106_127_360,
                "candidate": {
                    "target_path": TARGET,
                    "expected_size_bytes": SIZE,
                    "expected_sha256": SHA,
                    "protected_evidence_dir": "outputs/tools_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _runner(*, sha: str = SHA, active_processes: str = "", provenance_sha: str = SHA):
    def run(_remote: str, command: str) -> str:
        if "ps aux" in command:
            return active_processes
        if command.startswith("nvidia-smi"):
            return "0 %, 15 MiB, 49140 MiB\n"
        if command.startswith("df -B1"):
            return "/dev/vda5 209709965312 186576179200 12406591488 94% /\n"
        if command.startswith("realpath"):
            return TARGET + "\n"
        if command.startswith("stat -c") and TARGET in command:
            return f"{SIZE} {TARGET}\n"
        if command.startswith("sha256sum"):
            return f"{sha}  {TARGET}\n"
        if "fairness_provenance.json" in command and "python" not in command:
            return (
                "10 /project/fairness_provenance.json\n"
                "10 /project/server_final_evidence_audit.json\n"
                "10 /project/tables/ranking_eval_records.csv\n"
                "10 /project/scores.csv\n"
            )
        if "fairness_provenance.json" in command and "python" in command:
            return json.dumps(
                {
                    "implementation_status": "official_completed",
                    "blockers": [],
                    "score_coverage_rate": 1.0,
                    "qwen3_item_embedding_sha256": provenance_sha,
                }
            )
        if "server_final_evidence_audit.json" in command and "python" in command:
            return json.dumps(
                {
                    "ok": True,
                    "scores_present": True,
                    "prediction_present": False,
                    "ranking_eval_records_present": True,
                }
            )
        raise AssertionError(f"unexpected command: {command}")

    return run


def test_preapproval_audit_allows_only_disk_failure(tmp_path):
    plan = _write_plan(tmp_path / "plan.json")

    audit = build_audit(plan_json=plan, command_runner=_runner())

    assert audit["ok"] is False
    assert audit["preapproval_checks_ready_except_disk"] is True
    assert audit["failures"] == ["disk_below_min_free_before_cleanup"]
    assert audit["actual_sha256"] == SHA


def test_preapproval_audit_rejects_sha_mismatch(tmp_path):
    plan = _write_plan(tmp_path / "plan.json")

    audit = build_audit(plan_json=plan, command_runner=_runner(sha="bad"))

    assert audit["preapproval_checks_ready_except_disk"] is False
    assert "target_sha256_mismatch" in audit["failures"]


def test_preapproval_audit_rejects_active_process(tmp_path):
    plan = _write_plan(tmp_path / "plan.json")

    audit = build_audit(plan_json=plan, command_runner=_runner(active_processes="ajifang 1 python baseline\n"))

    assert audit["preapproval_checks_ready_except_disk"] is False
    assert "active_project_python_processes_present" in audit["failures"]
