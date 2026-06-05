import hashlib
import json
from pathlib import Path

from scripts.audit.main_audit_phase2_5_retention_decision_packet import build_audit


TARGET = "/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_packet(root: Path, *, shell: str | None = None, recommended_target: str = TARGET) -> Path:
    plan_json = root / "plan.json"
    plan_sh = root / "plan.sh"
    plan_md = root / "plan.md"
    retention_json = root / "storage.json"
    packet_sha = root / "plan.sha256"
    plan = {
        "plan_id": "test_plan",
        "status_label": "planning_only_not_executed",
        "will_delete": False,
        "will_delete_files": False,
        "will_execute_cleanup": False,
        "will_start_experiment": False,
        "requires_explicit_approval": True,
        "expected_to_clear_min_free_gate": True,
        "current_free_bytes": 12_406_620_160,
        "expected_free_bytes_after_delete": 18_069_307_520,
        "approval_token_required": "APPROVE_DELETE_COMPLETED_TOOLS_LLM2REC_UPSTREAM_EMBEDDING_20260605",
        "ranked_retention_audit_source": str(retention_json),
        "candidate": {
            "candidate": "tools_llm2rec_upstream_embedding",
            "target_path": TARGET,
        },
    }
    plan_json.write_text(json.dumps(plan) + "\n", encoding="utf-8")
    plan_sh.write_text(
        shell
        or "#!/usr/bin/env bash\nexit 2\nsha256sum /tmp/x\nrm -- /tmp/x\n",
        encoding="utf-8",
    )
    plan_md.write_text(
        "Will delete now: `False`\nRequires explicit approval: `True`\nDeletion remains prohibited\n",
        encoding="utf-8",
    )
    retention_json.write_text(
        json.dumps(
            {
                "phase2_5_disk_gate": {"experiment_launch_allowed": False},
                "recommended_approval_candidate": {
                    "path": recommended_target,
                    "expected_free_bytes_after_delete": 18_069_307_520,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    packet_sha.write_text(
        "\n".join(
            [
                f"{_sha(plan_json)}  {plan_json.name}",
                f"{_sha(plan_sh)}  {plan_sh.name}",
                f"{_sha(plan_md)}  {plan_md.name}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return plan_json


def test_packet_audit_accepts_non_destructive_packet(tmp_path):
    plan_json = _write_packet(tmp_path)

    audit = build_audit(plan_json=plan_json)

    assert audit["ok"] is True
    assert audit["read_only"] is True
    assert audit["will_delete"] is False
    assert audit["will_start_experiment"] is False
    assert audit["plan_summary"]["target_path"] == TARGET


def test_packet_audit_rejects_shell_guard_after_delete(tmp_path):
    plan_json = _write_packet(tmp_path, shell="#!/usr/bin/env bash\nrm -- /tmp/x\nexit 2\n")

    audit = build_audit(plan_json=plan_json)

    assert audit["ok"] is False
    assert "shell_exit_guard_after:rm --" in audit["failures"]


def test_packet_audit_rejects_recommended_target_mismatch(tmp_path):
    plan_json = _write_packet(tmp_path, recommended_target="/tmp/other.npy")

    audit = build_audit(plan_json=plan_json)

    assert audit["ok"] is False
    assert "retention_audit_recommends_different_target" in audit["failures"]
