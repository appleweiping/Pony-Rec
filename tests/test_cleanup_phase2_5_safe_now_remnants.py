import json
from pathlib import Path

from scripts.audit.main_cleanup_phase2_5_safe_now_remnants import SAFE_NOW_TARGETS, cleanup_safe_now


def _seed_targets(root: Path) -> None:
    for index, rel_path in enumerate(SAFE_NOW_TARGETS):
        path = root / rel_path
        path.mkdir(parents=True, exist_ok=True)
        (path / f"file_{index}.txt").write_text(f"payload-{index}\n", encoding="utf-8")


def test_cleanup_safe_now_dry_run_manifests_without_delete(tmp_path):
    _seed_targets(tmp_path)
    output = tmp_path / "outputs/summary/manifest.json"

    payload = cleanup_safe_now(
        root=tmp_path,
        output_json=output,
        execute=False,
        skip_process_check=True,
    )

    assert payload["delete_performed"] is False
    assert payload["total_size_bytes"] > 0
    assert all((tmp_path / rel_path).exists() for rel_path in SAFE_NOW_TARGETS)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["targets"][0]["files"][0]["sha256"]


def test_cleanup_safe_now_execute_removes_only_fixed_targets(tmp_path):
    _seed_targets(tmp_path)
    protected = tmp_path / "outputs/tools_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate/scores.csv"
    protected.parent.mkdir(parents=True, exist_ok=True)
    protected.write_text("source_event_id,user_id,item_id,score\n", encoding="utf-8")
    output = tmp_path / "outputs/summary/manifest.json"

    payload = cleanup_safe_now(
        root=tmp_path,
        output_json=output,
        execute=True,
        skip_process_check=True,
    )

    assert payload["delete_performed"] is True
    assert sorted(payload["deleted_targets"]) == sorted(SAFE_NOW_TARGETS)
    assert all(not (tmp_path / rel_path).exists() for rel_path in SAFE_NOW_TARGETS)
    assert protected.exists()
