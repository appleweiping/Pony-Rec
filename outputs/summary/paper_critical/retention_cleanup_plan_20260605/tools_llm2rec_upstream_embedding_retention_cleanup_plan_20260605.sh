#!/usr/bin/env bash
set -euo pipefail

echo 'GUARDED PLAN ONLY: this file documents an explicit retention decision path.'
echo 'It is intentionally non-runnable as generated and exits before any deletion command.'
exit 2

cd /home/ajifang/projects/pony-rec-rescue-shadow-v6
TARGET=/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy
EXPECTED_SIZE=5662687360
APPROVAL_TOKEN_REQUIRED=APPROVE_DELETE_COMPLETED_TOOLS_LLM2REC_UPSTREAM_EMBEDDING_20260605

# 1. Read-only server preflight.
date '+%F %T %Z'; ps aux | grep python | grep -v grep | grep -i 'pony-rec\|ccrp\|baseline\|uncertainty' || true; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; df -h /home/ajifang; df -B1 /home/ajifang

# 2. Evidence precondition check for the completed row.
test -f /home/ajifang/projects/pony-rec-rescue-shadow-v6/outputs/tools_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate/fairness_provenance.json && test -f /home/ajifang/projects/pony-rec-rescue-shadow-v6/outputs/tools_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate/server_final_evidence_audit.json && test -f /home/ajifang/projects/pony-rec-rescue-shadow-v6/outputs/tools_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate/tables/ranking_eval_records.csv && test -f /home/ajifang/projects/pony-rec-rescue-shadow-v6/outputs/tools_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate/scores.csv
/home/ajifang/miniconda3/bin/python scripts/audit/main_audit_domain_official_gate.py --root . --domain tools --output_json outputs/summary/phase2_5_retention_cleanup_domain_gate_recheck.json --output_csv outputs/summary/phase2_5_retention_cleanup_domain_gate_recheck.csv --quiet

# 3. Exact target guard.
test -f "$TARGET"
RESOLVED_TARGET=$(realpath "$TARGET")
test "$RESOLVED_TARGET" = "$TARGET"
ACTUAL_SIZE=$(stat -c '%s' "$TARGET")
test "$ACTUAL_SIZE" -ge "$EXPECTED_SIZE"
test "$(sha256sum "$TARGET" | awk '{print $1}')" = 306618d974eb4133d9cda87bae3251e17d793aa6f5a8cb38d558b549ed31d56e

# 4. Explicit approval guard. Set APPROVAL_TOKEN only after the retention decision is recorded.
test "${APPROVAL_TOKEN:-}" = APPROVE_DELETE_COMPLETED_TOOLS_LLM2REC_UPSTREAM_EMBEDDING_20260605

# 5. Manifest before delete.
mkdir -p outputs/summary && sha256sum /home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy > outputs/summary/tools_llm2rec_upstream_embedding_retention_cleanup_APPROVAL_REQUIRED_20260605.sha256 && stat -c '%s  %n' /home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy > outputs/summary/tools_llm2rec_upstream_embedding_retention_cleanup_APPROVAL_REQUIRED_20260605.size.txt

# 6. Delete only the approved target.
rm -- /home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy
test ! -e "$TARGET"

# 7. Recheck disk and domain evidence.
df -h /home/ajifang; df -B1 /home/ajifang
/home/ajifang/miniconda3/bin/python scripts/audit/main_audit_domain_official_gate.py --root . --domain tools --output_json outputs/summary/phase2_5_retention_cleanup_post_delete_domain_gate.json --output_csv outputs/summary/phase2_5_retention_cleanup_post_delete_domain_gate.csv --quiet
/home/ajifang/miniconda3/bin/python scripts/experiments/main_build_domain_official_comparison.py --root . --domain tools --gate_json outputs/summary/phase2_5_retention_cleanup_post_delete_domain_gate.json --output_dir outputs/summary/phase2_5_retention_cleanup_comparison_recheck --stamp tools_retention_cleanup_post_delete --n_bootstrap 0 --quiet
