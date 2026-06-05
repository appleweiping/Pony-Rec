$ErrorActionPreference = 'Stop'
Write-Host 'GUARDED PLAN ONLY: confirm the runner completed and remove this throw after all preconditions pass.'
throw 'This completion gate plan is intentionally non-runnable as generated.'

# Run from the local repository root: D:\Research\Uncertainty

# server_final_audit
python scripts\audit\main_remote_official_evidence_audit.py `
  --remote_evidence_dir outputs/tools_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate `
  --mode server_final `
  --expected_users 10000 `
  --expected_candidates_per_user 101 `
  --quiet

# server_large_artifact_manifest
python scripts\audit\main_remote_server_large_artifact_manifest.py `
  --remote_evidence_dir outputs/tools_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate `
  --quiet

# local_light_sync
python scripts\audit\main_sync_official_evidence_package.py `
  --remote_evidence_dir outputs/tools_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate `
  --local_evidence_dir outputs\baselines\official_adapters\tools_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate `
  --copy `
  --quiet

# local_light_audit
python scripts\audit\main_audit_official_evidence_package.py `
  --evidence_dir outputs\baselines\official_adapters\tools_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate `
  --mode local_light `
  --expected_users 10000 `
  --expected_candidates_per_user 101 `
  --output_json outputs\baselines\official_adapters\tools_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate\local_light_evidence_audit.json `
  --quiet
