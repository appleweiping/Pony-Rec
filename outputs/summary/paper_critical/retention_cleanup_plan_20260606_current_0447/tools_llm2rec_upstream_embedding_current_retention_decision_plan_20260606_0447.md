# Phase 2.5 Retention Decision Packet

- Plan ID: `tools_llm2rec_upstream_embedding_current_retention_decision_plan_20260606_0447`
- Status: `planning_only_not_executed`
- Will delete now: `False`
- Will start experiment: `False`
- Requires explicit approval: `True`
- Approval token required: `APPROVE_DELETE_COMPLETED_TOOLS_LLM2REC_UPSTREAM_EMBEDDING_20260605`

## Candidate

- Description: Completed Tools LLM2Rec upstream Qwen3 item embedding
- Target path: `/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`
- Expected size bytes: `5662687360`
- Expected sha256: `306618d974eb4133d9cda87bae3251e17d793aa6f5a8cb38d558b549ed31d56e`
- Risk tier: `approval_required_external_embedding_cache`
- Risk rank: `20`
- Retention audit source: `outputs/summary/paper_critical/server_storage_phase2_5_retention_audit_current_20260606_0432.json`
- Current free bytes: `12397707264`
- Expected free bytes after delete: `18060394624`
- Clears 15GiB floor: `True`

## Required Preconditions

- User explicitly approves this exact target and records the retention/archive decision.
- The ranked retention audit recommendation is accepted for this exact target.
- User records that no cheap rerun/resume need depends on this embedding.
- No relevant Python experiment or matching baseline process is active.
- Target realpath exactly matches the plan target_path and size is at least expected_size_bytes.
- Target sha256 matches expected_sha256 from provenance before deletion.
- Completed row evidence remains official_completed with blockers=[] and score_coverage_rate=1.0.
- Domain gate passes before deletion.
- SHA256 and byte-size manifests are written before deletion.

## Required Postconditions

- The target path is absent.
- Disk is above min_free_bytes.
- The domain gate still passes.
- The comparison/paired-test gate still passes without overwriting prior server-final evidence.
- Protected scores, provenance, audits, run summaries, and imported tables remain present.
- Canonical docs and shared memory record the deletion manifest, freed bytes, and preserved evidence.

## Verdict

This packet is non-destructive. Deletion remains prohibited until the exact target is approved, the approval token is set, the generated shell guard is deliberately removed, and all manifest/gate checks in the packet are run successfully.
