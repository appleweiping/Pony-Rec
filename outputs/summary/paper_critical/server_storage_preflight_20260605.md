# Server Storage Preflight for Phase 2.5

Generated UTC: `2026-06-05T14:31:22.882122+00:00`

## Verdict

`NO_IMMEDIATE_HIGH_IMPACT_SAFE_DELETION_WITHOUT_ARCHIVE_DECISION`.

Free space is `7849136128` bytes and used percentage is `97%`; this is below the project threshold for signal-row regeneration.

## Immediate Safe Cleanup

Cache/tmp cleanup candidates are small (`241299456` bytes upper-bound in the scan) and do not solve the threshold.

## Conditional Candidate

- `outputs/baselines/paper_adapters/tools_large10000_100neg_llmesr_official_adapter/llm_esr/handled/itm_emb_np.pkl` (`4418945194` bytes). This would likely raise free space above 10GiB, but it is an external embedding artifact recorded in provenance, so it needs an explicit archive/resume decision before deletion.

## Protected

- `outputs/baselines/external_tasks/* candidate/ranking/item_metadata task splits`
- `outputs/*_official_qwen3base_same_candidate/*_official_model.pt final method checkpoints`
- `outputs/*_official_qwen3base_same_candidate/scores.csv final scores`
- `outputs/*_official_qwen3base_same_candidate/tables/ imported tables`
- `/home/ajifang/models/Qwen/Qwen3-8B model weights`
- `/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy unless separately archived/accepted`
- `other projects under /home/ajifang/projects/*`

## Next Action

Make an explicit archive/resume decision on deleting the completed Tools LLM-ESR intermediate adapter embedding, or provision additional disk, before starting signal-row regeneration.
