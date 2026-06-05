# Phase 2.5 Storage Retention Audit

- Generated UTC: `2026-06-05T20:33:08+00:00`
- Remote: `pony-rec-gpu`
- Project: `~/projects/pony-rec-rescue-shadow-v6`
- Active project Python processes: `0`
- GPU: `0 %, 15 MiB, 49140 MiB`
- Free bytes: `12397707264`
- Deficit to minimum: `3708420096`
- Experiment launch allowed: `False`
- Safe-now recoverable bytes: `0`
- Safe-now sufficient: `False`

## Recommended Approval Candidate

- Path: `/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`
- Risk tier: `approval_required_external_embedding_cache`
- Expected free bytes after delete: `18060394624`
- Clears minimum gate: `True`
- Note: This is the lowest-risk high-yield candidate under the audit's current policy. It still must not be deleted without an explicit archive/retention approval and post-delete gate checks.

## High-Yield Approval-Required Candidates

- `/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`: `5662687360` bytes; `approval_required_external_embedding_cache`; expected free `18060394624`; Completed upstream embedding/cache outside the final evidence directory. High-yield, but protected unless an explicit retention/archive decision is recorded.
- `outputs/home_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/llmesr_official_model.pt`: `6812135931` bytes; `approval_required_final_model_checkpoint`; expected free `19209843195`; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/home_large10000_100neg_llmemb_official_qwen3base_same_candidate/llmemb_official_model.pt`: `6446871382` bytes; `approval_required_final_model_checkpoint`; expected free `18844578646`; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/tools_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/llmesr_official_model.pt`: `4798848507` bytes; `approval_required_final_model_checkpoint`; expected free `17196555771`; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/tools_large10000_100neg_llmemb_official_qwen3base_same_candidate/llmemb_official_model.pt`: `4522405462` bytes; `approval_required_final_model_checkpoint`; expected free `16920112726`; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/sports_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/llmesr_official_model.pt`: `4167963387` bytes; `approval_required_final_model_checkpoint`; expected free `16565670651`; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/sports_large10000_100neg_llmemb_official_qwen3base_same_candidate/llmemb_official_model.pt`: `3919354786` bytes; `approval_required_final_model_checkpoint`; expected free `16317062050`; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/toys_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/llmesr_official_model.pt`: `3847029499` bytes; `approval_required_final_model_checkpoint`; expected free `16244736763`; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.

## Safe-Now Low-Yield Candidates


## Verdict

Do not launch Phase 2.5 signal-row regeneration until disk is expanded or one high-yield completed artifact receives an explicit archive/retention decision.
