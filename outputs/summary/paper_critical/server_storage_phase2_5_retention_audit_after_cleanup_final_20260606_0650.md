# Phase 2.5 Storage Retention Audit

- Generated UTC: `2026-06-05T22:38:53+00:00`
- Remote: `pony-rec-gpu`
- Project: `/home/ajifang/projects/pony-rec-rescue-shadow-v6`
- Active project Python processes: `0`
- GPU: `0 %, 15 MiB, 49140 MiB`
- Free bytes: `25656160256`
- Deficit to minimum: `0`
- Experiment launch allowed: `True`
- Safe-now recoverable bytes: `0`
- Safe-now sufficient: `True`

## Recommended Approval Candidate

- Path: `/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`
- Risk tier: `approval_required_external_embedding_cache`
- Expected free bytes after delete: `31318847616`
- Clears minimum gate: `True`
- Note: This is the lowest-risk high-yield candidate under the audit's current policy. It still must not be deleted without an explicit archive/retention approval and post-delete gate checks.

## High-Yield Approval-Required Candidates

- `/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`: `5662687360` bytes; `approval_required_external_embedding_cache`; expected free `31318847616`; Completed upstream embedding/cache outside the final evidence directory. High-yield, but protected unless an explicit retention/archive decision is recorded.
- `outputs/tools_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/llmesr_official_model.pt`: `4798848507` bytes; `approval_required_final_model_checkpoint`; expected free `30455008763`; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/tools_large10000_100neg_llmemb_official_qwen3base_same_candidate/llmemb_official_model.pt`: `4522405462` bytes; `approval_required_final_model_checkpoint`; expected free `30178565718`; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/sports_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/llmesr_official_model.pt`: `4167963387` bytes; `approval_required_final_model_checkpoint`; expected free `29824123643`; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/sports_large10000_100neg_llmemb_official_qwen3base_same_candidate/llmemb_official_model.pt`: `3919354786` bytes; `approval_required_final_model_checkpoint`; expected free `29575515042`; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/toys_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/llmesr_official_model.pt`: `3847029499` bytes; `approval_required_final_model_checkpoint`; expected free `29503189755`; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/toys_large10000_100neg_llmemb_official_qwen3base_same_candidate/llmemb_official_model.pt`: `3612579746` bytes; `approval_required_final_model_checkpoint`; expected free `29268740002`; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/home_large10000_100neg_elmrec_graph_official_qwen3base_same_candidate/elmrec_official_model.pt`: `202495061` bytes; `approval_required_final_model_checkpoint`; expected free `25858655317`; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/tools_large10000_100neg_elmrec_graph_official_qwen3base_same_candidate/elmrec_official_model.pt`: `143280725` bytes; `approval_required_final_model_checkpoint`; expected free `25799440981`; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/sports_large10000_100neg_elmrec_graph_official_qwen3base_same_candidate/elmrec_official_model.pt`: `124725333` bytes; `approval_required_final_model_checkpoint`; expected free `25780885589`; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/toys_large10000_100neg_elmrec_graph_official_qwen3base_same_candidate/elmrec_official_model.pt`: `115286101` bytes; `approval_required_final_model_checkpoint`; expected free `25771446357`; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.

## Safe-Now Low-Yield Candidates


## Verdict

Safe-now cleanup could clear the minimum disk gate after a no-process recheck.
