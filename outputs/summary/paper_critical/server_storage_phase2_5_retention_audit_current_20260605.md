# Phase 2.5 Storage Retention Audit

- Generated UTC: `2026-06-05T15:53:35+00:00`
- Remote: `pony-rec-gpu`
- Project: `~/projects/pony-rec-rescue-shadow-v6`
- Active project Python processes: `0`
- GPU: `0 %, 15 MiB, 49140 MiB`
- Free bytes: `12342640640`
- Deficit to minimum: `3763486720`
- Experiment launch allowed: `False`
- Safe-now recoverable bytes: `64611717`
- Safe-now sufficient: `False`

## High-Yield Approval-Required Candidates

- `outputs/home_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/llmesr_official_model.pt`: `6812135931` bytes; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/home_large10000_100neg_llmemb_official_qwen3base_same_candidate/llmemb_official_model.pt`: `6446871382` bytes; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/tools_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/llmesr_official_model.pt`: `4798848507` bytes; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/tools_large10000_100neg_llmemb_official_qwen3base_same_candidate/llmemb_official_model.pt`: `4522405462` bytes; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/sports_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/llmesr_official_model.pt`: `4167963387` bytes; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/sports_large10000_100neg_llmemb_official_qwen3base_same_candidate/llmemb_official_model.pt`: `3919354786` bytes; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `outputs/toys_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/llmesr_official_model.pt`: `3847029499` bytes; Completed model/checkpoint artifact; protected unless a separate retention decision permits removal.
- `/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`: `5662687360` bytes; Completed upstream embedding/cache outside the final evidence directory. High-yield, but protected unless an explicit retention/archive decision is recorded.

## Safe-Now Low-Yield Candidates

- `outputs/baselines/paper_adapters/tools_large10000_100neg_llm2rec_official_adapter`: `59186235` bytes; `SAFE_NOW_LOW_YIELD`
- `outputs/baselines/paper_adapters/tools_large10000_100neg_llmesr_official_adapter`: `5389043` bytes; `SAFE_NOW_LOW_YIELD`
- `tmp_llm2rec_sync`: `36439` bytes; `SAFE_NOW_LOW_YIELD`

## Verdict

Do not launch Phase 2.5 signal-row regeneration until disk is expanded or one high-yield completed artifact receives an explicit archive/retention decision.
