# Phase 2.5 Storage Retention Audit

Created: 2026-06-05T14:49:38Z

Verdict: `below_phase2_5_launch_threshold`.

Server preflight found no relevant Python experiment process, GPU idle at
`0 % / 15 MiB`, and `/home/ajifang` at `12,342,841,344` bytes free / `94%`
used. This is below the Phase 2.5 minimum of `15GiB`, with a deficit of
`3,763,286,016` bytes.

Routine cleanup is not enough. The only clear safe-now project candidates are
low-yield completed staging/temp remnants, mainly the Tools LLM2Rec paper
adapter directory at about `57M` and `tmp_llm2rec_sync` at about `40K`.

High-yield candidates require an explicit archive or retention decision before
deletion:

- `/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`
  (`5,662,687,360` bytes): completed Tools LLM2Rec upstream embedding; referenced
  by the run summary and protected as an external embedding artifact unless a
  retention decision is made.
- Completed LLMEmb/LLM-ESR final model checkpoints for Tools/Home/Sports/Toys
  (`3.8G` to `6.8G` each): enough to recover the disk gate individually in most
  cases, but protected as final method checkpoints unless a separate retention
  policy permits deletion.

Protected and not deletion candidates: task splits, candidate rows, scores,
provenance, score audits, run summaries, imported `tables/`, legacy Electronics
ELMRec prediction with failed server-final audit, other projects, and installed
runtime packages.

No deletion was performed. Do not launch Phase 2.5 signal-row regeneration
until disk is expanded or one high-yield completed artifact receives an explicit
archive/retention decision with sha256/size manifesting and post-delete gate
checks.
