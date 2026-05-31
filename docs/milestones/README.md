# Milestone Map

This directory is the canonical navigation layer for the project. It preserves
the important historical milestones without treating every experiment log as an
equal paper claim.

## Project Spine

```text
M0 Week1-4 / pony12 observation
-> M1 Pony framework
-> M2 Light series boundary test
-> M3 Shadow series task-grounded signals
-> M4 Baseline system and fairness contract
-> M5 Small-domain to four-domain validation
-> M6 Complete recommendation-system roadmap
```

The defended paper story is narrower than the full roadmap:

```text
LLM recommendation confidence is informative but unreliable.
Task-grounded calibrated uncertainty can turn it into a decision-useful signal
under a controlled same-candidate recommendation protocol.
```

The larger system story remains staged:

```text
observation -> framework -> reproducible baseline system
-> four-domain validation -> official baselines
-> signal/decision/generative recommendation modules
```

## Evidence Levels

| level | meaning | main-claim eligible |
| --- | --- | --- |
| L0 | observation and diagnostic evidence | context only |
| L1 | small-domain controlled experiments | supporting evidence |
| L2 | Pony12 / Light / Shadow internal framework progression | supporting or ablation evidence |
| L3 | classical same-candidate baselines | main evidence if complete |
| L4 | four-domain same-candidate 100neg validation | main robustness evidence if complete |
| L5 | official-code-level external LLM-rec baselines | main external-baseline evidence only after provenance passes |
| L6 | artifact-ready reproducibility package | submission/artifact evidence |

## Milestone Index

- [M0 Week1-4 / pony12 Observation](M0_week1_4_pony12_observation.md)
- [M1 Pony Framework](M1_pony_framework_week5_6.md)
- [M2 Light Series](M2_light_series.md)
- [M3 Shadow Series](M3_shadow_series.md)
- [M4 Baseline System](M4_baseline_system.md)
- [M5 Four-Domain Same-Candidate Validation](M5_four_domain_same_candidate.md)
- [M6 Complete Recommendation-System Roadmap](M6_complete_recommender_system.md)

## Current Working Position

The repository is currently between M4 and M5:

- the same-candidate baseline system exists;
- the official external-baseline contract is written;
- the paper-style external rows are supplementary, not official reproductions;
- the four-domain 100neg protocol is the active robustness gate;
- the complete recommendation system remains a roadmap until the official
  baselines, Shadow large-scale diagnostics, and LoRA/generative modules are
  completed under the same protocol.

## Stable Agent Roles

Use these roles in future multi-agent work:

| role | responsibility |
| --- | --- |
| Milestone Architect | README, milestone map, file index, claim boundaries |
| Server Runner | server commands, nohup logs, resume checks, output validation |
| Baseline Engineer | official repos, provenance, score schema, importer coverage |
| Literature Scout | top-conference sources, related work, protocol expectations |
| Reviewer Agent | overclaim audit, fairness audit, ablations, statistical validity |

## Multi-Agent Handoff Rules

For broad research or implementation tasks, use multi-agent collaboration.
Every agent handoff should state:

- current milestone and files inspected;
- whether the work affects a main claim, supplementary evidence, or diagnostic
  evidence;
- blockers and server commands needed next;
- output/provenance paths;
- whether a result is eligible for a main table.

The Reviewer Agent can veto overclaims, toy shortcuts, and baseline rows that
do not pass the fairness/provenance gates. The Server Runner reports commands,
logs, PIDs, output paths, and failures; it does not change claim boundaries.

Root-level [AGENTS.md](../../AGENTS.md) is the operating contract for future
agents. If a milestone status changes, update this file and the relevant M-file
rather than relying on a chat handoff.

## Current Working Position (updated 2026-05-31)

The repository is now in M5 (multi-domain SOTA validation):

- C-CRP v3 completed on all 8 domains
- Official external baselines completed on original 4 domains (8 methods each)
- New domains (sports/toys/home/tools) official baselines are pending; the
  Phase 2 runner is now reconciled to the canonical 8-method block and imports
  full `@5/@10/@20 + MRR` metrics after each successful score audit
- Strategy: achieve SOTA only after the new-domain official baselines pass
  same-candidate score/provenance/import gates

### C-CRP v3 Results (all domains)

| Domain | Users | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR | Status |
|--------|-------|------|-------|-------|--------|---------|---------|-----|--------|
| beauty | 973 | 0.157 | 0.229 | 0.369 | 0.111 | 0.134 | 0.169 | 0.128 | #2 (ProEx=0.253) |
| books | 10000 | 0.374 | **0.476** | 0.592 | 0.300 | **0.333** | 0.362 | 0.306 | **SOTA** |
| electronics | 10000 | 0.218 | **0.299** | 0.418 | 0.157 | **0.183** | 0.213 | 0.168 | **SOTA** |
| movies | 10000 | 0.145 | 0.208 | 0.331 | 0.108 | 0.128 | 0.159 | 0.127 | #5 |
| sports | 10000 | 0.275 | 0.382 | 0.517 | 0.198 | 0.233 | 0.267 | 0.208 | baselines pending |
| toys | 10000 | 0.317 | 0.396 | 0.506 | 0.245 | 0.271 | 0.298 | 0.250 | baselines pending |
| home | 10000 | 0.156 | 0.226 | 0.351 | 0.110 | 0.132 | 0.164 | 0.126 | baselines pending |
| tools | 10000 | 0.194 | 0.270 | 0.393 | 0.142 | 0.166 | 0.197 | 0.156 | baselines pending |

Original-domain C-CRP v3 formal reports are under
`outputs/ccrp_v3_formal/<domain>/report.json`; the four-domain comparison with
the canonical official baseline block is
`outputs/ccrp_v3_formal/main_comparison_table.csv`. New-domain artifact
completeness: each of sports/toys/home/tools has `report.json`, `scores.csv`
with 1,010,000 candidate-score rows plus header, and `user_ranks.jsonl` with
10,000 user-rank rows.

### Experiment Execution Plan

1. C-CRP v3 on all 8 domains (Phase 1) — complete
2. 8 official baselines on 4 new domains (Phase 2) — runner reconciled;
   launch with a single-domain production loop such as
   `DOMAINS_OVERRIDE=sports bash scripts/run_baselines_new_domains.sh`
3. Full comparison table + statistical tests (Phase 3)
4. Paper writing with ARIS skill (Phase 4)
5. GPT-5.5/Codex review cycle until 8/10 (Phase 5)

### Server State

- Batch script complete: `run_ccrp_v3_all_new_domains.sh` (sports/toys/home/tools)
- GPU: RTX 4090, idle after C-CRP completion
- Disk: 44 GB free (checked 2026-05-31)
- All experiments use: Qwen3-8B, vLLM, 10k users, 101 candidates (1+100neg)
