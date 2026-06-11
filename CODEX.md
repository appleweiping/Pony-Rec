# CODEX.md

Read `AGENTS.md` first. It is the authoritative operating contract for this repository.

This file is the Codex (GPT-5.5) entry point for the Uncertainty project.

## Quick Orientation

- **Title**: Task-Grounded Uncertainty for LLM-based Recommendation
- **Stage**: M5 (multi-domain SOTA validation in progress)
- **Core Claim**: Task-grounded calibrated uncertainty improves controlled candidate ranking/reranking reliability under same-schema evaluation.
- **Methods**: C-CRP v3 (main), SRPD (ablation/supplementary)
- **Baselines**: 8 official external (ELMRec, IRLLRec, LLM2Rec, LLMEmb, LLMESR, ProEx, ProMax, RLMRec, SetRec)
- **Domains**: beauty, books, electronics, movies (original 4) + sports, toys, home, tools (new 4)
- **Server**: `pony-rec-gpu` (SSH key-based auth, directly accessible)
  - SSH: `ssh pony-rec-gpu` or `ssh -p 15302 ajifang@125.71.97.70`
  - GPU: RTX 4090 (49GB VRAM)
  - Server project: `~/projects/pony-rec-rescue-shadow-v6`
  - Local project: `D:\Research\Uncertainty`

## Project Goal（硬性目标）

1. 必须达到 SOTA（C-CRP v3 在多个域超越 8 个 official baselines）
2. 创新非缝合（原创方法，有理论动机）
3. 禁止 toy 化（full-scale: 10k 用户、101 候选）
4. 公平比较（setting 完全对齐：用户数、候选数、指标 @5/@10/@20、Qwen3-8B）
5. GPT-5.5 review 达到 8/10（ARIS 审核标准）
6. 实验做完再写（不能半成品提交 review）
7. 每步 commit + memory
8. 2 小时监控实验进度
9. 服务器跑实验，本地不跑

## Experiment Roadmap（2026-05-31）

```text
Phase 1: C-CRP v3 on 8 domains ← IN PROGRESS (6/8 done)
Phase 2: 8 official baselines on 4 new domains (scripts/run_baselines_new_domains.sh)
Phase 3: Full comparison table + statistical significance tests
Phase 4: Paper writing (ARIS paper-write skill)
Phase 5: GPT-5.5/Codex review cycle (must reach 8/10)
```

### Current Results (C-CRP v3)

| Domain | HR@10 | NDCG@10 | Status |
|--------|-------|---------|--------|
| books | **0.476** | **0.333** | SOTA |
| electronics | **0.299** | **0.183** | SOTA |
| toys | 0.396 | 0.271 | baselines pending |
| sports | 0.382 | 0.233 | baselines pending |
| beauty | 0.229 | 0.134 | #2 |
| movies | 0.208 | 0.128 | #5 |
| home | — | — | running |
| tools | — | — | queued |

## Start Every Task

### ⚠️ 第一步：检查服务器是否有实验在跑

**每次接手任务，必须先检查服务器状态。不检查就操作 = 违规。**

```bash
ssh pony-rec-gpu "ps aux | grep python | grep -v grep | grep -i 'pony-rec\|ccrp\|baseline\|uncertainty'"
ssh pony-rec-gpu "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
ssh pony-rec-gpu "grep 'DONE:\|FAILED:\|C-CRP v3:' ~/projects/pony-rec-rescue-shadow-v6/ccrp_v3_all_domains.log 2>/dev/null"
```

**严禁：** 不检查就启动新实验、kill 正在运行的进程、重跑已完成的实验。
**正确做法：** 确认 GPU 空闲 → 检查 outputs/ 里哪些域已有 report.json → 再决定下一步。

1. Read `AGENTS.md` and `README.md`
2. Read `docs/milestones/README.md` for current milestone position
3. Read `docs/paper_claims_and_status.md` for frozen claims
4. Read `docs/top_conference_review_gate.md` for reviewer gate
5. Read `docs/server_runbook.md` for server execution patterns

For complex tasks, run an `rg` discovery pass for the subsystem. Cover project route, task plan, implementation details, and execution/evidence rules.

## Route By Task

| Task Type | Extra Files to Read |
|-----------|-------------------|
| Claims, paper text, tables | `docs/paper_claims_and_status.md`, `docs/top_conference_review_gate.md`, relevant `docs/milestones/M*.md` |
| Official external baselines | `docs/baseline_protocol.md`, `docs/archive/legacy_root_reports/OFFICIAL_EXTERNAL_BASELINE_UPGRADE_PLAN_2026-05-07.md`, `configs/official_external_baselines.yaml` |
| C-CRP / Shadow / uncertainty | `docs/shadow_method.md`, `configs/shadow/*.yaml`, candidate-score importer/exporter scripts |
| SRPD / LoRA training | `configs/srpd/`, `configs/lora/`, `docs/server_runbook.md` (leakage, teacher, weighted-loss gates) |
| Server / long-running experiments | `docs/server_runbook.md` — do not guess server state |
| Reviews / audits | Lead with risks, table eligibility, overclaiming, fairness, leakage, reproducibility |

## Evidence Rules (Summary)

- Keep the defended claim narrow. Do not expand into full-catalog SOTA, generative-title, LoRA distillation, or universal winner claims.
- Main-table rows require: `completed_result`, valid status label, same train/valid/test discipline, exact same-candidate score coverage, finite scores, provenance, candidate audit, paired tests.
- Official external baselines require: pinned official repo/commit, preserved official algorithm/loss/scoring head, Qwen3-8B policy provenance, `implementation_status=official_completed`, `blockers=[]`, exact score coverage, import, paired-test gates.
- `style_adapter_only`, `partial_official_adapter_exists`, scaffold scorers = supplementary/blocked, NOT main-table.
- When in doubt, downgrade the claim, not the evidence standard.

## Server Workflow

Server is directly accessible via SSH. Run commands with:
```bash
ssh pony-rec-gpu "<command>"
```

Useful diagnostic commands:
```bash
ssh pony-rec-gpu "nvidia-smi"
ssh pony-rec-gpu "ps aux | grep python | grep -v grep"
ssh pony-rec-gpu "git -C ~/projects/pony-rec-rescue-shadow-v6 status --short --branch"
ssh pony-rec-gpu "tail -n 80 <LOG>"
ssh pony-rec-gpu "wc -l <output_file>"
```

Do not guess server state — always verify with a command.
For storage-heavy baselines: one method-domain production loop. Run → verify → import → package → confirm local copy → clean → next.

## Codex Role in This Project

- **Primary**: Task decomposition, parallel execution, server command generation, wiki/doc maintenance, comparison table builders
- **Coordination**: Dispatch to Opus for architecture/security review, Sonnet for code review, OpenCode for long-running analysis
- **Discipline**: Run readiness checks before pushing. Update canonical docs when plan/status/commands change. Stage only related files.

## Change Discipline

- Use `rg` and existing patterns before editing
- Keep changes scoped to the requested subsystem
- Do not push `outputs/`, raw logs, datasets, model weights, checkpoints, private keys
- Update canonical docs when plan, status, command surface, or claim boundary changes
- Run `python -m scripts.audit.main_project_readiness_check` after meaningful changes

## Final Handoff

End substantial work with:
- What changed or was learned
- Files inspected or changed
- Claim status and table eligibility
- Blockers or missing evidence
- Next server command, code/doc action, audit, or stopping condition

When enough gates are complete, say the experiment phase is closed and the project should move to writing. When gates remain, name the minimum remaining gates, not an open-ended wishlist.

## Memory Checklist (Codex 每次任务后必做)

**不需要判断"是否重要"。做完事 → 写 memory → 再汇报。**

1. 跑完命令/实验 → `memory_save(type=workflow)` 写结果数字+配置+耗时
2. 遇到报错 → `memory_lesson_save()` 写错误+根因+修复
3. 改了文件/配置 → 更新项目 docs (milestones/README.md 或对应文件)
4. 做了选择 → 写 markdown `decisions/<topic>.md`
5. 不确定要不要写 → **写。宁多勿少。**

**口诀：每一步都写，不是做完所有事再统一写。**

## Git & 备份 Checklist

1. 本地改了文件 → commit + push 到 GitHub（不要攒着）
2. 服务器不 push → 只 pull + 执行
3. 服务器跑出重要结果 → scp 回本地备份（report.json、metrics、关键 log）
4. 不可再生的数据 → 必须有本地副本
