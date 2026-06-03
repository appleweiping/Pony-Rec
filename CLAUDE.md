# CLAUDE.md

Read `AGENTS.md` first. It is the authoritative operating contract for this repository.

This is the Uncertainty project: Task-Grounded Uncertainty for LLM-based Recommendation.

## Quick Orientation

- **Current collaboration routing**: Claude reviewer tooling is unavailable in
  this thread; when multi-agent review is required, use GPT-5.5 xhigh
  sub-agents instead.
- **Stage**: M5 (C-CRP v3 complete; sports/toys official gates passed; home 5/8 official rows complete; home RLMRec active)
- **Core Claim**: Task-grounded calibrated uncertainty improves controlled candidate ranking/reranking reliability under same-schema evaluation.
- **Methods**: C-CRP v3 (main), SRPD (ablation/supplementary)
- **Baselines**: 8 official external (ELMRec, IRLLRec, LLM2Rec, LLMEmb, LLMESR, ProEx, ProMax, RLMRec). SETRec is blocked/supplementary unless future official gates pass.
- **Domains**: beauty, books, electronics, movies (original 4) + sports, toys, home, tools (new 4)
- **Server**: `pony-rec-gpu` (SSH accessible via `ssh pony-rec-gpu`, key-based auth configured)
  - Server project path: `~/projects/pony-rec-rescue-shadow-v6`
  - Local project path: `D:\Research\Uncertainty`
  - SSH config: `125.71.97.70:15302`, user `ajifang`
  - GPU: RTX 4090 (49GB VRAM)

## Experiment Roadmap (2026-05-31)

### Phase 1: C-CRP v3 Scoring (COMPLETE)
Run C-CRP v3 on all 8 domains with Qwen3-8B via vLLM.
- beauty (973 users): DONE — HR@10=0.229
- books (10k users): DONE — HR@10=0.476 **SOTA**
- electronics (10k users): DONE — HR@10=0.299 **SOTA**
- movies (10k users): DONE — HR@10=0.208
- sports (10k users): DONE — HR@5/10/20=0.275/0.382/0.517, NDCG@5/10/20=0.198/0.233/0.267, MRR=0.208
- toys (10k users): DONE — HR@5/10/20=0.317/0.396/0.506, NDCG@5/10/20=0.245/0.271/0.298, MRR=0.250
- home (10k users): DONE — HR@5/10/20=0.156/0.226/0.351, NDCG@5/10/20=0.110/0.132/0.164, MRR=0.126
- tools (10k users): DONE — HR@5/10/20=0.194/0.270/0.393, NDCG@5/10/20=0.142/0.166/0.197, MRR=0.156

For new-domain C-CRP v3 status reports, always include the full metric set:
HR@5/@10/@20, NDCG@5/@10/@20, MRR, `n_users`, `n_prompts`, data path, and
score/rank row counts. Do not summarize only @10.

Original-domain C-CRP v3 reports are not in the new-domain flat directory
layout. Use `outputs/ccrp_v3_formal/<domain>/report.json` for
beauty/books/electronics/movies and `outputs/ccrp_v3_formal/main_comparison_table.csv`
for their metric-complete 8-baseline comparison. As of the 2026-05-31 artifact
audit, those old-domain baseline metrics exist, but some method-specific
evidence packs still need provenance/audit reconciliation under the current
strict gate.

### Phase 2: Official Baselines on New Domains
Run 8 official baselines on sports/toys/home/tools (same protocol as original 4).
Script: `scripts/run_baselines_new_domains.sh` (already on server).
The script is aligned to the canonical 8-method block and excludes SETRec while
SETRec remains blocked/supplementary in `configs/official_external_baselines.yaml`.
It uses `${PYTHON:-/home/ajifang/miniconda3/bin/python}` to avoid non-interactive
SSH sessions failing on a missing bare `python`.
Use the documented single-domain loop under current storage pressure, e.g.
`DOMAINS_OVERRIDE=sports bash scripts/run_baselines_new_domains.sh`. The runner
audits exact score coverage and imports complete `@5/@10/@20 + MRR` metrics
after each completed score file.
Current status (2026-06-03): sports and toys are each 8/8 official baselines
complete and have passed their domain/comparison/paired-test gates against
C-CRP. Home has 5/8 audited official rows complete: `proex_profile`,
`promax_profile`, `elmrec_graph`, `llmemb`, and `irllrec_intent`. The latest
row, home `irllrec_intent`, completed at 2026-06-03 20:05 CST with
`implementation_status=official_completed`, `blockers=[]`, exact
`score_coverage_rate=1.0`, server-final audit PASS, lightweight sync PASS, and
local-light audit PASS. After the completed IRLLRec intermediate adapter was
removed with cleanup manifest
`outputs/summary/home_irllrec_completed_adapter_cleanup_manifest_20260603.sha256`,
disk recovered to about `12G` free. Home `rlmrec_graphcl` launched at
2026-06-03 20:28 CST as the sixth home row after fresh preflight; runner PID
`3178395`, adapter PID `3178403`, log
`baselines_new_domains_home_rlmrec_20260603_2028.log`. It is not
table-eligible until final score/provenance/import/server-final and local-light
gates pass. At 2026-06-03 21:14 CST it remained active in Qwen3 embedding at
about `176768/385364` with no fatal markers. A bounded cleanup removed only
server-side prediction JSONLs from already gated Sports/Toys official rows,
recorded
`outputs/summary/sports_toys_completed_predictions_deleted_for_home_rlmrec_disk_20260603.sha256`,
and copied post-cleanup Sports/Toys domain gate outputs showing
`gate_ok=true`, `official_ok_count=8`, and `ccrp_ok=true`; disk recovered to
about `19G` free.

Storage note: before the home launch, completed sports/toys LLMEmb and LLM-ESR
upstream staging directories were removed only after final server audits and
local-light packages passed. The cleanup manifest is
`outputs/summary/upstream_completed_sports_toys_llmemb_llmesr_cleanup_manifest_20260602.sha256`;
final official evidence directories and local lightweight packages were
preserved. Disk recovered from about `5.9G` to `17G` free.

### Phase 3: Full Comparison Table + Statistical Tests
Build @5/@10/@20 table across all domains. Paired t-test / bootstrap.

### Phase 2.5 / Paper-Critical Method Evidence
Before claiming paper readiness, complete four method-story gates:

- observation/motivation study showing why uncertainty is needed, using
  representative completed baselines/domains under fair same-candidate settings
  and producing a paper-ready figure or table;
- leave-one-component-out C-CRP ablations over the actual nontrivial
  implementation components, including score mode, boundary uncertainty,
  calibration gap, evidence support/insufficiency, counterevidence, risk
  penalty, eta, confidence weight, and C-CRP weight triples where supported;
- hyperparameter sensitivity curves for real controls such as eta, C-CRP
  weights, confidence weight, uncertainty gates/thresholds, anchor penalties,
  and SRPD learning-rate/lambda controls when SRPD rows are used;
- a clean framework overview figure showing the full pipeline and where
  uncertainty/components enter the ranking decision.

Each finished module needs commands, configs, seeds when applicable, git
commit, row counts, provenance notes, plots/tables, key logs, and a
lightweight local evidence package with server/local manifest comparison.

### Phase 4: Paper Writing (ARIS paper-write skill)

### Phase 5: GPT-5.5/Codex Review Cycle (must reach 8/10)

## Project Goal

本项目的最终目标是产出一篇达到顶会投稿水平的论文。具体要求：

1. **必须达到 SOTA** — C-CRP v3 在多个域上超越所有 8 个 official baselines
2. **创新非缝合** — 方法必须是原创的、有理论动机的，不是拼凑已有方法
3. **禁止 toy 化** — 所有实验必须是 full-scale（10k 用户、101 候选），不能用小样本代替
4. **8 个 official baselines** — 每个域都必须有完整的 8 个 baseline 对比
5. **公平比较** — 所有 setting 对齐（用户数、候选数、指标 @5/@10/@20、统一 backbone）
6. **GPT-5.5 review 达到 8/10** — 按 ARIS 审核标准，多维度评价，总分必须 ≥8
7. **实验做完再写** — 不能实验做一半就提交 review，必须结果完整后再写作
8. **每步 commit + memory** — 每个关键产物都要 commit 到 GitHub，更新项目文档
9. **目标续跑监控** — 长期项目使用当前线程 heartbeat 每 2 小时激活一次；每次续跑只做一个有界监控周期，禁止刚结束就连续自触发或无脑重复检查
10. **服务器跑实验，本地不跑** — GPU 实验全在服务器，本地只做版本控制和写作
11. **motivation/ablation/hyperparameter/figure 四件套必做** — 这些是 paper-ready 前置门禁，不是写作润色项

## Artifact Management Rules

### 本地保留（commit 到 GitHub）：
- `report.json` — 每个域/方法的指标结果
- `user_ranks.jsonl` — 每用户排名（统计检验用）
- `main_comparison_table.csv` — 对比表
- `fairness_provenance.json` — baseline 公平性证据
- `*_score_audit.json` — 分数审计
- `*_run_summary.json` — 运行摘要
- Paper 源文件、scripts、configs、docs

### 只留服务器（不下载、不提交）：
- `scores.csv`（87MB/域，可重新生成）
- `predictions/`（600MB+）
- `embeddings/`、`checkpoints/`
- 原始数据、模型权重

### 不提交到 GitHub：
- API keys、credentials
- `__pycache__/`、`.pyc`、editor swap files
- 上述大文件

## Your Role (Claude Code / Opus / Sonnet)

### ⚠️ 接手前必做：检查服务器运行状态

**每次新会话开始，第一件事检查服务器是否有正在运行的实验。不检查就操作 = 违规。**

```bash
ssh pony-rec-gpu "ps aux | grep python | grep -v grep | grep -i 'pony-rec\|ccrp\|baseline\|uncertainty'"
ssh pony-rec-gpu "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
ssh pony-rec-gpu "tail -5 ~/projects/pony-rec-rescue-shadow-v6/ccrp_v3_all_domains.log 2>/dev/null"
```

**严禁：** 不检查就启动新实验、kill 正在运行的进程、重跑已完成的实验。
**正确做法：** 确认 GPU 空闲 → 检查哪些域已完成 → 再决定下一步。

When invoked by Codex or OpenCode as a reviewer:
- You are **read-only**. Do not edit files, stage, commit, or push.
- Challenge rigor, novelty, technical depth, ablation completeness, baseline coverage, leakage risk, table eligibility.
- Reviewer objections remain blockers until addressed.
- Apply top-conference reviewer standards (RecSys/SIGIR/WSDM/KDD/NeurIPS).

When the user opens Claude Code directly with write scope:
- Follow `AGENTS.md` operating contract fully.
- Update canonical docs after meaningful changes.
- Run readiness checks before pushing.

## Key Docs (Read Order)

1. `AGENTS.md` — full operating contract (280 lines)
2. `docs/milestones/README.md` — M0-M6 milestone map
3. `docs/paper_claims_and_status.md` — frozen claims, status labels
4. `docs/top_conference_review_gate.md` — standing reviewer gate
5. `docs/server_runbook.md` — server execution patterns
6. `configs/official_external_baselines.yaml` — baseline configs

## Evidence Rules (Summary)

- Only `completed_result` rows enter main tables
- Score gates: exact key match, no missing/extra/duplicate, finite scores, coverage=1.0
- Paired statistical tests required for significance claims
- `official_completed` requires: pinned repo/commit, default hparams, full provenance, score coverage=1.0
- When in doubt, downgrade the claim, not the evidence standard

## Multi-Agent System

This project participates in the 6-agent collaboration system:

| Agent | Role in this project |
|-------|---------------------|
| OpenCode (像素猫) | Implementation, analysis, comparison builders, doc updates |
| Codex | Coordination, parallel execution, server command generation |
| Opus | Architecture review, complex reasoning, security/leakage audit |
| Sonnet | Code review, table eligibility checks, second opinion |
| Haiku | Quick lint, format checks |
| DeepSeek (鲸鱼) | Translation, bulk text, Chinese content |

Agent Hub daemon on port 9800 provides real-time collaboration when running.
