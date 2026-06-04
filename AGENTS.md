# Agent Operating Contract

This file is the first-read contract for future Codex/agent work in this
repository. It exists because chat memory is not a research artifact.

## Start Here Every Time

Before changing code, documents, experiments, or claims, read:

1. `README.md`
2. `docs/milestones/README.md`
3. `docs/active_todo_pony_uncertainty.md`
4. `docs/paper_claims_and_status.md`
5. `docs/top_conference_review_gate.md`
6. `docs/server_runbook.md`

For baseline or experiment implementation, also read:

1. `docs/baseline_protocol.md`
2. `OFFICIAL_EXTERNAL_BASELINE_UPGRADE_PLAN_2026-05-07.md`
3. `configs/official_external_baselines.yaml`
4. `PROJECT_LINEAGE_AND_FILE_INDEX_2026-05-06.md`

Legacy Week8 handoff/roadmap files are historical context only. Do not start
from them, and do not treat their roadmap items as completed evidence.

For any complex task, the agent must also do a short discovery pass before
acting: use `rg` to find the most relevant local docs, configs, scripts, and
prior summaries for the requested task, then decide which extra files to read.
The default extra reads should cover four needs: project route, task plan,
implementation details, and execution/evidence rules. Do not rely only on the
fixed first-read list when the task clearly touches a narrower subsystem.

## ⚠️ 接手前必做：检查服务器运行状态

**每次新会话/新 agent 接手本项目，第一件事必须检查服务器上是否有正在运行的实验进程。**

```bash
ssh pony-rec-gpu "ps aux | grep python | grep -v grep | grep -i 'pony-rec\|ccrp\|baseline\|uncertainty'"
ssh pony-rec-gpu "tail -5 ~/projects/pony-rec-rescue-shadow-v6/ccrp_v3_all_domains.log 2>/dev/null"
ssh pony-rec-gpu "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
```

**严禁：**
- 在不检查的情况下启动新实验（会 OOM 或覆盖正在写入的文件）
- kill 掉正在运行的进程（除非明确知道它已经卡死）
- 重跑已经在跑或已经完成的实验（浪费 GPU 时间）

**正确做法：**
1. 先跑上面的命令确认 GPU 是否空闲
2. 检查 `outputs/` 目录看哪些域已经有 `scores.csv` 或 `report.json`
3. 如果有进程在跑，记录 PID 和进度，等它跑完再做下一步
4. 如果不确定进程状态，看日志而不是盲目重启

## 断点续接：如何确定当前状态

每次新会话开始，按以下步骤确定项目处于哪个阶段，从哪里继续：

### Step 1: 读取 Roadmap 确定当前 Phase

查看本文件的 "Experiment Roadmap" 部分，确定当前在哪个 Phase。

### Step 2: 检查服务器实际完成情况

```bash
# 查看哪些域的 C-CRP v3 已完成（旧四域和新四域路径不同）
ssh pony-rec-gpu "cd ~/projects/pony-rec-rescue-shadow-v6 && for d in beauty books electronics movies; do echo -n \"\$d: \"; test -s outputs/ccrp_v3_formal/\$d/report.json && echo DONE || echo NOT_DONE; done; for d in sports toys home tools; do echo -n \"\$d: \"; test -s outputs/\${d}_large10000_100neg_ccrp_v3/report.json && echo DONE || echo NOT_DONE; done"

# 查看哪些新域的 official baselines 已完成（scores.csv 不是充分条件；
# 还必须有 fairness_provenance.json、score audit、imported metrics table）
ssh pony-rec-gpu "for d in sports toys home tools; do echo \"=== \$d ===\"; find ~/projects/pony-rec-rescue-shadow-v6/outputs -maxdepth 2 -type f -path \"*/\${d}_large10000_100neg_*official*same_candidate/fairness_provenance.json\" 2>/dev/null | wc -l; done"

# 查看正在运行的进程
ssh pony-rec-gpu "ps aux | grep python | grep -v grep | grep -i 'pony-rec\|ccrp\|baseline'"
```

### Step 3: 确定下一步动作

| 如果状态是... | 则下一步是... |
|--------------|-------------|
| C-CRP v3 还有域在跑 | 等待，监控进度 |
| C-CRP v3 全部完成，baselines 未开始 | 用单域生产循环启动 `scripts/run_baselines_new_domains.sh`，例如 `DOMAINS_OVERRIDE=sports bash scripts/run_baselines_new_domains.sh` |
| Baselines 正在跑 | 等待，监控进度 |
| Baselines 全部完成 | scp 轻量产物到本地 → 构建对比表 → 统计检验 |
| 对比表和统计检验完成 | 开始论文写作（ARIS paper-write skill） |
| 论文初稿完成 | 提交 GPT-5.5/Codex review（目标 8/10） |
| Review 返回修改意见 | 修改 → 重跑必要实验 → 再提交 review |

### Step 4: 同步本地仓库

```bash
cd D:\Research\Uncertainty
git pull origin main
```

确认本地文档和服务器状态一致后再开始工作。

## Project Direction

The project is not a toy demo and not a loose collection of scripts. The
research spine is:

```text
Week1-4 / pony12 observation
-> Pony framework
-> Light series boundary tests
-> Shadow task-grounded uncertainty
-> same-candidate baseline system
-> small-domain to four-domain 100neg validation
-> complete recommendation-system roadmap
```

The defended paper claim is narrower than the full roadmap:

```text
Task-grounded calibrated uncertainty improves controlled candidate ranking /
reranking reliability under same-schema evaluation.
```

Do not silently expand this into a full-catalog recommender SOTA claim, a LoRA
distillation claim, or a generative-title recommender claim. Those are future
modules unless their own protocol, baselines, audits, and claims are completed.

## Senior Baseline Advice

The adopted main comparison policy follows the senior advice:

```text
official or official-code-level implementations
+ our same-candidate dataset/protocol
+ unified Qwen3-8B base model
+ baseline official/default or recommended hyperparameters
+ our method validation-selected or pre-fixed hyperparameters
+ no test-set model or hyperparameter selection
```

This is a normal academic fairness setting. We do not need to fully retune every
baseline for the primary table, but we must state the policy cleanly. If a
baseline-retuned variant is run, label it as sensitivity/robustness, not as the
primary default-hyperparameter table.

LoRA/full fine-tuning policy must be explicit. The primary official external
baseline variant uses Qwen3-8B base plus the method-declared adapter,
identifier, representation learner, graph/intent module, or downstream
checkpoint required by that baseline's official algorithm. Full fine-tuning,
original-backbone, and retuned-baseline rows are supplementary/sensitivity
variants unless a new experiment-wide policy is declared.

Senior-recommended paper projects and top-conference repositories may be read
carefully as protocol and baseline-design references, but they are not a parts
library for our method. It is acceptable to borrow high-level evidence
discipline: pinned official implementations, same-candidate score export,
provenance fields, default-hyperparameter policy, validation-only selection,
and paired-test gates. Do not stitch together, rename, or silently transplant
another paper's architecture, loss, identifier system, intent module, graph
objective, adapter, scoring head, or training recipe into C-CRP or SRPD. If an
external method is run, it remains an external baseline with its own provenance
and status label; if our method changes, the change must be motivated by our
uncertainty claim and ablated under the same protocol.

## Non-Toy Experiment Standard

Do not replace hard baseline work with toy shortcuts. A result is not
paper-facing unless it has:

- real task data, not synthetic placeholders;
- the same train/valid/test discipline used by the claim;
- the same candidate rows and metric importer for comparable rows;
- score/provenance coverage checks;
- paired tests or a clear statement that it is diagnostic only;
- status labels that match the evidence level.

Small smoke tests are useful for code health, but they are never main evidence.

## Official External Baseline Guardrails

1. An `official` filename, wrapper, plan row, or `--plan_stage run` is not
   official evidence. Only provenance with `implementation_status=official_completed`,
   `blockers=[]`, and passing score/provenance/import/paired-test gates can be
   called an official baseline result.
2. Current `main_run_*_official_same_candidate_adapter.py` files are thin
   wrappers. If the unified runner writes `official_blocked`,
   `runner_support_level=inspect_scaffold`, or blocker
   `run_stage_not_implemented_for_method`, do not import the row as
   `completed_result`, do not put it in a main table, and do not call it an
   official reproduction.
3. `style_adapter_only` and `partial_official_adapter_exists` are not official
   main-table rows. `*_style_*`, scaffold scorers, hash/centroid embeddings, and
   adapter smoke-test scores are supplementary/pilot/protocol checks only.
4. Official rows must come from the pinned repo/commit in
   `configs/official_external_baselines.yaml`. Missing repos, missing commits,
   dirty/unverified checkouts, and commit mismatches are blockers.
5. Preserve the official architecture, loss/objective, train/eval entrypoint,
   adapter/representation/graph/intent/identifier module, and scoring head. The
   allowed changes are data adapters, Qwen3-8B representation bridge, and exact
   same-candidate score export.
6. Provenance is not "there is a file." Required fields must be filled or
   explicitly marked `none` with a reason: official entrypoint, repo commit,
   default hparam source, overrides/reason, adaptation mode,
   adapter/checkpoint path or not-applicable reason, task sources, score path,
   score sha256, and score coverage.
7. The plan generator is not a truth source. `implementation_status` passed on
   the CLI cannot override blocked provenance.
8. Score gates must pass before import: exact `source_event_id,user_id,item_id`
   key match to `candidate_items.csv`, no missing/extra/duplicate keys, finite
   numeric scores, and `score_coverage_rate=1.0`. Candidate order, row index,
   constant scores, or tie-only scaffolds cannot be method scores.
9. Full-catalog metrics from official repos do not enter same-candidate main
   tables. Use official repos to produce method-native candidate scores; compute
   final metrics through `main_import_same_candidate_baseline_scores.py`.
10. If documents, runner output, score files, and provenance conflict, use the
    most conservative status: blocked > partial > style/scaffold >
    official_completed.

## Multi-Agent Collaboration

Use multi-agent work for broad or complex research/engineering tasks by
default. A task is complex if it touches claims, baselines, experiments,
server runs, paper readiness, method design, major refactors, or any result
that could affect a table. At minimum:

- one implementation/engineering agent;
- one literature/protocol scout when claims or baselines are involved;
- one reviewer/auditor agent for overclaim, fairness, and table eligibility.

The literature/protocol scout should compare this project against multiple
top-conference papers or official projects, not just one or two convenient
examples. It may read senior-recommended projects and other RecSys/SIGIR/WSDM/
KDD/NeurIPS-style work carefully, but only for rigor, protocol, and design
expectations; it must not recommend stitching or copying another method into
ours.

The reviewer/auditor should behave like a top-conference reviewer. It should
challenge rigor, novelty, technical depth, ablation completeness, baseline
coverage, leakage risk, table eligibility, and whether the claim is still too
broad. Reviewer objections remain blockers until addressed, downgraded, or
explicitly documented as accepted limitations.

Each sub-agent handoff should report:

- milestone touched;
- files inspected or changed;
- claim status and table eligibility;
- blockers;
- server commands needed;
- whether results can enter main, supplementary, or diagnostic tables.

Reviewer/auditor findings can veto wording and table inclusion. Do not average
away a serious reviewer objection.

Current tool-routing note (2026-06-03): Claude reviewer tooling is unavailable
in this thread. When multi-agent review/collaboration is required, use
GPT-5.5 xhigh sub-agents instead of Claude reviewer sessions.

Every substantial final report must include both what changed or was learned
and a concrete next-step plan. The plan must name the next server command,
code/doc action, audit, or stopping condition. If no further experiment is
needed, say so and explain why.

Agents must keep an explicit project/experiment endgame. Do not create an
infinite chain of "next steps." When results, baselines, ablations, audits,
paired tests, provenance, and reviewer objections are complete enough for the
current defended claim, tell the user that the experiment phase is basically
closed and the project should move to writing. If important gates remain,
state the minimum remaining gates, not an open-ended wishlist.

## Server Collaboration

Server `pony-rec-gpu` is now directly accessible via SSH key-based auth:
```bash
ssh pony-rec-gpu "<command>"
```
- Host: `125.71.97.70:15302`, User: `ajifang`
- GPU: NVIDIA RTX 4090 (49GB VRAM)
- Server project path: `~/projects/pony-rec-rescue-shadow-v6`
- Local project path: `D:\Research\Uncertainty`

Agents can directly run commands, check logs, monitor GPU, and manage files.
Do not guess server state — run a command to verify before claiming status.

Useful diagnostic commands:
```bash
ssh pony-rec-gpu "git -C ~/projects/pony-rec-rescue-shadow-v6 status --short --branch"
ssh pony-rec-gpu "nvidia-smi"
ssh pony-rec-gpu "ps aux | grep python | grep -v grep"
ssh pony-rec-gpu "tail -n 80 <LOG>"
ssh pony-rec-gpu "wc -l <output_file>"
```

If a long job is needed, use `nohup` or the runbook pattern. Record the log path
and PID path. The user will paste logs or errors; update code/docs and push
fixes as needed.

For storage-heavy official baselines, especially LLM2Rec/LLM-ESR on large
domains, default to a single-domain production loop rather than launching every
domain at once. The loop is: run one method-domain row, verify unblocked
provenance and exact score coverage, import multi-k metrics, package a light
evidence artifact, have the user copy it off the server and confirm it exists
locally, then clean only the documented intermediate files before starting the
next domain. The light evidence package should mirror the completed LLM2Rec
large-domain artifacts: score CSV, fairness provenance, score audit, run
summary, training/server log, imported prediction/table outputs, comparison
summary tables, and checkpoint/embedding sha256 manifests. Do not make a huge
checkpoint tarball by default; record the model/checkpoint hash and keep or
delete the checkpoint only according to the documented storage decision. Do not
recommend deleting final scores, provenance, audits, compact checkpoints,
external embedding artifacts, or method checkpoints unless the corresponding
evidence archive has been copied off the server and the user confirms the
archive. Do not delete imported `outputs/*_same_candidate/tables/` summaries
needed by comparison builders until the method-level and final cross-baseline
tables have been rebuilt and archived; otherwise completed domains will drop
out of later comparison tables.
For disk emergencies after a completed row has passed `server_final` audit and
local-light sync, `predictions/rank_predictions.jsonl` may be removed only with
an exact sha256 manifest and a doc/memory note. The domain gate and comparison
builder accept a missing prediction JSONL only when
`server_final_evidence_audit.json` proves the file existed with the expected
line count. Keep `scores.csv`, provenance, score audits, imported `tables/`,
models/checkpoints, and local packages protected unless a separate archive
decision explicitly says otherwise.

Use the completed LLM2Rec official four-domain run as the template for future
official external baselines. A baseline is not complete when one domain finishes
or when a raw score file exists. It is complete only after every declared domain
has `implementation_status=official_completed`, `blockers=[]`, exact score
coverage, local evidence backup, cleaned server intermediates, imported
same-candidate summaries, and a four-domain summary table. If a method's run
stage still returns `run_stage_not_implemented_for_method`, stop and implement
the official-code adapter first; do not substitute style/scaffold rows.

For our formal methods, C-CRP and SRPD must also pass same-candidate gates
before any table claim. C-CRP is the main task-grounded uncertainty method:
select score mode, weights, eta, and ablations on validation only, export exact
`source_event_id,user_id,item_id,score` rows, then import as
`same_schema_internal_method`. SRPD is a trainable framework/ablation line:
teacher data must not be derived from final test events, leakage audits must
pass, sample weights must enter the loss when claimed, and rank-order fallback
scores must be labeled as internal ablation evidence rather than external
baseline-equivalent native scores.

## GitHub And Documentation Hygiene

After meaningful code/config/doc changes:

1. run the relevant smoke/readiness checks;
2. update canonical docs if the plan, status, or command surface changed;
3. stage only related files;
4. commit with a clear milestone/status message;
5. push to GitHub.

Do not push `outputs/` bulk artifacts, private keys, raw logs, model weights, or
large local data. Push code, configs, manifests, provenance schema, and concise
docs. If generated summaries are important but ignored by git, describe the
server path and regeneration command instead.

## Change Discipline

Do real read/modify/verify work. Do not only add Markdown when code/config is
needed, and do not only add code when the claim boundary or server runbook has
changed. Use `rg` first, inspect existing patterns, preserve user changes, and
avoid broad deletions unless they are explicitly requested and audited.

When in doubt, downgrade the claim, not the evidence standard.

## 强制记录规则（所有 agent 必须遵守）

每完成一个阶段、一个 step、一次错误排除、一次贡献，都必须同时更新：
1. agentmemory MCP（共享 memory）
2. 项目自身文档 (docs/paper_claims_and_status.md, docs/milestones/README.md)
3. CLAUDE.md 和 AGENTS.md（如果路线/规则变化）

不是做完所有事再统一更新，而是每一步都更新。不写 = 违规。

实验公平性：和 baseline 对比时，指标 @5/@10/@20、数据、用户数必须完全对齐。

## Project Goal（硬性目标，不可降级）

1. **必须达到 SOTA** — C-CRP v3 在多个域上超越所有 8 个 official baselines
2. **创新非缝合** — 方法必须是原创的、有理论动机的，不是拼凑已有方法
3. **禁止 toy 化** — 所有实验必须是 full-scale（10k 用户、101 候选）
4. **8 个 official baselines** — 每个域都必须有完整的 8 个 baseline 对比
5. **公平比较** — setting 完全对齐（用户数、候选数、指标、统一 Qwen3-8B backbone）
6. **GPT-5.5 review 达到 8/10** — 按 ARIS 审核标准，多维度评价
7. **实验做完再写** — 结果完整后再写作，不能半成品提交 review
8. **每步 commit + memory** — 每个关键产物 commit 到 GitHub，更新文档
9. **目标续跑监控** — 长期项目使用当前线程 heartbeat 每 2 小时激活一次；每次续跑只做一个有界监控周期，禁止刚结束就连续自触发或无脑重复检查
10. **服务器跑实验，本地不跑** — GPU 实验全在服务器，本地只做版本控制和写作
11. **motivation 必做** — 论文 ready 前必须有代表性 observation/motivation study，解释为什么 uncertainty 在该框架中必要，并产出可入文的图或表
12. **组件 ablation 必做** — C-CRP 每个非平凡设计组件都要做 leave-one-component-out 审计；删掉不降反升要诚实报告为弱组件或设计问题
13. **超参曲线必做** — eta、C-CRP weights、confidence weight、uncertainty gates/thresholds、SRPD lr/lambda 等真实控制量需要合理 sweep 和 matplotlib 曲线
14. **框架图必做** — 论文 ready 前必须准备清晰 pipeline/framework overview figure，标出 uncertainty、calibration/evidence 和 risk-adjusted ranking 位置

## Experiment Roadmap（2026-06-04 更新）

```text
Phase 1: C-CRP v3 on 8 domains (sports✓ toys✓ home✓ tools✓)
Phase 2: 8 official baselines on 4 new domains (sports✓ 8/8 + gate✓; toys✓ 8/8 + gate✓; home✓ 8/8 + gate✓ with C-CRP rank 1 on all 7 metrics and 56/56 positive Holm-significant paired tests; tools 5/8 gated✓ with proex_profile, promax_profile, elmrec_graph, llmemb, irllrec_intent; tools rlmrec_graphcl active since 2026-06-05 05:41 CST, runner PID 3347729, adapter PID 3347738, log baselines_new_domains_tools_rlmrec_20260605_054158.log, heartbeat monitor-tools-rlmrec, latest monitor 2026-06-05 06:15 CST at Qwen3 embedding progress 128944/269711 with disk about 12.35G free / 94% used and no final artifacts yet; remaining after active RLMRec are llm2rec_sasrec and llmesr_sasrec; scripts/run_baselines_new_domains.sh; SETRec excluded while blocked; single-domain loop; full @5/@10/@20+MRR import after each score audit)
Phase 2.5: Paper-critical modules before readiness claim: uncertainty observation/motivation figure, full C-CRP component ablations, real hyperparameter curves, framework overview figure
Phase 3: Full comparison table + statistical significance tests
Phase 4: Paper writing (ARIS paper-write skill)
Phase 5: GPT-5.5/Codex review cycle (must reach 8/10)
```

## Artifact Management（产物管理规则）

### Reporting Completeness（汇报完整性）

Pony-rec experiment status reports must not summarize only HR@10/NDCG@10.
For every completed domain/method result, report the full metric set whenever
available: HR@5/@10/@20, NDCG@5/@10/@20, MRR, user count, prompt/candidate
score count, data path, score file row count, rank file row count, and any
FAILED/OOM/Traceback log scan result. If a metric is absent from an older
artifact, mark it as missing rather than silently omitting it.

### 本地保留 + commit 到 GitHub：
- `report.json` — 指标结果
- `user_ranks.jsonl` — 每用户排名（统计检验用）
- `main_comparison_table.csv` — 对比表
- `fairness_provenance.json` — baseline 公平性证据
- `*_score_audit.json` — 分数审计
- `*_run_summary.json` — 运行摘要
- Paper 源文件、scripts、configs、docs

For completed official baseline rows, copy a lightweight but complete local
evidence package before any server cleanup: inspect/final provenance,
JSON/TXT score audits, run summary, imported metric/coverage/exposure summary
tables, and the per-event `tables/ranking_eval_records.csv` needed for paired
or statistical follow-up. Keep huge `scores.csv`, full predictions,
checkpoints, and embedding files server-side unless a separate archive is
explicitly requested; provenance/run summaries must record their hashes.
If disk pressure threatens active progress, a completed row's full prediction
JSONL can be deleted after `server_final` audit + local-light audit PASS, with
sha256 recorded before deletion. This is not allowed for `scores.csv`, imported
`tables/`, provenance, score audits, run summaries, or models/checkpoints
without an explicit separate archive decision.
### 只留服务器（不下载、不提交）：
- `scores.csv`（87MB/域）
- `predictions/`（600MB+）
- `embeddings/`、`checkpoints/`
- 原始数据、模型权重

### 不提交到 GitHub：
- API keys、credentials
- `__pycache__/`、`.pyc`、editor swap files

## Git 工作流

- 本地 (D:esearch\Uncertainty) 是主仓库，所有 commit/push 从本地发起
- 服务器 (pony-rec-gpu) 只是实验场所，跑推理和训练，不做 commit/push
- GitHub 更新用本地提交
- 服务器产出通过 scp 或记录到文档同步回本地
