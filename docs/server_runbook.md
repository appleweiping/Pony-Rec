# Server Runbook

This file is the stable entry point for server-side work. It avoids relying on
chat memory.

## Always Start Here

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
git pull --ff-only
python main_project_readiness_check.py
```

Read these files before launching heavy work:

```text
AGENTS.md
docs/milestones/README.md
docs/top_conference_review_gate.md
docs/archive/legacy_root_reports/CODEX_HANDOFF_WEEK8_2026-05-06.md
docs/archive/legacy_root_reports/WEEK8_LARGE_SCALE_10K_100NEG_PLAN_2026-05-06.md
OFFICIAL_EXTERNAL_BASELINE_UPGRADE_PLAN_2026-05-07.md
```

The agent normally cannot see this server. Do not assume server state from
local files. Paste back command outputs when something is run, especially logs,
PIDs, audit summaries, and missing-file errors.

## Current Priority Order

```text
1. Pull latest repo state.
2. Run the readiness check and confirm canonical docs are present.
3. Confirm no stale nohup job is still running.
4. Finish or restart the Week8 four-domain 100neg paper-style supplementary run
   only if that is the intended diagnostic.
5. Audit official external repositories.
6. Implement official adapters in order: LLM2Rec, LLM-ESR, LLMEmb, RLMRec,
   IRLLRec, SETRec.
7. Run Shadow large-scale diagnostics only after the 100neg task packages are
   confirmed healthy.
8. Build Signal/Decision/Generative LoRA artifacts only after teacher data and
   validation gates exist.
```

## Safe Nohup Pattern

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
mkdir -p outputs/summary/logs
LOG=outputs/summary/logs/week8_fourdomain_100neg_full_external_$(date +%F_%H%M%S).log
nohup bash scripts/run_week8_large_scale_10k_100neg.sh > "$LOG" 2>&1 &
echo $! > outputs/summary/logs/week8_fourdomain_100neg_full_external.pid
echo "$LOG"
```

Monitor:

```bash
tail -f "$LOG"
ps -p $(cat outputs/summary/logs/week8_fourdomain_100neg_full_external.pid) -o pid=,etime=,cmd=
```

Paste-back template for the user:

```bash
git status --short --branch
echo "LOG=$LOG"
tail -n 80 "$LOG"
ps -p $(cat outputs/summary/logs/week8_fourdomain_100neg_full_external.pid) -o pid=,etime=,cmd=
ls -lh outputs/summary outputs/baselines 2>/dev/null | head -80
```

If a command fails, paste the full traceback plus the command that produced it.
The next agent should patch locally, push to GitHub, and give a `git pull
--ff-only` recovery command.

Stop:

```bash
kill $(cat outputs/summary/logs/week8_fourdomain_100neg_full_external.pid)
```

## Official Baseline Audit

```bash
python main_audit_official_external_repos.py
python main_audit_official_fairness_policy.py
python main_make_official_external_adapter_plan.py
```

The default adapter plan is `inspect` stage. It writes provenance and blocker
reports for all 24 domain/method rows, but it does not import scores. This is
intentional: rows must not become `official_completed` until a method adapter
actually calls the pinned official implementation and emits exact candidate
scores.

Use run stage only after the target method adapter is implemented:

```bash
python main_make_official_external_adapter_plan.py --plan_stage run
```

Do not import official-baseline rows into main comparison tables until:

```text
comparison variant recorded
implementation_status=official_completed for main-table official rows
official repo commit pinned
official training/scoring entrypoint recorded
Qwen3-8B base path recorded
method-declared adaptation mode recorded
baseline hyperparameter source and overrides recorded
method-specific adapter/checkpoint path recorded
source_event_id,user_id,item_id,score score file emitted
exact score-key coverage verified
finite numeric scores verified
paired-test inputs generated
```

The unified runner entry point is:

```bash
python main_run_official_same_candidate_adapter.py \
  --method llm2rec \
  --stage inspect \
  --domain books \
  --task_dir outputs/baselines/external_tasks/books_large10000_100neg_test_same_candidate \
  --output_scores_path outputs/baselines/official_adapters/books_large10000_100neg_llm2rec_official/llm2rec_official_scores.csv \
  --provenance_output_path outputs/baselines/official_adapters/books_large10000_100neg_llm2rec_official/fairness_provenance.json \
  --fairness_policy_id official_code_qwen3base_default_hparams_declared_adaptation_v1 \
  --comparison_variant official_code_qwen3base_default_hparams_declared_adaptation \
  --backbone_path /home/ajifang/models/Qwen/Qwen3-8B \
  --allow_blocked_exit_zero
```

## LLM2Rec Single-Domain Production Loop

LLM2Rec is the first official-code-level adapter wired for execution. Its run
stage is not a toy scorer: it exports the same-candidate task into LLM2Rec's
native `data/<alias>/downstream` layout, patches the pinned official checkout's
dataset maps, generates Qwen3-8B item text embeddings, invokes the official
`evaluate_with_seqrec.py` / `seqrec.runner.Runner` SASRec training path, then
exports exact same-candidate scores with the shared schema.

For large domains, the official LLM2Rec SASRec state dict includes a duplicate
copy of the precomputed item embedding table. The runner compacts the checkpoint
after official training by removing that duplicated `item_embedding.weight` and
records the operation in provenance; scoring injects the same external Qwen3
`.npy` table before loading the model. Pass `--llm2rec_keep_full_checkpoint`
only when you intentionally want the much larger original checkpoint.

Large domains are storage-heavy. The default production policy is one domain at
a time:

```text
run one domain
-> verify implementation_status=official_completed, blockers=[], audit_ok=True
-> package the evidence artifact
-> copy it to local storage with scp
-> verify the local archive exists
-> delete only documented server-side intermediates
-> start the next domain
```

The completed domain package should include the score CSV, fairness provenance,
score audit, run summary, compact checkpoint or checkpoint manifest, Qwen3 item
embedding metadata/path/digest, and the command/log needed to reproduce the run.

Single-domain command template:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
DOMAIN=books
EXP=books_large10000_100neg
mkdir -p outputs/summary/logs
LOG=outputs/summary/logs/week8_llm2rec_official_${DOMAIN}_$(date +%F_%H%M%S).log
PID=outputs/summary/logs/week8_llm2rec_official_${DOMAIN}.pid
nohup python main_run_llm2rec_official_same_candidate_adapter.py \
  --stage run \
  --domain "$DOMAIN" \
  --task_dir "outputs/baselines/external_tasks/${EXP}_test_same_candidate" \
  --valid_task_dir "outputs/baselines/external_tasks/${EXP}_valid_same_candidate" \
  --output_scores_path "outputs/baselines/official_adapters/${EXP}_llm2rec_official/llm2rec_official_scores.csv" \
  --provenance_output_path "outputs/baselines/official_adapters/${EXP}_llm2rec_official/fairness_provenance.json" \
  --fairness_policy_id official_code_qwen3base_default_hparams_declared_adaptation_v1 \
  --comparison_variant official_code_qwen3base_default_hparams_declared_adaptation \
  --backbone_path /home/ajifang/models/Qwen/Qwen3-8B \
  --llm_adaptation_mode frozen_base_embedding \
  --hparam_policy official_default_or_recommended \
  --embedding_backend hf_mean_pool \
  --embedding_max_length 128 \
  --hf_device_map auto > "$LOG" 2>&1 &
echo $! > "$PID"
disown
echo "log=$LOG"
echo "pid_file=$PID"
```

Monitor:

```bash
tail -f "$LOG"
ps -p $(cat "$PID") -o pid=,etime=,stat=,cmd=
```

Package after the domain completes:

```bash
DOMAIN=books
EXP=books_large10000_100neg
STAMP=$(date +%F_%H%M%S)
mkdir -p outputs/exports
tar -czf "outputs/exports/llm2rec_${DOMAIN}_official_qwen3base_${STAMP}.tar.gz" \
  "outputs/baselines/official_adapters/${EXP}_llm2rec_official" \
  "outputs/baselines/paper_adapters/${EXP}_llm2rec_official_adapter" \
  "/home/ajifang/projects/LLM2Rec/item_info/BooksLarge10000_100Neg"
sha256sum "outputs/exports/llm2rec_${DOMAIN}_official_qwen3base_${STAMP}.tar.gz"
```

Copy that archive from the local machine, then confirm the local file exists
before deleting server intermediates:

```powershell
scp pony-rec-gpu:~/projects/pony-rec-rescue-shadow-v6/outputs/exports/llm2rec_books_official_qwen3base_<STAMP>.tar.gz .
Get-Item .\llm2rec_books_official_qwen3base_<STAMP>.tar.gz
```

Only after local confirmation, clean the completed domain on the server:

```bash
DOMAIN=books
EXP=books_large10000_100neg
rm -rf "outputs/baselines/official_adapters/${EXP}_llm2rec_official"
rm -rf "outputs/baselines/paper_adapters/${EXP}_llm2rec_official_adapter"
rm -rf /home/ajifang/projects/LLM2Rec/item_info/BooksLarge10000_100Neg
df -h /
```

Do not delete final scores, provenance, audits, compact checkpoints, or Qwen3
embedding artifacts before the archive has been copied off the server and
confirmed by the user.

## LLM2Rec Four-Domain Convenience Wrapper

The four-domain wrapper is not the default production path on storage-limited
servers. Use it only when disk space is sufficient and the user explicitly wants
one batch job:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
mkdir -p outputs/summary/logs
LOG=outputs/summary/logs/week8_llm2rec_official_fourdomain_$(date +%F_%H%M%S).log
PID=outputs/summary/logs/week8_llm2rec_official_fourdomain.pid
nohup bash scripts/run_week8_llm2rec_official_fourdomain.sh > "$LOG" 2>&1 &
echo $! > "$PID"
disown
echo "log=$LOG"
echo "pid_file=$PID"
```

Only after each domain writes `implementation_status=official_completed`,
`blockers=[]`, and the score audit prints `audit_ok=True`, import its row into
the same-candidate summary table. Failed domains can be rerun after fixing the
reported blocker; the adapter package and embeddings are deterministic and can
be reused unless `--force_embeddings` is passed.

## Output Interpretation

- `*_style_*` rows are paper-style supplementary diagnostics.
- `*_official_qwen3base_*` rows are the target official external-baseline
  family.
- Full fine-tuning and retuned-baseline variants are supplementary/sensitivity
  rows unless a new experiment-wide policy is explicitly declared.
- Beauty is supplementary smaller-N unless the eligible user count reaches the
  main-domain target.
- Week7.7 compact six-candidate results and Week8 101-candidate results must
  not be mixed as direct row-level comparisons.

## Legacy Entry Points

These remain in the tree for history and compatibility, but they are not the
preferred first-read files:

- `docs/archive/legacy_root_reports/CODEX_HANDOFF_WEEK8_2026-05-06.md`
- `docs/archive/legacy_root_reports/WEEK8_FUTURE_FRAMEWORK_ROADMAP_2026-05-06.md`
- `docs/archive/legacy_root_reports/WEEK8_LARGE_SCALE_10K_100NEG_PLAN_2026-05-06.md`
- `docs/archive/legacy_root_reports/WEEK8_FUSION_EXTERNAL_ONLY_CONTRIBUTION_UPDATE_2026-05-06.md`
- `docs/archive/legacy_root_reports/WEEK8_OURS_EXTERNAL_COMBO_AND_EXTERNAL_ONLY_PLAN_2026-05-06.md`

Use them only when you need historical detail for that specific stage.

## GitHub Push Convention

After code/config/doc changes that affect server commands or project status:

```bash
python main_project_readiness_check.py
python main_audit_official_fairness_policy.py
git status --short
git add <related files only>
git commit -m "<milestone/status message>"
git push origin main
```

Do not push bulk `outputs/`, raw logs, model weights, local datasets, or keys.
Push source, configs, provenance schemas, manifests, and concise docs. If a
server artifact is too large or ignored by git, record its path and regeneration
command in the final answer or runbook.
