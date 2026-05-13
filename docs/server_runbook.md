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
-> import the row with --ks 5,10,20 and rebuild the comparison table
-> package the evidence artifact
-> copy it to local storage with scp
-> verify the local archive exists
-> delete only documented server-side intermediates, not imported summary tables
-> start the next domain
```

This is also the template for future official external baselines. Do not call a
baseline complete after a single raw score file. A method-level official
baseline is complete only after all declared domains have unblocked provenance,
exact score coverage, local evidence backup, server intermediates cleaned, and
same-candidate imports included in a method-level summary table. If the runner
prints `run_stage_not_implemented_for_method`, stop and implement the pinned
official adapter before launching expensive jobs.

For LLM-rec official large domains, use the completed LLM2Rec large-domain
packages as the default archive standard. The lightweight evidence package must
include the score CSV, fairness provenance, score audit, run summary,
training/server log, imported same-candidate predictions/tables, comparison
summary tables, and checkpoint/embedding sha256 manifests when the checkpoint
or embedding file is too large to archive immediately. Do not build a huge
checkpoint tarball by default. Full checkpoints can be archived separately only
when time and storage allow.

Important: comparison tables are rebuilt from imported
`outputs/*_same_candidate/tables/same_candidate_external_baseline_summary.csv`
files. After a domain is imported, keep that `tables/` summary available until
the method-level and final cross-baseline comparison tables have been rebuilt
and archived. If storage is tight, delete predictions or large model/adapter
intermediates first, but do not remove the imported summary directory in a way
that makes a completed domain disappear from later comparisons.

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

Package after the domain completes. Prefer this lightweight evidence package on
large domains:

```bash
DOMAIN=books
EXP=books_large10000_100neg
BASE=llm2rec_official_qwen3base_sasrec
OUTDIR="outputs/baselines/official_adapters/${EXP}_llm2rec_official"
LOG=$(ls -t outputs/summary/logs/week8_llm2rec_official_${DOMAIN}_*.log 2>/dev/null | head -1)
STAMP=$(date +%F_%H%M%S)
mkdir -p outputs/exports

find "$OUTDIR" -maxdepth 3 -type f \( -name "*.pt" -o -name "*.pth" \) -print0 \
  | xargs -0 -r sha256sum > "${OUTDIR}/checkpoint_manifest.sha256"
ADAPTER_DIR="outputs/baselines/paper_adapters/${EXP}_llm2rec_official_adapter"
if [ -d "$ADAPTER_DIR" ]; then
  find "$ADAPTER_DIR" -maxdepth 3 -type f \( -name "*.npy" -o -name "*.pkl" \) -print0 \
    | xargs -0 -r sha256sum > "${OUTDIR}/embedding_manifest.sha256"
else
  echo "missing_adapter_dir $ADAPTER_DIR" > "${OUTDIR}/embedding_manifest.sha256"
fi

FILES=(
  "${OUTDIR}/fairness_provenance.json"
  "${OUTDIR}/llm2rec_official_score_audit.json"
  "${OUTDIR}/llm2rec_official_run_summary.json"
  "${OUTDIR}/llm2rec_official_scores.csv"
  "${OUTDIR}/checkpoint_manifest.sha256"
  "${OUTDIR}/embedding_manifest.sha256"
  "outputs/${EXP}_${BASE}_same_candidate"
  "outputs/summary/week8_official_external_qwen3base_multik_comparison.csv"
  "outputs/summary/week8_official_external_qwen3base_multik_comparison.md"
)
[ -n "$LOG" ] && [ -f "$LOG" ] && FILES+=("$LOG")
for f in \
  "${ADAPTER_DIR}/adapter_metadata.json" \
  "${ADAPTER_DIR}/llm2rec_embedding_metadata.json" \
  "${ADAPTER_DIR}/llm2rec_upstream_prepare_summary.json"
do
  [ -f "$f" ] && FILES+=("$f")
done

tar -czf "outputs/exports/llm2rec_${DOMAIN}_official_qwen3base_evidence_${STAMP}.tar.gz" \
  "${FILES[@]}"
sha256sum "outputs/exports/llm2rec_${DOMAIN}_official_qwen3base_evidence_${STAMP}.tar.gz"
```

Copy that archive from the local machine, then confirm the local file exists
before deleting server intermediates:

```powershell
scp pony-rec-gpu:~/projects/pony-rec-rescue-shadow-v6/outputs/exports/llm2rec_books_official_qwen3base_evidence_<STAMP>.tar.gz .
Get-Item .\llm2rec_books_official_qwen3base_evidence_<STAMP>.tar.gz
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

Do not delete final scores, provenance, audits, compact checkpoints, Qwen3
embedding artifacts, or method checkpoints before the evidence archive has
been copied off the server and confirmed by the user. If the archive check
prints `gzip: unexpected end of file` or `tar: unexpected EOF`, delete only the
bad archive and keep all domain outputs.

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

LLM2Rec official Qwen3-base status as of 2026-05-09:

```text
beauty supplementary smaller-N: completed/imported
books large10000 100neg: completed/imported
electronics large10000 100neg: completed/imported
movies large10000 100neg: completed/imported
summary:
  outputs/summary/week8_llm2rec_official_qwen3base_fourdomain_summary.csv
  outputs/summary/week8_llm2rec_official_qwen3base_fourdomain_summary.md
```

LLM-ESR also has run-stage support wired in the unified runner. It imports the
pinned repo's `models.LLMESR.LLMESR_SASRec` class and preserves the official
SASRec-style architecture/loss/predict path while local code only adapts the
same-candidate handled files, Qwen3 item embeddings, and exact score export.
As with LLM2Rec, it is not a completed result until a server run writes
`implementation_status=official_completed`, `blockers=[]`, and exact score
coverage.

LLM-ESR single-domain smoke/production template:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
DOMAIN=beauty
EXP=beauty_supplementary_smallerN_100neg
mkdir -p outputs/summary/logs
LOG=outputs/summary/logs/week8_llmesr_official_${DOMAIN}_$(date +%F_%H%M%S).log
PID=outputs/summary/logs/week8_llmesr_official_${DOMAIN}.pid
nohup python main_run_llmesr_official_same_candidate_adapter.py \
  --stage run \
  --domain "$DOMAIN" \
  --task_dir "outputs/baselines/external_tasks/${EXP}_test_same_candidate" \
  --valid_task_dir "outputs/baselines/external_tasks/${EXP}_valid_same_candidate" \
  --output_scores_path "outputs/baselines/official_adapters/${EXP}_llmesr_official/llmesr_official_scores.csv" \
  --provenance_output_path "outputs/baselines/official_adapters/${EXP}_llmesr_official/fairness_provenance.json" \
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

LLM-ESR follows the same one-domain archive-and-clean loop as LLM2Rec. Its
large-domain checkpoints can be several GB, so the default archive is a light
evidence package plus a model sha256 manifest. After the run and score audit
complete, import the domain with `--ks 5,10,20`, rebuild
`week8_official_external_qwen3base_multik_comparison`, then package:

```bash
DOMAIN=books
EXP=books_large10000_100neg
BASE=llmesr_official_qwen3base_sasrec
OUTDIR="outputs/baselines/official_adapters/${EXP}_llmesr_official"
LOG=$(ls -t outputs/summary/logs/week8_llmesr_official_${DOMAIN}_*.log 2>/dev/null | head -1)
STAMP=$(date +%F_%H%M%S)
mkdir -p outputs/exports

if [ -f "${OUTDIR}/llmesr_official_model.pt" ]; then
  sha256sum "${OUTDIR}/llmesr_official_model.pt" > "${OUTDIR}/llmesr_official_model.pt.sha256"
else
  echo "missing_model ${OUTDIR}/llmesr_official_model.pt" > "${OUTDIR}/llmesr_official_model.pt.sha256"
fi

FILES=(
  "${OUTDIR}/fairness_provenance.json"
  "${OUTDIR}/llmesr_official_run_summary.json"
  "${OUTDIR}/llmesr_official_score_audit.json"
  "${OUTDIR}/llmesr_official_scores.csv"
  "${OUTDIR}/llmesr_official_model.pt.sha256"
  "outputs/${EXP}_${BASE}_same_candidate"
  "outputs/summary/week8_official_external_qwen3base_multik_comparison.csv"
  "outputs/summary/week8_official_external_qwen3base_multik_comparison.md"
)
[ -n "$LOG" ] && [ -f "$LOG" ] && FILES+=("$LOG")

tar -czf "outputs/exports/llmesr_${DOMAIN}_official_qwen3base_evidence_${STAMP}.tar.gz" \
  "${FILES[@]}"
sha256sum "outputs/exports/llmesr_${DOMAIN}_official_qwen3base_evidence_${STAMP}.tar.gz"
ls -lh "outputs/exports/llmesr_${DOMAIN}_official_qwen3base_evidence_${STAMP}.tar.gz"
```

Copy the archive to local storage and verify it before deleting any server
intermediates. Only after confirmation, clean the completed LLM-ESR domain:

```bash
DOMAIN=books
EXP=books_large10000_100neg
BASE=llmesr_official_qwen3base_sasrec
rm -rf "outputs/baselines/official_adapters/${EXP}_llmesr_official"
rm -rf "outputs/baselines/paper_adapters/${EXP}_llmesr_official_adapter"
# Keep outputs/${EXP}_${BASE}_same_candidate/tables/ so comparison builders
# continue to include this completed domain. If space is tight, prune only
# large/non-table files after confirming the archive:
find "outputs/${EXP}_${BASE}_same_candidate" -mindepth 1 -maxdepth 1 ! -name tables -exec rm -rf {} +
df -h /
```

Keep this sequence domain-by-domain. Do not start cleanup for a domain whose
evidence archive has not been confirmed locally, and do not let a slow full
checkpoint archive block the next domain unless the user explicitly asks for
full checkpoint preservation. Beauty must be restored or retained in the same
way as the large domains; do not publish a method-level comparison table where
an official baseline is missing a completed domain because its imported summary
was cleaned or left only in a local archive.

The next official external LLM-rec baselines after LLM2Rec/LLM-ESR/LLMEmb are
RLMRec, IRLLRec, and SETRec. RLMRec imports the pinned repo's
`encoder.models.general_cf.simgcl_plus.SimGCL_plus` and preserves the official
BPR, graph contrastive, and semantic alignment losses. IRLLRec imports the
pinned repo's `encoder.models.general_cf.lightgcn_int.LightGCN_int` and
preserves the official BPR, semantic alignment, and intent representation
losses while supplying same-candidate graph data, Qwen3 item embeddings, and
Qwen3-PCA64 intent artifacts. On large domains, IRLLRec's official
`ssl_con_loss` would materialize an all-node N x N matrix; the runner applies a
documented deterministic node cap (`--irllrec_ssl_con_max_nodes`, default
4096) for that term and records the bridge in provenance. SETRec imports the
pinned repo's `code.model_qwen.Qwen4Rec` and preserves the official
query-guided simultaneous decoding, LoRA path, CF token projection, semantic AE
tokenizer, and item scoring path while supplying same-candidate dictionaries,
Qwen3 item/semantic features, and exact score export. Use the same one-domain
archive-and-clean loop. Do not import blocked scaffold rows.

For baseline comparison tables, keep the main reading order at
`NDCG@5`, `NDCG@10`, `HR@5`, `HR@10`, then use `@20` as the extended-check
column when the exporter provides it. The working target for the official
external block is eight baselines, not six: the current six are the floor, and
two additional current-year recommendation baselines from DBLP/GitHub should be
added as separate official-code-level rows when they are ready.

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

## C-CRP And SRPD Formal Internal Methods

C-CRP and SRPD are our internal method lines, but they still use the same
candidate-score gate as external baselines. Do not call a prompt-only Shadow
run, a local C-CRP CSV, or an SRPD training artifact paper-facing until it has
exact score coverage and an imported same-candidate summary.

C-CRP production flow:

```text
build pointwise shadow rows
-> run Qwen3-8B shadow_v1 on valid/test
-> calibrate on valid only
-> select C-CRP mode/weights/eta/ablation on valid only
-> export test source_event_id,user_id,item_id,score
-> import with status_label=same_schema_internal_method
```

Generate and run the server command script:

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
python main_make_week8_future_framework_commands.py \
  --stage shadow \
  --domains books,electronics,movies \
  --output_path outputs/summary/week8_large10000_100neg_ccrp_shadow_commands.sh

mkdir -p outputs/summary/logs
LOG=outputs/summary/logs/week8_ccrp_formal_$(date +%F_%H%M%S).log
PID=outputs/summary/logs/week8_ccrp_formal.pid
nohup bash outputs/summary/week8_large10000_100neg_ccrp_shadow_commands.sh > "$LOG" 2>&1 &
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

SRPD production flow is stricter because it is trainable. The formal configs
live at:

```text
configs/srpd/{books,electronics,movies}_large10000_100neg_srpd_v6_formal.yaml
configs/lora/{books,electronics,movies}_large10000_100neg_srpd_v6_formal.yaml
```

They are intentionally fail-fast: they require validation-side teacher files,
reject test-derived teacher paths, require leakage audit against final test
events, require `training.use_sample_weights=true`, and default to
`status_label=same_schema_internal_ablation`.

```text
teacher data must come from train/valid-compatible sources
-> leakage audit must pass against final eval events
-> sample weights must be enabled if the variant claims weighting
-> LoRA train/eval
-> export exact candidate scores
-> import as same_schema_internal_ablation unless native candidate scores and
   all main gates are complete
```

If SRPD predictions are already available, export/import them with:

```bash
python main_export_srpd_scores_from_predictions.py \
  --ranking_input_path outputs/baselines/external_tasks/books_large10000_100neg_test_same_candidate/ranking_test.jsonl \
  --candidate_items_path outputs/baselines/external_tasks/books_large10000_100neg_test_same_candidate/candidate_items.csv \
  --prediction_path outputs/books_srpd_formal/predictions/rank_predictions.jsonl \
  --output_scores_path outputs/summary/week8_srpd_formal/books/srpd_scores.csv \
  --provenance_output_path outputs/summary/week8_srpd_formal/books/srpd_internal_provenance.json \
  --method_variant SRPD-formal

python main_import_same_candidate_baseline_scores.py \
  --baseline_name books_srpd_formal \
  --exp_name books_srpd_formal_same_candidate \
  --domain books \
  --ranking_input_path outputs/baselines/external_tasks/books_large10000_100neg_test_same_candidate/ranking_test.jsonl \
  --scores_path outputs/summary/week8_srpd_formal/books/srpd_scores.csv \
  --method_provenance_path outputs/summary/week8_srpd_formal/books/srpd_internal_provenance.json \
  --status_label same_schema_internal_ablation \
  --artifact_class completed_result
```

Generate SRPD formal commands:

```bash
python main_make_srpd_formal_commands.py \
  --domains books,electronics,movies \
  --stage all \
  --output_path outputs/summary/week8_srpd_formal_commands.sh
```

Before GPU training, run startup checks after the validation-side teacher files
exist:

```bash
python main_make_srpd_formal_commands.py \
  --domains books,electronics,movies \
  --stage train \
  --startup_check_only \
  --output_path outputs/summary/week8_srpd_formal_startup_commands.sh
bash outputs/summary/week8_srpd_formal_startup_commands.sh
```

This startup script intentionally runs the SRPD data-build step first, so the
teacher existence checks and leakage audit execute before LoRA startup checks.

The formal teacher files are required inputs, not generated by fallback:

```text
outputs/summary/week8_srpd_formal_teachers/books/valid_teacher_rank_reranked.jsonl
outputs/summary/week8_srpd_formal_teachers/electronics/valid_teacher_rank_reranked.jsonl
outputs/summary/week8_srpd_formal_teachers/movies/valid_teacher_rank_reranked.jsonl
```

If any of those paths are missing, SRPD formal should fail. Do not point SRPD
formal configs at `ranking_test.jsonl` teachers to make it run.

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
