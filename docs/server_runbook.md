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
docs/milestones/README.md
docs/top_conference_review_gate.md
CODEX_HANDOFF_WEEK8_2026-05-06.md
WEEK8_LARGE_SCALE_10K_100NEG_PLAN_2026-05-06.md
OFFICIAL_EXTERNAL_BASELINE_UPGRADE_PLAN_2026-05-07.md
```

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

- `CODEX_HANDOFF_WEEK8_2026-05-06.md`
- `WEEK8_FUTURE_FRAMEWORK_ROADMAP_2026-05-06.md`
- `WEEK8_LARGE_SCALE_10K_100NEG_PLAN_2026-05-06.md`
- `WEEK8_FUSION_EXTERNAL_ONLY_CONTRIBUTION_UPDATE_2026-05-06.md`
- `WEEK8_OURS_EXTERNAL_COMBO_AND_EXTERNAL_ONLY_PLAN_2026-05-06.md`

Use them only when you need historical detail for that specific stage.
