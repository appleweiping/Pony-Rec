# Server Runbook

This file is the stable entry point for server-side work. It avoids relying on
chat memory.

## Always Start Here

```bash
cd ~/projects/pony-rec-rescue-shadow-v6
git pull --ff-only
git status --short
```

Read these files before launching heavy work:

```text
docs/milestones/README.md
docs/top_conference_review_gate.md
CODEX_HANDOFF_WEEK8_2026-05-06.md
WEEK8_LARGE_SCALE_10K_100NEG_PLAN_2026-05-06.md
OFFICIAL_EXTERNAL_BASELINE_UPGRADE_PLAN_2026-05-07.md
```

Then run the lightweight readiness check:

```bash
python main_project_readiness_check.py
```

## Current Priority Order

```text
1. Pull latest repo state.
2. Confirm no stale nohup job is still running.
3. Finish or restart the Week8 four-domain 100neg paper-style supplementary run
   only if that is the intended diagnostic.
4. Audit official external repositories.
5. Implement official adapters in order: LLM2Rec, LLM-ESR, LLMEmb, RLMRec,
   IRLLRec, SETRec.
6. Run Shadow large-scale diagnostics only after the 100neg task packages are
   confirmed healthy.
7. Build Signal/Decision/Generative LoRA artifacts only after teacher data and
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
python main_make_official_external_adapter_plan.py
```

Do not import official-baseline rows into main comparison tables until:

```text
official repo commit pinned
official training/scoring entrypoint recorded
Qwen3-8B base path recorded
method-specific adapter/checkpoint path recorded
source_event_id,user_id,item_id,score score file emitted
score coverage verified
paired-test inputs generated
```

## Output Interpretation

- `*_style_*` rows are paper-style supplementary diagnostics.
- `*_official_qwen3_lora_*` rows are the target official external-baseline
  family.
- Beauty is supplementary smaller-N unless the eligible user count reaches the
  main-domain target.
- Week7.7 compact six-candidate results and Week8 101-candidate results must
  not be mixed as direct row-level comparisons.
