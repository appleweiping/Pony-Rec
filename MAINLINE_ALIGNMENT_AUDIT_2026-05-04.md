# Mainline Alignment Audit - 2026-05-04

This audit is a guardrail document. It records how the local `shadow-v6`
branch, the remote `origin/main` branch, and the paper planning files line up.
It does not merge, rebase, or promote any experiment result.

## Scope

- Workspace checked: `D:\Research\Uncertainty`
- Local branch checked: `codex/shadow-v6-restart`
- Remote checked: `origin/main`
- Merge base: `95a74f309c8421238bfa0fd73460666bb2e85875`
- Local-only commits: `1`
- Remote-main-only commits: `36`
- Audit action performed before writing this file: `git fetch origin`

## Executive Decision

Do not treat `origin/main` as a trusted research mainline.

After the first version of this audit, we checked commit metadata more closely.
The current `origin/main` line from `40b328c` through `6caf031` is effectively
Cursor-coauthored. Therefore, `origin/main` should be treated as a Cursor
engineering donor branch, not as the source of truth for the research direction.

Use the paper/Pony files and the non-Cursor protocol artifact branch as the
research anchor. Then selectively salvage verified engineering pieces from
`origin/main` and the local `codex/shadow-v6-restart` branch.

Practical decision:

- Research source of truth: `Paper/version1,2`, `Paper/pony*`, and
  `backup/protocol-artifact-b1aa04f`.
- Pre-Cursor main recovery anchor: `backup/protocol-artifact-b1aa04f`
  (`Harden research protocol artifact`, 2026-04-30 22:11 +0200).
- Optional first v6 draft patch: `40b328c` (`Add shadow v6 decision bridge`,
  2026-05-01 00:33 +0200). This commit has no Cursor coauthor in current git
  metadata, but it should still be reviewed before adoption.
- Shadow patch source: `codex/shadow-v6-restart`.
- Cursor engineering donor: `origin/main`.
- Do not merge `origin/main` wholesale.
- Do not expand directly from `codex/shadow-v6-restart` either; it is a useful
  patch branch, not a full project base.

## Research Mainline

The paper planning files describe a coherent progression:

1. `Paper/version1,2` is the early evidence chain:
   confidence diagnosis, validation-to-test calibration, uncertainty-aware
   reranking, multi-domain and robustness expansion.
2. `Paper/pony2/version pony` upgrades the project from a pointwise confidence
   study into a multi-level uncertainty framework:
   pointwise observation, pairwise preference, and candidate ranking.
3. `Paper/pony2/version pony week7.9` reframes `shadow-v1` to `shadow-v5` as
   task-grounded uncertainty signal candidates. `shadow-v6` is a bridge that
   tests whether a winning signal can enter ranking, reranking, preference, or
   later generative-title verification.
4. `Paper/pony2/version pony week8` says shadow must be reported honestly by
   status: design only, prompt-only, Signal LoRA ready, or decision ready.
   Shadow should not enter a main ranking table before same-split ranking or
   rerank evidence exists.
5. `Paper/pony2/version pony week9` points the final system toward:
   `user history -> generated recommendation title -> catalog-grounded
   verification -> accept / revise / fallback`.

Therefore, the current research mainline is not "old confidence rerank wins".
It is:

```text
reliability observation
-> task-grounded uncertainty signal
-> calibrated signal-to-decision bridge
-> recommendation / exposure / generative verification control
```

## Remote Mainline Status

`origin/main` contains useful engineering work, but it is Cursor-contaminated
from a research-governance perspective. Its current rules come from
`AGENTS.md`, `README.md`, and the pilot documents under `docs/`, but those
rules should be audited before being adopted as project doctrine.

Important current facts:

- The supported experiment interface is `python -m src.cli.*`.
- Former repo-root `main_*.py` entrypoints have been moved to
  `legacy/root_main/` and are deprecated for new experiments.
- The current project name and framing in remote docs is CARE-Rec:
  confidence-aware reliable exposure recommendation with LLMs.
- Official processed domains are:
  `amazon_beauty`, `amazon_books`, `amazon_electronics`, `amazon_movies`.
- Processed source tables are accepted for pilots only after verification:
  `interactions.csv`, `items.csv`, `popularity_stats.csv`, `users.csv`.
- Recompute splits, candidates, and popularity. Do not reuse old split
  artifacts as paper evidence.
- Every output must clearly separate smoke, pilot, diagnostic, and
  paper-result artifacts.
- Mock or smoke outputs are never paper evidence.
- Test split must not be used for calibration, threshold selection,
  hyperparameter selection, pruning, or rerank weight design.

Potentially useful `origin/main` content:

- Clean `src.cli` module surface.
- Manifest and artifact-class ideas.
- Processed-source reprocessing code.
- DeepSeek pilot runner and prompt/parser hardening utilities.
- RecBole smoke baseline scaffolding.
- Some tests around data protocol, parsing, calibration, and rerank logic.

Potentially risky `origin/main` content:

- It reframes the project as CARE-Rec, which may or may not match the Pony
  paper direction exactly.
- It deletes or replaces several earlier protocol docs from
  `backup/protocol-artifact-b1aa04f`.
- It may over-privilege c99/CARE rerank/LoRA pilot artifacts relative to the
  Pony shadow/generative-title trajectory.
- It moves root scripts into `legacy/root_main`, which may be good engineering,
  but the migration should be validated against the actual scripts used in
  prior results.
- All pilot claims from this line need independent verification before they
  influence paper direction.

Current remote pilot gates, if retained after audit:

- c99 DeepSeek + CARE rerank is pilot-only.
- `care_full` does not yet show a clean main-method win over listed pilot
  baselines.
- Books improved under cleaned prompt-view rerun, but remains pilot-only.
- Movies remains unstable; metadata gaps and raw/catalog quality are still a
  blocker.
- CARE-LoRA strict JSON generation remains blocked. Safe-repaired ranking is
  engineering usability evidence, not strict generation quality.
- No full-scale API sweep and no `is_paper_result=true` should be introduced
  before the relevant gates are accepted.

## Local Shadow Branch Status

Local branch `codex/shadow-v6-restart` has one unique commit:

```text
b86784c Rebuild shadow v6 decision bridge
```

Files changed relative to the shared base:

- `README.md`
- `configs/shadow/week7_9_shadow_runtime.yaml`
- `main_build_shadow_v6_bridge.py`
- `main_shadow_make_commands.py`
- `src/shadow/__init__.py`
- `src/shadow/decision_bridge.py`
- `src/shadow/scoring.py`
- `tests/conftest.py`
- `tests/test_shadow_v6_bridge.py`

Useful content in the local branch:

- `src/shadow/decision_bridge.py` adds event-aware signal lookup. It prefers
  `(source_event_id, user_id, item_id)` over `(user_id, item_id)`, reducing the
  risk that repeated user-item pairs across events are matched to the wrong
  shadow signal.
- The local CLI script adds metadata fields such as dataset, domain, split,
  seed, backend, prompt id, config hash, artifact class, and
  `is_paper_result`.
- The local tests add a CLI smoke test and verify event-specific matching.

Main risks in the local branch:

- It adds `main_build_shadow_v6_bridge.py` at repo root, but `origin/main`
  explicitly moved new work away from repo-root `main_*.py` entrypoints.
- It lacks the current `AGENTS.md` guardrails and most current `docs/` pilot
  gates from `origin/main`.
- It can mislead future work into treating shadow-v6 as the active mainline,
  even though the paper docs say shadow-v6 is a bridge/interface until backed
  by same-split evidence.
- It should not be merged wholesale because doing so would reintroduce an old
  project surface and discard remote mainline documentation.

## Branch Roles

### `backup/protocol-artifact-b1aa04f`

This branch is the best current match for the user's intended pre-Cursor main
state: shadow v1-v5 and the server/result protocol context already exist, and
the project is positioned just before or around v6 implementation. It adds
protocol docs, candidate audit, calibration/statistical checks, shadow method
notes, generative-title bridge notes, and paper-claim status material without
the later Cursor-coauthored history.

Use this branch to recover:

- `docs/baseline_protocol.md`
- `docs/calibration_protocol.md`
- `docs/candidate_protocol.md`
- `docs/experiments.md`
- `docs/generative_title_bridge.md`
- `docs/paper_claims_and_status.md`
- `docs/shadow_method.md`
- `docs/tables.md`
- `src/eval/candidate_protocol_audit.py`
- `src/eval/statistical_tests.py`
- `src/shadow/ccrp.py`
- `src/uncertainty/baseline_reliability_proxy.py`

### `codex/shadow-v6-restart`

This branch has one useful local patch around shadow-v6 event-aware matching
and metadata propagation. Treat it as a patch source only.

### `40b328c`

This commit is the earliest visible v6-code commit after the protocol anchor:
`Add shadow v6 decision bridge`. It is not Cursor-coauthored according to the
current commit metadata. Treat it as the first v6 draft to compare against the
later `codex/shadow-v6-restart` rebuild.

### `origin/main`

This branch is not the research source of truth. Treat it as a donor branch for
specific engineering pieces after review.

## Integration Recommendation

Create a new rescue branch from the pre-Cursor anchor:

```text
backup/protocol-artifact-b1aa04f
```

Then port selectively:

From `40b328c` and `codex/shadow-v6-restart`:

1. Keep the event-aware `SignalLookup` idea.
2. Keep metadata propagation and artifact-class fields.
3. Keep or adapt the event-specific tests.
4. Do not treat shadow-v6 as a completed main method.

From `origin/main`, only after review:

1. Clean CLI structure if it preserves actual experiment semantics.
2. Manifest and artifact-class utilities.
3. Processed-source reprocessing if it passes leakage checks.
4. Data cleaning / prompt hardening if it does not change candidate protocol.
5. Tests that verify behavior rather than encode Cursor assumptions.

Suggested integration target:

```text
trusted protocol anchor
-> rescue branch
-> restore Pony/shadow/generative docs as research north star
-> selectively port verified CLI/manifest utilities
-> selectively port shadow-v6 event-aware bridge
-> run tests and smoke checks
-> document all imported Cursor pieces as reviewed or rejected
```

## Cursor And Server Guardrails

Before using Cursor, running on the server, or accepting automated edits:

1. Do not start blindly from `origin/main`.
2. Do not start blindly from `codex/shadow-v6-restart`.
3. Use the rescue branch once created.
4. Read this audit, the Pony paper files, and recovered protocol docs before
   starting server work.
5. Use `python -m src.cli.*` only after the CLI implementation has been
   reviewed against the recovered protocol.
6. Do not copy old local outputs into paper tables.
7. Do not mark any c99, CARE rerank, shadow, or LoRA run as paper evidence
   unless the gate doc explicitly allows it.
8. Keep RecBole, API LLM, local LoRA, and candidate-set protocols separated
   unless a document explicitly maps them.
9. When bringing server results back, require manifests with run type,
   backend type, split protocol, candidate protocol, seed, config hash, and
   `is_paper_result`.

## Immediate Next Steps

Recommended next work order:

1. Create a rescue branch from `backup/protocol-artifact-b1aa04f`.
2. Compare `40b328c` and `codex/shadow-v6-restart` to decide the clean v6 patch.
3. Restore the protocol docs and Pony-aligned research status as the north star.
4. Review `origin/main` piece by piece; import only verified engineering
   utilities.
5. Port the local shadow-v6 event-aware matching patch only if shadow-v6 is
   the next immediate task.
6. Run tests and a tiny smoke check before any server job.
7. Only after the rescue branch is stable, decide which pilot line deserves
   server resources.

## One-Line Rule

Do not trust either `origin/main` or `codex/shadow-v6-restart` wholesale:
build a rescue mainline from the trusted protocol/paper anchor, then import
only reviewed pieces.
