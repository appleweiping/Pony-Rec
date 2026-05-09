# Top-Conference Review Gate

This file turns the literature-scout and reviewer-agent passes into a standing
pre-submission gate. It is not a literature review draft; it is a claim and
evidence defense checklist.

## Venue Fit

The project is best framed as:

```text
from uncertainty observation to a reproducible same-candidate recommendation
decision framework, with staged official-baseline and complete-system
extensions
```

Most natural venues:

- RecSys / SIGIR-family / WSDM for recommender-system and retrieval framing.
- KDD for data-mining, benchmark, and applied evaluation framing.
- NeurIPS Evaluations & Datasets if the evaluation protocol and artifact become
  the main contribution.

## Official Source Anchors

- ACM RecSys 2026 Call for Contributions:
  `https://recsys.acm.org/recsys26/call/`
- WSDM 2026 Call for Papers:
  `https://wsdm-conference.org/2026/index.php/call-for-papers/`
- KDD 2026 Datasets and Benchmarks Track:
  `https://kdd2026.kdd.org/datasets-and-benchmarks-track-call-for-papers/`
- KDD 2026 Artifact Badging:
  `https://kdd2026.kdd.org/call-for-artifact-badging/`
- NeurIPS 2026 Evaluations & Datasets Track:
  `https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets`
- ACM SIGIR Artifact Badging:
  `https://sigir.org/general-information/acm-sigir-artifact-badging/`

## Reviewer Gate

Before a claim enters the paper, answer these questions.

For complex work, run this as a multi-agent review: implementation/engineering,
literature/protocol scout, and top-conference reviewer/auditor. The literature
pass should compare against multiple relevant top-conference papers or official
projects, using them to calibrate rigor, novelty, technical depth, ablation
coverage, and reproducibility expectations. It must not turn those papers into
a parts library for C-CRP or SRPD.

### Scope

- Is this a completed result, a supplementary diagnostic, or a roadmap item?
- Does the claim fit controlled candidate ranking/reranking, or does it imply
  full-catalog recommendation?
- Are Shadow, LoRA, and generated-title modules labeled as extensions unless
  fully evaluated?

### Fairness

- Same train/valid/test rows?
- Same candidate rows?
- Same negative sampler and seed policy?
- Same metric importer?
- Same score schema: `source_event_id,user_id,item_id,score`?
- Same paired-test unit?
- Beauty marked as supplementary smaller-N?

### Baselines

- Classical baselines complete and fairly tuned?
- Does the baseline block follow the senior-advice policy: official/default
  baselines first, method variants second?
- Is any strong baseline absent, and if so is the reason documented instead of
  hidden?
- Paper-style LLM-rec rows labeled as supplementary?
- Official-code-level rows used only after pinned repo, checkpoint, and score
  provenance are recorded?
- Qwen3-8B used as the shared base model while method-declared adapters or
  representation artifacts follow each baseline's official algorithm?
- Primary table uses baseline official default/recommended hyperparameters,
  while our method uses validation-selected or pre-fixed hyperparameters?
- Full fine-tuning and retuned-baseline rows kept out of the primary comparison
  variant unless explicitly labeled as supplementary/sensitivity?
- Official external rows require `implementation_status=official_completed`
  and exact finite same-candidate score coverage?

### Protocol Rigor

Every main-table row should record:

```text
dataset/domain
selected user count
candidate count
negative sampling strategy and seed
training command
checkpoint or adapter path
comparison variant
finetuning mode
baseline hyperparameter source and overrides
score CSV path
score schema
metric script
git commit
official repo commit when applicable
score coverage
paired-test output
```

### Ablations

At least one main paper should make these boundaries visible:

- observation-only vs framework;
- Pony/Light/Shadow progression;
- calibration removed;
- uncertainty/risk component removed;
- rescue or fallback component removed, if claimed;
- small-domain vs four-domain behavior;
- compact six-candidate vs 100neg protocol separation;
- official vs paper-style external-baseline status.

### Endgame Gate

Before asking for more experiments, decide whether the experiment phase should
stop. It is reasonable to move from experiments to writing when all are true
for the defended claim:

- the claim is scoped to controlled same-candidate candidate ranking/reranking;
- core internal method rows have validation-only selection, exact score export,
  same-schema import, paired tests, and ablation coverage;
- the required baseline block has either completed official-code-level rows or
  explicitly documented limitations/supplementary status;
- four-domain or declared-domain robustness is complete enough for the stated
  claim, with Beauty smaller-N labeled if used;
- leakage, candidate protocol, score coverage, provenance, and reproducibility
  audits pass;
- a top-conference reviewer/auditor has no unresolved major objection about
  fairness, novelty, technical depth, or overclaiming.

If these gates pass, tell the user the project is ready to enter paper writing
and artifact packaging rather than continuing to add new experiments. If they
do not pass, report the minimum remaining gates and avoid open-ended
"one more baseline" drift.

## Likely Reviewer Objections

| risk | defense |
| --- | --- |
| claim too broad | use milestone eligibility and keep full-system claims staged |
| official baselines unfinished | label paper-style rows as supplementary or finish official adapters |
| mixed protocols | keep protocol, candidate count, N, and status in every evidence table |
| weak reproducibility | maintain server runbook, provenance, git commit, and output manifests |
| agent drift / undocumented server state / unpushed changes | use `AGENTS.md`, paste-back logs, readiness checks, and push code/config/doc updates |
| test-selected fusion | label as diagnostic unless validation-selected |
| Beauty N mismatch | report Beauty as supplementary smaller-N |

Before submission-facing work, run a reviewer-agent pass that reads
`AGENTS.md`, `docs/paper_claims_and_status.md`, and this file. The reviewer
agent should explicitly check for toy substitutes, unpushed command changes,
unfinished official baselines, and claims that outrun the completed evidence.
