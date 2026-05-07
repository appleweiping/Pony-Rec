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
- Paper-style LLM-rec rows labeled as supplementary?
- Official-code-level rows used only after pinned repo, checkpoint, and score
  provenance are recorded?
- Qwen3-8B used as the shared base model while LoRA/adapters follow each
  baseline's official algorithm?

### Protocol Rigor

Every main-table row should record:

```text
dataset/domain
selected user count
candidate count
negative sampling strategy and seed
training command
checkpoint or adapter path
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

## Likely Reviewer Objections

| risk | defense |
| --- | --- |
| claim too broad | use milestone eligibility and keep full-system claims staged |
| official baselines unfinished | label paper-style rows as supplementary or finish official adapters |
| mixed protocols | keep protocol, candidate count, N, and status in every evidence table |
| weak reproducibility | maintain server runbook, provenance, git commit, and output manifests |
| test-selected fusion | label as diagnostic unless validation-selected |
| Beauty N mismatch | report Beauty as supplementary smaller-N |
