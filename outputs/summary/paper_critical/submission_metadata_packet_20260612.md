# Submission Metadata Packet

Generated: 2026-06-12T11:56:02.988152+00:00

- OK: `true`
- Submission metadata packet ready: `true`
- Final submission ready: `false`
- Target profile: `sigir2026_full_paper_acm_anonymous`
- Paper type: `full research paper`
- Title: Actionable Uncertainty for LLM-Based Recommendation
- Abstract words/chars: `212` / `1565`
- Keywords: LLM-based recommendation, uncertainty estimation, calibration, candidate reranking, same-candidate evaluation
- Topic areas: Recommender systems, Evaluation and reproducibility, Large language models, Uncertainty and calibration
- PDF: `Paper\main.pdf`, `9` pages, `546669` bytes
- Source manifest files: `21`
- Source manifest sha256: `4f2a9856f722c98ffaf6b7073af27f6890c3086fffe23fa596ebe9fc62aa3cfa`

## Abstract

Large language models (LLMs) are increasingly used as scoring and representation engines for recommendation, but high ranking scores do not by themselves tell us when a recommendation decision is reliable. We study this problem under a controlled same-candidate protocol: every method ranks the same 101 candidates for each user, uses the same Qwen3-8B backbone when an LLM is required, and is evaluated with the same importer, metrics, provenance checks, and paired tests. We propose C-CRP, a task-grounded calibrated candidate relevance posterior that decomposes uncertainty into boundary ambiguity, calibration gap, evidence support, and counterevidence, and defines a validation-controlled ranking family whose zero-risk special case is explicitly allowed. Across Sports, Toys, Home, and Tools, C-CRP ranks first against eight official-code-level LLM-based recommendation baselines on HR@5/@10/@20, NDCG@5/@10/@20, and MRR. All 56 per-domain paired C-CRP-vs-baseline tests are positive and Holm-significant in each domain. We also provide paper-critical diagnostic evidence: event-level uncertainty stratifies ranking reliability in all four domains, component ablations reveal that several uncertainty terms are weak or redundant rather than uniformly necessary, and validation-selected hyperparameters are stable within the pre-registered tolerance on NDCG@10. The result is a scoped claim: the C-CRP ranking line improves controlled same-candidate ranking, while its uncertainty signal stratifies reliability; it is not a full-catalog recommender SOTA claim.

## Remaining Blockers

- ProMax final ACM page range and ACM/Crossref registry visibility must be rechecked immediately before submission.
- promax:final_page_range_missing_in_bib
- promax:crossref_registry_not_visible:status=404
- promax:doi_resolver_not_visible:status=404
- Final manual submission-system metadata/format checklist is not closed.

## Manual Fields Not Stored

- author names and affiliations
- conflicts of interest
- reviewer suggestions or exclusions
- submission-system declarations

## Failures

- None

## Warnings

- None

## Next Actions

- Use these fields to fill the submission system metadata.
- Complete conflict-of-interest, author, and declaration fields inside the submission system.
- Recheck ProMax final ACM page range and ACM/Crossref visibility immediately before submission.
- Keep final_submission_ready=false until manual submission-system and external metadata blockers are closed.
