# Submission Metadata Packet

Generated: 2026-06-14T21:21:55.502977+00:00

- OK: `false`
- Submission metadata packet ready: `false`
- Final submission ready: `false`
- Target profile: `sigir2026_full_paper_acm_anonymous`
- Paper type: `full research paper`
- Title: A Pointwise LLM Relevance Posterior Is a Strong Reranker: A Controlled Same-Candidate Study of Uncertainty-Adjusted LLM Recommendation
- Abstract words/chars: `419` / `2899`
- Keywords: LLM-based recommendation, uncertainty estimation, calibration, candidate reranking, same-candidate evaluation
- Topic areas: Recommender systems, Evaluation and reproducibility, Large language models, Uncertainty and calibration
- PDF: `Paper\main.pdf`, `15` pages, `850448` bytes
- Source manifest files: `26`
- Source manifest sha256: `2acac6e54318be410e9e216429195cad580fd870b91ef95a6bddb9f361909a08`

## Abstract

Large language models (LLMs) are increasingly used as scoring and representation engines for recommendation, yet comparisons across LLM-based recommenders are often confounded by different candidate sets, backbones, importers, and evaluation schemas. We first contribute a rigorous unified same-candidate reranking protocol: every method ranks the same 101 candidates (one positive and 100 popularity-sampled negatives) for each of 10,000 users, uses the same Qwen3-8B backbone when an LLM is required, and is evaluated with a shared importer, identical metrics, exact coverage checks, and paired Holm-corrected bootstrap tests, with full provenance. Against eight official-code-level LLM-based recommendation baselines across eight Amazon domains (Beauty, Books, Electronics, Movies, Sports, Toys, Home, Tools), our central empirical finding is that a task-grounded pointwise LLM relevance posterior -- the model's own per-candidate relevance probability, used directly as the ranking score -- ranks first in six of the eight domains, improving NDCG@10 by +21.6\ seven metrics HR@5/@10/@20, NDCG@5/@10/@20, MRR in five of these six domains --- Electronics, Sports, Toys, Home, Tools; 56/56 paired-significant --- and on six of seven on Books, HR@20 versus LLMEmb the exception). In the remaining two domains it is not first: rank 2 on Beauty (-11\ this shortfall is not Holm-significant on any metric, leaving C-CRP statistically indistinguishable from the strongest baseline) and mid-pack at rank 5 on Movies ($-24\ baselines). We report these two losses explicitly rather than dropping them. We compute paired Holm-corrected bootstrap tests on all eight domains: on the six winning domains C-CRP is significantly ahead on all 56 metric/baseline pairs (54/56 on Books), while on Movies it is significantly behind the stronger baselines on 22 of 56 pairs. We then report a rigorous negative result on the design space this work set out to study. We instrument the pointwise posterior with a principled calibrated-uncertainty and risk-adjusted ranking family that decomposes uncertainty into boundary ambiguity, a calibration gap, evidence support, and counterevidence. Under a leave-one-component-out ablation and a raw-probability attribution, this machinery does not improve same-candidate ranking: the zero-risk setting is test-best in all four diagnostic domains, a confidence-only variant matches or exceeds the full family in three of those four domains, and individual uncertainty terms are inert or mildly harmful. The working ranking signal lies in the posterior itself, not in the uncertainty decomposition. As a minor, descriptive observation -- and one we caveat as partly circular -- the event-level uncertainty signal does stratify reliability, but we do not claim it improves ranking. This is a scoped result about same-candidate reranking, not a full-catalog recommender SOTA claim.

## Remaining Blockers

- ProMax final ACM page range and ACM/Crossref registry visibility must be rechecked immediately before submission.
- Final submission package still needs the external submission-target-specific formatting pass.
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

- `submission_package_audit_not_ok`
- `submission_package_not_ready_for_target_formatting`
- `target_profile_not_ok`

## Warnings

- `abstract_word_count_outside_common_range:419`

## Next Actions

- Use these fields to fill the submission system metadata.
- Complete conflict-of-interest, author, and declaration fields inside the submission system.
- Recheck ProMax final ACM page range and ACM/Crossref visibility immediately before submission.
- Keep final_submission_ready=false until manual submission-system and external metadata blockers are closed.
