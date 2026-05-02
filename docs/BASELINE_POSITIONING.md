# Baseline Positioning

| Baseline | Official repo or paper | Status | Required data format | Expected output format | Reason for inclusion | Limitation |
| --- | --- | --- | --- | --- | --- | --- |
| P5/OpenP5 | P5/OpenP5 generative recommendation | Pending | Text-to-text recommendation prompts | Ranked item IDs and confidence if elicited | Canonical generative LLM4Rec baseline | Not integrated in this repo yet |
| CoLLM | CoLLM collaborative LLM recommendation | Pending | Collaborative embeddings plus text prompts | Ranked candidates | Tests collaborative signal injection | Requires official implementation mapping |
| SLMRec | SLMRec small language model rec | Pending | Sequential recommendation SFT data | Ranked candidates | Local small-model comparison | Requires server integration |
| LLM-ESR | LLM-enhanced semantic recommendation | Pending | Item text and long-tail semantic features | Ranked candidates and exposure stats | Long-tail semantic enhancement baseline | Approximation would need clear labeling |
| LLM2Rec/LLMEmb style | LLM embedding recommendation papers | Pending approximation | Item/user text embeddings | Dense retrieval ranking | Embedding-style LLM4Rec comparison | Exact official code not integrated |
| DeepSeek zero-shot listwise | This repo implementation | Integrated | Candidate JSONL from `src.cli.build_candidates` | `predicted_ranking`, confidence, parser flags | Primary API LLM baseline | Requires API key for real evidence |
| DeepSeek few-shot listwise | Prompt variant | Pending | Candidate JSONL plus examples | Same as listwise | Tests prompt demonstration sensitivity | Prompt/config not finalized |
| DeepSeek self-consistency | `src.uncertainty` framework | Pending partial | Repeated listwise API samples | Vote distribution and uncertainty | Main uncertainty estimator family | Needs cost-controlled API runner |

Do not claim pending baselines as implemented. Official integrations must record repository URL, commit, dataset mapping, config, and local modifications.
