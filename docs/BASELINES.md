# Baselines

Implemented sanity baselines:

- Random
- Popularity
- BM25 text overlap
- ItemKNN helper functions

Configured but requiring server/dependency integration:

- RecBole: BPR-MF, LightGCN, GRU4Rec, SASRec, BERT4Rec, FMLPRec.
- LLM4Rec families: P5/OpenP5, CoLLM, LLM2Rec/LLMEmb, LLM-ESR, SLMRec.
- Uncertainty/debias families: uncertainty decomposition, GUIDER-style logit-guided reranking, decoding/amplification analyses.

Rule: if an official implementation is integrated, record repository URL, commit hash, dataset mapping, config, and modifications here. If approximated, label it as an approximation and state differences.
