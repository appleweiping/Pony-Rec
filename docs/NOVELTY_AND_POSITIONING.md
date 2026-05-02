# Novelty And Positioning

The contribution should be framed as recommendation-specific uncertainty operationalization, not as generic confidence elicitation.

Core positioning:

- Measure whether LLM confidence predicts recommendation correctness.
- Identify confident errors and under-confidence on long-tail items.
- Test whether high-confidence head-item recommendations amplify exposure concentration.
- Calibrate uncertainty on validation data and apply it to test-time reranking, abstention, list truncation, exploration, and LoRA data selection.

Required literature to inspect before paper claims:

- Bryan Hooi group work on LLM uncertainty expression and uncertainty of thoughts.
- P5/OpenP5, CoLLM, LLM2Rec/LLMEmb, SLMRec, LLM-ESR, AGRec, CoVE, Decoding Matters, Lost in Sequence, LLM4RSR, RecExplainer, SPRec.
- Uncertainty-aware LLM recommendation, GUIDER, uncertainty-aware semantic decoding, and related calibration work.

Do not claim novelty from simply asking for confidence.
