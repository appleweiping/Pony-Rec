# Gap Decomposition: why C-CRP wins 6/8 domains

Measured (not hand-waved) explanation of the per-domain ranking gap of the
**Candidate-Conditioned Relevance Posterior** (C-CRP v3, the pointwise LLM
per-candidate relevance probability used directly as the ranking score) vs the
strongest same-candidate baseline (and vs LLMEmb specifically), across all 8
domains, using interpretable **domain features**.

This is the #1 highest-leverage "gap-to-8" item from the ARIS review: convert
the qualitative "wins 6/8 but loses Beauty/Movies" into a quantified
feature x gap analysis. **CPU-only**, computed from existing results +
dataset statistics. No GPU; the running TGL/cc-pace LoRA job was not touched.

## Headline result

C-CRP ranks first in **6/8** domains (electronics +53.2%, tools +43.3%, home
+41.0%, toys +32.2%, sports +29.7%, books +21.6% NDCG@10 over the strongest
baseline) and is **not first** in **beauty (-11.0% vs ProEx)** and
**movies (-24.2% vs LLMEmb)**. Against LLMEmb specifically, C-CRP still wins
beauty (+9.3%) and only truly trails on movies (-24.2%).

Three directional associations (n=8 -> correlational/descriptive, not causal):

| Feature | Spearman rho with gap% vs best | reading |
|---|---|---|
| **Mean user-history length** | **-0.69** (p=0.058, OLS R2=0.61) | wins in the SHORT-history / cold-start regime |
| Popularity Gini | +0.57 | margin GROWS on skewed/long-tail catalogs (prior "loses on skew" REVERSED) |
| Top-1% interaction share | +0.52 | same direction as Gini |
| Cand. title TF-IDF cosine | -0.24 | loses on text-HOMOGENEOUS panels |
| Cand. title token Jaccard | -0.45 | same direction |
| Cand. panel vocab size | +0.76 (top \|rho\| but Beauty-outlier-driven; not used as figure headline) | |

**Dual mechanism** — the two non-winning domains fail for *different* reasons,
so no single feature classifies both:
- **Beauty**: tiny (1,184-item), near-uniform catalog (Gini 0.26) where the
  dense-ID baseline ProEx has ample collaborative signal; the LLM posterior
  adds little (still beats LLMEmb +9.3%).
- **Movies**: most text-homogeneous candidate panels + longest histories +
  low item-metadata completeness (~51%).

## Files

- `gap_decomposition_results.json` — full results: per-domain gaps (vs strongest
  baseline and vs LLMEmb), all features, Spearman correlations (+ approx p),
  single-feature OLS, win-vs-loss separation, interpretation, data-quality
  disclosures.
- `gap_decomposition_feature_table.csv` — per-domain feature x gap table.
- `gap_decomposition_table.tex` — paper table (`\input`-able).
- `gap_decomposition_paragraph.txt` — draft 1-paragraph gap analysis.
- `gap_vs_top_feature.png` / `.pdf` — headline scatter: per-domain gap vs the
  most robust predictor (mean user-history length); `\includegraphics`-able.
- `gap_features_all.json` — raw per-domain features (copy of the server compute output).

## Reproduce

1. Per-domain features (server, CPU-only; reads dataset csvs + same-candidate
   eval records, no GPU):
   ```
   CUDA_VISIBLE_DEVICES='' python scripts/gap_decomposition_features.py --out /tmp/gap_features_all.json
   ```
   (run on `pony-rec-rescue-shadow-v6`; output copied to `tmp_outputs/gap_features_all.json`)
2. Analysis + figure (local, CPU-only):
   ```
   python scripts/gap_decomposition_analysis.py
   ```

## Feature sources (per domain)

- catalog size + popularity skew (Gini, top-1% share): `data/processed/amazon_<d>/popularity_stats.csv`
- candidate-text homogeneity: 101-candidate panels from
  `outputs/<d>_..._llmemb_official_qwen3base_same_candidate/tables/ranking_eval_records.csv`
  (`candidate_item_ids`, identical across methods under same-candidate protocol),
  joined to titles in `data/processed/amazon_<d>/items.csv`. TF-IDF cosine /
  Jaccard / length / vocab averaged over a 400-user sample.
- mean user-history length: `history` in the panel `ranking_test.jsonl`
  (7/8 domains); Movies uses an `interactions.csv` all-user proxy (panel pruned).
- #test users: beauty 973, others 10,000.

## Disclosed caveats

- **n=8** domains: every correlation is descriptive, not causal; p-values are
  approximate (t on Spearman rho) and only history-length / vocab-size approach
  p<0.06.
- **Movies titles ~38% missing** in Amazon metadata: TF-IDF cosine, title
  length, and vocab are computed over non-empty titles only; metadata
  completeness is reported as its own feature.
- **Movies@10k panel pruned** from server for storage, so its history length is
  an interactions-table all-user proxy (`hist_source` column flags this).
- Candidate homogeneity uses item **titles**; richer `candidate_text` is
  available but titles are the discriminative surface form the LLM ranks on.
