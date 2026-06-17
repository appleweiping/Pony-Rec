# Second-Backbone External Validity — C-CRP v3 on Llama-3.1-8B

**Purpose.** Show the C-CRP v3 pointwise relevance-posterior ranking signal is not specific to
Qwen3-8B: re-run the *identical* method on a different LLM family (Llama-3.1-8B-Instruct), same
protocol, and check whether it still beats the 8 official baselines.

**Scope (SLMRec-aligned, ICLR 2025).** 4 representative domains (sports, toys, home, tools) — SLMRec
itself evaluates on 4 Amazon18 domains, not the full catalog. Single alternative backbone (different
family, same 8B scale) is sufficient for an external-validity claim; SLMRec varies one LLM family by
depth, not across families. Protocol held identical to the Qwen3-8B main table (10k users, 101
candidates, guided-JSON decoding, temp 0.1, prob01 scale) so the comparison is apples-to-apples.

**Backbone.** `/home/ajifang/models/Llama-3.1-8B-Instruct`, vLLM guided-JSON
(`{relevance_probability:[0,1], reason}`). Qwen path is byte-identical when guided-JSON is off.

## Per-domain comparison (NDCG@10 = headline)

Each domain: Qwen3-8B (main, internal) vs Llama-3.1-8B (2nd backbone) vs strongest official baseline.
"Llama win/loss" = Llama-3.1-8B C-CRP vs the strongest baseline on that domain.

### sports — strongest baseline: LLMEmb
| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR |
|---|---|---|---|---|---|---|---|
| Qwen3-8B (main) | 0.2745 | 0.3819 | 0.5172 | 0.1985 | 0.2329 | 0.2670 | 0.2076 |
| **Llama-3.1-8B** | 0.2559 | 0.3624 | 0.4812 | 0.1710 | **0.2054** | 0.2354 | 0.1761 |
| LLMEmb (baseline) | 0.2124 | 0.3384 | 0.4900 | 0.1389 | 0.1795 | 0.2177 | 0.1539 |
| Llama vs LLMEmb | win | win | **loss** | win | win (+14.4%) | win | win |

→ Llama rank-1 on **6/7** (only HR@20 lost). NDCG@10 +14.4% over strongest baseline.

### toys — strongest baseline: LLMEmb
| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR |
|---|---|---|---|---|---|---|---|
| Qwen3-8B (main) | 0.3172 | 0.3964 | 0.5059 | 0.2452 | 0.2708 | 0.2983 | 0.2503 |
| **Llama-3.1-8B** | 0.2809 | 0.3463 | 0.4356 | 0.2062 | **0.2274** | 0.2499 | 0.2075 |
| LLMEmb (baseline) | 0.2499 | 0.3505 | 0.4866 | 0.1725 | 0.2049 | 0.2391 | 0.1814 |
| Llama vs LLMEmb | win | **loss** | **loss** | win | win (+11.0%) | win | win |

→ Llama rank-1 on **5/7** (HR@10/@20 lost). NDCG@10 +11.0% over strongest baseline.

### home — strongest baseline: LLMEmb
| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR |
|---|---|---|---|---|---|---|---|
| Qwen3-8B (main) | 0.1561 | 0.2264 | 0.3505 | 0.1098 | 0.1324 | 0.1635 | 0.1259 |
| **Llama-3.1-8B** | 0.1339 | 0.1980 | 0.3023 | 0.0897 | **0.1103** | 0.1365 | 0.1046 |
| LLMEmb (baseline) | 0.1079 | 0.1856 | 0.3169 | 0.0690 | 0.0939 | 0.1267 | 0.0901 |
| Llama vs LLMEmb | win | win | **loss** | win | win (+17.5%) | win | win |

→ Llama rank-1 on **6/7** (only HR@20 lost). NDCG@10 +17.5% over strongest baseline.

### tools — strongest baseline: LLMEmb
| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR |
|---|---|---|---|---|---|---|---|
| Qwen3-8B (main) | 0.1937 | 0.2696 | 0.3931 | 0.1419 | 0.1661 | 0.1970 | 0.1559 |
| **Llama-3.1-8B** | 0.1731 | 0.2438 | 0.3420 | 0.1180 | **0.1407** | 0.1654 | 0.1289 |
| LLMEmb (baseline) | 0.1365 | 0.2257 | 0.3637 | 0.0875 | 0.1159 | 0.1505 | 0.1065 |
| Llama vs LLMEmb | win | win | **loss** | win | win (+21.4%) | win | win |

→ Llama rank-1 on **6/7** (only HR@20 lost). NDCG@10 +21.4% over strongest baseline.

## Pattern across ALL FOUR domains (sports + toys + home + tools — COMPLETE)

- **Headline NDCG@10 rank-1 transfers across backbone families** in all four domains
  (sports +14.4%, toys +11.0%, home +17.5%, tools +21.4% over the strongest official baseline) → the
  C-CRP ranking-quality advantage is **not Qwen-specific**.
- **Absolute level tracks backbone quality**: Llama-3.1-8B ≈ 12–17% below Qwen3-8B on NDCG@10 —
  expected, sensible, and reassuring that the metric discriminates.
- **Consistent honest nuance — deep recall is backbone-sensitive.** Llama wins all NDCG ranks + MRR +
  HR@5 in every domain, and also HR@10 in 3/4 domains, but cedes the deepest-recall metric (HR@20 in
  all four; HR@10 additionally in toys) to LLMEmb. Mechanism: the coarser Llama verbalized posterior
  floors off-category candidates at 0.0 → ties deep in the list → sharper top-of-ranking but weaker
  @20 recall. To be stated plainly in the paper, not hidden.
- Per-domain rank-1 count vs the 8 official baselines: sports 6/7, toys 5/7, home 6/7, tools 6/7
  (**23/28 metric-domain cells; 4/4 on the headline NDCG@10**).

### Compact 4-domain summary (NDCG@10)
| Domain | Qwen3-8B | Llama-3.1-8B | Strongest baseline (LLMEmb) | Llama vs baseline | Llama vs Qwen |
|---|---|---|---|---|---|
| sports | 0.2329 | 0.2054 | 0.1795 | **+14.4%** | −11.8% |
| toys | 0.2708 | 0.2274 | 0.2049 | **+11.0%** | −16.0% |
| home | 0.1324 | 0.1103 | 0.0939 | **+17.5%** | −16.7% |
| tools | 0.1661 | 0.1407 | 0.1159 | **+21.4%** | −15.3% |

All four: Llama C-CRP rank-1 on NDCG@10 over all 8 official baselines; ~15% below Qwen3 absolute.

## Status
- sports / toys / home / tools: **ALL DONE + committed to main.** Run finished 2026-06-17 ~11:10 CST, GPU freed.
- Remaining: (1) η=0 ablation on the 4 Llama domains' scores (confirm the uncertainty/risk decomposition
  is inert on Llama too = replicated negative result); (2) wire a compact version of this table into the
  paper's external-validity subsection (Paper/sections/results.tex); (3) dual Codex-high + Opus-ultracode
  re-review → ≥8 → ensure on main; (4) re-audit assets/README, clean server garbage.
