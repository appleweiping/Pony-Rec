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

### home — strongest baseline: TBD (Qwen NDCG@10 0.132)
| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR |
|---|---|---|---|---|---|---|---|
| Qwen3-8B (main) | 0.156 | 0.226 | 0.351 | 0.110 | 0.132 | 0.164 | 0.126 |
| **Llama-3.1-8B** | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| (baseline) | _TBD_ | | | | | | |

### tools — strongest baseline: TBD (Qwen NDCG@10 0.166)
| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR |
|---|---|---|---|---|---|---|---|
| Qwen3-8B (main) | 0.194 | 0.270 | 0.393 | 0.142 | 0.166 | 0.197 | 0.156 |
| **Llama-3.1-8B** | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| (baseline) | _TBD_ | | | | | | |

## Two-domain pattern (sports + toys, both COMPLETE)

- **Headline NDCG@10 rank-1 transfers across backbone families** in both domains
  (sports +14.4%, toys +11.0% over strongest baseline) → the C-CRP ranking-quality advantage is
  **not Qwen-specific**.
- **Absolute level tracks backbone quality**: Llama-3.1-8B ≈ 12–16% below Qwen3-8B on NDCG@10 —
  expected, sensible, and reassuring that the metric discriminates.
- **Consistent honest nuance — deep recall is backbone-sensitive.** Llama wins all NDCG ranks + MRR +
  HR@5 in both domains, but cedes the deep-recall metrics (sports HR@20; toys HR@10 & HR@20) to
  LLMEmb. Mechanism: the coarser Llama verbalized posterior floors off-category candidates at 0.0 →
  ties deep in the list → sharper top-of-ranking but weaker @10/@20 recall. To be stated plainly in
  the paper, not hidden.

## Status
- sports: DONE + committed (main). toys: DONE + committed (main).
- home: running (~9h). tools: queued.
- After all 4: finalize this table, run the η=0 ablation on the Llama scores (confirm the
  uncertainty/risk decomposition is inert on Llama too = replicated negative result), wire a compact
  version into the paper's external-validity subsection, then dual Codex-high + Opus-ultracode review.
