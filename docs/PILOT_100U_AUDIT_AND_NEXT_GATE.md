# 100-User Pilot Audit and Next Gate

Audit date: 2026-05-03  
Workspace: `fresh/uncertainty-llm4rec`  
Scope: completed 100-user pilot outputs only; **no new experiments** in this audit.

---

## 1) Scope and artifact class

Checked roots:

- `outputs/reprocessed_processed_source_100u_c19_seed42/`
- `outputs/pilots/deepseek_v4_flash_processed_100u_c19_seed42/`
- `outputs/pilots/care_rerank_deepseek_v4_flash_processed_100u_c19_seed42/`
- `outputs/pilots/recbole_processed_100u_c19_seed42/`
- `outputs/pilots/care_lora_qwen3_8b_beauty_100u_c19_seed42_pilot/`
- (post-audit) `outputs/pilots/care_lora_qwen3_8b_beauty_100u_c19_seed42_strictgen/` — **strictgen attempt C**, recorded only as a **negative** decode/prompt iteration; **not** a new baseline.

Findings:

- All phases are recorded as pilot-class workflows (`run_type=pilot` in manifests/summaries).
- `is_paper_result=false` is preserved across the pilot artifacts.
- No output in this document is treated as paper evidence.

---

## 2) Reprocess protocol audit

Source/target:

- Source: `data/processed`
- Output: `outputs/reprocessed_processed_source_100u_c19_seed42`
- Seed: `42`
- Candidate size: `19`

Per-domain counts (`train/valid/test`):

- `amazon_beauty`: `100/100/100`
- `amazon_books`: `100/100/100`
- `amazon_electronics`: `100/100/100`
- `amazon_movies`: `100/100/100`

Protocol checks:

- Leakage check: pass in all domains (`leakage_report.json -> passed=true`).
- Popularity source: `train_only` in manifests.
- No `srpd`, old split, or old prediction path references detected in this output root.

---

## 3) DeepSeek 100u summary

Root:

- `outputs/pilots/deepseek_v4_flash_processed_100u_c19_seed42/`

### 3.1 Ranking quality (per domain/split)

From `eval/metrics.json`:

- `amazon_beauty`
  - valid: `HR@5=0.39`, `Recall@5=0.39`, `NDCG@5=0.2877`, `MRR@5=0.2543`
  - test: `HR@5=0.36`, `Recall@5=0.36`, `NDCG@5=0.2328`, `MRR@5=0.1915`
- `amazon_books`
  - valid: `0.70`, `0.70`, `0.5490`, `0.4992`
  - test: `0.69`, `0.69`, `0.5852`, `0.5505`
- `amazon_electronics`
  - valid: `0.58`, `0.58`, `0.3916`, `0.3303`
  - test: `0.53`, `0.53`, `0.3352`, `0.2713`
- `amazon_movies`
  - valid: `0.42`, `0.42`, `0.3097`, `0.2740`
  - test: `0.38`, `0.38`, `0.2984`, `0.2708`

### 3.2 Invalid output and confidence availability

From `pilot_run_summary.json`:

- `amazon_beauty`: invalid `0.12` (valid), `0.09` (test); confidence available `0.88/0.91`
- `amazon_books`: invalid `0.61/0.67`; confidence available `0.42/0.35`
- `amazon_electronics`: invalid `0.13/0.16`; confidence available `0.87/0.84`
- `amazon_movies`: invalid `0.39/0.48`; confidence available `0.63/0.54`

### 3.3 Latency and token usage

From `raw_responses.jsonl`:

- Average latency is mostly ~`6.0s` to `7.0s` per request (p95 ~`6.8s` to `8.5s`).
- Prompt token load varies significantly by domain:
  - lower on beauty (~`1.7k`)
  - highest on books/electronics (~`4.2k`–`4.9k`)
- Completion tokens average around ~`360`–`435`.

### 3.4 Calibration diagnostics

From `calibration_diagnostics/calibration_aggregate.csv`:

- ECE range across splits roughly `0.38` to `0.68`
- adaptive ECE roughly `0.38` to `0.67`
- Brier roughly `0.33` to `0.58`

Confidence-correctness relation:

- AUC proxy is mixed (`~0.44` to `~0.64`), indicating confidence is not consistently well-aligned with correctness.

High-confidence wrong rates:

- Often high (e.g., beauty test `0.9011`, beauty valid `0.8409`, electronics test `0.8690`).

Head/mid/tail confidence and exposure:

- `head_confidence_mean`/`mid_confidence_mean`/`tail_confidence_mean` available in aggregate CSV.
- Some splits show sparse head/mid exposure (`nan` where buckets are not represented enough), indicating uneven exposure distribution.

Blockers before c99/full:

- Domain-level reliability is uneven (books/movies invalid-output rates remain high).
- Calibration remains weak in absolute terms (high ECE/Brier + high-confidence-wrong).

---

## 4) CARE rerank 100u summary

Root:

- `outputs/pilots/care_rerank_deepseek_v4_flash_processed_100u_c19_seed42/`

Compared variants:

- `CARE_full` vs `original_deepseek`
- `CARE_full` vs `confidence_only`
- `CARE_full` vs `popularity_penalty_only`
- `CARE_full` vs `uncertainty_only`

### 4.1 Ranking metric deltas

From `care_rerank_aggregate.csv`:

- Most domain/split cells are unchanged (or near-identical).
- Meaningful movement is mainly in `amazon_beauty`:
  - valid: slight drop vs original (`HR@5 -0.02`, `NDCG@5 -0.0066`, `MRR@5 -0.0027`)
  - test: slight gain vs original (`HR@5 +0.01`, `NDCG@5 +0.0054`, `MRR@5 +0.0038`)

### 4.2 Exposure shifts

From `exposure_shift.csv`:

- Beauty head top-1 share reduced under `CARE_full`:
  - test: `0.08 -> 0.03`
  - valid: `0.04 -> 0.02`
- Tail share increases correspondingly on beauty.
- Other domains are largely unchanged.

### 4.3 High-confidence wrong fixed/hurt

From `high_confidence_wrong_changes.csv` + aggregate rates:

- Net reduction is small in most splits.
- Beauty shows top-1 changes but not a strong broad reduction in high-confidence-wrong rate.
- Most non-beauty splits show near-zero effect.

### 4.4 Consistency and c99 readiness signal

- Effects are **not consistent** across all domains/splits.
- Current signal supports keeping CARE rerank as an active method, but still in pilot/ablation posture.
- Gains are not yet large/consistent enough to justify immediately treating c99 expansion as “clearly de-risked”.

No statistical significance is claimed.

---

## 5) RecBole 100u summary

Root:

- `outputs/pilots/recbole_processed_100u_c19_seed42/`

From `smoke_run_summary.json`:

- All configured models report run success across domains:
  - Pop, BPR, LightGCN, SASRec, BERT4Rec (`ok=true`)
- Metrics are available in per-model `result.json` (quality varies widely by model/domain; this is pilot-grade).

Protocol gap (explicit):

- RecBole path uses **atomic files** and standard baseline training/eval.
- It does **not** consume LLM listwise candidate JSONL directly unless explicitly adapted.
- Therefore this baseline track is useful but not directly protocol-equivalent to LLM candidate-based pipelines.

---

## 6) CARE-LoRA 100u summary (beauty only)

Root:

- `outputs/pilots/care_lora_qwen3_8b_beauty_100u_c19_seed42_pilot/`

Adapters:

- `vanilla_lora_baseline`
- `care_full_training`

### 6.1 Training data and policy stats

- vanilla train rows: `100`
- CARE_full train rows: `88`
- pruned rows (policy gate): `24` (from `pruned_samples.jsonl`)
- CARE sample weight range: min `0.4639`, max `1.2727`

### 6.2 GPU + loss

From `train_manifest.json`:

- GPU reserved max bytes:
  - vanilla: `17857249280`
  - care_full: `17865637888`
- Loss summary:
  - vanilla: first `3.0568` -> last `2.0495`
  - care_full: first `12.9027` -> last `8.4131`

### 6.3 Strict-generation metrics

From per-split `repair_summary.json`:

- vanilla valid: strict valid `0.47`, strict invalid `0.53`, strict confidence `0.47`
- vanilla test: `0.43`, `0.57`, `0.43`
- care_full valid: `0.47`, `0.53`, `0.47`
- care_full test: `0.35`, `0.65`, `0.35`

Strict-only ranking slice (computed on strict-valid rows):

- vanilla valid (`n=47`): `HR@5=0.2979`, `NDCG@5=0.2542`, `MRR@5=0.2401`
- vanilla test (`n=43`): `0.1860`, `0.1139`, `0.0907`
- care_full valid (`n=47`): `0.2766`, `0.2180`, `0.1993`
- care_full test (`n=35`): `0.2000`, `0.1434`, `0.1248`

### 6.4 Safe-repaired metrics (separate family)

- usable ranking after repair: `1.00` for all adapter/split cells
- invalid after repair: `0.00` for all cells
- confidence available after repair:
  - unchanged from strict confidence (repair does not hallucinate confidence)
- repaired ranking metrics (`eval/metrics.json`):
  - vanilla valid: `HR@5=0.32`, `NDCG@5=0.2469`, `MRR@5=0.2233`
  - vanilla test: `0.20`, `0.1441`, `0.1260`
  - care_full valid: `0.32`, `0.2465`, `0.2227`
  - care_full test: `0.18`, `0.1189`, `0.0990`
- repair reason counts:
  - vanilla valid: `none=47`, `strict_json_invalid_then_candidate_completion=53`
  - vanilla test: `none=43`, `...=57`
  - care_full valid: `none=47`, `...=53`
  - care_full test: `none=35`, `...=65`

Critical interpretation:

- Safe-repaired metrics are **engineering usability** metrics, not pure generation-quality metrics.

### 6.5 Strictgen attempt C (negative; inference-only)

See `docs/PILOT_CARE_LORA_100U.md` for commands and decode knobs. **Accepted as a negative result:**

- **Mainline CARE-LoRA strict JSON validity** remains the **original 100u pilot** band **~0.35–0.47** (`repair_summary.json` under `..._pilot/`). **Strict output is not fixed**; do not claim otherwise.
- Strictgen (attempt C + **v3** re-inference) **did not improve strict validity vs the registered pilot**: latest on-disk `repair_summary.json` under `.../strictgen/` matches pilot strict rates (**vanilla 0.47 / 0.43**, **CARE 0.47 / 0.35**). Intermediate strictgen snapshots with lower CARE strict cells are **superseded** by the current files.
- **`usable_ranking_rate_after_safe_repair=1.00`** still holds in strictgen outputs, but that is **engineering usability**, not proof of strict generative quality.

**Paper-result / scaling implication:** CARE-LoRA remains **blocked** for paper-result scaling until **strict generation** improves (true constrained decoding) or the method shifts to **non-generative scoring** over candidates.

---

## 7) Decision gate (A–E)

### A. Is DeepSeek 100u stable enough for c99?

**Partially**. Beauty/electronics look reasonably stable; books/movies still have high invalid-output rates and weaker confidence quality.  
Recommendation: do not treat c99 as fully de-risked across all domains yet.

### B. Is CARE rerank strong enough as main method?

**Keep as main candidate method, but still pilot/ablation-labeled.**  
Signal exists (especially beauty exposure shift), but cross-domain consistency is limited.

### C. Is CARE-LoRA ready for 100u multi-domain?

**No.**  
Strict JSON validity (`~0.35–0.47`) on the **original beauty 100u pilot** remains the primary blocker for multi-domain expansion. **Strictgen attempt C** (decode/prompt iteration only) is an **accepted negative result** and does **not** change this gate.

### D. Should RecBole be expanded/adapted/kept?

**Keep as standard baseline track now; consider adaptation later.**  
RecBole protocol gap vs LLM candidate protocol is still real and should remain explicit.

### E. Exact next phase recommendation?

See section 8 (single recommended next phase).

---

## 8) Recommended next phase (single choice)

Selected next phase:

- **candidate_size=99 DeepSeek pilot + CARE rerank on those outputs** (API / rerank mainline), still `run_type=pilot` and `is_paper_result=false`.

Rationale:

- **CARE-LoRA strict JSON validity** on the beauty 100u pilot remains **~0.35–0.47**; **strictgen** (C + v3) did **not** raise strict validity **above** that pilot (latest strictgen `repair_summary` matches pilot strict rates) and remains **accepted as negative** for blocker clearance. Safe-repaired **`usable_ranking=1.00`** is **engineering usability**, not proof that strict generative JSON is solved. **Do not run a full experiment** or claim LoRA strict output is fixed.
- The API + rerank stack already supports the next scaling knob (**c99**) without depending on LoRA free-form JSON.
- LoRA remains a **blocked** paper-result track until **strict generation** or **true constrained decoding** / **non-generative scoring** lands.

**Recommended LoRA-side research (parallel, not mainline gate unblocker by itself):** true constrained decoding (grammar / token masks over the candidate set) or a non-generative scoring head—**not** further decode-only heuristics on ASIN strings.

Not recommended yet:

- full experiment preparation (blockers still open)
- all-domain LoRA scale-up
- treating CARE-LoRA repaired metrics as paper-result evidence of generation quality

---

## Final audit stance

100-user pilot phases are complete and useful for gating decisions, but they remain pilot evidence only.  
**Mainline:** proceed toward a **c99 DeepSeek + CARE rerank** pilot while keeping protocol gaps explicit. **CARE-LoRA:** strict generation remains the hard blocker; strictgen (C + v3) is a **negative** control on decode-only fixes (**no strict lift vs pilot** on latest disk)—continue only as **constrained decoding / non-generative** research, not as the primary scaling path until strict validity materially improves.
