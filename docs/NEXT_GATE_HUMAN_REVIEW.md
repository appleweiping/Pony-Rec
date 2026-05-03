# Human review: c99 DeepSeek + CARE rerank gate

Purpose: concise pilot-only summary so reviewers can decide whether to proceed.  
Scope: `candidate_size=99`, `n=100` users per split per domain, `seed=42`. **Not** a paper result (`is_paper_result=false`).  
Evidence: `docs/PILOT_C99_DEEPSEEK_CARE_RERANK.md`, `docs/PILOT_100U_AUDIT_AND_NEXT_GATE.md`, `docs/PILOT_CARE_LORA_100U.md`, and CSVs under `outputs/pilots/deepseek_v4_flash_processed_100u_c99_seed42/` and `outputs/pilots/care_rerank_deepseek_v4_flash_processed_100u_c99_seed42/`.

---

## 1) Is c99 DeepSeek stable enough to continue?

**Verdict: continue only as controlled pilot work; not “stable enough” to treat as de-risked for scaling or a full study.**

**Invalid output rate** (from `pilot_run_summary.json`, tabulated in `PILOT_C99` §3):

| Domain | valid | test |
| --- | --- | --- |
| amazon_beauty | 0.20 | 0.15 |
| amazon_books | 0.50 | 0.59 |
| amazon_electronics | 0.35 | 0.29 |
| amazon_movies | 0.32 | 0.41 |

Rates did **not** blow up toward ~1.0; rerank was **not** aborted. Versus c19 on the same summary: beauty and electronics **invalid rose**; books **improved modestly** but stays **~0.5–0.59**; movies **mixed** (valid better than c19, test still high).

**Confidence availability** (`care_rerank_aggregate.csv` / pilot summary): beauty and electronics show **high** availability (e.g. beauty valid **0.96**, test **0.95**; electronics **0.95–0.97**). Books and movies remain **weaker** (books valid/test **0.68 / 0.59**; movies **0.80 / 0.72**).

**Books / movies risk:** **High.** Invalid fractions stay large on books; movies test invalid **0.41** with weaker confidence coverage. Calibration rows for books often have **`nan` head/mid confidence means** (sparse buckets in `calibration_aggregate.csv`), which is an additional interpretability risk.

---

## 2) Does c99 make uncertainty phenomena clearer than c19?

**Verdict: no — harder listwise (K=99) generally makes miscalibration and overconfident-wrong signals worse or more extreme, not “cleaner,” versus c19.**

**High-confidence wrong** (`calibration_aggregate.csv`): beauty valid/test **~0.94 / 0.95** (c19 was ~0.84 / 0.90); electronics test **~0.94** (c19 ~0.87); movies test **~0.94** (c19 ~0.85). Books test **~0.88** vs c19 ~0.54 on the same diagnostic table in `PILOT_C99` §5 — **directionally worse** in several cells.

**ECE / Brier:** same source — beauty and electronics show **higher** ECE and Brier at c99 than c19 on both splits; movies test ECE/Brier also **worse** than c19. This is consistent with a **harder ranking task**, not a cleaner uncertainty readout.

**Head / mid / tail confidence:** beauty shows **higher** bucket means at c99 than c19 (per `PILOT_C99` §5). Books: head/mid **`nan`** (empty buckets). Electronics: head often **`nan`**, mid/tail elevated. Movies: bucket means **shift** (e.g. test mid **0.815**) — useful as a stress test, not as a simplified story.

**Exposure shift** (`exposure_shift.csv`): at c99, **only amazon_beauty** shows small **head top-1 share** moves under `care_full` (e.g. valid **0.03 → 0.02**, test **0.04 → 0.03**). Books, electronics, and movies: **no change** in recorded top-1 bucket shares for `care_full` vs `original_deepseek` (books/electronics remain all-tail top-1 **1.0**).

**`high_confidence_wrong_changes.csv`:** row-level file is large; aggregate rerank columns in `care_rerank_aggregate.csv` show **HC-wrong before ≈ after** for most variants (e.g. beauty test **0.9 → 0.9** for `care_full`). Rerank does **not** materially collapse HC-wrong in this pilot.

---

## 3) Does `care_full` beat or behave better than the listed baselines?

Source: `care_rerank_aggregate.csv`.

| Baseline | vs `care_full` (c99) |
| --- | --- |
| **original_deepseek** | **No consistent beat.** Same **HR@5** on every domain/split in the aggregate. Beauty: tiny **NDCG@5 / MRR@5** deltas vs original (noise-scale). |
| **confidence_only** | **Identical** metrics on all domain/split rows. |
| **popularity_penalty_only** | **No HR@5 gain.** Beauty: small **NDCG@5 / MRR@5** differences vs popularity-only; other domains flat on HR@5. |
| **uncertainty_only** | **Identical** to `original_deepseek` / `confidence_only` on listed HR@5 cells. |

---

## 4) Is CARE rerank strong enough to remain a main method candidate?

**Choice: maybe, keep as ablation/diagnostic.**

Rationale: at c99, **`care_full` does not separate** from **`confidence_only`** or **`uncertainty_only`** on aggregate ranking metrics; gains vs **`original_deepseek`** are **at most** tiny beauty NDCG/MRR moves with **unchanged HR@5**. Exposure shift is **material only on beauty**. That supports **keeping the machinery** for diagnostics and beauty-focused ablations, **not** promoting it to a primary “wins on API listwise” method on this evidence alone.

---

## 5) What should be the next phase? (exactly one)

**prompt/parser hardening for unstable domains**

Rationale: c99 shows **books/movies** still dominate invalid-rate and weak-bucket risk; beauty/electronics **invalid rose** vs c19 under larger K. Until parse/invalid and calibration are under control per domain, **tuning CARE coefficients** or **a larger c99 API pilot** mostly stress a **broken observability** layer. **CARE-LoRA** remains a **separate** track blocked on strict JSON (`PILOT_CARE_LORA_100U.md`); the right LoRA research is **constrained decoding / non-generative** scoring, but the **immediate** API-listwise gate is **input/output contract stability** on weak domains—not more unbounded API volume.

---

## 6) What must not be done yet?

- **No full experiment** until c99 findings are manually accepted and domain follow-ups are chosen (`PILOT_C99` gate status).
- **No paper-result marking** (`is_paper_result=false` on manifests; this summary is pilot evidence only).
- **No repaired LoRA metric presented as strict generation quality** — repaired slates are **engineering usability** only (`PILOT_CARE_LORA_100U.md`, `PILOT_100U_AUDIT` §6.4).
- **No RecBole / LLM same-protocol claim** — RecBole remains a separate atomic baseline track (`PILOT_100U_AUDIT` §5).

---

## Reviewer checklist (one line each)

| Question | Short answer |
| --- | --- |
| Stable enough to continue blindly? | **No** — pilot-only; books/movies weak; beauty/electronics invalid up vs c19. |
| Clearer uncertainty than c19? | **No** — ECE/Brier/HC-worse generally **worse** at c99. |
| `care_full` beats baselines? | **No** meaningful HR@5 separation; identical to confidence/uncertainty paths. |
| CARE rerank role? | **Maybe** — ablation/diagnostic, not sole main method on this run. |
| Next phase? | **Prompt/parser hardening** (unstable domains first). |
| Hard stops? | No full experiment, no paper flags, no repaired-LoRA-as-strict, no RecBole–LLM protocol equivalence. |
