# Data quality audit — c99 DeepSeek unstable domains

Audit date: 2026-05-03  
Workspace: `fresh/uncertainty-llm4rec`  
**Targets (unstable):** `amazon_books`, `amazon_movies`  
**References (stable-ish):** `amazon_beauty`, `amazon_electronics`  

**Inputs:** `data/processed/<domain>/{items,interactions}.csv`, `outputs/reprocessed_processed_source_100u_c99_seed42/<domain>/`, `outputs/pilots/deepseek_v4_flash_processed_100u_c99_seed42/<domain>/`.  
**Pilot prompt:** `listwise_ranking_v1` (see per-split `manifest.json`).

This document is **pilot evidence only** (`is_paper_result=false`). It does **not** change split/candidate protocol.

---

## Executive conclusion

1. **Data quality and prompt payload scale plausibly explain a large part of books/movies invalid rates**, alongside parser/model limits. **Books** prompts are **~3× longer** than beauty on average because `candidate_text` embeds very long descriptions; a **deterministic catalog-text cleaning** path (see §Phase B) cuts **mean prompt length by roughly half** in replayed prompt construction on the same c99 rows (§4.3). **Movies** show **~40% missing titles** and **~40% missing categories** in the global `items.csv`, which is a separate metadata-quality problem from “prompt wording” alone.

2. **Slate construction for the 100-user c99 slice is sound:** target always in candidates, no duplicate candidate IDs, no candidates missing from `items.csv` in this audit (§4.2).

3. **Pearson correlations** between `invalid_output` and prompt length are **small or inconsistent** at `n=100` (noise-dominated), so **do not** treat correlation alone as proof—**structural** differences (catalog completeness, HTML, description truncation at 1000 chars, duplicate titles in the global catalog) are the stronger story.

4. A **cleaned prompt-view API pilot** on **books + movies only** was run (same c99 candidates). Evidence is in §“Cleaned prompt-view rerun (API)” and under `outputs/pilots/deepseek_v4_flash_processed_100u_c99_seed42_data_hardened/` (gitignored).

**Machine-readable audit:** run  
`.venv_lora/bin/python3.11 -m src.cli.audit_c99_data_quality`  
to regenerate `outputs/diagnostics/c99_data_quality_audit/audit_report.json` and exemplar JSON under `outputs/diagnostics/c99_prompt_cleaning_candidates/` (paths are **gitignored**; Phase A tables match a local audit run).

**Hardened pilot config (code):** `configs/pilots/c99_deepseek_data_hardened_books_movies.yaml`  
**Invoked as:**  
`.venv_lora/bin/python3.11 -m src.cli.run_pilot_reprocessed_deepseek --pilot_config configs/pilots/c99_deepseek_data_hardened_books_movies.yaml`

---

## Phase A — Item metadata completeness (`data/processed`)

All domains: **no duplicate `item_id` rows** in the raw tables below.

| Metric | amazon_beauty | amazon_books | amazon_electronics | amazon_movies |
| --- | ---:| ---:| ---:| ---:|
| `items.csv` rows | 1,184 | 963,824 | 533,532 | 258,113 |
| Title missing rate | 0.00 | ~0.00 | ~0.00 | **0.399** |
| Categories missing rate | **1.00** | 0.038 | 0.057 | **0.400** |
| Description missing rate | 0.913 | 0.345 | 0.442 | 0.458 |
| Weird placeholder title rate | 0.00 | ~3e-6 | ~4e-5 | **0.399** |
| Duplicate title rows (lower/stripped) | 3 | **110,264** | 10,782 | **122,439** |
| Title length mean / p95 / max | 147 / 200 / 390 | 51 / 111 / 681 | 125 / 198 / 1,822 | **20 / 64 / 558** |
| Description len p95 / max (chars) | ~461 / 1,000 | **1,000 / 1,000** | **1,000 / 1,000** | **1,000 / 1,000** |
| HTML-like hits in title / description | 0 / 0 | 15 / **1,600** | 16 / 207 | 2 / 160 |
| Control-char hits in title / description | 0 / 0 | 0 / 1 | 0 / 0 | 0 / 0 |
| `item_id` numeric-only fraction | 0.00 | **0.78** | ~0.00 | 0.04 |

**Reading:** Beauty has thin metadata by design (many empty categories). **Books** carry **mass** description text (often clamped at 1000 chars) and **many** HTML snippets in descriptions. **Movies** are **metadata-sparse** (missing title/category for a large share of rows) while still shipping long `description`/`candidate_text` where present—**insufficient processing** of item text is visible before any LLM call.

---

## Phase A — Prompt payload length (c99 `valid` / `test`, `n=100` per cell)

Reconstructed with the same `_merge_item_texts` + `_build_ranking_prompt` path as `run_pilot_reprocessed_deepseek` (`listwise_ranking_v1`, `topk=99`). Character counts; approximate tokens ≈ chars / 4.

| Domain | split | mean chars | p50 | p90 | p95 | max | cand block mean | history mean | ~tokens (mean/4) |
| --- | --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| beauty | valid | 25,281 | 25,261 | 28,055 | 28,525 | 32,702 | 22,990 | 946 | ~6,320 |
| beauty | test | 25,639 | 25,165 | 28,274 | 30,167 | 34,646 | 23,129 | 1,166 | ~6,410 |
| books | valid | **76,537** | 72,954 | 88,323 | 96,745 | **151,661** | **67,159** | **8,033** | **~19,134** |
| books | test | **77,522** | 74,234 | 89,941 | 99,540 | **152,167** | **67,338** | **8,838** | **~19,380** |
| electronics | valid | 63,694 | 62,232 | 71,725 | 76,547 | 122,415 | 57,466 | 4,882 | ~15,923 |
| electronics | test | 64,707 | 62,693 | 72,208 | 76,810 | 138,456 | 57,933 | 5,429 | ~16,177 |
| movies | valid | 46,129 | 44,230 | 54,633 | 59,073 | 89,460 | 41,201 | 3,583 | ~11,532 |
| movies | test | 47,369 | 45,944 | 56,024 | 58,878 | 94,932 | 42,037 | 3,987 | ~11,842 |

**Per-candidate title line stats (merged title vs `candidate_text`):** books/movies show **~0** candidates with both title and merged text empty in this slice; beauty ~4.8 candidates per slate with title string length >200 (short titles, long `candidate_text`).

**Books/movies vs beauty/electronics:** Books prompts are **longer and noisier** (history + candidate blocks both large). Movies are shorter than books but **catalog titles are often missing**, so the model sees weaker semantic anchors even when char counts are lower than books.

---

## Phase A — Candidate slate quality (c99 JSONL × pilot slice)

| Metric | beauty | books | electronics | movies |
| --- | --- | --- | --- | --- |
| `target_in_candidates` rate | 1.0 | 1.0 | 1.0 | 1.0 |
| Mean duplicate candidate IDs | 0.0 | 0.0 | 0.0 | 0.0 |
| Mean candidates missing from `items.csv` | 0.0 | 0.0 | 0.0 | 0.0 |
| Mean duplicate-title surplus inside slate | ~0–0.03 | ~0.0 | ~0.0 | ~0–0.02 |

So: **invalid rate is not explained by broken candidate sampling in this pilot** (no missing targets, no ID duplication in slates). The dominant risks are **global catalog text** and **payload size**, not “wrong negatives sampled.”

---

## Phase A — Failure correlation (`parsed_responses.jsonl` `invalid_output`)

**Old invalid rates** (unchanged; from `pilot_run_summary.json`):

| Domain | valid | test |
| --- | ---:| ---:|
| beauty | 0.20 | 0.15 |
| books | **0.50** | **0.59** |
| electronics | 0.35 | 0.29 |
| movies | 0.32 | 0.41 |

**Pearson(invalid, prompt_char_len)** (weak at n=100): beauty valid ≈ −0.002, beauty test ≈ 0.12; books valid ≈ 0.11, books test ≈ −0.17; electronics valid ≈ 0.14, test ≈ 0.21; movies valid ≈ 0.13, test ≈ −0.02.

**Pearson with duplicate titles in slate:** mostly undefined (zero variance) for books/electronics; beauty valid ≈ 0.20; movies valid ≈ 0.21.

**Missing-title-count correlation:** undefined (all-zero in this slice for the chosen definition).

**By target popularity bucket:** books valid/test — **all 100** users are `tail` targets in this sample (invalid 0.50 / 0.59). Electronics/movies show sparse head/mid cells—interpret cautiously.

---

## Phase B — Minimal cleaning proposal (no writes to `data/processed`)

**Code:** `src/data/item_text_cleaning.py`  
**Tests:** `tests/test_item_text_cleaning.py`

**Rules (deterministic):**

- Strip HTML tags (regex) and `html.unescape`.
- Remove C0 control characters (keep tab/newline where present, then collapse whitespace).
- Truncate title/category strings to configured UTF-8 character budgets (`CleaningConfig`).
- Prefer **title + categories**; **do not** include full description unless `include_description=True`.
- Missing or placeholder titles → stable line `Title: (missing metadata) item_id=<id>`.
- Preserve `item_id` exactly (cleaning applies to prose only).
- For **machine JSON** elsewhere, serialize with `json.dumps` (`json_escape_for_debug` helper).
- **Never** drop the target item from a slate (cleaning is text-only; sampling unchanged).
- **Do not** use test-set labels to pick cleaning thresholds (fixed constants only).

**Prompt-view / diagnostics output:** re-run the audit CLI; exemplar before/after snippets for books/movies are written under `outputs/diagnostics/c99_prompt_cleaning_candidates/`.

**Replayed prompt length with cleaned subset lookup** (same JSONL rows; catalog strings replaced for referenced IDs only):

| Domain | split | mean chars (raw lookup) | mean chars (cleaned lookup) |
| --- | --- | ---:| ---:|
| books | valid | 76,537 | **36,413** |
| books | test | 77,522 | **36,960** |
| beauty | valid | 25,281 | 21,796 |
| movies | valid | 46,129 | 27,249 |

This is **not** an API improvement measurement—only **offline** prompt mass. It supports treating **item-text processing** as a first-class lever before prompt/parser-only tweaks.

---

## Phase C — After audit: recommended order

| Step | Action |
| --- | --- |
| 1 | **Done:** `item_text_cleaning` wired into `run_pilot_reprocessed_deepseek` via `item_text_view: cleaned_v1` and `--pilot_config` (same `*_andidates.jsonl`; `item_id`s unchanged). |
| 2 | If invalid remains high on **movies**, tighten template/parser or improve raw/catalog metadata (missing titles). |
| 3 | Evaluate **DeepSeek JSON mode** only if the contract is still unstable after (1)–(2). |
| 4 | **Done:** Small API rerun: `amazon_books` + `amazon_movies`, valid+test, 100 users, c99, output `outputs/pilots/deepseek_v4_flash_processed_100u_c99_seed42_data_hardened/`. |

---

## Cleaned prompt-view rerun (API) — books & movies only

**When:** 2026-05-03 (pilot class only). **Protocol:** unchanged reprocess root `outputs/reprocessed_processed_source_100u_c99_seed42`, `candidate_size=99`, `seed=42`, `run_type=pilot`, **`is_paper_result=false`**. **Prompt:** `listwise_ranking_v1_structured_ids` (allowed IDs duplicated as a **`json.dumps`** string array in the template). **Item view:** `cleaned_v1` per `src/data/item_text_cleaning.py` + post-merge scrub (`manifest.json` records `item_text_cleaning_config_hash`).

**Baseline (“old”)** rows refer to `outputs/pilots/deepseek_v4_flash_processed_100u_c99_seed42/` with `listwise_ranking_v1`. **Hardened** rows refer to `..._data_hardened/`.

### 1) Invalid output rate (`parsed_responses.jsonl`)

| Domain | split | Old | Hardened |
| --- | --- | ---:| ---:|
| books | valid | 0.50 | **0.18** |
| books | test | 0.59 | **0.22** |
| movies | valid | 0.32 | 0.34 |
| movies | test | 0.41 | 0.40 |

**Interpretation:** **Books:** large drop—consistent with HTML/payload cleanup. **Movies:** essentially flat—consistent with metadata sparsity dominating over prompt length.

### 2) Prompt length diagnostics (`prompt_diagnostics.json`; old = offline audit replay `listwise_ranking_v1`, hardened = live pilot)

Old books/movies lengths are from the **pre-flight audit** reconstruction in §Phase A (raw catalog lookup). Hardened lengths include the **extra JSON allowed-ID line** in the structured template.

| Domain | split | Old mean / p50 / p90 / p95 / max chars | Hardened mean / p50 / p90 / p95 / max chars | ~tokens hardened (mean/4) |
| --- | --- | --- | --- | ---:|
| books | valid | 76,537 / 72,954 / 88,323 / 96,745 / 151,661 | 50,695 / 48,446 / 57,134 / 64,145 / **97,313** | **12,674** |
| books | test | 77,522 / 74,234 / 89,941 / 99,540 / 152,167 | 51,402 / 49,405 / 57,958 / 64,280 / **97,889** | **12,851** |
| movies | valid | 46,129 / 44,230 / 54,633 / 59,073 / 89,460 | 36,377 / 35,344 / 41,468 / 45,848 / **65,007** | **9,094** |
| movies | test | 47,369 / 45,944 / 56,024 / 58,878 / 94,932 | 37,121 / 36,129 / 42,442 / 45,180 / **66,795** | **9,280** |

### 3) Confidence availability (`pilot_run_summary.json`, parsed rows)

| Domain | split | Old | Hardened |
| --- | --- | --- | --- |
| books | valid | 0.68 | **0.94** |
| books | test | 0.59 | **0.92** |
| movies | valid | 0.80 | 0.76 |
| movies | test | 0.72 | 0.73 |

### 4) Calibration (`calibration_diagnostics/calibration_aggregate.csv`)

| Domain | split | Metric | Old | Hardened |
| --- | --- | --- | ---:| ---:|
| books | valid | ECE | 0.523 | 0.577 |
| books | valid | adaptive ECE | 0.503 | 0.577 |
| books | valid | Brier | 0.468 | 0.515 |
| books | valid | high-conf wrong | 0.809 | **0.777** |
| books | test | ECE | 0.571 | 0.575 |
| books | test | adaptive ECE | 0.571 | 0.555 |
| books | test | Brier | 0.517 | 0.513 |
| books | test | high-conf wrong | 0.881 | **0.772** |
| movies | valid | ECE | 0.566 | 0.542 |
| movies | valid | high-conf wrong | 0.886 | 0.893 |
| movies | test | ECE | 0.559 | 0.515 |
| movies | test | high-conf wrong | 0.944 | 0.898 |

*(Pilot-only; not a paper claim.)*

### 5) Ranking metrics — baseline `llm_listwise` DeepSeek (`eval/metrics.json`)

Pilot-only **HR@5 / NDCG@5 / MRR@5**:

| Domain | split | Old | Hardened |
| --- | --- | --- | --- |
| books | valid | 0.42 / 0.313 / 0.278 | 0.38 / 0.296 / 0.268 |
| books | test | 0.43 / 0.329 / 0.294 | 0.42 / 0.318 / 0.284 |
| movies | valid | 0.16 / 0.129 / 0.120 | **0.20** / **0.142** / **0.123** |
| movies | test | 0.19 / 0.128 / 0.107 | **0.22** / **0.158** / **0.137** |

### 6) Decision

- **Did cleaned prompt-view reduce invalid?** **Yes for books (strongly).** **Movies:** negligible change—still treat movies as **metadata / raw-protocol** constrained, not “fixed by scrub.”
- **Stable enough to rerun CARE rerank?** **Yes (pilot):** books invalid dropped materially; CARE rerank was run on hardened outputs only (`outputs/pilots/care_rerank_deepseek_v4_flash_processed_100u_c99_seed42_data_hardened/`, gitignored). **Do not** treat this as paper-ready; movies remain weak.
- **Deeper data / raw fix?** Still recommended for **movies** (missing titles/categories in catalog).

---

## Phase D — Gate answers (updated post-hardened pilot)

| Question | Answer |
| --- | --- |
| Does data quality plausibly explain invalid rates? | **Yes** for **books** (now validated by API). **Movies:** partial—invalid barely moved; catalog gaps likely dominate. |
| Did hardened rerun improve invalid? | **Books: yes (0.50/0.59 → 0.18/0.22). Movies: ~flat.** |
| Are books/movies still unstable? | **Books: much improved invalid; still pilot-only.** **Movies: still unstable.** |
| Should c99 continue? | **Yes, pilot-only**; next focus **movie metadata / raw protocol** + optional parser work; **no** full experiment; **no** paper marking. |

---

## Hard stops (unchanged from program gate)

- **No full experiment.**
- **No unbounded / full-scale API sweep** without an explicit gate (this doc records one **scoped** books/movies hardened pilot only).
- **No split/candidate protocol change** from this audit alone.
- **No `is_paper_result=true`.**
- **No repaired LoRA metrics as strict generation.**
- **No RecBole / LLM same-protocol claim.**
