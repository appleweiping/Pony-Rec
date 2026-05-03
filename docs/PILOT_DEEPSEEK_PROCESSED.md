# DeepSeek pilot on reprocessed processed-source outputs

This document records a **small API pilot** (`run_type=pilot`, `backend_type=api`, `is_paper_result=false`) on candidate JSONL produced by `reprocess_processed_source`, using **`deepseek-v4-flash`** and listwise prompts. No RecBole, LoRA, full-data, or paper-labeled runs.

## Scope

| Setting | Value |
|--------|--------|
| Domains | `amazon_beauty`, `amazon_books`, `amazon_electronics`, `amazon_movies` |
| Users per domain / split | 20 |
| Candidate list size | 19 negatives + target (20 IDs in file; inference `topk=19` for full candidate ranking) |
| Seed | 42 |
| Backend config | `configs/backends/deepseek_v4_flash.yaml` |
| Model (API) | `deepseek-v4-flash` |
| Prompt template | `listwise_ranking_v1` |
| Reprocessed inputs | `outputs/reprocessed_processed_source/<domain>/{valid,test}_candidates.jsonl` |
| Processed item text | `data/processed/<domain>/items.csv` (merged with JSONL `candidate_texts` / titles) |
| Pilot output root | `outputs/pilots/deepseek_v4_flash_processed_20u_c19_seed42/` |

## Commands run

From repository root (`/home/ajifang/projects/fresh/uncertainty-llm4rec`), with API key loaded (key is never echoed):

```bash
set -a
source .env
set +a
```

Inspect reprocessed layout (sample):

```bash
find outputs/reprocessed_processed_source -maxdepth 3 -type f | sort | head -100
```

Pilot driver (inference + per-split `evaluate` + aggregate CSV + summary JSON):

```bash
.venv_lora/bin/python3.11 -m src.cli.run_pilot_reprocessed_deepseek
```

Defaults match the table above (`--reprocess_dir outputs/reprocessed_processed_source`, `--output_root outputs/pilots/deepseek_v4_flash_processed_20u_c19_seed42`, four domains, `valid` and `test`).

Full test suite after code changes:

```bash
.venv_lora/bin/python3.11 -m pytest
```

Result: **70 passed**.

## Pipeline

- **`src.cli.run_pilot_reprocessed_deepseek`**: httpx-only DeepSeek `/chat/completions` (avoids OpenAI client / `pydantic` in this venv). Writes per `{domain}/{split}/predictions/`: `raw_responses.jsonl`, `parsed_responses.jsonl`, `rank_predictions.jsonl`; per `{domain}/{split}/manifest.json` via shared manifest helpers; then **`python -m src.cli.evaluate`** on `rank_predictions.jsonl`, passing **`--candidates_source_path`** so eval manifests list the reprocess JSONL alongside predictions.
- **`src.cli.evaluate`**: HR@K, Recall@K, NDCG@K, MRR@K (and MAP@K), ECE/MCE/Brier on `raw_confidence` vs hit @1, exposure metrics, `risk_coverage.csv`.

## Output layout (pilot)

Under `outputs/pilots/deepseek_v4_flash_processed_20u_c19_seed42/`:

- `{domain}/{split}/predictions/raw_responses.jsonl` — model text, full `api_response`, `token_usage`, `latency_seconds`, `error`
- `{domain}/{split}/predictions/parsed_responses.jsonl` — parse validity, hallucination / duplicate / missing-confidence / not-in-candidate flags, ranked IDs, confidence
- `{domain}/{split}/predictions/rank_predictions.jsonl` — rows consumed by `evaluate` (includes `config_hash`, `prompt_template_id`, `run_type`, `backend_type`, `is_paper_result`)
- `{domain}/{split}/manifest.json` — run manifest (command, paths, model, prompt, candidate size)
- `{domain}/{split}/eval/metrics.json`, `eval/manifest.json`, `eval/risk_coverage.csv`
- `pilot_run_summary.json` — per domain/split invalid rate, confidence availability, HTTP errors
- `pilot_metrics_aggregate.csv` — flattened `metrics.json` from all eight eval runs

Git commit baked into manifests/summary: `485cee9fdc544d7872c16c09a6f1efa770162c77`. Config hash for the **successful** pilot config: **`1303000896c6106b`**.

## Success / failure notes

1. **Item lookup bug (fixed before any successful API batch):** `_load_item_lookup` used `itertuples` with string column names → `TypeError`. Replaced with `iterrows`-based lookup.
2. **First live batch (discarded):** With default DeepSeek V4 **thinking** behavior, the API returned long `reasoning_content` and **empty `content`**, while `max_tokens` was 800 — completions hit `finish_reason: length` before any JSON reached `content`. Parser saw empty strings → **invalid_output_rate = 1.0** everywhere, ranking metrics zero.
3. **Fix for the successful pilot:** Honor `generation.thinking_mode: false` in YAML by sending `"thinking": {"type": "disabled"}` on the chat request, and raise **`max_tokens` to 4096** in `configs/backends/deepseek_v4_flash.yaml`. Optional fallback: if `content` is still empty, use `reasoning_content` (not relied on for the final successful run).

**Final pilot:** completed all four domains × (`valid`, `test`) with **zero HTTP errors** in `pilot_run_summary.json`.

## Invalid output rate & confidence availability

Rates below are from `pilot_run_summary.json` (successful run).

| Domain | Split | n | invalid_output_rate | confidence_availability_rate |
|--------|-------|---|---------------------|------------------------------|
| amazon_beauty | valid | 20 | 0.25 | 0.75 |
| amazon_beauty | test | 20 | 0.10 | 0.90 |
| amazon_books | valid | 20 | 0.70 | 0.40 |
| amazon_books | test | 20 | 0.55 | 0.45 |
| amazon_electronics | valid | 20 | 0.05 | 0.95 |
| amazon_electronics | test | 20 | 0.10 | 0.90 |
| amazon_movies | valid | 20 | 0.45 | 0.55 |
| amazon_movies | test | 20 | 0.45 | 0.55 |

**Pilot-wide (160 rows):** invalid fraction ≈ **33.1%** (53/160); confidence present on **67.5%** (108/160) of parsed rows.

## Ranking metrics (HR / Recall / NDCG / MRR @ 1,5,10)

Summarized from `pilot_metrics_aggregate.csv` (each file is 20 users; metrics include all rows passed to `evaluate`, including invalid parses where ranking may be empty — same convention as `src.cli.evaluate`).

| Path (eval) | HR@1 | NDCG@10 | MRR@10 | Recall@10 |
|-------------|------|---------|--------|-----------|
| …/amazon_beauty/valid/eval/metrics.json | 0.15 | 0.333 | 0.230 | 0.70 |
| …/amazon_beauty/test/eval/metrics.json | 0.20 | 0.274 | 0.237 | 0.40 |
| …/amazon_books/valid/eval/metrics.json | 0.50 | 0.615 | 0.573 | 0.75 |
| …/amazon_books/test/eval/metrics.json | 0.25 | 0.491 | 0.396 | 0.80 |
| …/amazon_electronics/valid/eval/metrics.json | 0.10 | 0.344 | 0.225 | 0.75 |
| …/amazon_electronics/test/eval/metrics.json | 0.15 | 0.370 | 0.259 | 0.75 |
| …/amazon_movies/valid/eval/metrics.json | 0.20 | 0.390 | 0.302 | 0.70 |
| …/amazon_movies/test/eval/metrics.json | 0.20 | 0.408 | 0.332 | 0.65 |

Full JSON for each split is in the corresponding `eval/metrics.json` (includes MAP@K, exposure, calibration block).

## Confidence–correctness diagnostic

`evaluate` reports **calibration** on hit-at-1 vs `raw_confidence` (with default 0.5 when missing): **ECE, MCE, Brier, avg_confidence, accuracy** per split. Example (`amazon_beauty/valid`): accuracy @1 = 0.15, avg_confidence = 0.5, ECE = 0.35, Brier = 0.25 (see that file for exact numbers). Interpreting ECE on sparse 20-user pilots is noisy; use mainly as a sanity check until invalid rate drops.

## Latency and token logging

- **Latency:** each `raw_responses.jsonl` row includes `latency_seconds` (wall time for the HTTP round trip).
- **Tokens:** `token_usage` / `api_response.usage` are populated when the API returns usage (observed in practice: `prompt_tokens`, `completion_tokens`, `total_tokens`).

## Blockers before scaling to ~100 users / domain

1. **Domain-specific JSON contract:** `amazon_books` (and to a lesser extent `amazon_movies`) show **high invalid_output_rate** on a 20-user slice — likely prompt length, item text ambiguity, or model drift on long listwise JSON. Tune prompt, shorten candidate text, add repair/retry, or switch to a stricter output schema / `response_format` if supported without breaking the parser.
2. **Thinking vs non-thinking:** Any regression that omits `"thinking": {"type": "disabled"}` or lowers `max_tokens` can silently zero out `content` again — add CI or smoke checks on response shape.
3. **Cost and rate limits:** 100 users/domain × 2 splits × 4 domains = 800 calls at pilot candidate size; linear in prompt tokens. Monitor quotas and backoff (`runtime` in YAML).
4. **Evaluation semantics:** current `evaluate` scores **all** rows including invalid parses; for paper-scale runs you may want a secondary metric restricted to `is_valid` rows or impute ranks — not changed in this pilot.

## Related entrypoints

- Reprocess driver: `src.cli.reprocess_processed_source`
- Standard API infer (OpenAI stack): `src.cli.infer` (requires full optional deps such as `pydantic` in the environment)
- This pilot: **`src.cli.run_pilot_reprocessed_deepseek`**
