"""C-CRP v3 scoring — SEED- and MODEL-parameterized variant.

Identical pipeline to experiments/rsc/run_ccrp_v3_domain.py (same prompt, same
parse_score regex, same metric computation, same output layout:
scores.csv / report.json / user_ranks.jsonl) with TWO additions required by the
two gap-to-8 reviewer experiments:

  1. --seed  : threads a generation seed into vLLM SamplingParams (the shared
               VLLMBackend never sets SamplingParams.seed, so run-to-run
               variance at temp 0.1 could not be measured with it). Used for the
               MULTI-SEED run-variance CI experiment (seeds 2026/2027/2028).
  2. --model : already supported by the original script, kept here. Used for the
               SECOND-BACKBONE experiment (Llama-3.1-8B-Instruct vs Qwen3-8B).
               The probability is parsed from generated JSON text (parse_score
               regex) and the chat template is applied generically via the
               tokenizer's own apply_chat_template — so NO Qwen-specific token
               id / template is hard-coded; swapping --model is sufficient.

This script builds the vLLM LLM + SamplingParams directly (instead of through
VLLMBackend) ONLY so it can set the seed; every other knob mirrors VLLMBackend's
defaults as used by run_ccrp_v3_domain.py:
    max_model_len=1024, gpu_memory_utilization=<--gpu_mem>, enable_prefix_caching=True,
    dtype=float16, temperature=0.1, top_p=0.95, max_tokens=100,
    use_chat_template=True (enable_thinking=False), trust_remote_code=False.

The report.json gains "model", "seed", and "backbone" fields for provenance.
"""
import json, numpy as np, time, re, argparse, csv
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build_v3_prompt(history, candidate_title, candidate_text=""):
    # IDENTICAL to run_ccrp_v3_domain.py build_v3_prompt
    hist_block = "\n".join([f"- {h}" for h in history[-5:]])
    meta = candidate_text[:200] if candidate_text else ""
    desc_line = f"\nDescription: {meta}" if meta else ""
    return (
        "You are an expert recommendation system.\n\n"
        f"User purchase history (most recent first):\n{hist_block}\n\n"
        f"Candidate item:\nTitle: {candidate_title}{desc_line}\n\n"
        "Based on the purchase pattern, estimate probability (0.0-1.0) this candidate is their next purchase. "
        "Consider category alignment, attribute match, and purchase trajectory.\n\n"
        'Return ONLY JSON: {"relevance_probability": 0.0, "reason": "one sentence"}'
    )


def parse_score(text):
    # IDENTICAL to run_ccrp_v3_domain.py parse_score
    try:
        m = re.search(r'"relevance_probability"\s*:\s*([\d.]+)', text)
        if m:
            return float(m.group(1))
        m = re.search(r'(0\.\d+|1\.0)', text)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="ranking_test.jsonl path")
    parser.add_argument("--n_users", type=int, default=None, help="Limit users (None=all)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="/home/ajifang/models/Qwen/Qwen3-8B")
    parser.add_argument("--seed", type=int, default=None,
                        help="Generation seed (SamplingParams.seed + LLM seed). None=vLLM default.")
    parser.add_argument("--gpu_mem", type=float, default=0.85, help="GPU memory utilization (0-1)")
    parser.add_argument("--max_model_len", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--backbone", type=str, default=None,
                        help="Provenance label, e.g. qwen3-8b / llama3.1-8b. Default=model dir name.")
    args = parser.parse_args()

    backbone = args.backbone or Path(args.model).name

    print(f"Loading {args.data}...")
    records = [json.loads(l) for l in open(args.data)]
    if args.n_users and args.n_users < len(records):
        records = records[:args.n_users]
    print(f"Users: {len(records)}  model={args.model}  seed={args.seed}  backbone={backbone}")

    # --- vLLM init (direct, to thread seed). Mirrors VLLMBackend defaults. ---
    from vllm import LLM, SamplingParams
    print("Initializing vLLM...")
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        dtype="float16",
        enforce_eager=False,
        trust_remote_code=False,
        seed=(args.seed if args.seed is not None else 0),
    )
    tokenizer = llm.get_tokenizer()

    sampling_kwargs = dict(
        max_tokens=args.max_new_tokens,
        top_p=0.95,
        temperature=(args.temperature if args.temperature > 0 else 0.0),
    )
    if args.seed is not None:
        sampling_kwargs["seed"] = args.seed
    sampling_params = SamplingParams(**sampling_kwargs)

    # --- chat template applied GENERICALLY (no hard-coded Qwen/Llama format). ---
    def format_prompt(p):
        if not getattr(tokenizer, "chat_template", None):
            return p
        msg = [{"role": "user", "content": p}]
        try:
            return tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )

    # Build all prompts (user x candidate), user-contiguous for prefix caching.
    all_prompts, all_meta = [], []
    for rec in records:
        history = rec["history"]
        candidates = rec["candidate_titles"]
        candidate_texts = rec.get("candidate_texts", [""] * len(candidates))
        for i, (title, text) in enumerate(zip(candidates, candidate_texts)):
            all_prompts.append(format_prompt(build_v3_prompt(history, title, text)))
            all_meta.append({"user_id": rec["user_id"], "candidate_idx": i})

    n_candidates = len(records[0]["candidate_titles"])
    print(f"Total prompts: {len(all_prompts)} ({len(records)} users x {n_candidates} candidates)")

    CHUNK_USERS = 5000
    chunk_size = CHUNK_USERS * n_candidates
    start = time.time()
    results = []
    for i in range(0, len(all_prompts), chunk_size):
        chunk = all_prompts[i:i + chunk_size]
        outs = llm.generate(chunk, sampling_params)
        for o in outs:
            results.append(o.outputs[0].text.strip() if o.outputs else "")
        done = min(i + chunk_size, len(all_prompts))
        el = time.time() - start
        print(f"  [{done}/{len(all_prompts)}] {done/el:.0f} prompts/s" if el > 0 else f"  [{done}]")
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({len(all_prompts)/elapsed:.1f} samples/s)")

    user_scores = defaultdict(list)
    for meta, text in zip(all_meta, results):
        user_scores[meta["user_id"]].append((meta["candidate_idx"], parse_score(text)))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_csv_path = out_dir / "scores.csv"
    with open(scores_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_event_id", "user_id", "item_id", "score"])
        for rec in records:
            uid = rec["user_id"]
            source_event_id = rec.get("source_event_id", uid)
            candidate_ids = rec["candidate_item_ids"]
            for candidate_idx, score in user_scores.get(uid, []):
                writer.writerow([source_event_id, uid, candidate_ids[candidate_idx], score])
    print(f"Saved scores CSV: {scores_csv_path}")

    scores_output = []
    for rec in records:
        uid = rec["user_id"]
        pos_idx = rec["positive_item_index"]
        scores = user_scores.get(uid, [])
        if not scores:
            continue
        scores.sort(key=lambda x: -x[1])
        ranked_indices = [idx for idx, _ in scores]
        rank_0based = ranked_indices.index(pos_idx) if pos_idx in ranked_indices else len(ranked_indices)
        scores_output.append({"user_id": uid, "positive_rank": rank_0based + 1, "n_candidates": len(scores)})

    pos_ranks = np.array([s["positive_rank"] for s in scores_output])
    report = {}
    for k in [5, 10, 20]:
        report[f"HR@{k}"] = float(np.mean(pos_ranks <= k))
        report[f"NDCG@{k}"] = float(np.mean([1.0 / np.log2(r + 1) if r <= k else 0.0 for r in pos_ranks]))
    report["MRR"] = float(np.mean([1.0 / r for r in pos_ranks]))
    report["n_users"] = len(pos_ranks)
    report["n_prompts"] = len(all_prompts)
    report["inference_time_s"] = elapsed
    report["data_path"] = args.data
    report["model"] = args.model
    report["backbone"] = backbone
    report["seed"] = args.seed
    report["temperature"] = args.temperature

    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(out_dir / "user_ranks.jsonl", "w") as f:
        for s in scores_output:
            f.write(json.dumps(s) + "\n")

    print("=" * 50)
    print(f"C-CRP v3 [{backbone} seed={args.seed}] ({report['n_users']} users)")
    print(f"HR@10={report['HR@10']:.4f}  NDCG@10={report['NDCG@10']:.4f}  MRR={report['MRR']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
