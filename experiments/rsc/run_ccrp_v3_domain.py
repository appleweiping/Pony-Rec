"""C-CRP v3: Run on any domain using ranking_test.jsonl directly."""
import json, numpy as np, time, re, argparse
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.llm.vllm_backend import VLLMBackend

def build_v3_prompt(history, candidate_title, candidate_text=""):
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
    try:
        m = re.search(r'"relevance_probability"\s*:\s*([\d.]+)', text)
        if m:
            return float(m.group(1))
        m = re.search(r'(0\.\d+|1\.0)', text)
        if m:
            return float(m.group(1))
    except:
        pass
    return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="ranking_test.jsonl path")
    parser.add_argument("--n_users", type=int, default=None, help="Limit users (None=all)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="/home/ajifang/models/Qwen/Qwen3-8B")
    parser.add_argument("--gpu_mem", type=float, default=0.85, help="GPU memory utilization (0-1)")
    args = parser.parse_args()

    print(f"Loading {args.data}...")
    records = [json.loads(l) for l in open(args.data)]
    if args.n_users and args.n_users < len(records):
        records = records[:args.n_users]
    print(f"Users: {len(records)}")

    print("Initializing vLLM...")
    backend = VLLMBackend(
        model_name_or_path=args.model,
        max_model_len=1024,
        gpu_memory_utilization=args.gpu_mem,
        enable_prefix_caching=True,
        temperature=0.1,
        max_new_tokens=100,
        batch_size=512,
    )

    # Build all prompts (user × candidate)
    all_prompts = []
    all_meta = []
    for rec in records:
        history = rec["history"]
        candidates = rec["candidate_titles"]
        candidate_texts = rec.get("candidate_texts", [""] * len(candidates))
        for i, (title, text) in enumerate(zip(candidates, candidate_texts)):
            prompt = build_v3_prompt(history, title, text)
            all_prompts.append(prompt)
            all_meta.append({"user_id": rec["user_id"], "candidate_idx": i})

    n_candidates = len(records[0]['candidate_titles'])
    print(f"Total prompts: {len(all_prompts)} ({len(records)} users x {n_candidates} candidates)")

    # Process in chunks — vLLM handles scheduling internally via continuous batching.
    # With clean GPU/CPU, large batches maximize prefix cache hits and throughput.
    CHUNK_USERS = 5000
    chunk_size = CHUNK_USERS * n_candidates
    print(f"Running inference in chunks of {CHUNK_USERS} users ({chunk_size} prompts)...")
    start = time.time()
    results = []
    for i in range(0, len(all_prompts), chunk_size):
        chunk = all_prompts[i:i + chunk_size]
        chunk_results = backend.batch_generate(chunk)
        results.extend(chunk_results)
        done = min(i + chunk_size, len(all_prompts))
        elapsed_so_far = time.time() - start
        rate = done / elapsed_so_far if elapsed_so_far > 0 else 0
        print(f"  [{done}/{len(all_prompts)}] {rate:.0f} prompts/s")
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({len(all_prompts)/elapsed:.1f} samples/s)")

    # Build per-user score arrays
    user_scores = defaultdict(list)
    for meta, result in zip(all_meta, results):
        score = parse_score(result["raw_text"])
        user_scores[meta["user_id"]].append((meta["candidate_idx"], score))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-candidate scores CSV (aligned with official baseline import format)
    import csv
    scores_csv_path = out_dir / "scores.csv"
    with open(scores_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_event_id", "user_id", "item_id", "score"])
        for rec in records:
            uid = rec["user_id"]
            source_event_id = rec.get("source_event_id", uid)
            candidate_ids = rec["candidate_item_ids"]
            scores = user_scores.get(uid, [])
            for candidate_idx, score in scores:
                writer.writerow([source_event_id, uid, candidate_ids[candidate_idx], score])
    print(f"Saved scores CSV: {scores_csv_path} ({sum(len(v) for v in user_scores.values())} rows)")

    # Compute metrics using 1-based rank (aligned with official ranking_task_metrics.py)
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
        positive_rank = rank_0based + 1  # 1-based, aligned with official eval
        scores_output.append({"user_id": uid, "positive_rank": positive_rank, "n_candidates": len(scores)})

    pos_ranks = np.array([s["positive_rank"] for s in scores_output])

    report = {}
    for k in [5, 10, 20]:
        hr = float(np.mean(pos_ranks <= k))
        ndcg = float(np.mean([1.0 / np.log2(r + 1) if r <= k else 0.0 for r in pos_ranks]))
        report[f"HR@{k}"] = hr
        report[f"NDCG@{k}"] = ndcg

    report["MRR"] = float(np.mean([1.0 / r for r in pos_ranks]))
    report["n_users"] = len(pos_ranks)
    report["n_prompts"] = len(all_prompts)
    report["inference_time_s"] = elapsed
    report["data_path"] = args.data

    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Save per-user ranks for post-hoc analysis
    with open(out_dir / "user_ranks.jsonl", "w") as f:
        for s in scores_output:
            f.write(json.dumps(s) + "\n")

    print(f"\n{'='*50}")
    print(f"C-CRP v3 Results ({report['n_users']} users)")
    print(f"{'='*50}")
    print(f"HR@5:    {report['HR@5']:.4f}")
    print(f"HR@10:   {report['HR@10']:.4f}")
    print(f"HR@20:   {report['HR@20']:.4f}")
    print(f"NDCG@5:  {report['NDCG@5']:.4f}")
    print(f"NDCG@10: {report['NDCG@10']:.4f}")
    print(f"NDCG@20: {report['NDCG@20']:.4f}")
    print(f"MRR:     {report['MRR']:.4f}")

if __name__ == "__main__":
    main()
