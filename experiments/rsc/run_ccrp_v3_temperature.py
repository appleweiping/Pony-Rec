"""C-CRP v3 with domain-adaptive temperature scaling.
1. Run validation set → get raw scores → grid search optimal temperature T
2. Run test set → apply T → evaluate
Outputs: report.json with all metrics, optimal T, comparison with T=1."""
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


def run_inference(backend, records):
    """Run batch inference, return per-user scores dict."""
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

    print(f"  Prompts: {len(all_prompts)} ({len(records)} users x {len(records[0]['candidate_titles'])} candidates)")
    start = time.time()
    results = backend.batch_generate(all_prompts)
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s ({len(all_prompts)/elapsed:.1f} samples/s)")

    user_scores = defaultdict(list)
    for meta, result in zip(all_meta, results):
        score = parse_score(result["raw_text"])
        user_scores[meta["user_id"]].append((meta["candidate_idx"], score))

    return user_scores, elapsed


def evaluate_with_temperature(records, user_scores, temperature):
    """Evaluate ranking metrics with temperature-scaled scores."""
    pos_ranks = []
    for rec in records:
        uid = rec["user_id"]
        pos_idx = rec["positive_item_index"]
        scores = user_scores.get(uid, [])
        if not scores:
            continue
        # Apply temperature: score^(1/T) preserves order when T=1, sharpens when T<1
        if temperature != 1.0:
            scaled = [(idx, s ** (1.0 / temperature) if s > 0 else 0.0) for idx, s in scores]
        else:
            scaled = scores
        scaled.sort(key=lambda x: -x[1])
        ranked_indices = [idx for idx, _ in scaled]
        rank = ranked_indices.index(pos_idx) if pos_idx in ranked_indices else len(ranked_indices)
        pos_ranks.append(rank)

    pos_ranks = np.array(pos_ranks)
    metrics = {}
    for k in [5, 10, 20]:
        metrics[f"hr{k}"] = float(np.mean(pos_ranks < k))
        metrics[f"ndcg{k}"] = float(np.mean([1.0 / np.log2(r + 2) if r < k else 0.0 for r in pos_ranks]))
    metrics["mrr"] = float(np.mean([1.0 / (r + 1) for r in pos_ranks]))
    metrics["n_users"] = len(pos_ranks)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="/home/ajifang/models/Qwen/Qwen3-8B")
    parser.add_argument("--domain", type=str, required=True)
    args = parser.parse_args()

    print(f"=== C-CRP v3 + Temperature Scaling: {args.domain} ===")

    # Load data
    print(f"Loading valid: {args.valid_data}")
    valid_records = [json.loads(l) for l in open(args.valid_data)]
    print(f"Loading test: {args.test_data}")
    test_records = [json.loads(l) for l in open(args.test_data)]
    print(f"Valid: {len(valid_records)} users, Test: {len(test_records)} users")

    # Init model
    print("Initializing vLLM...")
    backend = VLLMBackend(
        model_name_or_path=args.model,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
        temperature=0.1,
        max_new_tokens=100,
        batch_size=512,
    )

    # Run validation inference
    print("\n[Phase 1] Validation inference...")
    valid_scores, valid_time = run_inference(backend, valid_records)

    # Grid search temperature on validation
    print("\n[Phase 2] Temperature grid search on validation...")
    temperatures = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
    best_t = 1.0
    best_ndcg10 = 0.0
    valid_sweep = []
    for t in temperatures:
        m = evaluate_with_temperature(valid_records, valid_scores, t)
        valid_sweep.append({"temperature": t, **m})
        print(f"  T={t:.1f}: HR@10={m['hr10']:.4f} NDCG@10={m['ndcg10']:.4f}")
        if m["ndcg10"] > best_ndcg10:
            best_ndcg10 = m["ndcg10"]
            best_t = t

    print(f"\n  Best T={best_t} (valid NDCG@10={best_ndcg10:.4f})")

    # Run test inference
    print("\n[Phase 3] Test inference...")
    test_scores, test_time = run_inference(backend, test_records)

    # Evaluate test with best temperature
    print(f"\n[Phase 4] Test evaluation with T={best_t}...")
    test_metrics = evaluate_with_temperature(test_records, test_scores, best_t)
    test_metrics_t1 = evaluate_with_temperature(test_records, test_scores, 1.0)

    # Report
    report = {
        "domain": args.domain,
        "best_temperature": best_t,
        "valid_best_ndcg10": best_ndcg10,
        "test_with_best_T": test_metrics,
        "test_with_T1": test_metrics_t1,
        "improvement_ndcg10": test_metrics["ndcg10"] - test_metrics_t1["ndcg10"],
        "valid_sweep": valid_sweep,
        "valid_inference_time_s": valid_time,
        "test_inference_time_s": test_time,
        "n_valid_users": len(valid_records),
        "n_test_users": len(test_records),
        "model": args.model,
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report_temperature.json", "w") as f:
        json.dump(report, f, indent=2)

    # Save raw scores for future use
    with open(out_dir / "valid_raw_scores.json", "w") as f:
        json.dump({uid: scores for uid, scores in valid_scores.items()}, f)
    with open(out_dir / "test_raw_scores.json", "w") as f:
        json.dump({uid: scores for uid, scores in test_scores.items()}, f)

    print(f"\n{'='*60}")
    print(f"C-CRP v3 + Temperature Scaling: {args.domain}")
    print(f"{'='*60}")
    print(f"Best T (validation-selected): {best_t}")
    print(f"\nTest results (T={best_t}):")
    for k in [5, 10, 20]:
        print(f"  HR@{k}:   {test_metrics[f'hr{k}']:.4f} (T=1: {test_metrics_t1[f'hr{k}']:.4f}, delta: {test_metrics[f'hr{k}']-test_metrics_t1[f'hr{k}']:+.4f})")
        print(f"  NDCG@{k}: {test_metrics[f'ndcg{k}']:.4f} (T=1: {test_metrics_t1[f'ndcg{k}']:.4f}, delta: {test_metrics[f'ndcg{k}']-test_metrics_t1[f'ndcg{k}']:+.4f})")
    print(f"  MRR:    {test_metrics['mrr']:.4f} (T=1: {test_metrics_t1['mrr']:.4f}, delta: {test_metrics['mrr']-test_metrics_t1['mrr']:+.4f})")
    print(f"\n[AUTO-MEMORY] Saved to {out_dir / 'report_temperature.json'}")


if __name__ == "__main__":
    main()
