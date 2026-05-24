"""C-CRP v4: Enhanced prompt with category extraction for better cross-domain performance.
Key change: extract categories from candidate_texts, show them explicitly in prompt.
Uses more description text (400 chars vs 200) and structured category matching."""
import json, numpy as np, time, re, argparse
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.llm.vllm_backend import VLLMBackend


def extract_category(text):
    """Extract category from candidate_text format 'Title: X Categories: Y Description: Z'"""
    m = re.search(r'Categories:\s*(.+?)\s*Description:', text)
    if m:
        return m.group(1).strip()
    return ""


def build_v4_prompt(history_titles, history_texts, candidate_title, candidate_text):
    """Enhanced prompt with explicit category matching."""
    # Extract categories from history
    hist_lines = []
    for i, (title, text) in enumerate(zip(history_titles[-8:], history_texts[-8:])):
        cat = extract_category(text) if text else ""
        if cat:
            hist_lines.append(f"- {title} [{cat}]")
        else:
            hist_lines.append(f"- {title}")
    hist_block = "\n".join(hist_lines)

    # Extract candidate info
    cand_cat = extract_category(candidate_text) if candidate_text else ""
    # Use more description (400 chars)
    desc_match = re.search(r'Description:\s*(.+)', candidate_text or "")
    desc = desc_match.group(1)[:400] if desc_match else ""

    cat_line = f"\nCategory: {cand_cat}" if cand_cat else ""
    desc_line = f"\nDescription: {desc}" if desc else ""

    return (
        "You are an expert recommendation system.\n\n"
        f"User's recent interaction history (with categories):\n{hist_block}\n\n"
        f"Candidate item:\nTitle: {candidate_title}{cat_line}{desc_line}\n\n"
        "Task: Estimate the probability (0.0-1.0) that this user will interact with this candidate next.\n"
        "Consider: (1) category alignment with history, (2) content/theme similarity, (3) interaction trajectory.\n\n"
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
    parser.add_argument("--domain", type=str, default="unknown")
    args = parser.parse_args()

    print(f"=== C-CRP v4 (enhanced prompt): {args.domain} ===")
    print(f"Loading {args.data}...")
    records = [json.loads(l) for l in open(args.data)]
    if args.n_users and args.n_users < len(records):
        records = records[:args.n_users]
    print(f"Users: {len(records)}")

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

    # Build prompts with enhanced category info
    all_prompts = []
    all_meta = []
    for rec in records:
        history_titles = rec["history"]
        # Get history texts if available (for category extraction)
        # History texts aren't in ranking_test.jsonl directly, use empty
        history_texts = rec.get("history_texts", [""] * len(history_titles))
        candidates = rec["candidate_titles"]
        candidate_texts = rec.get("candidate_texts", [""] * len(candidates))
        for i, (title, text) in enumerate(zip(candidates, candidate_texts)):
            prompt = build_v4_prompt(history_titles, history_texts, title, text)
            all_prompts.append(prompt)
            all_meta.append({"user_id": rec["user_id"], "candidate_idx": i})

    print(f"Total prompts: {len(all_prompts)} ({len(records)} users x {len(records[0]['candidate_titles'])} candidates)")

    # Show sample prompt
    print(f"\n--- Sample prompt (first user, first candidate) ---")
    print(all_prompts[0][:500])
    print("...")

    print("\nRunning inference...")
    start = time.time()
    results = backend.batch_generate(all_prompts)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({len(all_prompts)/elapsed:.1f} samples/s)")

    # Build per-user score arrays
    user_scores = defaultdict(list)
    for meta, result in zip(all_meta, results):
        score = parse_score(result["raw_text"])
        user_scores[meta["user_id"]].append((meta["candidate_idx"], score))

    # Evaluate
    scores_output = []
    for rec in records:
        uid = rec["user_id"]
        pos_idx = rec["positive_item_index"]
        scores = user_scores.get(uid, [])
        if not scores:
            continue
        scores.sort(key=lambda x: -x[1])
        ranked_indices = [idx for idx, _ in scores]
        rank = ranked_indices.index(pos_idx) if pos_idx in ranked_indices else len(ranked_indices)
        scores_output.append({"user_id": uid, "pos_rank": rank, "n_candidates": len(scores)})

    pos_ranks = np.array([s["pos_rank"] for s in scores_output])

    report = {}
    for k in [5, 10, 20]:
        report[f"hr{k}"] = float(np.mean(pos_ranks < k))
        report[f"ndcg{k}"] = float(np.mean([1.0 / np.log2(r + 2) if r < k else 0.0 for r in pos_ranks]))
    report["mrr"] = float(np.mean([1.0 / (r + 1) for r in pos_ranks]))
    report["n_users"] = len(pos_ranks)
    report["n_prompts"] = len(all_prompts)
    report["inference_time_s"] = elapsed
    report["data_path"] = args.data
    report["domain"] = args.domain
    report["method"] = "C-CRP_v4_enhanced_prompt"

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(out_dir / "user_ranks.jsonl", "w") as f:
        for s in scores_output:
            f.write(json.dumps(s) + "\n")

    print(f"\n{'='*50}")
    print(f"C-CRP v4 Results: {args.domain} ({report['n_users']} users)")
    print(f"{'='*50}")
    print(f"HR@5:    {report['hr5']:.4f}")
    print(f"HR@10:   {report['hr10']:.4f}")
    print(f"HR@20:   {report['hr20']:.4f}")
    print(f"NDCG@5:  {report['ndcg5']:.4f}")
    print(f"NDCG@10: {report['ndcg10']:.4f}")
    print(f"NDCG@20: {report['ndcg20']:.4f}")
    print(f"MRR:     {report['mrr']:.4f}")

    # Auto-save markdown
    summary_file = out_dir / f"result_summary.md"
    summary_file.write_text(
        f"---\ntitle: C-CRP v4 {args.domain} result\ntype: fact\n"
        f"created: {time.strftime('%Y-%m-%dT%H:%M:%S')}\nagent: script-auto\n"
        f"tags: [experiment, ccrp-v4, {args.domain}]\n---\n\n"
        f"## C-CRP v4 (enhanced prompt) — {args.domain}\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        + "".join(f"| {k} | {v:.4f} |\n" for k, v in report.items() if isinstance(v, float) and k.startswith(("hr", "ndcg", "mrr")))
        + f"\n- Users: {report['n_users']}\n- Inference: {elapsed:.0f}s\n"
    )
    print(f"\n[AUTO-MEMORY] Saved to {summary_file}")


if __name__ == "__main__":
    main()
