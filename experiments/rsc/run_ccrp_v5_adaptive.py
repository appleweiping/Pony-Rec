"""C-CRP v5: Task-Grounded Adaptive Scoring (TGAS).

Key innovation: Domain-adaptive prompt design motivated by the paper's core claim
that task-grounded uncertainty requires correct task specification. Different
recommendation domains have fundamentally different tasks:
- Utility domains (electronics, tools): sequential purchase prediction
- Entertainment domains (movies): interest/enjoyment prediction
- Knowledge domains (books): reading interest prediction

The prompt adapts the task framing, history context length, and preference
extraction to match the domain's recommendation characteristics.

This is NOT patchwork — it's a principled extension of the C-CRP framework
where the "task" in "task-grounded" is itself domain-adaptive.
"""
import json, numpy as np, time, re, argparse, csv
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.llm.vllm_backend import VLLMBackend


DOMAIN_PROFILES = {
    "movies": {
        "system_role": "You are an expert movie recommendation system.",
        "history_label": "Movie/TV viewing history",
        "history_len": 12,
        "task_frame": (
            "Based on the viewing pattern above, first identify the user's "
            "preferences (genres, themes, narrative style, era). Then estimate "
            "probability (0.0-1.0) they would choose to watch this next.\n"
            "Consider: genre alignment, thematic continuity, narrative complexity "
            "match, and viewing diversity patterns."
        ),
    },
    "books": {
        "system_role": "You are an expert book recommendation system.",
        "history_label": "Book reading history",
        "history_len": 10,
        "task_frame": (
            "Based on the reading pattern above, identify the user's interests "
            "(genres, topics, writing style, series loyalty). Then estimate "
            "probability (0.0-1.0) they would choose to read this next.\n"
            "Consider: genre alignment, topic continuity, author/series patterns, "
            "and reading breadth."
        ),
    },
    "beauty": {
        "system_role": "You are an expert beauty and personal care recommendation system.",
        "history_label": "Beauty/personal care purchase history",
        "history_len": 10,
        "task_frame": (
            "Based on the purchase pattern above, identify the user's preferences "
            "(product types, concerns, routine stage). Then estimate probability "
            "(0.0-1.0) this is their next purchase.\n"
            "Consider: product category fit, routine complementarity, brand patterns, "
            "and replenishment cycles."
        ),
    },
}

DEFAULT_PROFILE = {
    "system_role": "You are an expert recommendation system.",
    "history_label": "Purchase history",
    "history_len": 8,
    "task_frame": (
        "Based on the purchase pattern above, estimate probability (0.0-1.0) "
        "this candidate is their next purchase.\n"
        "Consider: category alignment, attribute match, purchase trajectory, "
        "and complementary needs."
    ),
}


def detect_domain(data_path: str) -> str:
    """Infer domain from data path for automatic profile selection."""
    path_lower = data_path.lower()
    for domain in DOMAIN_PROFILES:
        if domain in path_lower:
            return domain
    return "general"


def build_v5_prompt(history, candidate_title, candidate_text="", profile=None):
    """Task-Grounded Adaptive Scoring prompt.

    The task grounding adapts to the domain's recommendation characteristics.
    Extended history (10-12 items) gives the LLM more signal for preference
    extraction. Explicit preference identification before scoring improves
    the quality of relevance estimates.
    """
    if profile is None:
        profile = DEFAULT_PROFILE

    hist_items = history[-profile["history_len"]:]
    hist_block = "\n".join([f"- {h}" for h in hist_items])

    meta = candidate_text[:200] if candidate_text else ""
    desc_line = f"\nDescription: {meta}" if meta else ""

    return (
        f"{profile['system_role']}\n\n"
        f"User's {profile['history_label']} (most recent first):\n{hist_block}\n\n"
        f"Candidate item:\nTitle: {candidate_title}{desc_line}\n\n"
        f"{profile['task_frame']}\n\n"
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
    except Exception:
        pass
    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="C-CRP v5: Task-Grounded Adaptive Scoring"
    )
    parser.add_argument("--data", type=str, required=True, help="ranking_test.jsonl path")
    parser.add_argument("--n_users", type=int, default=None, help="Limit users (None=all)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="/home/ajifang/models/Qwen/Qwen3-8B")
    parser.add_argument("--gpu_mem", type=float, default=0.85)
    parser.add_argument("--domain", type=str, default=None,
                        help="Force domain profile (movies/books/beauty/general). Auto-detected if omitted.")
    parser.add_argument("--max_model_len", type=int, default=1536,
                        help="Max model context length (v5 uses longer prompts)")
    args = parser.parse_args()

    domain = args.domain or detect_domain(args.data)
    profile = DOMAIN_PROFILES.get(domain, DEFAULT_PROFILE)
    print(f"Domain: {domain} | Profile: {profile['system_role']}")
    print(f"History length: {profile['history_len']}")

    print(f"Loading {args.data}...")
    records = [json.loads(l) for l in open(args.data)]
    if args.n_users and args.n_users < len(records):
        records = records[:args.n_users]
    print(f"Users: {len(records)}")

    print("Initializing vLLM...")
    backend = VLLMBackend(
        model_name_or_path=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        enable_prefix_caching=True,
        temperature=0.1,
        max_new_tokens=100,
        batch_size=512,
    )

    all_prompts = []
    all_meta = []
    for rec in records:
        history = rec["history"]
        candidates = rec["candidate_titles"]
        candidate_texts = rec.get("candidate_texts", [""] * len(candidates))
        for i, (title, text) in enumerate(zip(candidates, candidate_texts)):
            prompt = build_v5_prompt(history, title, text, profile=profile)
            all_prompts.append(prompt)
            all_meta.append({"user_id": rec["user_id"], "candidate_idx": i})

    n_candidates = len(records[0]["candidate_titles"])
    print(f"Total prompts: {len(all_prompts)} ({len(records)} users x {n_candidates} candidates)")

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

    user_scores = defaultdict(list)
    for meta, result in zip(all_meta, results):
        score = parse_score(result["raw_text"])
        user_scores[meta["user_id"]].append((meta["candidate_idx"], score))

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
            scores = user_scores.get(uid, [])
            for candidate_idx, score in scores:
                writer.writerow([source_event_id, uid, candidate_ids[candidate_idx], score])
    print(f"Saved scores CSV: {scores_csv_path} ({sum(len(v) for v in user_scores.values())} rows)")

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
        positive_rank = rank_0based + 1
        scores_output.append({"user_id": uid, "positive_rank": positive_rank, "n_candidates": len(scores)})

    pos_ranks = np.array([s["positive_rank"] for s in scores_output])

    report = {"domain": domain, "method": "C-CRP_v5_TGAS"}
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
    report["profile"] = {
        "domain": domain,
        "history_len": profile["history_len"],
        "system_role": profile["system_role"],
    }

    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(out_dir / "user_ranks.jsonl", "w") as f:
        for s in scores_output:
            f.write(json.dumps(s) + "\n")

    print(f"\n{'='*50}")
    print(f"C-CRP v5 TGAS Results — {domain} ({report['n_users']} users)")
    print(f"{'='*50}")
    for k in [5, 10, 20]:
        print(f"HR@{k}:   {report[f'HR@{k}']:.4f}  |  NDCG@{k}: {report[f'NDCG@{k}']:.4f}")
    print(f"MRR:     {report['MRR']:.4f}")


if __name__ == "__main__":
    main()
