"""SRPD Step 1: Anchor rank inference using vLLM batch (title-only compact prompts).

Full-scale: 10,000 users × 101 candidates per domain.
Uses title-only prompts (~2.5K tokens) to fit vLLM's efficient batching.
Same evaluation protocol as C-CRP v3 and all 8 baselines.
"""
import json
import re
import time
import argparse
from pathlib import Path
from collections import defaultdict

def build_ranking_prompt(history_titles, candidate_ids, candidate_titles, topk=10):
    hist_block = "\n".join([f"- {t}" for t in history_titles[-10:]])
    cand_lines = []
    for idx, (cid, title) in enumerate(zip(candidate_ids, candidate_titles), 1):
        cand_lines.append(f"{idx}. id={cid} | {title}")
    cand_block = "\n".join(cand_lines)
    allowed = ", ".join(candidate_ids)

    return (
        "You are a recommendation ranking assistant.\n\n"
        f"User purchase history (most recent first):\n{hist_block}\n\n"
        f"Rank the following {len(candidate_ids)} candidates by relevance to this user.\n\n"
        f"Candidates:\n{cand_block}\n\n"
        f"Return the top {topk} most relevant item IDs as JSON.\n"
        "Do not output reasoning or chain-of-thought.\n\n"
        'Return ONLY: {"ranked_item_ids": ["id1", "id2", ...]}'
    )


def parse_ranking(text, allowed_ids, topk=10):
    try:
        m = re.search(r'\{[^}]*"ranked_item_ids"\s*:\s*\[([^\]]*)\]', text, re.DOTALL)
        if m:
            ids_str = m.group(1)
            ids = re.findall(r'"([^"]+)"', ids_str)
            valid = [i for i in ids if i in allowed_ids]
            return valid[:topk]
    except:
        pass
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--split", default="valid", choices=["valid", "test"])
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.88)
    args = parser.parse_args()

    domain = args.domain
    split = args.split

    if domain == "beauty":
        prefix = "beauty_supplementary_smallerN_100neg"
    else:
        prefix = f"{domain}_large10000_100neg"

    data_root = Path("outputs/baselines/external_tasks")
    input_path = data_root / f"{prefix}_{split}_same_candidate" / f"ranking_{split}.jsonl"
    output_dir = Path(f"outputs/{prefix}_srpd_anchor_rank_{split}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions" / "rank_predictions.jsonl"

    if pred_path.exists():
        print(f"[{domain}/{split}] Already done: {pred_path}")
        return

    print(f"[{domain}/{split}] Loading data from {input_path}")
    samples = [json.loads(line) for line in open(input_path)]
    print(f"  Loaded {len(samples)} samples")

    # Check for partial results
    partial_path = output_dir / "predictions" / "rank_predictions_partial.jsonl"
    done_ids = set()
    existing_results = []
    if partial_path.exists():
        existing_results = [json.loads(l) for l in open(partial_path)]
        done_ids = {r["user_id"] for r in existing_results}
        print(f"  Resuming from {len(done_ids)} existing predictions")

    remaining = [s for s in samples if s.get("user_id") not in done_ids]
    if not remaining:
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w") as f:
            for r in existing_results:
                f.write(json.dumps(r) + "\n")
        print(f"  All done, saved to {pred_path}")
        return

    print(f"  Remaining: {len(remaining)} samples")
    print(f"  Loading vLLM (max_model_len={args.max_model_len})...")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model="/home/ajifang/models/Qwen/Qwen3-8B",
        tokenizer="/home/ajifang/models/Qwen/Qwen3-8B",
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        tensor_parallel_size=1,
        trust_remote_code=False,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
    )

    tokenizer = llm.get_tokenizer()
    results = list(existing_results)
    total_done = len(results)
    t0 = time.time()

    # Process in batches
    for batch_start in range(0, len(remaining), args.batch_size):
        batch = remaining[batch_start:batch_start + args.batch_size]

        # Build prompts
        prompts = []
        for sample in batch:
            history = sample.get("history", sample.get("candidate_titles", [])[:3])
            if isinstance(history, list) and len(history) > 0 and isinstance(history[0], dict):
                history = [h.get("title", "") for h in history]
            candidate_ids = [str(cid) for cid in sample["candidate_item_ids"]]
            candidate_titles = sample["candidate_titles"]
            prompt_text = build_ranking_prompt(history, candidate_ids, candidate_titles, topk=args.topk)

            messages = [{"role": "user", "content": prompt_text}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(formatted)

        # vLLM batch inference
        outputs = llm.generate(prompts, sampling_params)

        # Parse results
        for sample, output in zip(batch, outputs):
            text = output.outputs[0].text
            candidate_ids = [str(cid) for cid in sample["candidate_item_ids"]]
            allowed_set = set(candidate_ids)
            ranked = parse_ranking(text, allowed_set, topk=args.topk)

            record = {
                "user_id": sample.get("user_id", ""),
                "source_event_id": sample.get("source_event_id", ""),
                "positive_item_id": str(sample.get("positive_item_id", "")),
                "positive_item_index": sample.get("positive_item_index", -1),
                "split_name": sample.get("split_name", split),
                "candidate_item_ids": candidate_ids,
                "pred_ranked_item_ids": ranked,
                "topk_item_ids": ranked[:args.topk],
                "parse_success": len(ranked) > 0,
                "raw_text": text[:500],
            }
            results.append(record)

        total_done = len(results)
        elapsed = time.time() - t0
        speed = (total_done - len(existing_results)) / elapsed if elapsed > 0 else 0
        eta = (len(remaining) - (total_done - len(existing_results))) / speed if speed > 0 else 0
        print(f"  [{domain}/{split}] {total_done}/{len(samples)} done | {speed:.1f} samples/s | ETA: {eta/60:.1f} min")

        # Checkpoint
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        with open(partial_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    # Final save
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    elapsed = time.time() - t0
    print(f"\n[{domain}/{split}] COMPLETE: {len(results)} predictions in {elapsed/60:.1f} min")
    print(f"  Saved to {pred_path}")

    # Summary stats
    parse_ok = sum(1 for r in results if r["parse_success"])
    print(f"  Parse success: {parse_ok}/{len(results)} ({100*parse_ok/len(results):.1f}%)")


if __name__ == "__main__":
    main()
