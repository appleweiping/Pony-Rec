"""C-CRP v3 diagnostic: Run pointwise scoring on a subset and compute calibration metrics.
Outputs: diagnostic_metrics.csv, reliability_bins.csv, raw_scores.jsonl"""
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


def compute_calibration_metrics(scores, labels):
    """Compute ECE, MCE, AUROC, Brier score."""
    scores = np.array(scores)
    labels = np.array(labels)
    n = len(scores)

    # Brier score
    brier = float(np.mean((scores - labels) ** 2))

    # ECE and MCE (10 bins)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    mce = 0.0
    reliability_rows = []
    for i in range(10):
        mask = (scores >= bins[i]) & (scores < bins[i+1])
        if i == 9:
            mask = (scores >= bins[i]) & (scores <= bins[i+1])
        count = mask.sum()
        if count > 0:
            avg_conf = float(scores[mask].mean())
            acc = float(labels[mask].mean())
            gap = abs(avg_conf - acc)
            ece += gap * count / n
            mce = max(mce, gap)
        else:
            avg_conf = (bins[i] + bins[i+1]) / 2
            acc = 0.0
        reliability_rows.append({
            "bin_lower": float(bins[i]),
            "bin_upper": float(bins[i+1]),
            "bin_center": float((bins[i] + bins[i+1]) / 2),
            "count": int(count) if count else 0,
            "avg_confidence": avg_conf,
            "accuracy": acc,
        })

    # AUROC
    from sklearn.metrics import roc_auc_score
    try:
        auroc = float(roc_auc_score(labels, scores))
    except:
        auroc = 0.5

    return {
        "num_samples": n,
        "accuracy": float(labels.mean()),
        "avg_score": float(scores.mean()),
        "brier_score": brier,
        "ece": ece,
        "mce": mce,
        "auroc": auroc,
    }, reliability_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--n_users", type=int, default=None, help="Limit users (None=all)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="/home/ajifang/models/Qwen/Qwen3-8B")
    args = parser.parse_args()

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

    all_prompts = []
    all_meta = []
    for rec in records:
        history = rec["history"]
        candidates = rec["candidate_titles"]
        candidate_texts = rec.get("candidate_texts", [""] * len(candidates))
        pos_idx = rec["positive_item_index"]
        for i, (title, text) in enumerate(zip(candidates, candidate_texts)):
            prompt = build_v3_prompt(history, title, text)
            all_prompts.append(prompt)
            all_meta.append({
                "user_id": rec["user_id"],
                "candidate_idx": i,
                "label": 1 if i == pos_idx else 0,
            })

    print(f"Total prompts: {len(all_prompts)} ({len(records)} users x {len(records[0]['candidate_titles'])} candidates)")
    print("Running inference...")
    start = time.time()
    results = backend.batch_generate(all_prompts)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({len(all_prompts)/elapsed:.1f} samples/s)")

    # Extract scores and labels
    all_scores = []
    all_labels = []
    raw_rows = []
    for meta, result in zip(all_meta, results):
        score = parse_score(result["raw_text"])
        label = meta["label"]
        all_scores.append(score)
        all_labels.append(label)
        raw_rows.append({
            "user_id": meta["user_id"],
            "candidate_idx": meta["candidate_idx"],
            "score": score,
            "label": label,
        })

    # Compute metrics
    metrics, reliability_rows = compute_calibration_metrics(all_scores, all_labels)
    metrics["inference_time_s"] = elapsed
    metrics["n_users"] = len(records)
    metrics["n_prompts"] = len(all_prompts)
    metrics["data_path"] = args.data

    # Save outputs
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "diagnostic_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "reliability_bins.csv", "w") as f:
        f.write("bin_lower,bin_upper,bin_center,count,avg_confidence,accuracy\n")
        for row in reliability_rows:
            f.write(f"{row['bin_lower']},{row['bin_upper']},{row['bin_center']},{row['count']},{row['avg_confidence']},{row['accuracy']}\n")

    with open(out_dir / "raw_scores.jsonl", "w") as f:
        for row in raw_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\n{'='*50}")
    print(f"Calibration Diagnostic ({metrics['n_users']} users, {metrics['num_samples']} samples)")
    print(f"{'='*50}")
    print(f"AUROC:       {metrics['auroc']:.4f}")
    print(f"ECE:         {metrics['ece']:.4f}")
    print(f"MCE:         {metrics['mce']:.4f}")
    print(f"Brier:       {metrics['brier_score']:.4f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Avg score:   {metrics['avg_score']:.4f}")
    print(f"\nReliability bins:")
    for row in reliability_rows:
        if row["count"] > 0:
            print(f"  [{row['bin_lower']:.1f}-{row['bin_upper']:.1f}] n={row['count']:>6} conf={row['avg_confidence']:.3f} acc={row['accuracy']:.4f}")

    # Auto-save markdown
    domain = Path(args.data).parent.name.split("_")[0]
    summary_file = out_dir / f"diagnostic_summary_{domain}.md"
    summary_file.write_text(
        f"---\ntitle: C-CRP v3 {domain} calibration diagnostic\ntype: fact\n"
        f"created: {time.strftime('%Y-%m-%dT%H:%M:%S')}\nagent: script-auto\n"
        f"tags: [diagnostic, calibration, {domain}]\n---\n\n"
        f"## Calibration Metrics ({metrics['n_users']} users)\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| AUROC | {metrics['auroc']:.4f} |\n"
        f"| ECE | {metrics['ece']:.4f} |\n"
        f"| MCE | {metrics['mce']:.4f} |\n"
        f"| Brier | {metrics['brier_score']:.4f} |\n"
        f"| Accuracy | {metrics['accuracy']:.4f} |\n"
        f"| Avg Score | {metrics['avg_score']:.4f} |\n"
    )
    print(f"\n[AUTO-MEMORY] Saved to {summary_file}")


if __name__ == "__main__":
    main()
