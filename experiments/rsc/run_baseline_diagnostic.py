"""Compute calibration diagnostics for official baselines from existing score files.
No inference needed — scores already exist. Just join with labels and compute metrics."""
import json, csv, numpy as np, argparse
from pathlib import Path
from collections import defaultdict


def load_ranking_test_labels(ranking_test_path):
    """Load ground truth: {(user_id, item_id): label} from ranking_test.jsonl"""
    labels = {}
    with open(ranking_test_path) as f:
        for line in f:
            rec = json.loads(line)
            uid = rec["user_id"]
            pos_idx = rec["positive_item_index"]
            for i, item_id in enumerate(rec["candidate_item_ids"]):
                labels[(uid, item_id)] = 1 if i == pos_idx else 0
    return labels


def load_baseline_scores(score_path):
    """Load baseline scores: list of (user_id, item_id, score)"""
    rows = []
    with open(score_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["user_id"], row["item_id"], float(row["score"])))
    return rows


def normalize_scores_per_event(scores_with_labels):
    """Normalize scores to [0,1] per event (user) using min-max."""
    events = defaultdict(list)
    for uid, item_id, score, label in scores_with_labels:
        events[uid].append((item_id, score, label))

    normalized = []
    for uid, items in events.items():
        scores_arr = [s for _, s, _ in items]
        s_min, s_max = min(scores_arr), max(scores_arr)
        rng = s_max - s_min if s_max > s_min else 1.0
        for item_id, score, label in items:
            norm_score = (score - s_min) / rng
            normalized.append((uid, item_id, norm_score, label))
    return normalized


def compute_calibration_metrics(scores, labels):
    """Compute ECE, MCE, AUROC, Brier score."""
    scores = np.array(scores)
    labels = np.array(labels)
    n = len(scores)

    brier = float(np.mean((scores - labels) ** 2))

    bins = np.linspace(0, 1, 11)
    ece = 0.0
    mce = 0.0
    reliability_rows = []
    for i in range(10):
        mask = (scores >= bins[i]) & (scores < bins[i+1])
        if i == 9:
            mask = (scores >= bins[i]) & (scores <= bins[i+1])
        count = int(mask.sum())
        if count > 0:
            avg_conf = float(scores[mask].mean())
            acc = float(labels[mask].mean())
            gap = abs(avg_conf - acc)
            ece += gap * count / n
            mce = max(mce, gap)
        else:
            avg_conf = float((bins[i] + bins[i+1]) / 2)
            acc = 0.0
        reliability_rows.append({
            "bin_lower": float(bins[i]),
            "bin_upper": float(bins[i+1]),
            "bin_center": float((bins[i] + bins[i+1]) / 2),
            "count": count,
            "avg_confidence": avg_conf,
            "accuracy": acc,
        })

    try:
        from sklearn.metrics import roc_auc_score
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
    parser.add_argument("--score_file", type=str, required=True, help="Baseline score CSV")
    parser.add_argument("--ranking_test", type=str, required=True, help="ranking_test.jsonl with labels")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--method_name", type=str, required=True)
    parser.add_argument("--domain", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading labels from {args.ranking_test}...")
    labels_map = load_ranking_test_labels(args.ranking_test)
    print(f"  {len(labels_map)} (user, item) pairs")

    print(f"Loading scores from {args.score_file}...")
    raw_scores = load_baseline_scores(args.score_file)
    print(f"  {len(raw_scores)} score rows")

    # Join scores with labels
    joined = []
    missing = 0
    for uid, item_id, score in raw_scores:
        label = labels_map.get((uid, item_id))
        if label is not None:
            joined.append((uid, item_id, score, label))
        else:
            missing += 1
    print(f"  Joined: {len(joined)}, Missing labels: {missing}")

    # Normalize per event
    normalized = normalize_scores_per_event(joined)
    norm_scores = [s for _, _, s, _ in normalized]
    norm_labels = [l for _, _, _, l in normalized]

    # Compute metrics
    metrics, reliability_rows = compute_calibration_metrics(norm_scores, norm_labels)
    metrics["method"] = args.method_name
    metrics["domain"] = args.domain
    metrics["score_file"] = args.score_file

    # Save
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "diagnostic_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "reliability_bins.csv", "w") as f:
        f.write("bin_lower,bin_upper,bin_center,count,avg_confidence,accuracy\n")
        for row in reliability_rows:
            f.write(f"{row['bin_lower']},{row['bin_upper']},{row['bin_center']},{row['count']},{row['avg_confidence']},{row['accuracy']}\n")

    print(f"\n{'='*50}")
    print(f"Calibration Diagnostic: {args.method_name} / {args.domain}")
    print(f"{'='*50}")
    print(f"AUROC:       {metrics['auroc']:.4f}")
    print(f"ECE:         {metrics['ece']:.4f}")
    print(f"MCE:         {metrics['mce']:.4f}")
    print(f"Brier:       {metrics['brier_score']:.4f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Avg score:   {metrics['avg_score']:.4f}")


if __name__ == "__main__":
    main()
