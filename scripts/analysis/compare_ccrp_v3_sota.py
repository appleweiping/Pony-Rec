"""Compare C-CRP v3 against all baselines for SOTA determination.

Reads ranking_metrics.csv from each method's output directory and produces
a comparison table. Run after all baselines and C-CRP v3 are imported.
"""
import csv
import json
from pathlib import Path
import argparse


def load_metrics(csv_path: Path) -> dict:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return {k: float(v) for k, v in row.items() if v.replace('.', '').replace('-', '').replace('e', '').isdigit()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="sports,toys,home,tools")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--output", default="outputs/summary/ccrp_v3_sota_comparison.json")
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",")]
    root = Path(args.output_root)

    results = {}
    for domain in domains:
        exp_prefix = f"{domain}_large10000_100neg"
        domain_results = {}

        # Find all ranking_metrics.csv for this domain
        pattern = f"{exp_prefix}_*_same_candidate/tables/ranking_metrics.csv"
        metrics_files = list(root.glob(pattern))

        for mf in metrics_files:
            method_dir = mf.parent.parent.name
            method_name = method_dir.replace(f"{exp_prefix}_", "").replace("_same_candidate", "")
            try:
                metrics = load_metrics(mf)
                domain_results[method_name] = metrics
            except Exception as e:
                print(f"  WARN: {method_dir}: {e}")

        if domain_results:
            results[domain] = domain_results

    # Determine SOTA
    key_metrics = ["HR@5", "NDCG@5", "MRR", "HR@10", "NDCG@10", "HR@20", "NDCG@20"]
    sota_summary = {}

    for domain, methods in results.items():
        ccrp_key = [k for k in methods if "ccrp" in k.lower()]
        if not ccrp_key:
            sota_summary[domain] = {"status": "NO_CCRP_FOUND", "methods": list(methods.keys())}
            continue

        ccrp_metrics = methods[ccrp_key[0]]
        is_sota = True
        beaten_by = []

        for method_name, method_metrics in methods.items():
            if "ccrp" in method_name.lower():
                continue
            for metric in ["HR@5", "NDCG@5", "MRR"]:
                if method_metrics.get(metric, 0) > ccrp_metrics.get(metric, 0):
                    beaten_by.append(f"{method_name}:{metric}={method_metrics[metric]:.4f}>{ccrp_metrics[metric]:.4f}")
                    is_sota = False

        sota_summary[domain] = {
            "is_sota": is_sota,
            "ccrp_method": ccrp_key[0],
            "ccrp_hr5": ccrp_metrics.get("HR@5", 0),
            "ccrp_ndcg5": ccrp_metrics.get("NDCG@5", 0),
            "ccrp_mrr": ccrp_metrics.get("MRR", 0),
            "beaten_by": beaten_by if beaten_by else None,
            "n_baselines": len(methods) - 1,
        }

    # Print summary
    print(f"\n{'='*60}")
    print("C-CRP v3 SOTA Comparison")
    print(f"{'='*60}")
    sota_count = 0
    for domain, info in sota_summary.items():
        status = "SOTA" if info.get("is_sota") else "NOT SOTA"
        if info.get("is_sota"):
            sota_count += 1
        print(f"\n{domain}: {status}")
        if "ccrp_hr5" in info:
            print(f"  C-CRP: HR@5={info['ccrp_hr5']:.4f} NDCG@5={info['ccrp_ndcg5']:.4f} MRR={info['ccrp_mrr']:.4f}")
        if info.get("beaten_by"):
            for b in info["beaten_by"][:3]:
                print(f"  Beaten: {b}")

    print(f"\n{'='*60}")
    print(f"SOTA domains: {sota_count}/{len(sota_summary)}")
    print(f"{'='*60}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"sota_summary": sota_summary, "full_results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
