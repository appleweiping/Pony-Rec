"""Aggregate multi-seed C-CRP v3 reports -> mean +/- std per domain.

Reads outputs/<domain>_large10000_100neg_ccrp_v3_seed{2026,2027,2028}/report.json
for each of {sports,toys,home,tools} and writes:
  outputs/summary/paper_critical/gpu_queue_to8/multiseed_ndcg_summary.json
  outputs/summary/paper_critical/gpu_queue_to8/multiseed_ndcg_table.csv

For every metric (HR@5/10/20, NDCG@5/10/20, MRR) it reports mean and std over the
3 seeds; the headline for the paper is NDCG@10 mean +/- std. It also prints, for
context, the original (unseeded) Qwen NDCG@10 from outputs/<...>_ccrp_v3/report.json
so you can confirm the seeds bracket the published number.

Run on the server (after run_ccrp_v3_multiseed_qwen.sh finishes), CPU-only:
  PYTHONPATH=... python experiments/rsc/aggregate_multiseed_ndcg.py
Or locally after scp-ing the small report.json files down.
"""
import json, statistics, csv
from pathlib import Path

DOMAINS = ["sports", "toys", "home", "tools"]
SEEDS = [2026, 2027, 2028]
METRICS = ["HR@5", "NDCG@5", "HR@10", "NDCG@10", "HR@20", "NDCG@20", "MRR"]

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/summary/paper_critical/gpu_queue_to8"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_report(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main():
    summary = {}
    rows = []
    for d in DOMAINS:
        prefix = f"{d}_large10000_100neg"
        per_seed = {}
        for s in SEEDS:
            rep = load_report(ROOT / f"outputs/{prefix}_ccrp_v3_seed{s}/report.json")
            if rep is not None:
                per_seed[s] = rep
        orig = load_report(ROOT / f"outputs/{prefix}_ccrp_v3/report.json")  # original unseeded Qwen

        dom = {"n_seeds_found": len(per_seed), "seeds": sorted(per_seed.keys()),
               "original_unseeded": {m: orig.get(m) for m in METRICS} if orig else None}
        for m in METRICS:
            vals = [per_seed[s][m] for s in per_seed if m in per_seed[s]]
            if vals:
                mean = statistics.mean(vals)
                std = statistics.pstdev(vals) if len(vals) > 1 else 0.0  # population std over the seeds
                dom[m] = {"mean": mean, "std": std, "values": vals}
        summary[d] = dom

        nd = dom.get("NDCG@10", {})
        rows.append({
            "domain": d,
            "n_seeds": len(per_seed),
            "NDCG@10_mean": round(nd.get("mean", float("nan")), 6) if nd else "",
            "NDCG@10_std": round(nd.get("std", float("nan")), 6) if nd else "",
            "NDCG@10_original_unseeded": round(orig["NDCG@10"], 6) if orig else "",
        })
        if nd:
            print(f"{d:8s} NDCG@10 = {nd['mean']:.4f} +/- {nd['std']:.4f}  "
                  f"(seeds {dom['seeds']}, orig {orig['NDCG@10']:.4f} )" if orig
                  else f"{d:8s} NDCG@10 = {nd['mean']:.4f} +/- {nd['std']:.4f}")
        else:
            print(f"{d:8s} NDCG@10 = <no seed reports found yet>")

    (OUT_DIR / "multiseed_ndcg_summary.json").write_text(json.dumps(summary, indent=2))
    with open(OUT_DIR / "multiseed_ndcg_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "n_seeds", "NDCG@10_mean",
                                          "NDCG@10_std", "NDCG@10_original_unseeded"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_DIR/'multiseed_ndcg_summary.json'}")
    print(f"Wrote {OUT_DIR/'multiseed_ndcg_table.csv'}")


if __name__ == "__main__":
    main()
