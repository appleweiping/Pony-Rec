#!/usr/bin/env python3
"""Posterior-degeneracy diagnostic for the C-CRP v3 second-backbone study.

For each (domain, backbone) pointwise score file, quantify how "graded" vs
"degenerate" the verbalized relevance posterior is, grouped by ranking event
(101 candidates each). This explains the deep-recall (HR@20) softness of the
coarser Llama posterior and gives the rationale for ranking by the RAW posterior
(eta=0): a risk reweighting has little graded signal to exploit when most scores
are floored at 0.0.

Metrics per (domain, backbone):
  floor_rate     : fraction of all candidate scores == 0.0
  sat_rate       : fraction == 1.0
  graded_rate    : fraction strictly in (0,1)
  mean_distinct  : mean # distinct score values per 101-candidate event
  mean_top_ties  : mean # candidates sharing the per-event MAX score (>1 = ambiguous top)
  pos_floor_rate : fraction of events whose ground-truth positive item is floored at 0.0

Pure stdlib (csv), CPU-only, streams the file once. No GPU, no pandas.
Run on the server where scores.csv live.
"""
import csv, json, sys, os
from collections import defaultdict

# domain -> {backbone: scores.csv path}
DOMAINS = ["sports", "toys", "home", "tools"]
BACKBONES = {
    "qwen3-8b": "outputs/{d}_large10000_100neg_ccrp_v3/scores.csv",
    "llama3.1-8b": "outputs/{d}_large10000_100neg_ccrp_v3_llama/scores.csv",
}


def positive_items(domain):
    """Map source_event_id -> positive item_id from the ranking_test.jsonl.

    The same-candidate panel stores the ground-truth positive; we read it so we
    can measure how often the positive is floored at 0.0 (a direct driver of
    recall loss). Best-effort: if the field layout differs, returns {} and the
    pos_floor_rate is reported as None.
    """
    path = ("outputs/baselines/external_tasks/"
            f"{domain}_large10000_100neg_test_same_candidate/ranking_test.jsonl")
    pos = {}
    if not os.path.exists(path):
        return pos
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            eid = obj.get("source_event_id") or obj.get("event_id") or obj.get("id")
            p = (obj.get("positive_item_id") or obj.get("positive") or
                 obj.get("target_item_id") or obj.get("label_item_id"))
            if eid is not None and p is not None:
                pos[str(eid)] = str(p)
    return pos


def diagnose(domain, backbone, path, pos_map):
    if not os.path.exists(path):
        return None
    by_event = defaultdict(list)          # eid -> [(item, score)]
    n_scores = floor = sat = graded = 0
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                s = float(row["score"])
            except (KeyError, ValueError):
                continue
            by_event[row["source_event_id"]].append((row["item_id"], s))
            n_scores += 1
            if s == 0.0:
                floor += 1
            elif s == 1.0:
                sat += 1
            else:
                graded += 1
    if n_scores == 0:
        return None
    n_events = len(by_event)
    distinct_sum = top_ties_sum = 0
    pos_floored = pos_known = 0
    for eid, lst in by_event.items():
        scores = [s for _, s in lst]
        distinct_sum += len(set(scores))
        mx = max(scores)
        top_ties_sum += sum(1 for s in scores if s == mx)
        if eid in pos_map:
            pos_known += 1
            pitem = pos_map[eid]
            for it, s in lst:
                if it == pitem:
                    if s == 0.0:
                        pos_floored += 1
                    break
    return {
        "domain": domain,
        "backbone": backbone,
        "n_events": n_events,
        "n_scores": n_scores,
        "floor_rate": round(floor / n_scores, 4),
        "sat_rate": round(sat / n_scores, 4),
        "graded_rate": round(graded / n_scores, 4),
        "mean_distinct_per_event": round(distinct_sum / n_events, 3),
        "mean_top_ties_per_event": round(top_ties_sum / n_events, 3),
        "pos_floor_rate": (round(pos_floored / pos_known, 4) if pos_known else None),
        "pos_known": pos_known,
    }


def main():
    rows = []
    for d in DOMAINS:
        pos_map = positive_items(d)
        for bk, tmpl in BACKBONES.items():
            res = diagnose(d, bk, tmpl.format(d=d), pos_map)
            if res:
                rows.append(res)
                print(json.dumps(res), flush=True)
    out = "outputs/summary/paper_critical/posterior_degeneracy_diagnostic.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
