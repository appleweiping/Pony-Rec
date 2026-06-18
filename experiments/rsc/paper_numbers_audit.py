"""ARIS paper-claim-audit (numbers slice): main_results table vs report.json source-of-truth.

Cross-checks the \\method{} (C-CRP v3) HR@10 + NDCG@10 in Paper/tables/main_results.tex
against the committed Qwen report.json for all 8 domains. Handles BOTH report schemas:
  - original 4 (beauty/books/electronics/movies): outputs/ccrp_v3_formal/<dom>/report.json, keys hr10/ndcg10
  - new 4 (sports/toys/home/tools): outputs/<dom>_*_ccrp_v3/report.json, keys HR@10/NDCG@10
Re-run after the 3-backbone table integration to re-audit the delta.
"""
import json, glob

# \method{} (C-CRP) rows from Paper/tables/main_results.tex (HR@10, NDCG@10)
TABLE = {
    'beauty': (0.2292, 0.1341), 'books': (0.4758, 0.3328),
    'electronics': (0.2993, 0.1833), 'movies': (0.2083, 0.1281),
    'sports': (0.3819, 0.2329), 'toys': (0.3964, 0.2708),
    'home': (0.2264, 0.1324), 'tools': (0.2696, 0.1661),
}
ORIG = {'beauty', 'books', 'electronics', 'movies'}

def report_path(dom):
    if dom in ORIG:
        return f'outputs/ccrp_v3_formal/{dom}/report.json'
    for p in glob.glob(f'outputs/{dom}_*_ccrp_v3/report.json'):
        if 'mistral' not in p and 'llama' not in p:
            return p
    return None

def get(d, *keys):
    for k in keys:
        if k in d:
            return d[k]
    return None

bad = 0
for dom, (hr, nd) in TABLE.items():
    p = report_path(dom)
    if not p:
        print(f"  {dom}: NO report.json"); bad += 1; continue
    d = json.load(open(p))
    rh = get(d, 'HR@10', 'hr10'); rn = get(d, 'NDCG@10', 'ndcg10')
    if rh is None or rn is None:
        print(f"  {dom}: keys not found in {p}"); bad += 1; continue
    rh, rn = round(rh, 4), round(rn, 4)
    ok = abs(rh - hr) < 0.0006 and abs(rn - nd) < 0.0006
    if not ok:
        bad += 1
    print(f"  {dom:12s} table(HR@10={hr}, NDCG@10={nd}) vs report({rh}, {rn})  {'OK' if ok else 'MISMATCH <<<'}")
print(f"\n=== VERDICT: {'ALL 8 MATCH (numbers faithful to source)' if bad == 0 else f'{bad} ISSUE(S)'} ===")
