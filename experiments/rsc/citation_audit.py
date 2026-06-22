"""ARIS citation-audit for the pony C-CRP paper: unresolved cites, orphans, dup keys."""
import re, glob, os
from collections import Counter

texs = glob.glob('Paper/sections/*.tex') + ['Paper/main.tex'] + glob.glob('Paper/tables/*.tex')
cite_re = re.compile(r'\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}')
cited = {}
for t in texs:
    txt = open(t, encoding='utf-8', errors='replace').read()
    for m in cite_re.finditer(txt):
        for k in m.group(1).split(','):
            k = k.strip()
            if k:
                cited.setdefault(k, set()).add(os.path.basename(t))

bib = open('Paper/references.bib', encoding='utf-8', errors='replace').read()
allbib = re.findall(r'@\w+\s*\{\s*([^,\s]+)', bib)
bibkeys = set(allbib)
citedkeys = set(cited)

print(f"=== {len(citedkeys)} distinct cite-keys used; {len(bibkeys)} bib entries ===")
missing = sorted(citedkeys - bibkeys)
print(f"\n--- UNRESOLVED cites (used, NOT in .bib) [{len(missing)}] ---")
for k in missing:
    print(f"  {k}  (in {sorted(cited[k])})")
orphan = sorted(bibkeys - citedkeys)
print(f"\n--- ORPHAN bib entries (defined, never cited) [{len(orphan)}] ---")
for k in orphan:
    print(f"  {k}")
dups = [k for k, c in Counter(allbib).items() if c > 1]
print(f"\n--- DUPLICATE bib keys [{len(dups)}] ---")
for k in dups:
    print(f"  {k}")
print("\n=== VERDICT:", "CLEAN" if not (missing or dups) else "ISSUES FOUND", "===")
