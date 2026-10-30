"""Pool the mis_drone tables over seeds: mean crash % over test, steep and shallow dives, mean +- s.e. per row.
usage: python work/pool_drone.py [--base] prop50mv prop50mv-gust10 ...  (--base pools the base_drone baselines)"""
import json
import re
import sys
from collections import defaultdict

import numpy as np

pre = "base" if "--base" in sys.argv else "mis"
for f in [a for a in sys.argv[1:] if a != "--base"]:
    t = defaultdict(list)
    for s in range(1, 6):
        try:
            d = json.load(open(f"work/{pre}_{f}_s{s}.json"))
        except FileNotFoundError:
            continue
        for r in d["rows"]:
            m = re.sub(r"(from|add) (\d+) (dives|crash)", lambda x: f"{x.group(1)} {x.group(2) if x.group(2) in ('1', '3') else 'all'} {x.group(3)}", r["model"])
            t[m].append([r["test dives"], r["steep dives"], r["shallow dives"]])
    print(f"== {f}")
    for m, v in t.items():
        a = 100 * np.array(v); avg = a.mean(1)
        se = avg.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
        print(f"  {m[:56]:56s} n={len(v)} mean {avg.mean():5.1f} +- {se:4.1f} | test {a[:, 0].mean():5.1f} steep {a[:, 1].mean():5.1f} shallow {a[:, 2].mean():5.1f}")
