"""Crazyflie comparison of curation and learning to reweight: crash and reach per row, over every seed that has the row
(rows collected across all tag files of a seed)."""
import glob, json, re
import numpy as np
T = ("test dives", "steep dives", "shallow dives")
KEY = ["faulty (all flights)", "oracle (flights below 3.35 V)", "add 1 crash dives, W=20", "add 3 crash dives, W=20",
       "ours from 1 dives, sign rule", "ours from 3 dives, sign rule",
       "ours from 1 dives, sign rule, and add them (W=20)", "ours from 3 dives, sign rule, and add them (W=20)",
       "learning to reweight, 1 crash dives as clean set", "learning to reweight, 3 crash dives as clean set",
       "learning to reweight, 1 crash dives as clean set, and add them (W=20)",
       "learning to reweight, 3 crash dives as clean set, and add them (W=20)"]
per = {}
for f in [g for g in glob.glob("cf_3.1_eq_s*_*.json") if re.search(r"_s\d+_(base|A|B|C|L)\.json$", g)]:
    s = int(re.search(r"_s(\d+)_", f).group(1))
    for r in json.load(open(f))["rows"]:
        per.setdefault(r["model"], {})[s] = (100 * np.mean([r[t] for t in T]), 100 * np.mean([r[t + " reach"] for t in T]))
for k in KEY:
    d = per.get(k, {})
    if not d:
        print(f"{k:72s} -"); continue
    c = np.array([d[s][0] for s in sorted(d)]); re_ = np.array([d[s][1] for s in sorted(d)])
    se = c.std(ddof=1) / np.sqrt(len(c)) if len(c) > 1 else float("nan")
    print(f"{k:72s} n={len(c)} crash {c.mean():5.1f} +- {se:4.1f} reach {re_.mean():3.0f} seeds {dict((s, round(d[s][0], 1)) for s in sorted(d))}")
