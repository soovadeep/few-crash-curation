"""Pool the equal-steps Crazyflie tables over seeds: mean crash % (test/steep/shallow) and reach, +- s.e.
usage: python pool_eq.py [--split] [--seeds 1,2,3]   (--split: the held-out-plant runs, cfsplit_*.json, instead)"""
import glob, json, re, sys
import numpy as np
SPLIT = "--split" in sys.argv
seeds = [int(a) for a in sys.argv[sys.argv.index("--seeds") + 1].split(",")] if "--seeds" in sys.argv else [1, 2, 3]
tab = {}
for s in seeds:
    # the held-out-plant runs reuse the main table's row names, so the two experiments live under different prefixes
    pat = f"cfsplit_3.1_eq_s{s}*.json" if SPLIT else f"cf_3.1_eq_s{s}_*.json"
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f))
        assert d.get("split", False) == SPLIT, f"{f} is from the {'held-out' if d.get('split') else 'main'} experiment"
        for r in d["rows"]:
            m = re.sub(r"random flights \(draw \d\)", "random flights", r["model"])
            c = 100 * (r["test dives"] + r["steep dives"] + r["shallow dives"]) / 3
            g = 100 * (r["test dives reach"] + r["steep dives reach"] + r["shallow dives reach"]) / 3
            tab.setdefault(m, {}).setdefault(s, []).append((c, g, r.get("bad_removed"), r.get("removed")))
for m, d in tab.items():
    per = [np.mean([x[0] for x in d[s]]) for s in d]
    reach = np.mean([x[1] for s in d for x in d[s]])
    bad = [x[2] for s in d for x in d[s] if x[2] is not None]
    se = np.std(per, ddof=1) / np.sqrt(len(per)) if len(per) > 1 else float("nan")
    print(f"{m[:58]:58s} n={len(per)} crash {np.mean(per):5.1f} +- {se:4.1f}  per seed {[round(p, 1) for p in per]}  reach {reach:3.0f}"
          + (f"  bad removed {bad}/{d[next(iter(d))][0][3]}" if bad else ""))
