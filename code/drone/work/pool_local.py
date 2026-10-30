"""Pool the within-flight fault runs (local_drone.py): crash rate over test, steep and shallow dives, and what each
removal rule takes out, per seed and mean +- s.e.

usage: python work/pool_local.py [tag]
"""
import glob
import json
import sys

import numpy as np

tag = sys.argv[1] if len(sys.argv) > 1 else "sev50_n11_on25-50"
TASKS = ("test dives", "steep dives", "shallow dives")
ds = [json.load(open(f)) for f in sorted(glob.glob(f"work/local_{tag}_s*.json"))]
ds = [d for d in ds if "flown" in d and len(d["flown"]) == 6]
print(f"{len(ds)} seeds; faulty share of descent {ds[0]['faulty_share_of_descent']:.2f}, faulty flights' descent "
      f"before onset {ds[0]['faulty_flights_descent_before_onset']:.2f}")
print("AUROC on descent windows", [round(d["detection"]["window_auroc_descent"], 3) for d in ds],
      "samples", [round(d["detection"]["sample_auroc_descent"], 3) for d in ds])
out = {}
for k in ds[0]["flown"]:
    c = [100 * np.mean([d["flown"][k][t] for t in TASKS]) for d in ds]
    reach = [100 * np.mean([d["flown"][k][t + " reach"] for t in TASKS]) for d in ds]
    row = dict(crash=c, reach=reach)
    s = f"{k:15s} crash {np.mean(c):5.1f} +- {np.std(c, ddof=1) / np.sqrt(len(c)):3.1f} [{'/'.join(f'{x:.1f}' for x in c)}]"
    s += f" reach {np.mean(reach):4.0f}"
    if k in ds[0]["removal"]:
        for m in ("faulty_descent_removed", "correct_descent_removed", "correct_removed", "kept"):
            v = [100 * d["removal"][k][m] for d in ds]
            row[m] = v
            s += f" | {m} {min(v):.0f}-{max(v):.0f}"
    out[k] = row
    print(s)
json.dump(out, open(f"work/pool_local_{tag}.json", "w"), indent=1)
