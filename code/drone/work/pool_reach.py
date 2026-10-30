"""Pool the per-dive flights of reach_drone.py: crash, reach and safe completion (% of the 72 evaluation dives, mean
+- s.e. over seeds) per fault and model, with FACT's removals and, for the reference study, the reference's coverage.
Also the pipeline variants of pipeline_drone.py. Reads work/reach_*.json and work/pipeline_*.json; writes
work/pool_reach.json.

usage (from code/drone/): python work/pool_reach.py
"""
import glob
import json
import re

import numpy as np

TASKS = ("test dives", "steep dives", "shallow dives")


def se(v):
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


out = {}
for f in sorted(glob.glob("work/reach_*.json")):
    tag, fault, seed = re.match(r"work/reach_(.+)_([a-z0-9]+)_s(\d+)\.json", f).groups()
    d = json.load(open(f))
    row = {k: 100 * np.mean([d["summary"][t][k] for t in TASKS]) for k in ("crash", "reach", "safe")}
    for k in ("removed", "bad_removed", "auroc"):
        if k in d["info"]:
            row[k] = d["info"][k]
    ref = d["info"].get("reference", {})
    for k in ("axial_descent_share", "faulty_err", "transitions"):
        if k in ref:
            row["ref_" + k] = ref[k]
    out.setdefault(fault, {}).setdefault(tag, {})[int(seed)] = row
for f in sorted(glob.glob("work/pipeline_*.json")):
    var, fault, seed = re.match(r"work/pipeline_([a-z]+)_([a-z0-9]+)_s(\d+)\.json", f).groups()
    d = json.load(open(f))
    row = {k: d["info"][k] for k in ("removed", "bad_removed", "auroc")}
    if d["summary"]:
        row.update({k: 100 * np.mean([d["summary"][t][k] for t in TASKS]) for k in ("crash", "reach", "safe")})
    out.setdefault(fault, {}).setdefault("pipeline-" + var, {})[int(seed)] = row

pooled = {}
for fault, models in out.items():
    for tag, seeds in models.items():
        keys = sorted({k for r in seeds.values() for k in r})
        p = {"seeds": sorted(seeds)}
        for k in keys:
            v = [seeds[s][k] for s in sorted(seeds) if k in seeds[s]]
            p[k] = float(np.mean(v)); p[k + "_se"] = se(v)
            p[k + "_per_seed"] = v
        pooled.setdefault(fault, {})[tag] = p
        print(f"{fault:10s} {tag:22s} n={len(seeds)} crash {p['crash']:5.1f}+-{p['crash_se']:4.1f} "
              f"safe {p.get('safe', float('nan')):5.1f}+-{p.get('safe_se', float('nan')):4.1f}"
              + (f"  removed {p['removed']:.1f} bad {p['bad_removed']:.1f} AUROC {p['auroc']:.2f}" if "removed" in p else "")
              + (f"  coverage {p['ref_axial_descent_share']:.2f}" if "ref_axial_descent_share" in p else ""))
json.dump(pooled, open("work/pool_reach.json", "w"), indent=1)
