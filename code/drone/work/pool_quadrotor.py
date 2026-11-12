"""Numbers for the paper's quadrotor tables: crash rate (mean +- s.e. over seeds, test/steep/shallow dives
pooled) and goal reach for every fix, from base_*.json (failure-free and crude rules), cupid_*.json, mis_*.json
(adding failures, FACT), sup_*.json (supervised runs) and controls_*.json (reweighting, fine-tuning, budget controls).

usage: python work/pool_quadrotor.py  -> prints a block per benchmark and writes work/pool_quadrotor.json
"""
import glob
import json

import numpy as np

T = ("test dives", "steep dives", "shallow dives")
BENCH = ("prop50j8", "gain40j8", "prop50j8m")


def seeds(pat):
    return sorted(glob.glob(pat))


def crash(r):
    return 100 * np.mean([r[t] for t in T])


def reach(r):
    return 100 * np.mean([r[t + " reach"] for t in T]) if all(t + " reach" in r for t in T) else None


def rows_by_name(files, key="rows", name="model"):
    out = {}
    for f in files:
        for r in json.load(open(f))[key]:
            out.setdefault(r[name], []).append(r)
    return out


def summ(rs, extra=()):
    c = [crash(r) for r in rs]
    re = [x for x in (reach(r) for r in rs) if x is not None]
    d = dict(n=len(c), crash=float(np.mean(c)), se=float(np.std(c, ddof=1) / np.sqrt(len(c))) if len(c) > 1 else None,
             per_seed=[round(x, 1) for x in c], reach=float(np.mean(re)) if re else None, n_reach=len(re))
    for k in extra:
        v = [r[k] for r in rs if k in r]
        if v:
            d[k] = [min(v), max(v)]
    return d


OUT = {}
for b in BENCH:
    res = {}
    base = rows_by_name(seeds(f"work/base_{b}_s?.json"))
    mis = rows_by_name(seeds(f"work/mis_{b}_s?.json"))
    sup = rows_by_name(seeds(f"work/sup_{b}_s?.json"))
    for name, src, key in [("None (all logs)", mis, "faulty"), ("Oracle", mis, "clean (bad flights removed)"),
                           ("Huber", base, "huber loss"), ("L1", base, "l1 loss"),
                           ("Worst-fit k=40", base, "remove 40 worst-fit flights"),
                           ("Ensemble mean", base, "ensemble of 5, mean"),
                           ("Ensemble beta=1", base, "ensemble of 5, disagreement penalty beta=1"),
                           ("Ensemble beta=10", base, "ensemble of 5, disagreement penalty beta=10"),
                           ("Crash-like k=40", base, "remove 40 most crash-like flights (most axial descent)"),
                           ("Drop descent", base, "remove all axial-descent samples"),
                           ("Add 1, W=20", mis, "add 1 crash dives, W=20"), ("Add 3, W=20", mis, "add 3 crash dives, W=20"),
                           ("Add 24, W=20", mis, "add 24 crash dives, W=20"),
                           ("FACT sign 1", mis, "ours (sign rule) from 1 dives"),
                           ("FACT sign 3", mis, "ours (sign rule) from 3 dives"),
                           ("FACT sign 1 + add", mis, "ours (sign rule) from 1 dives, and add them (W=20)"),
                           ("FACT sign 3 + add", mis, "ours (sign rule) from 3 dives, and add them (W=20)"),
                           ("FACT k=40, 1", mis, "ours from 1 dives, remove 40"),
                           ("FACT k=40, 3", mis, "ours from 3 dives, remove 40"),
                           ("FACT top-2s, sign count, 1", mis, "ours (worst 2 s, sign-rule count) from 1 dives"),
                           ("Supervised, sign", sup, "FACT from 1 supervised run with takeover, sign rule"),
                           ("Supervised, k=40", sup, "FACT from 1 supervised run with takeover, remove k"),
                           ("Before takeover, sign", sup, "FACT from 1 supervised run before takeover, sign rule"),
                           ("Before takeover, k=40", sup, "FACT from 1 supervised run before takeover, remove k")]:
        if key in src:
            res[name] = summ(src[key], ("removed", "bad_removed", "distr_removed"))
    cup = {}
    for f in seeds(f"work/cupid_{b}_s?.json"):
        for r in json.load(open(f))["removal"]:
            r = dict(r, **{"test dives": r["crash"], "test dives reach": r["reach"]})
            cup.setdefault((r["method"], r["k"]), []).append(r)
    for (m, k), rs in sorted(cup.items()):
        res[f"{m} k={k}"] = summ(rs, ("bad_removed", "decoys_removed"))
    rev = [json.load(open(f)) for f in seeds(f"work/controls_{b}_s?.json")]
    if rev:
        names = sorted({k for d in rev for k, v in d["flown"].items() if isinstance(v, dict)})
        for k in names:
            res[f"controls {k}"] = summ([d["flown"][k] for d in rev if isinstance(d["flown"].get(k), dict)])
        z = [d["diag_l2rw_zero_weight_batches"] for d in rev if "diag_l2rw_zero_weight_batches" in d]
        if z:
            res["controls l2rw diag"] = dict(zero_weight_batches=z, err_z=[d["diag_l2rw_mean_err_z"] for d in rev if "diag_l2rw_mean_err_z" in d])
        res["controls removal"] = [d["removal"] for d in rev]
    OUT[b] = res
    print(f"== {b}")
    for k, v in res.items():
        if "crash" in v:
            se = f"{v['se']:.1f}" if v["se"] is not None else "-"
            rr = f"{v['reach']:.0f} ({v['n_reach']})" if v["reach"] is not None else "-"
            ext = {kk: vv for kk, vv in v.items() if kk in ("removed", "bad_removed", "distr_removed", "decoys_removed")}
            print(f"  {k:34s} n={v['n']} crash {v['crash']:5.1f} +- {se:>4s} reach {rr:8s} {v['per_seed']} {ext}")
        else:
            print(f"  {k:34s} {v}")
json.dump(OUT, open("work/pool_quadrotor.json", "w"), indent=1)
