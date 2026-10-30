"""Pool the evaluator-plant panel (plant_*.json) and the failure-record study (failrec_*.json).

usage: python work/pool_robust.py
"""
import glob
import json

import numpy as np


def ms(v):
    v = 100 * np.asarray(v)
    se = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return f"{v.mean():5.1f} +- {se:3.1f} (n={len(v)}) [{'/'.join(f'{x:.1f}' for x in v)}]"


def rng(v):
    return f"{min(v)}" if min(v) == max(v) else f"{min(v)}-{max(v)}"


out = {}
print("Evaluator plants (prop50j8)")
for p in ("wide", "subset", "gbt", "mismatch", "heldout"):
    fs = sorted(glob.glob(f"work/plant_{p}_prop50j8_s*.json"))
    ds = [json.load(open(f)) for f in fs]
    ds = [d for d in ds if len(d["rows"]) == 4]
    if not ds:
        continue
    print(f" {p}: R2 {np.round(ds[0]['plant_r2'], 2)}, gap to default {ds[0]['mean_gap_to_default_plant']:.2f}")
    rows = {}
    for j, name in enumerate(["unfixed", "add", "fact", "oracle"]):
        v = [d["rows"][j]["crash"] for d in ds]
        rows[name] = v
        print(f"   {name:8s} {ms(v)}")
    det = [d["rows"][2] for d in ds]
    rem = [r.get("removed") for r in det]; bad = [r.get("bad_removed") for r in det]
    if None in rem:
        rem = [d.get("detection", {}).get("sign_removed") for d in ds]; bad = [d.get("detection", {}).get("sign_bad") for d in ds]
    print(f"   removed {rng(rem)}, bad {rng(bad)}")
    out[p] = dict(rows=rows, removed=rem, bad=bad, r2=ds[0]["plant_r2"], gap=ds[0]["mean_gap_to_default_plant"])

for setting in ("prop50j8", "sweep_sev35_n8_l4"):
    fs = sorted(glob.glob(f"work/failrec_{setting}_s*.json"))
    ds = [json.load(open(f)) for f in fs]
    if not ds:
        continue
    print(f"\nFailure records: {setting} ({len(ds)} seeds)")
    if "unfixed" in ds[0]:
        print(f"   unfixed  {ms([d['unfixed']['crash'] for d in ds if 'unfixed' in d])}")
    o = {}
    for v in ("clean", "noise1", "noise3", "first75", "first50", "sup_m0.1", "sup_m0.5", "sup_m1.0", "sup_late", "combined"):
        rr = [d["variants"][v] for d in ds if v in d["variants"]]
        if not rr:
            continue
        o[v] = rr
        print(f"   {v:9s} {ms([r['crash'] for r in rr])}  removed {rng([r['removed'] for r in rr])}, "
              f"bad {rng([r['bad_removed'] for r in rr])} of {ds[0]['n_bad']}, look-alikes {rng([r['lookalike_removed'] for r in rr])}")
    sup = [d["supervisor"] for d in ds]
    for v in ("sup_m0.1", "sup_m0.5", "sup_m1.0", "sup_late"):
        hs = [s[v]["takeover_h"] for s in sup if v in s]; sv = [s[v]["saved"] for s in sup if v in s]
        if hs:
            print(f"   {v}: takeover at {min(hs):.2f}-{max(hs):.2f} m, saved {sum(sv)}/{len(sv)}")
    out[setting] = o
json.dump(out, open("work/pool_robust.json", "w"), indent=1)
