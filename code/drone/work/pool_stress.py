"""Per-seed table for the three stress settings: sweep_*.json (FACT sign rule) + stressref_*.json (unfixed, oracle).

usage: python work/pool_stress.py
"""
import json
import os

import numpy as np

N_FLIGHTS = 95
TASKS = ("test dives", "steep dives", "shallow dives")
SETS = {"sev20_n8_l0": "mild fault (+20%)", "sev50_n11_l0": "80% faulty (44 bad)", "sev50_n8_l8": "32 look-alikes"}


def crash(r):
    return 100 * np.mean([r[t] for t in TASKS])


def reach(r):
    return 100 * np.mean([r[t + " reach"] for t in TASKS])


out = {}
for key, name in SETS.items():
    print(name)
    rows = []
    for s in range(1, 6):
        fs, fr = f"work/sweep_{key}_s{s}.json", f"work/stressref_{key}_s{s}.json"
        if not os.path.exists(fs):
            continue
        a = json.load(open(fs)); det = a["detection"]
        ref = json.load(open(fr)) if os.path.exists(fr) else None
        row = dict(seed=s, n_bad=a["n_faulty"], n_look=a["n_look"], bad_found=det["sign_bad"],
                   correct_removed=det["sign_removed"] - det["sign_bad"], lookalike_removed=det["sign_lookalike"],
                   kept=N_FLIGHTS - det["sign_removed"], auroc=det["auroc"], ap=det["ap"],
                   fact=crash(a["sign_rule_flown"]), fact_reach=reach(a["sign_rule_flown"]),
                   unfixed=crash(ref["unfixed"]) if ref and "unfixed" in ref else None,
                   oracle=crash(ref["oracle"]) if ref and "oracle" in ref else None,
                   oracle_reach=reach(ref["oracle"]) if ref and "oracle" in ref else None)
        rows.append(row)
        f = lambda v: "  -  " if v is None else f"{v:5.1f}"
        print(f"  s{s}: unfixed {f(row['unfixed'])}  FACT {f(row['fact'])}  oracle {f(row['oracle'])} | bad found "
              f"{row['bad_found']}/{row['n_bad']}  correct removed {row['correct_removed']} (look-alikes "
              f"{row['lookalike_removed']})  kept {row['kept']}/{N_FLIGHTS} | AUROC {row['auroc']:.2f} AP {row['ap']:.2f}"
              f" | reach FACT {row['fact_reach']:.0f} oracle {f(row['oracle_reach'])}")
    out[key] = rows
    for k in ("unfixed", "fact", "oracle"):
        v = [r[k] for r in rows if r[k] is not None]
        if v:
            print(f"  {k}: {np.mean(v):.1f} +- {np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0:.1f} (n={len(v)})")
json.dump(out, open("work/pool_stress.json", "w"), indent=1)
