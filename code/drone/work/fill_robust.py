"""Write the rows of Table tab:plants in the paper from pool_robust.json (run pool_robust.py first).
Rows are found by their label prefix, as in fill_sweep.py.

usage (from code/drone/): python work/fill_robust.py ../../paper/ras/main.tex
"""
import json
import sys

import numpy as np

P = json.load(open("work/pool_robust.json"))
tex = sys.argv[1]
s = open(tex).read()


def m(v, se=True):
    v = 100 * np.asarray(v)
    if not se or len(v) == 1:
        return f"{v.mean():.1f}"
    return f"{v.mean():.1f}$\\pm${np.std(v, ddof=1) / np.sqrt(len(v)):.1f}"


def rng(v):
    return f"{min(v)}" if min(v) == max(v) else f"{min(v)}--{max(v)}"


def put(label, cells):
    global s
    i = s.index(label)
    j = s.index("\\\\", i)
    s = s[:i] + label + " & ".join(cells) + " " + s[j:]


for key, label in [("wide", "MLP 3$\\times$256 tanh & "), ("subset", "MLP 2$\\times$64, 70\\% of flights & "),
                   ("gbt", "Boosted trees & "), ("mismatch", "Default + mismatch & "),
                   ("heldout", "Cross-fitted$^\\ast$ & ")]:
    if key not in P:
        continue
    r = P[key]["rows"]
    put(label, [m(r["unfixed"], False), m(r["add"], False), m(r["fact"]), m(r["oracle"], False)])

B, S = P["prop50j8"], P["sweep_sev35_n8_l4"]
for key, label in [("clean", "Crash, as simulated & "), ("noise1", "State noise $\\times$1 & "), ("noise3", "\\quad $\\times$3 & "),
                   ("first75", "First 75\\% of the run & "), ("first50", "First 50\\% & "),
                   ("sup_m0.1", "Supervisor, margin 0.1~m & "), ("sup_m0.5", "\\quad 0.5~m & "), ("sup_m1.0", "\\quad 1.0~m & "),
                   ("sup_late", "\\quad 0.3~s reaction & "), ("combined", "\\quad 0.5~m + noise $\\times$1 & ")]:
    cells = []
    for D in (B, S):
        rr = D.get(key, [])
        cells += [m([x["crash"] for x in rr]), rng([x["bad_removed"] for x in rr])] if rr else ["--", "--"]
    put(label, cells)
open(tex, "w").write(s)
print("filled")
