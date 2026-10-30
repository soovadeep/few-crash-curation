"""Rows of Table tab:local (within-flight fault) from local_*.json (local_drone.py), localdiag_*.json (diag_local.py)
and localiter(w)_*.json (iter_local.py, flown and --nofly). Prints the LaTeX rows; with a .tex path, writes them
between the table's \\midrule and \\bottomrule.

usage (from code/drone/): python work/fill_local.py [../../paper/ras/main.tex]
"""
import glob
import json
import sys

import numpy as np

TAG = "sev50_n11_on25-50"
TASKS = ("test dives", "steep dives", "shallow dives")


def load(pat):
    return [json.load(open(f)) for f in sorted(glob.glob(f"work/{pat}_{TAG}_s?.json"))]


def crash(r):
    return 100 * np.mean([r[t] for t in TASKS])


def ms(v):
    return f"{np.mean(v):.1f}$\\pm${np.std(v, ddof=1) / np.sqrt(len(v)):.1f}"


def rng(v):
    lo, hi = round(min(v)), round(max(v))
    return f"{lo}" if lo == hi else f"{lo}--{hi}"


L, D = load("local"), load("localdiag")
I, W = load("localiter"), load("localiterw")
N = [json.load(open(f)) for f in sorted(glob.glob(f"work/localiter_{TAG}_s?_nofly.json"))]
assert len(L) == len(D) == len(I) == len(W) == len(N) == 5


def rem(rr):
    f = [100 * r["faulty_descent_removed"] for r in rr]
    c = [100 * r["correct_descent_removed"] for r in rr]
    k = [100 * r["kept"] for r in rr]
    return f"{rng(f)} / {rng(c)} & {rng(k)}"


rows = [("Unfixed", "--", ms([crash(d["flown"]["Unfixed"]) for d in L]), "-- & 100")]
for name, key, src in [("\\method{}, flights", "FACT flights", L), ("\\method{}, 2-s windows", "FACT windows", L)]:
    rows.append((name, "1", ms([crash(d["flown"][key]) for d in src]), rem([d["removal"][key] for d in src])))
    if key == "FACT flights":
        rows.append((name, "2", ms([crash(d["rounds"][1]["flown"]) for d in I]), rem([d["rounds"][1] for d in N])))
rows.append(("Oracle, flights", "--", ms([crash(d["flown"]["Oracle flights"]) for d in L]),
             rem([d["removal"]["Oracle flights"] for d in L])))
rows.append(("Oracle, 2-s windows", "--", ms([crash(d["flown"]["Oracle windows"]) for d in D]),
             rem([d["removal"]["Oracle windows"] for d in D])))
rows.append(("Oracle, samples", "--", ms([crash(d["flown"]["Oracle samples"]) for d in L]),
             rem([d["removal"]["Oracle samples"] for d in L])))
body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
print(body)
print("% text: windows 2 failures", ms([crash(d["rounds"][1]["flown"]) for d in W]),
      "| samples 1 failure", ms([crash(d["flown"]["FACT samples"]) for d in L]), rem([d["removal"]["FACT samples"] for d in L]),
      "| flights per seed", [round(crash(d["flown"]["FACT flights"]), 1) for d in L],
      "2 failures per seed", [round(crash(d["rounds"][1]["flown"]), 1) for d in I],
      "| bad found round 1", [d["rounds"][0]["bad_removed"] for d in N], "removed", [d["rounds"][0]["removed"] for d in N],
      "round 2 removed", [d["rounds"][1]["removed"] for d in N],
      "| window AUROC", rng([100 * d["detection"]["window_auroc_descent"] for d in L]))
if len(sys.argv) > 1:
    tex = open(sys.argv[1]).read()
    i = tex.index("\\label{tab:local}")
    a = tex.index("\\midrule\n", i) + len("\\midrule\n")
    b = tex.index("\\bottomrule", a)
    open(sys.argv[1], "w").write(tex[:a] + body + "\n" + tex[b:])
    print("filled")
