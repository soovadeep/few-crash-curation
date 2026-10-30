"""Recompute the numbers in the paper's tables, figures and text from the result files in results/, and compare them with
paper/ras/main.tex and paper/make_figs.py. Prints every cell it checks and a list of mismatches.

usage (from the repository root): python paper/ras/check_numbers.py

Crash and reach are the mean over the three test sets of the share of runs that crash / reach the goal, in %; the
paper reports the mean over seeds and its standard error (sample s.d. / sqrt(seeds)), crash to 0.1 and reach to 1.
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "results")
TEX = open(os.path.join(ROOT, "paper", "ras", "main.tex")).read()
BAD = []


def stats(crash, reach):
    c = np.array(crash, float)
    se = c.std(ddof=1) / np.sqrt(len(c)) if len(c) > 1 else float("nan")
    return dict(n=len(c), crash=c.mean(), se=se, reach=np.mean(reach) if reach else None, n_reach=len(reach))


# ---------------------------------------------------------------- quadrotor (NeuroBEM plant), results/drone
DT = ("test dives", "steep dives", "shallow dives")


def drone_rows(kind, bench, key="rows", name="model"):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, "drone", f"{kind}_{bench}_s[0-9].json"))):
        for r in json.load(open(f))[key]:
            out.setdefault(r[name], []).append(r)
    return out


def drone_stat(rs):
    crash = [100 * np.mean([r[t] for t in DT]) for r in rs]
    reach = [100 * np.mean([r[t + " reach"] for t in DT]) for r in rs if all(t + " reach" in r for t in DT)]
    return stats(crash, reach)


def drone(bench):
    base, mis, sup = drone_rows("base", bench), drone_rows("mis", bench), drone_rows("sup", bench)
    src = {
        "none": (mis, "faulty"), "oracle": (mis, "clean (bad flights removed)"),
        "huber": (base, "huber loss"), "l1": (base, "l1 loss"), "worst": (base, "remove 40 worst-fit flights"),
        "crashlike": (base, "remove 40 most crash-like flights (most axial descent)"),
        "dropdesc": (base, "remove all axial-descent samples"),
        "add1": (mis, "add 1 crash dives, W=20"), "add3": (mis, "add 3 crash dives, W=20"),
        "add24": (mis, "add 24 crash dives, W=20"),
        "sign1": (mis, "ours (sign rule) from 1 dives"), "sign3": (mis, "ours (sign rule) from 3 dives"),
        "sign1add": (mis, "ours (sign rule) from 1 dives, and add them (W=20)"),
        "sign3add": (mis, "ours (sign rule) from 3 dives, and add them (W=20)"),
        "sup": (sup, "FACT from 1 supervised run with takeover, sign rule"),
        "sup_before": (sup, "FACT from 1 supervised run before takeover, sign rule"),
        "topk": (mis, "ours from 1 dives, remove 40"),
    }
    out = {k: drone_stat(s[n]) for k, (s, n) in src.items() if n in s}
    # mis_*.json has no oracle reach for seeds 1-2; reach_oracle_*.json has it for all five seeds, per dive
    rf = sorted(glob.glob(os.path.join(RES, "drone", f"reach_oracle_{bench}_s[0-9].json")))
    if rf and "oracle" in out:
        out["oracle"]["reach"] = np.mean([100 * np.mean([r["reach"] for t in DT for r in json.load(open(f))["runs"][t]]) for f in rf])
        out["oracle"]["n_reach"] = len(rf)
    cup = {}
    for f in sorted(glob.glob(os.path.join(RES, "drone", f"cupid_{bench}_s[0-9].json"))):
        for r in json.load(open(f))["removal"]:
            cup.setdefault((r["method"], r["k"]), []).append(r)
    for tag, m in [("cupid_sq", "CUPID-sq sum"), ("cupid_pg", "CUPID-PG sum")]:
        if (m, 40) in cup:
            rs = cup[(m, 40)]
            # 'crash'/'reach' are the test dives; steep and shallow dives are stored by name
            out[tag] = stats([100 * np.mean([r["crash"], r["steep dives"], r["shallow dives"]]) for r in rs],
                             [100 * np.mean([r["reach"], r["steep dives reach"], r["shallow dives reach"]]) for r in rs])
    rev = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "drone", f"controls_{bench}_s[0-9].json")))]
    for v in sorted({k for d in rev for k, x in d["flown"].items() if isinstance(x, dict)}):
        out["controls " + v] = drone_stat([d["flown"][v] for d in rev if isinstance(d["flown"].get(v), dict)])
    return out


# ---------------------------------------------------------------- MuJoCo car, results/mujoco
CT = ("test", "late entry", "ice-edge entry")


def car():
    rows = {}
    for kind in ("major", "crude", "out_major"):
        for f in sorted(glob.glob(os.path.join(RES, "mujoco", f"{kind}_V2.5_dry300_m100_s[0-9].json"))):
            for r in json.load(open(f))["rows"]:
                rows.setdefault(r["model"], []).append(r)
    out = {}
    for k, rs in rows.items():
        s = stats([100 * np.mean([r[t]["crash"] for t in CT]) for r in rs], [])
        for c in ("removed", "bad_removed", "lookalike_removed"):
            if c in rs[0]:
                s[c] = [r[c] for r in rs]
        out[k] = s
    return out


# ---------------------------------------------------------------- Crazyflie (NanoBench), results/crazyflie
def crazyflie(seeds=(1, 2, 3, 4, 5), split=False):
    per = {}
    for s in seeds:
        # the held-out-plant runs reuse the row names but are a different experiment, stored as cfsplit_*.json
        pat = f"cfsplit_3.1_eq_s{s}*.json" if split else f"cf_3.1_eq_s{s}_*.json"
        for f in sorted(glob.glob(os.path.join(RES, "crazyflie", pat))):
            d = json.load(open(f))
            assert d.get("split", False) == split, f"{f}: wrong experiment"
            for r in d["rows"]:
                m = re.sub(r" \(draw \d\)", "", r["model"])
                per.setdefault(m, {}).setdefault(s, []).append(
                    (100 * np.mean([r[t] for t in DT]), 100 * np.mean([r[t + " reach"] for t in DT])))
    out = {}
    for m, d in per.items():
        out[m] = stats([np.mean([x[0] for x in d[s]]) for s in sorted(d)], [x[1] for s in sorted(d) for x in d[s]])
    return out


# ---------------------------------------------------------------- reading the paper
def table(label):
    """Data rows of the tabular after \\label{label}: list of (first cell, second cell, remaining cells)."""
    i = TEX.index("\\label{%s}" % label)
    body = TEX[TEX.index("\\begin{tabular}", i):TEX.index("\\end{tabular}", i)]
    body = body.split("\\midrule", 1)[1]
    out, last = [], ""
    for line in body.split("\n"):
        line = line.strip()
        if not line.endswith("\\\\") or re.match(r"\\(top|mid|bottom|c)rule", line):
            continue
        cells = [c.strip() for c in line[:-2].split("&")]
        last = cells[0] or last
        out.append((last, cells[1], cells[2:]))
    return out


def num(cell):
    """'\\pmse{3.1}{1.2}' -> (3.1, 1.2); '61$^\\dagger$' -> (61, None); '--' -> None."""
    cell = re.sub(r"\\textbf\{(.*)\}", r"\1", cell)
    m = re.search(r"\\pmse\{([\d.]+)\}\{([\d.]+)\}", cell)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.match(r"\$?([\d.]+)", cell.replace("{,}", ""))
    return (float(m.group(1)), None) if m else None


def check(where, paper, s, reach=True, digits=1):
    """paper: (crash, se) and reach cells; s: recomputed stats."""
    pc, pr = paper
    ok = True
    msg = []
    if pc is None:
        ok = s is None
        msg.append("paper '--'" + ("" if ok else f", data {s['crash']:.1f}"))
    elif s is None:
        ok = False
        msg.append(f"paper {pc}, NO DATA")
    else:
        c, se = round(s["crash"], digits), round(s["se"], 1) if s["n"] > 1 else 0.0
        ok &= abs(pc[0] - c) < 0.51 * 10 ** -digits
        if pc[1] is not None:
            ok &= abs(pc[1] - se) < 0.051
        msg.append(f"crash paper {pc[0]}+-{pc[1]} data {c}+-{se} (n={s['n']})")
        if reach and pr is not None:
            r = s["reach"]
            ok &= r is not None and abs(pr[0] - round(r)) < 0.51
            msg.append(f"reach paper {pr[0]:.0f} data {'-' if r is None else round(r)} (n={s['n_reach']})")
    print(("  ok   " if ok else "  DIFF ") + where + ": " + "; ".join(msg))
    if not ok:
        BAD.append(where)


BENCH = [("prop50j8", "Propeller"), ("gain40j8", "Thrust gain"), ("prop50j8m", "Look-alike")]
D = {b: drone(b) for b, _ in BENCH}

print("== Table tab:major")
MAJOR_ROWS = {("None (all logs)", "0"): "none", ("Oracle (bad logs removed)", "--"): "oracle", ("Huber loss", "0"): "huber",
      ("L1 loss", "0"): "l1", ("Remove $k$ worst-fit logs", "0"): "worst", ("Drop $k$ most crash-like logs", "0$^\\S$"): "crashlike",
      ("Drop all descent data", "0$^\\S$"): "dropdesc", ("CUPID, label-free surrogate", "100"): "cupid_sq",
      ("CUPID, REINFORCE", "100"): "cupid_pg", ("Add crash dives, $W=20$", "1"): "add1", ("Add crash dives, $W=20$", "3"): "add3",
      ("Add crash dives, $W=20$", "24"): "add24", ("\\textbf{\\method{}}, sign rule", "1"): "sign1",
      ("\\textbf{\\method{}}, sign rule", "3"): "sign3", ("\\method{}, sign rule, then add, $W=20$", "1"): "sign1add",
      ("\\method{}, supervised run, sign rule", "1$^\\P$"): "sup", ("\\method{}, remove top $k$ (count known)", "1"): "topk"}
for lab, fail, cells in table("tab:major"):
    key = MAJOR_ROWS.get((lab, fail))
    if key is None:
        print(f"  ??   no mapping for row {lab!r} {fail!r}"); BAD.append(lab); continue
    for j, (b, bn) in enumerate(BENCH):
        c, r = num(cells[2 * j]), num(cells[2 * j + 1])
        check(f"tab:major {key} {bn}", (c, r), D[b].get(key))
        if r is not None and "dagger" in cells[2 * j + 1] and D[b].get(key) and D[b][key]["n_reach"] != 3:
            print(f"  DIFF tab:major {key} {bn}: reach marked 3 of 5 seeds but has {D[b][key]['n_reach']}"); BAD.append(key)

print("== Table tab:reweight")
REWEIGHT_ROWS = {"\\method{}, sign rule, fixed epochs": "sign1", "\\quad unnormalised score, fixed epochs": "controls unnorm",
      "\\quad 2,640 updates": None, "As many random flights, fixed epochs": "controls random", "Add, $W=20$, fixed epochs": "add1",
      "Add, $W=200$": "controls add_w200", "Add, $W=2{,}000$": "controls add_w2000",
      "Add, balanced batches ($W=4{,}784$)": "controls add_5050", "Add, $W=20{,}000$": "controls add_w20000",
      "Fine-tune, half crash dive, half replay": "controls ft_replay", "Fine-tune on the crash dive only": "controls ft_only",
      "Learning to reweight, crash dive as clean set": "controls l2rw"}
prev = None
for lab, c0, cells in table("tab:reweight"):
    cells = [c0] + cells  # no Fail. column here
    key = REWEIGHT_ROWS.get(lab, "?")
    if lab == "\\quad 2,640 updates":
        key = "controls fact_matched" if prev == "sign1" or prev == "controls unnorm" else "controls random_matched"
    prev = key if key != "controls fact_matched" else prev
    if key == "?":
        print(f"  ??   no mapping for row {lab!r}"); BAD.append(lab); continue
    for j, (b, bn) in enumerate(BENCH):
        c, r = num(cells[2 * j]), num(cells[2 * j + 1])
        check(f"tab:reweight {key} {bn}", (c, r), D[b].get(key))

print("== Table tab:mj")
C = car()
MJ_ROWS = {("None (all logs)", "0"): "faulty", ("Oracle", "--"): "clean (bad logs removed)", ("Huber loss", "0"): "huber loss",
      ("L1 loss", "0"): "l1 loss", ("Remove 360 worst-fit", "0"): "remove 360 worst-fit logs",
      ("Drop 360 most crash-like", "0$^\\S$"): "drop 360 most crash-like logs",
      ("Drop all fast data on ice", "0$^\\S$"): "drop all fast data on the ice",
      ("Add crash runs, $W=20$", "1"): "add 1 runs, W=20", ("Add crash runs, $W=20$", "3"): "add 3 runs, W=20",
      ("\\textbf{\\method{}}, sign count$^\\dagger$", "1"): "ours (sign-rule count) from 1 runs",
      ("\\textbf{\\method{}}, sign count$^\\dagger$", "3"): "ours (sign-rule count) from 3 runs",
      ("Gradient-free, sign count$^\\dagger$", "1"): "output-space (sign-rule count) from 1 runs",
      ("\\method{}, remove 360", "3"): "ours from 3 runs, remove 360",
      ("Gradient-free, remove 360", "3"): "output-space from 3 runs, remove 360"}
for lab, fail, cells in table("tab:mj"):
    key = MJ_ROWS.get((lab, fail))
    if key is None:
        print(f"  ??   no mapping for row {lab!r} {fail!r}"); BAD.append(lab); continue
    s = C.get(key)
    check(f"tab:mj {key}", (num(cells[0]), None), s, reach=False)
    if s and "bad_removed" in s:
        rng = lambda v: f"{min(v)}" if min(v) == max(v) else f"{min(v)}--{max(v)}"
        data = f"{rng(s['bad_removed'])} / {rng(s['lookalike_removed'])}"
        paper = re.sub(r"\$\^\\ast\$", "", cells[1]).strip()
        ok = paper == data
        print(("  ok   " if ok else "  DIFF ") + f"tab:mj {key} removed: paper '{paper}' data '{data}' (logs removed {s['removed']})")
        if not ok:
            BAD.append(key + " removed")

print("== Table tab:cf")
F = crazyflie()
CF_ROWS = {("None (all flights)", "0"): "faulty (all flights)", ("Oracle (low voltage)", "--"): "oracle (flights below 3.35 V)",
      ("Huber loss", "0"): "huber loss", ("L1 loss", "0"): "l1 loss", ("Remove 76 worst-fit", "0"): "remove 76 worst-fit flights",
      ("Remove 76 random", "0"): "remove 76 random flights",
      ("Drop 76 most crash-like", "0$^\\S$"): "remove 76 most crash-like flights (most descent)",
      ("Add crash dives", "1"): "add 1 crash dives, W=20", ("Add crash dives", "3"): "add 3 crash dives, W=20",
      ("Learning to reweight", "1"): "learning to reweight, 1 crash dives as clean set",
      ("Learning to reweight", "3"): "learning to reweight, 3 crash dives as clean set",
      ("Learning to reweight + add", "1"): "learning to reweight, 1 crash dives as clean set, and add them (W=20)",
      ("Learning to reweight + add", "3"): "learning to reweight, 3 crash dives as clean set, and add them (W=20)",
      ("\\method{}, sign rule", "1"): "ours from 1 dives, sign rule", ("\\method{}, sign rule", "3"): "ours from 3 dives, sign rule",
      ("\\method{}, sign rule + add", "1"): "ours from 1 dives, sign rule, and add them (W=20)",
      ("\\method{}, sign rule + add", "3"): "ours from 3 dives, sign rule, and add them (W=20)",
      ("Gradient-free, sign + add", "3"): "output-space from 3 dives, sign rule, and add them (W=20)"}
for lab, fail, cells in table("tab:cf"):
    key = CF_ROWS.get((lab, fail))
    if key is None:
        print(f"  ??   no mapping for row {lab!r} {fail!r}"); BAD.append(lab); continue
    check(f"tab:cf {key}", (num(cells[0]), num(cells[1])), F.get(key))

print("== Figure fig:nfail (paper/make_figs.py PANELS)")
sys.path.insert(0, os.path.join(ROOT, "paper"))
src = open(os.path.join(ROOT, "paper", "make_figs.py")).read()
panels = eval(src[src.index("PANELS = [") + len("PANELS = "):src.index("\n]\n", src.index("PANELS = [")) + 2])
cfmap = dict(add=["add 1 crash dives, W=20", "add 3 crash dives, W=20"],
             cur=["ours from 1 dives, sign rule", "ours from 3 dives, sign rule"],
             both=["ours from 1 dives, sign rule, and add them (W=20)", "ours from 3 dives, sign rule, and add them (W=20)"],
             none="faulty (all flights)", oracle="oracle (flights below 3.35 V)")
drmap = dict(add=["add1", "add3"], cur=["sign1", "sign3"], both=["sign1add", "sign3add"], none="none", oracle="oracle")
for (title, d), bench in zip(panels, ["prop50j8", "gain40j8", "prop50j8m", None]):
    get = (lambda k: D[bench].get(k)) if bench else (lambda k: F.get(k))
    mp = drmap if bench else cfmap
    for k in ("add", "cur", "both"):
        for i, v in enumerate(d[k]):
            check(f"fig:nfail {title} {k} {[1, 3][i]}", ((v[0], v[1]) if v else None, None), get(mp[k][i]), reach=False)
    for k in ("none", "oracle"):
        check(f"fig:nfail {title} {k}", ((d[k], None), None), get(mp[k]), reach=False)

print("== Crazyflie text (Section sec:cf), 5 seeds")
m = re.search(r"Every fix without failures crashes in (\d+)--(\d+)\\% of dives, against ([\d.]+)\\% for the low-battery flights alone.*?"
              r"from one crash or three \((\d+)--(\d+)\\%\).*?\(AUROC ([\d.]+)--([\d.]+)\)\. Learning to reweight does better from three crashes "
              r"\(([\d.]+)\\%\)\. Curating with three crashes and then adding them gives the lowest rate, ([\d.]+)\\%, .*?(?:against|compares with) ([\d.]+)\\% for adding them "
              r"alone or after reweighting\..*?whose worst seed crashes in (\d+)\\% against (\d+)\\%", TEX, re.S)
g = [float(x) for x in m.groups()]
ff = [F[k]["crash"] for k in ("huber loss", "l1 loss", "remove 76 worst-fit flights", "remove 76 random flights",
                               "remove 76 most crash-like flights (most descent)")]
cur = [F[k]["crash"] for k in ("ours from 1 dives, sign rule", "ours from 3 dives, sign rule")]
def worst(k):
    return max(np.mean([100 * np.mean([r[t] for t in DT])]) for f in glob.glob(os.path.join(RES, "crazyflie", "cf_3.1_eq_s[0-9]_*.json"))
               for r in json.load(open(f))["rows"] if r["model"] == k)
pairs = [("failure-free low", g[0], round(min(ff))), ("failure-free high", g[1], round(max(ff))),
         ("oracle", g[2], round(F["oracle (flights below 3.35 V)"]["crash"], 1)),
         ("curation alone low", g[3], round(min(cur))), ("curation alone high", g[4], round(max(cur))),
         ("L2RW 3", g[7], round(F["learning to reweight, 3 crash dives as clean set"]["crash"], 1)),
         ("FACT 3 + add", g[8], round(F["ours from 3 dives, sign rule, and add them (W=20)"]["crash"], 1)),
         ("add 3", g[9], round(F["add 3 crash dives, W=20"]["crash"], 1)),
         ("L2RW 3 + add", g[9], round(F["learning to reweight, 3 crash dives as clean set, and add them (W=20)"]["crash"], 1)),
         ("L2RW 3 + add worst seed", g[10], round(worst("learning to reweight, 3 crash dives as clean set, and add them (W=20)"))),
         ("FACT 3 + add worst seed", g[11], round(worst("ours from 3 dives, sign rule, and add them (W=20)")))]
for where, p, d in pairs:
    ok = abs(p - d) < 0.051
    print(("  ok   " if ok else "  DIFF ") + f"CF text {where}: paper {p:g} data {d:g}")
    if not ok:
        BAD.append("CF text " + where)


def per_seed(k):
    d = {}
    for f in glob.glob(os.path.join(RES, "crazyflie", "cf_3.1_eq_s[0-9]_*.json")):
        s = int(re.search(r"_s(\d)_", f).group(1))
        for r in json.load(open(f))["rows"]:
            if re.sub(r" \(draw \d\)", "", r["model"]) == k:
                d.setdefault(s, []).append(100 * np.mean([r[t] for t in DT]))
    return np.array([np.mean(d[s]) for s in sorted(d)])


m = re.search(r"paired over seeds it is \$(-?[\d.]+)\\pm([\d.]+)\$ points against adding alone \(lower on (\w+) seeds, tied on (\w+)\) "
              r"and \$(-?[\d.]+)\\pm([\d.]+)\$ against reweighting", TEX)
A = per_seed("ours from 3 dives, sign rule, and add them (W=20)")
words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
for i, k in enumerate(["add 3 crash dives, W=20", "learning to reweight, 3 crash dives as clean set, and add them (W=20)"]):
    dd = A - per_seed(k)
    got = (round(dd.mean(), 1), round(dd.std(ddof=1) / np.sqrt(len(dd)), 1))
    paper = (float(m.group(1 + 4 * i)), float(m.group(2 + 4 * i)))
    ok = got == paper and (i or (words[m.group(3)], words[m.group(4)]) == ((dd < 0).sum(), (dd == 0).sum()))
    print(("  ok   " if ok else "  DIFF ") + f"CF paired difference vs {k}: paper {paper} data {got}")
    if not ok:
        BAD.append("CF paired " + k)

print("== Crazyflie with the plant fit on held-out flights (text of Section sec:cf: 26% against 36%; oracle 52%)")
FS = crazyflie(split=True)
m = re.search(r"it is again lowest \((\d+)\\% against (\d+)\\%; oracle (\d+)\\%\)", TEX)
for val, key in zip(m.groups(), ["ours from 3 dives, sign rule, and add them (W=20)", "add 3 crash dives, W=20",
                                 "oracle (flights below 3.35 V)"]):
    check(f"text held-out {key}", ((float(val), None), None), FS.get(key), reach=False, digits=0)

print("== Supervised run before takeover (text of Section sec:sup)")
m = re.search(r"so that part alone finds only (\d+)--(\d+) of the 32 bad flights and crashes in \\pmse\{([\d.]+)\}\{([\d.]+)\}", TEX)
check("text sup before takeover", ((float(m.group(3)), float(m.group(4))), None), D["prop50j8"].get("sup_before"), reach=False)
bad = [r["bad_removed"] for r in drone_rows("sup", "prop50j8")["FACT from 1 supervised run before takeover, sign rule"]]
ok = (min(bad), max(bad)) == (int(m.group(1)), int(m.group(2)))
print(("  ok   " if ok else "  DIFF ") + f"text sup before takeover, bad flights found: paper {m.group(1)}--{m.group(2)} data {min(bad)}--{max(bad)}")
if not ok:
    BAD.append("sup before takeover found")

print("== Per-log aggregation check (text after Proposition 5), results/drone/agg_*_s1.json")
A = [json.load(open(os.path.join(RES, "drone", f"agg_{b}_s1.json"))) for b, _ in BENCH]
grp = [[f for f in a["flights"] if f["bad"] == bad] for a in A for bad in (True, False)]
med = lambda k: [float(np.median([f[k] for f in g])) for g in grp]
got = dict(sample=[100 * a[k] for a in A for k in ("sample_err_bad", "sample_err_good")],
           window=[100 * a[k] for a in A for k in ("window_err_bad", "window_err_good")],
           wrong_good=[sum(f["mean"] >= 0 for f in a["flights"] if not f["bad"]) for a in A],
           n_good=[sum(not f["bad"] for f in a["flights"]) for a in A],
           tau_s=[t / 100 for t in med("tau")], neff=med("n_eff"),
           bound_ok=[sum(f["bound"] < 0.01 for f in a["flights"] if f["bad"]) for a in A],
           n_bad=[sum(f["bad"] for f in a["flights"]) for a in A], bad_wrong=[sum(f["mean"] <= 0 for f in a["flights"] if f["bad"]) for a in A])
m = re.search(r"([\d.]+)--([\d.]+)\\% of samples and ([\d.]+)--([\d.]+)\\% of 2-s windows score with the wrong sign, but no bad flight "
              r"and ([\d.]+)--([\d.]+) of the (\d+) correct ones\. Scores stay correlated for ([\d.]+)--([\d.]+)~s, .*? a flight "
              r"([\d.]+)--([\d.]+)\..*?below 1\\% for ([\d.]+)--([\d.]+) of the (\d+) bad flights", TEX, re.S)
p = [float(x) for x in m.groups()]
for where, (lo, hi), vals, tol in [("samples wrong sign (%)", p[0:2], got["sample"], 0.5), ("windows wrong sign (%)", p[2:4], got["window"], 0.5),
                                   ("correct flights wrong sign", p[4:6], got["wrong_good"], 0), ("correlation time (s)", p[7:9], got["tau_s"], 0.05),
                                   ("independent samples per flight", p[9:11], got["neff"], 3), ("bad flights with bound < 1%", p[11:13], got["bound_ok"], 0)]:
    ok = abs(min(vals) - lo) <= tol and abs(max(vals) - hi) <= tol
    print(("  ok   " if ok else "  DIFF ") + f"agg {where}: paper {lo:g}--{hi:g} data {min(vals):.2f}--{max(vals):.2f}")
    if not ok:
        BAD.append("agg " + where)
for where, ok in [("63 correct, 32 bad flights", set(got["n_good"]) == {int(p[6])} and set(got["n_bad"]) == {int(p[13])}),
                  ("no bad flight with the wrong sign", sum(got["bad_wrong"]) == 0)]:
    print(("  ok   " if ok else "  DIFF ") + "agg " + where)
    if not ok:
        BAD.append("agg " + where)

print("== Figure fig:scores text (Section sec:scores), results/drone/fig_scores_prop50j8m_s1.json")
Z = json.load(open(os.path.join(RES, "drone", "fig_scores_prop50j8m_s1.json")))
zb, zl = np.array(Z["bad"]), np.array(Z["lookalike"])
tl, tr = np.array(Z["train_loss"]), np.array(Z["tracin"])
o = np.argsort(-tl)
got = [int(zb[o[:40]].sum()), int(zl[o[:40]].sum()), int(np.where(zb[o])[0].max() + 1),
       int((tr[zb] > 0).sum()), int((tr[zl] > 0).sum()), int((tr[~zb & ~zl] > 0).sum()), int((tr > 0).sum())]
m = re.search(r"top 40 holds (\d+) of the 32 faulty flights and (\d+) of the 16 look-alikes, and the last faulty flight ranks "
              r"(\d+)nd of 95\. .*? positive for all (\d+) faulty flights, no look-alike and (\w+) other flights, so the sign rule removes (\d+)", TEX, re.S)
words = {"six": 6}
p = [int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), 0, words.get(m.group(5), m.group(5)), int(m.group(6))]
ok = p == got
print(("  ok   " if ok else "  DIFF ") + f"fig:scores text: paper {p} data {got}")
if not ok:
    BAD.append("fig:scores text")

print("== Timing (Section sec:reweight cost paragraph and Appendix app:protocol), results/drone/timing_l2rw.json")
tm = json.load(open(os.path.join(RES, "drone", "timing_l2rw.json")))
U, per = tm["updates_per_training"], tm["s_per_update"]
m = re.search(r"On one CPU core an ordinary training takes (\d+)~s\. Learning to reweight as we ran it, with per-sample gradients, "
              r"takes (\d+)~s; .*?it would take an estimated (\d+)--(\d+)~s .*?\\method\{\} scores "
              r"every sample in ([\d.]+)~s and retrains on the kept data in ([\d.]+)~s", TEX, re.S)
m2 = re.search(r"\\textbf\{Timing\.\}.*?([\d,]+) samples, an MLP with ([\d,]+) parameters.*?An ordinary update takes ([\d.]+)~ms \((\d+)~s per "
               r"training; a complete run of the standard routine on all data takes (\d+)~s\).*?: (\d+)~ms per update with per-sample gradients.*?; (\d+)~s per "
               r"training\), ([\d.]+)~ms with the look-ahead.*?\((\d+)~s\), and ([\d.]+)~ms with one forward-mode.*?\((\d+)~s\)\..*?"
               r"scoring takes (\d+)~s with per-sample gradients, or ([\d.]+)~s using.*?on the (\d+)\\% of samples kept takes ([\d.]+)~s", TEX, re.S)
p1 = [float(x) for x in m.groups()]
p2 = [float(x.replace(",", "")) for x in m2.groups()]
data = [(p1[0], per["plain"] * U, 0), (p1[1], per["l2rw_per_sample"] * U, -1), (p1[2], per["l2rw_jvp"] * U, 0),
        (p1[3], per["l2rw_lookahead"] * U, 0), (p1[4], tm["fact_scoring_fast_s"], 1), (p1[5], tm["fact_retrain_s"], 1),
        (p2[0], tm["samples"], 0), (p2[1], tm["params"], 0), (p2[2], 1e3 * per["plain"], 1), (p2[3], per["plain"] * U, 0),
        (p2[4], tm["full_fit_s"], 0), (p2[5], 1e3 * per["l2rw_per_sample"], -1), (p2[6], per["l2rw_per_sample"] * U, -1),
        (p2[7], 1e3 * per["l2rw_lookahead"], 1), (p2[8], per["l2rw_lookahead"] * U, 0), (p2[9], 1e3 * per["l2rw_jvp"], 1),
        (p2[10], per["l2rw_jvp"] * U, 0), (p2[11], tm["fact_scoring_s"], 0), (p2[12], tm["fact_scoring_fast_s"], 1),
        (p2[13], 100 * tm["fact_kept_samples"] / tm["samples"], 0), (p2[14], tm["fact_retrain_s"], 1)]
for i, (paper, d, dig) in enumerate(data):
    ok = paper == round(d, dig)
    print(("  ok   " if ok else "  DIFF ") + f"timing #{i}: paper {paper:g} data {round(d, dig):g}")
    if not ok:
        BAD.append(f"timing #{i}")
m = re.search(r"so it costs (\d+)--(\d+)~s \(\\ref\{app:protocol\}", TEX)
m3 = re.search(r"thus costs (\d+)~s with its default schedule and (\d+)~s with matched updates", TEX)
d = (round(tm["fact_scoring_fast_s"] + tm["fact_retrain_s"]), round(tm["fact_scoring_fast_s"] + per["plain"] * U))
for where, mm in (("sec:reweight", m), ("app:protocol", m3)):
    ok = mm is not None and (int(mm.group(1)), int(mm.group(2))) == d
    print(("  ok   " if ok else "  DIFF ") + f"FACT total cost {where}: paper {mm and mm.groups()} data {d}")
    if not ok:
        BAD.append("FACT total cost " + where)
r = [per[k] * U / d[1] for k in ("l2rw_jvp", "l2rw_lookahead", "l2rw_per_sample")]
ok = min(r) > 2 and "needs substantially more computation in our implementation" in TEX
print(("  ok   " if ok else "  DIFF ") + f"'substantially more computation' (sec:reweight): reweighting / FACT (matched) = {r[0]:.1f}, {r[1]:.1f} "
      f"(efficient, estimated), {r[2]:.0f} (as run)")
if not ok:
    BAD.append("timing ratio")

m = re.search(r"reaching the goal about as often \((\d+)--(\d+)\\% against (\d+)--(\d+)\\%\)", TEX)
for (lo, hi), key in (((m.group(1), m.group(2)), "sign1"), ((m.group(3), m.group(4)), "oracle")):
    v = [round(D[b][key]["reach"]) for b, _ in BENCH]
    ok = (int(lo), int(hi)) == (min(v), max(v))
    print(("  ok   " if ok else "  DIFF ") + f"sec:main reach range {key}: paper {lo}--{hi} data {v}")
    if not ok:
        BAD.append("sec:main reach " + key)

m = re.search(r"CUPID from 100 rollouts, told to remove 40, crashes in (\d+)--(\d+)\\% of dives against ([\d.]+)--([\d.]+)\\%", TEX)
cv = [D[b][k]["crash"] for b, _ in BENCH for k in ("cupid_sq", "cupid_pg") if k in D[b]]
fv = [D[b]["sign1"]["crash"] for b, _ in BENCH]
got = (round(min(cv)), round(max(cv)), round(min(fv), 1), round(max(fv), 1))
ok = tuple(float(x) for x in m.groups()) == got
print(("  ok   " if ok else "  DIFF ") + f"Related work CUPID vs FACT: paper {m.groups()} data {got}")
if not ok:
    BAD.append("Related work CUPID")

print("== Safe completion and final distance (Section sec:reweight and Appendix app:protocol), results/drone/reach_*_prop50j8_s*.json")
RE, NUMW = {}, {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
for mdl in ("fact", "oracle", "l2rw"):
    runs = [[r for t in DT for r in json.load(open(os.path.join(RES, "drone", f"reach_{mdl}_prop50j8_s{s}.json")))["runs"][t]]
            for s in range(1, 6)]
    safe = [100 * np.mean([r["safe"] for r in rr]) for rr in runs]
    ok_runs = [r for rr in runs for r in rr if not r["crash"]]
    RE[mdl] = dict(safe=safe, crash=np.mean([100 * np.mean([r["crash"] for r in rr]) for rr in runs]),
                   reach=np.mean([100 * np.mean([r["reach"] for r in rr]) for rr in runs]),
                   final=np.median([r["final_dist"] for r in ok_runs]), closest=np.median([r["min_dist"] for r in ok_runs]))
m = re.search(r"safe completion, reaching the goal without crashing, is \\pmse\{([\d.]+)\}\{([\d.]+)\}, \\pmse\{([\d.]+)\}\{([\d.]+)\} and "
              r"\\pmse\{([\d.]+)\}\{([\d.]+)\}\\%, and over dives without a crash the median final distance to the goal is ([\d.]+), ([\d.]+) "
              r"and ([\d.]+)~m and the median closest approach ([\d.]+), ([\d.]+) and ([\d.]+)~m", TEX)
m2 = re.search(r"it reaches the goal without crashing in (\d+)\\% of dives, against (\d+)\\% for \\method\{\} \(lower on (\w+) of five seeds\) "
               r"and (\d+)\\% for the oracle, and ends nearer the goal \(median ([\d.]+)~m against ([\d.]+) and ([\d.]+)~m\)", TEX)
p, q = [float(x) for x in m.groups()], m2.groups()
se = lambda v: np.std(v, ddof=1) / np.sqrt(len(v))
order = ("fact", "oracle", "l2rw")
data = [(p[2 * i], round(np.mean(RE[k]["safe"]), 1)) for i, k in enumerate(order)] + \
       [(p[2 * i + 1], round(se(RE[k]["safe"]), 1)) for i, k in enumerate(order)] + \
       [(p[6 + i], round(RE[k]["final"], 2)) for i, k in enumerate(order)] + [(p[9 + i], round(RE[k]["closest"], 2)) for i, k in enumerate(order)] + \
       [(float(q[0]), round(np.mean(RE["l2rw"]["safe"]))), (float(q[1]), round(np.mean(RE["fact"]["safe"]))),
        (NUMW[q[2]], int(sum(a < b for a, b in zip(RE["fact"]["safe"], RE["l2rw"]["safe"])))),
        (float(q[3]), round(np.mean(RE["oracle"]["safe"]))), (float(q[4]), round(RE["l2rw"]["final"], 2)),
        (float(q[5]), round(RE["fact"]["final"], 2)), (float(q[6]), round(RE["oracle"]["final"], 2))]
for i, (paper, d) in enumerate(data):
    ok = paper == d
    print(("  ok   " if ok else "  DIFF ") + f"reach re-fly #{i}: paper {paper:g} data {d:g}")
    if not ok:
        BAD.append(f"reach re-fly #{i}")
for k, key in (("fact", "sign1"), ("oracle", "oracle"), ("l2rw", "controls l2rw")):  # the re-flown models are the paper's models
    s = D["prop50j8"].get(key)
    ok = s is not None and abs(RE[k]["crash"] - s["crash"]) < 0.05
    print(("  ok   " if ok else "  DIFF ") + f"re-flown {k}: crash {RE[k]['crash']:.1f} reach {RE[k]['reach']:.0f}; "
          f"stored crash {s and round(s['crash'], 1)} reach {s and s['reach'] and round(s['reach'])}")
    if not ok:
        BAD.append(f"re-flown {k}")

print("\n" +(f"{len(BAD)} MISMATCHES: " + ", ".join(BAD) if BAD else "all checked numbers match the result files"))
