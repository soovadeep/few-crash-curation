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
    # on the propeller fault mis_*.json has no oracle reach for seeds 1-2; reach_oracle_*.json has it for all five seeds
    rf = sorted(glob.glob(os.path.join(RES, "drone", f"reach_oracle_{bench}_s[0-9].json")))
    if rf and "oracle" in out and out["oracle"]["n_reach"] < out["oracle"]["n"]:
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
      ("Gradient-free, sign + add", "3"): "output-space from 3 dives, sign rule, and add them (W=20)",
      ("Oracle + add", "3"): "oracle, and add 3 crash dives (W=20)"}
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
              r"from one crash or three \((\d+)--(\d+)\\%\).*?\(AUROC ([\d.]+)--([\d.]+)\).*?Learning to reweight does better from three crashes "
              r"\(([\d.]+)\\%\)\. Three crashes help mostly as data: adding them alone gives ([\d.]+)\\%, and curating with them first gives "
              r"([\d.]+)\\%, the same as the gradient-free score curated and added the same way \(([\d.]+)\\%\).*?the oracle's correct historical "
              r"flights crash in ([\d.]+)\\%.*?reweighting's worst seed crashes in (\d+)\\% against (\d+)\\%", TEX, re.S)
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
         ("add 3", g[8], round(F["add 3 crash dives, W=20"]["crash"], 1)),
         ("FACT 3 + add", g[9], round(F["ours from 3 dives, sign rule, and add them (W=20)"]["crash"], 1)),
         ("gradient-free 3 + add", g[10], round(F["output-space from 3 dives, sign rule, and add them (W=20)"]["crash"], 1)),
         ("oracle + add 3", g[11], round(F["oracle, and add 3 crash dives (W=20)"]["crash"], 1)),
         ("L2RW 3 + add worst seed", g[12], round(worst("learning to reweight, 3 crash dives as clean set, and add them (W=20)"))),
         ("FACT 3 + add worst seed", g[13], round(worst("ours from 3 dives, sign rule, and add them (W=20)")))]
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


from scipy import stats as _st
m = re.search(r"\\method\{\} plus adding minus adding alone is \$(-?[\d.]+)\$ points \(95\\% CI \$(-?[\d.]+)\$ to \$(-?[\d.]+)\$; "
              r"lower on (\w+) seeds, tied on (\w+)\); minus reweighting and adding, \$(-?[\d.]+)\$ \(\$(-?[\d.]+)\$ to \$(-?[\d.]+)\$;.*?"
              r"minus the gradient-free score, \$([+-]?[\d.]+)\$ \(\$(-?[\d.]+)\$ to \$(-?[\d.]+)\$\); and minus the oracle with the crashes "
              r"added, \$([+-]?[\d.]+)\$ \(\$(-?[\d.]+)\$ to \$(-?[\d.]+)\$\)", TEX, re.S)
A = per_seed("ours from 3 dives, sign rule, and add them (W=20)")
words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
G = [float(x) if i not in (3, 4) else x for i, x in enumerate(m.groups())]
for i, (k, at) in enumerate([("add 3 crash dives, W=20", 0), ("learning to reweight, 3 crash dives as clean set, and add them (W=20)", 5),
                             ("output-space from 3 dives, sign rule, and add them (W=20)", 8), ("oracle, and add 3 crash dives (W=20)", 11)]):
    dd = A - per_seed(k)
    mu, sd = dd.mean(), dd.std(ddof=1) / np.sqrt(len(dd))
    t = _st.t.ppf(0.975, len(dd) - 1)
    got = (round(mu, 1), round(mu - t * sd, 1), round(mu + t * sd, 1))
    paper = (G[at], G[at + 1], G[at + 2])
    ok = got == paper and (i or (words[G[3]], words[G[4]]) == ((dd < 0).sum(), (dd == 0).sum()))
    print(("  ok   " if ok else "  DIFF ") + f"CF paired difference vs {k}: paper {paper} data {got}")
    if not ok:
        BAD.append("CF paired " + k)

print("== Crazyflie with the plant fit on held-out flights (text of Section sec:cf: 26% against 36%; oracle 52%)")
FS = crazyflie(split=True)
m = re.search(r"curating and adding is again below adding alone \((\d+)\\% against (\d+)\\%; oracle (\d+)\\%\)", TEX)
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

print("== Timing (Table tab:safe, Section sec:reweight cost paragraph, Appendix app:protocol), results/drone/timing_e2e.json and timing_l2rw.json")
tm = json.load(open(os.path.join(RES, "drone", "timing_l2rw.json")))
te = json.load(open(os.path.join(RES, "drone", "timing_e2e.json")))["seconds"]
TP = te["prop50j8"]
m = re.search(r"An ordinary training takes (\d+)~s and adding the crash dive (\d+)~s\. \\method\{\} takes (\d+)~s with its default schedule, "
              r"of which ([\d.]+)~s is scoring, and (\d+)~s with as many updates as the full-data model\. Learning to reweight takes (\d+)~s with "
              r"the look-ahead of \\citet\{ren2018l2rw\}, (\d+)~s with one forward-mode product per batch, and (\d+)~s with per-sample gradients", TEX)
p = [float(x) for x in m.groups()]
data = [round(TP["plain"]), round(TP["add"]), round(TP["fact"]), round(tm["fact_scoring_fast_s"], 1), round(TP["fact_matched"]),
        round(TP["l2rw_la"]), round(TP["l2rw_jvp"]), round(TP["l2rw"])]
ok = p == [float(x) for x in data]
print(("  ok   " if ok else "  DIFF ") + f"cost paragraph: paper {p} data {data}")
if not ok:
    BAD.append("cost paragraph")
rng = lambda k: (lambda v: "%d" % v[0] if v[0] == v[1] else "%d--%d" % v)((min(round(te[b][k]) for b, _ in BENCH), max(round(te[b][k]) for b, _ in BENCH)))
m = re.search(r"an ordinary training takes ([\d-]+)~s, \\method\{\} ([\d-]+)~s \(([\d-]+)~s with matched updates\), and learning to reweight "
              r"([\d-]+)~s with the look-ahead and ([\d-]+)~s with forward-mode products; with per-sample gradients it takes (\d+)~s on the "
              r"propeller benchmark\. That benchmark has ([\d,]+) samples; the MLP has ([\d,]+) parameters, and a full-data training takes "
              r"([\d,]+) updates of batch ([\d,]+)\..*?largest relative difference \$2\\times10\^\{-7\}\$.*?scoring takes (\d+)~s with per-sample "
              r"gradients, or ([\d.]+)~s using.*?scores equal to \$2\\times10\^\{-5\}\$\); the sign rule keeps (\d+)\\% of the samples", TEX, re.S)
p = list(m.groups())
data = [rng("plain"), rng("fact"), rng("fact_matched"), rng("l2rw_la"), rng("l2rw_jvp"), "%d" % round(TP["l2rw"]),
        f"{tm['samples']:,}", f"{tm['params']:,}", f"{tm['updates_per_training']:,}", f"{tm['batch']:,}",
        "%d" % round(tm["fact_scoring_s"]), "%.1f" % tm["fact_scoring_fast_s"], "%d" % round(100 * tm["fact_kept_samples"] / tm["samples"])]
ok = p == data and round(max(tm["max_rel_weight_diff"].values()), 7) == 2e-7 and round(tm["fact_scoring_fast_max_abs_diff"], 5) == 2e-5
print(("  ok   " if ok else "  DIFF ") + f"app:protocol timing: paper {p} data {data}")
if not ok:
    BAD.append("app:protocol timing")
r = [TP[k] / TP["fact_matched"] for k in ("l2rw_jvp", "l2rw_la", "l2rw")]
ok = min(r) > 2 and "needs substantially more computation in our implementation" in TEX
print(("  ok   " if ok else "  DIFF ") + f"'substantially more computation' (sec:reweight): reweighting / FACT (matched updates) = "
      + ", ".join(f"{x:.1f}" for x in r))
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

print("== Table tab:safe (Section sec:main), results/drone/reach_*.json and timing_e2e.json")


def reach(tag, bench):
    """Per-dive re-flights of reach_drone.py (reach_{tag}_{bench}_s*.json), pooled over seeds."""
    ds = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "drone", f"reach_{tag}_{bench}_s[0-9].json")))]
    if not ds:
        return None
    runs = [[r for t in DT for r in d["runs"][t]] for d in ds]
    safe = [100 * np.mean([r["safe"] for r in rr]) for rr in runs]
    ok_runs = [r for rr in runs for r in rr if not r["crash"]]
    s = stats([100 * np.mean([r["crash"] for r in rr]) for rr in runs], [])
    s.update(safe=np.mean(safe), safe_se=np.std(safe, ddof=1) / np.sqrt(len(safe)), safe_seeds=safe,
             final=np.median([r["final_dist"] for r in ok_runs]), closest=np.median([r["min_dist"] for r in ok_runs]),
             info=[d["info"] for d in ds])
    ref = [i.get("reference") for i in s["info"]]
    if tag == "fact" and None in ref:  # the stored propeller FACT runs predate the reference record
        ref = list(json.load(open(os.path.join(RES, "drone", f"reference_failure_{bench}.json"))).values())
    if None not in ref:
        s["cov"] = np.mean([r["axial_descent_share"] for r in ref])
        s["err"] = np.mean([r["faulty_err"] for r in ref])
    for k in ("removed", "bad_removed"):
        if all(k in i for i in s["info"]):
            s[k] = np.mean([i[k] for i in s["info"]])
    return s


R = {(tag, b): reach(tag, b) for b, _ in BENCH
     for tag in ("unfixed", "oracle", "add", "fact", "fact_matched", "l2rw", "l2rw_la",
                 "fact-expert", "fact-routine", "fact-hover")}
TE = json.load(open(os.path.join(RES, "drone", "timing_e2e.json")))["seconds"]["prop50j8"]
SAFE_ROWS = {"None (all logs)": "unfixed", "Oracle (bad logs removed)": "oracle", "Add crash dive, $W=20$": "add",
             "\\method{}, sign rule": "fact", "\\quad 2,640 updates": "fact_matched",
             "Learning to reweight, look-ahead": "l2rw_la", "\\quad per-sample gradients": "l2rw"}
for lab, c0, cells in table("tab:safe"):
    tag = SAFE_ROWS[lab]
    cells = [c0] + cells
    for j, (b, bn) in enumerate(BENCH):
        s = R[(tag, b)]
        check(f"tab:safe {tag} {bn}", (num(cells[2 * j]), None), s, reach=False)
        ps, d = num(cells[2 * j + 1]), (round(s["safe"], 1), round(s["safe_se"], 1))
        ok = ps == d
        print(("  ok   " if ok else "  DIFF ") + f"tab:safe {tag} {bn} safe: paper {ps} data {d}")
        if not ok:
            BAD.append(f"tab:safe {tag} {bn} safe")
    pt, dt = cells[6], ("--" if tag not in TE else "%.0f" % TE[tag])
    ok = pt == dt
    print(("  ok   " if ok else "  DIFF ") + f"tab:safe {tag} time: paper {pt} data {dt}")
    if not ok:
        BAD.append(f"tab:safe {tag} time")
m = re.search(r"an ordinary training on all logs takes (\d+)~s", TEX)
ok = m is not None and int(m.group(1)) == round(TE["plain"])
print(("  ok   " if ok else "  DIFF ") + f"tab:safe caption, ordinary training: paper {m and m.group(1)} data {TE['plain']:.1f}")
if not ok:
    BAD.append("tab:safe caption plain")
for k, key in (("fact", "sign1"), ("oracle", "oracle"), ("l2rw", "controls l2rw")):  # re-flown propeller models = stored ones
    s, r = D["prop50j8"].get(key), R[(k, "prop50j8")]
    ok = s is not None and abs(r["crash"] - s["crash"]) < 0.05
    print(("  ok   " if ok else "  DIFF ") + f"re-flown {k} (propeller): crash {r['crash']:.1f}; stored {s and round(s['crash'], 1)}")
    if not ok:
        BAD.append(f"re-flown {k}")


def rg(v, d=0):
    lo, hi = round(min(v), d), round(max(v), d)
    f = "%." + str(d) + "f"
    return f % lo if lo == hi else (f % lo) + "--" + (f % hi)


B3 = [b for b, _ in BENCH]
L2 = ("l2rw", "l2rw_la")
m = re.search(r"on every fault \(([\d-]+)\\% against ([\d-]+)\\%\)\. Retrained for as many updates as the full-data model, it settles higher "
              r"above the floor on the two faults without look-alikes \(median final distance ([\d.-]+)~m without a crash, against ([\d.]+)~m\) "
              r"and completes fewer \(([\d-]+)\\%\), crashing in ([\d.-]+)\\%\. Learning to reweight with the crash dive as its clean set "
              r"completes more \(([\d-]+)\\%\), more even than the oracle, but on thrust gain it also crashes in ([\d.-]+)\\% of dives, against "
              r"([\d.]+)\\%\. Its planner ends closer to the goal and to the floor than the oracle's \(median final distance without a crash "
              r"([\d.-]+)~m against ([\d.]+)~m; closest approach ([\d.-]+)~m against ([\d.-]+)~m\).*?Adding the crash dive completes ([\d-]+)\\% "
              r"of dives safely", TEX, re.S)
p = list(m.groups())
data = [rg([R[("fact", b)]["safe"] for b in B3]), rg([R[("oracle", b)]["safe"] for b in B3]),
        rg([R[("fact_matched", b)]["final"] for b in ("prop50j8", "gain40j8")], 2), rg([R[("oracle", b)]["final"] for b in B3], 2),
        rg([R[("fact_matched", b)]["safe"] for b in B3]), rg([R[("fact_matched", b)]["crash"] for b in B3], 1),
        rg([R[(k, b)]["safe"] for k in L2 for b in B3]), rg([R[(k, "gain40j8")]["crash"] for k in L2], 1),
        "%.1f" % R[("fact", "gain40j8")]["crash"],
        rg([R[(k, b)]["final"] for k in L2 for b in B3], 2), rg([R[("oracle", b)]["final"] for b in B3], 2),
        rg([R[(k, b)]["closest"] for k in L2 for b in B3], 2), rg([R[("oracle", b)]["closest"] for b in B3], 2),
        rg([R[("add", b)]["safe"] for b in B3])]
ok = p == data and min(R[(k, b)]["safe"] for k in L2 for b in B3) > max(R[("oracle", b)]["safe"] for b in B3)
print(("  ok   " if ok else "  DIFF ") + f"safe-completion text (sec:main): paper {p} data {data}")
if not ok:
    BAD.append("safe-completion text")
m = re.search(r"reweighting completed more dives safely on all three faults \(([\d-]+)\\% against ([\d-]+)\\%\)", TEX)
ok = m is not None and list(m.groups()) == [data[6], data[0]]
print(("  ok   " if ok else "  DIFF ") + f"sec:practice safe completion: paper {m and m.groups()} data {[data[6], data[0]]}")
if not ok:
    BAD.append("sec:practice safe completion")

print("== Table tab:ref (Section sec:ref)")
REF_ROWS = {"Crash dive": "fact", "Expert dive": "fact-expert", "Routine flight": "fact-routine", "Hover": "fact-hover"}
g = lambda v: ("%d" % round(v)) if abs(v - round(v)) < 1e-9 else "%.1f" % v
for lab, c0, cells in table("tab:ref"):
    tag = REF_ROWS[lab]
    cov = [R[(tag, b)]["cov"] for b, _ in BENCH]
    lo, hi = min(cov), max(cov)
    d = "%.2f" % lo if round(lo, 2) == round(hi, 2) else "%.2f--%.2f" % (lo, hi)
    out = [(c0, d)]
    for j, (b, bn) in enumerate(BENCH):
        s = R[(tag, b)]
        out += [(cells[3 * j], "%.2f" % s["err"] if s["err"] < 0.1 else "%.1f" % s["err"]),
                (cells[3 * j + 1], g(s["bad_removed"]) + "/" + g(s["removed"]))]
        check(f"tab:ref {tag} {bn}", (num(cells[3 * j + 2]), None), s, reach=False)
    for i, (paper, data) in enumerate(out):
        ok = paper == data
        print(("  ok   " if ok else "  DIFF ") + f"tab:ref {tag} cell {i}: paper {paper} data {data}")
        if not ok:
            BAD.append(f"tab:ref {tag} cell {i}")
m = re.search(r"finding all or nearly all 32 bad flights \(crashes in ([\d.]+)--([\d.]+)\\% of dives, against ([\d.]+)--([\d.]+)\\%\); "
              r"the routine flight, with less than half the descent, removes more correct flights with the look-alikes \((\d+) on average, "
              r"against (\d+)\)\. .*?error ([\d.]+), against ([\d.]+)--([\d.]+) on the crash dive\): it finds (\d+)--(\d+) of the 32 bad flights, "
              r"and the planner crashes nearly as often as without a fix \((\d+)--(\d+)\\% against (\d+)--(\d+)\\%\)\. .*?hover finds all 32 \(([\d.]+)\\%\)", TEX, re.S)
p = [float(x) for x in m.groups()]
cr = lambda tags, bs: [round(R[(t, b)]["crash"], 1) for t in tags for b in bs]
B3, BP = [b for b, _ in BENCH], ["prop50j8", "prop50j8m"]
er = [R[("fact", b)]["err"] for b in BP]
hb = [R[("fact-hover", b)]["bad_removed"] for b in BP]
hbs = [i["bad_removed"] for b in BP for i in R[("fact-hover", b)]["info"]]
data = [min(cr(["fact-expert", "fact-routine"], B3)), max(cr(["fact-expert", "fact-routine"], B3)), min(cr(["fact"], B3)), max(cr(["fact"], B3)),
        round(R[("fact-routine", "prop50j8m")]["removed"] - 32), round(R[("fact", "prop50j8m")]["removed"] - 32),
        round(R[("fact-hover", "prop50j8")]["err"], 2), round(min(er), 1), round(max(er), 1), min(hbs), max(hbs),
        round(min(cr(["fact-hover"], BP))), round(max(cr(["fact-hover"], BP))), round(min(cr(["unfixed"], BP))), round(max(cr(["unfixed"], BP))),
        round(R[("fact-hover", "gain40j8")]["crash"], 1)]
ok = p == [float(x) for x in data] and R[("fact-hover", "gain40j8")]["bad_removed"] == 32 and round(R[("fact-hover", "prop50j8m")]["err"], 2) == p[6]
print(("  ok   " if ok else "  DIFF ") + f"sec:ref text: paper {p} data {data}")
if not ok:
    BAD.append("sec:ref text")

print("== Measurement pipeline sensitivity (Appendix app:protocol), results/drone/pipeline_*.json")
PL = {}
for v in ("filtered", "shifted", "differenced"):
    ds = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "drone", f"pipeline_{v}_prop50j8_s[0-9].json")))]
    PL[v] = dict(n=len(ds), crash=[100 * np.mean([d["summary"][t]["crash"] for t in DT]) for d in ds],
                 bad=[d["info"]["bad_removed"] for d in ds], auroc=[d["info"]["auroc"] for d in ds],
                 rel=[100 * a / b for d in ds[:1] for a, b in zip(d["info"]["rms_change"], d["info"]["rms_target"])])
m = re.search(r"These change the targets by ([\d.]+)--(\d+)\\% of their RMS, depending on the axis\. Over (\d) seeds the sign rule finds "
              r"(\d+)--(\d+) of the 32 bad flights in every case \(AUROC ([\d.]+)--([\d.]+)\), and the planner crashes in "
              r"\\pmse\{([\d.]+)\}\{([\d.]+)\}, \\pmse\{([\d.]+)\}\{([\d.]+)\} and \\pmse\{([\d.]+)\}\{([\d.]+)\}\\% of dives, against "
              r"\\pmse\{([\d.]+)\}\{([\d.]+)\}\\% with the dataset's targets", TEX)
p = [float(x) for x in m.groups()]
allv = lambda k: [x for v in PL.values() for x in v[k]]
se_ = lambda v: np.std(v, ddof=1) / np.sqrt(len(v))
data = [round(min(allv("rel")), 1), round(max(allv("rel"))), min(v["n"] for v in PL.values()), min(allv("bad")), max(allv("bad")),
        round(min(allv("auroc")), 2), round(max(allv("auroc")), 2)]
for v in ("filtered", "shifted", "differenced"):
    data += [round(np.mean(PL[v]["crash"]), 1), round(se_(PL[v]["crash"]), 1)]
data += [round(D["prop50j8"]["sign1"]["crash"], 1), round(D["prop50j8"]["sign1"]["se"], 1)]
ok = p == [float(x) for x in data]
print(("  ok   " if ok else "  DIFF ") + f"pipeline sensitivity: paper {p} data {data}")
if not ok:
    BAD.append("pipeline sensitivity")

print("\n" +(f"{len(BAD)} MISMATCHES: " + ", ".join(BAD) if BAD else "all checked numbers match the result files"))
