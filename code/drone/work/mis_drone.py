"""Few-failure test on a drone the learned model cannot fit exactly (plant with gusts or motor lag, see
attr_data.HiddenPlant): curation from n crash dives vs adding those n dives to the training data, and both.

Labels on the crash dives are the residuals the plant actually produced (attr_data.realized_residual), for our
score and for the add-data rows alike. Models are flown on test, steep and shallow dives.

usage: python work/mis_drone.py --fault prop50j8 --seed 1 --k 40
"""
import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.func import functional_call, grad, vmap

import attr_data as AD
import drone as Dr

p = argparse.ArgumentParser()
p.add_argument("--fault", default="prop50mv-gust20")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--w", type=int, default=20)
p.add_argument("--k", type=int, default=20)  # flights removed (raise it when more flights are bad, e.g. prop50j8)
args = p.parse_args()
t0 = time.time()
Xf, Rf, bad, distr = AD.setup(args.fault)
CFG = AD.CFG
base = args.fault.split("-")[0]
net = Dr.make_net(); net.load_state_dict(torch.load(f"work/model3_{base}_s{args.seed}.pt"))
TASKS = {"test dives": dict(AD.TASK), "steep dives": dict(AD.TASK, dive_dx=(0.5, 1.5)),
         "shallow dives": dict(AD.TASK, dive_dx=(5, 8))}


def log(*a):
    print(*a, f"({time.time() - t0:.0f}s)", flush=True)


def fly(m):
    out = {}
    for name, tk in TASKS.items():
        r = Dr.evaluate(m, c_hi=CFG["c_hi"], delay=3, every=3, T=250, n_each=24, seed=7, **tk)
        out[name] = float(np.mean([x["crash"] for x in r]))
        out[f"{name} reach"] = float(np.mean([x["reach"] for x in r]))  # goal reached (a timid planner never crashes)
    return out


# crash dives of the faulty model on the attribution tasks, with their own gusts
if hasattr(Dr.PLANT, "offset"):
    Dr.PLANT.offset = 50_000
crashes = []
with torch.no_grad():
    for i, (kind, a, b) in enumerate(Dr.tasks(24, seed=11, **AD.TASK)):
        s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
        S, U = AD.run_cl(s0, torch.tensor(b), net, seed=i)
        clr = Dr.clearance(S[:, :3])
        if clr.min() < 0:
            t_c = int(torch.nonzero(clr < 0)[0, 0])
            crashes.append(AD.realized_residual(S[:t_c + 1], U[:t_c], CFG["c_hi"]))
if hasattr(Dr.PLANT, "offset"):
    Dr.PLANT.offset = 0
log(f"{len(crashes)} crash dives")

params = {n: q.detach() for n, q in net.named_parameters()}
Xn = torch.tensor((Xf - Dr.mu) / Dr.sd, dtype=torch.float32)
Yn = torch.tensor(Rf / Dr.rs, dtype=torch.float32)


def loss_i(pd, x, y):
    return ((functional_call(net, pd, (x[None],))[0] - y) ** 2).sum()


g_i = vmap(grad(loss_i), in_dims=(None, 0, 0))
flights = np.unique(Dr.FLIGHT)
fbad = np.array([bad[Dr.FLIGHT == f].any() for f in flights])
fdistr = np.array([distr[Dr.FLIGHT == f].any() for f in flights])
N = len(Xn)


def subset(n):
    return crashes if n == "all" else crashes[:n]


AGG = {"mean": np.mean, "worst 2 s": lambda v: np.sort(v)[-200:].mean()}  # logs are at 100 Hz


def score(n, agg="mean"):
    """Our flight scores from the first n crash dives: TracIn-cos on prediction loss, realized labels, rolled up
    per flight by the mean or by the mean of the flight's worst 2 s of transitions."""
    xs = torch.cat([x for x, _ in subset(n)]); ys = torch.cat([y for _, y in subset(n)])
    feat, yt = (xs - Dr.MU) / Dr.SD, ys / Dr.RS
    pg = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    L = ((functional_call(net, pg, (feat,)) - yt) ** 2).sum(-1).mean()
    tg = torch.autograd.grad(L, list(pg.values()))
    sc = torch.zeros(N)
    for s in range(0, N, 2048):
        G = g_i(params, Xn[s:s + 2048], Yn[s:s + 2048])
        nrm = torch.sqrt(sum((G[k].flatten(1) ** 2).sum(1) for k in G))
        sc[s:s + 2048] = -sum((G[k].flatten(1) * t.flatten()[None]).sum(1) for k, t in zip(G, tg)) / (nrm + 1e-12)
    fs = np.array([AGG[agg](sc.numpy()[Dr.FLIGHT == f]) for f in flights])
    o = np.argsort(-fs)
    with torch.no_grad():
        err = float(((net(feat) - yt) ** 2).sum(-1).mean())
    return o, dict(n_dives=len(subset(n)), transitions=len(xs), faulty_err=err,
                   auroc=float(roc_auc_score(fbad, fs)), ap=float(average_precision_score(fbad, fs)),
                   bad_top12=int(fbad[o[:12]].sum()), distr_top20=int(fdistr[o[:20]].sum()),
                   n_positive=int((fs > 0).sum()))


def augmented(n, keep):
    xs = np.concatenate([x.numpy() for x, _ in subset(n)]); ys = np.concatenate([y.numpy() for _, y in subset(n)])
    Xa = np.concatenate([Xf[keep]] + [xs] * args.w); Ra = np.concatenate([Rf[keep]] + [ys] * args.w)
    return Xa, Ra


OUT = f"work/mis_{args.fault}_s{args.seed}.json"
try:  # resume after an interruption: keep rows already flown
    rows = json.load(open(OUT))["rows"]
except FileNotFoundError:
    rows = []
done = {r["model"] for r in rows}


def add(name, make, **kw):
    if name in done:
        return
    r = dict(model=name, **kw, **fly(make()))
    rows.append(r); log(r)
    json.dump(dict(args=vars(args), n_crash_dives=len(crashes), rows=rows),
              open(OUT, "w"), indent=1)


clean = Dr.make_net(); clean.load_state_dict(torch.load(f"work/model3_{base}_dropharm_s{args.seed}.pt"))
with torch.no_grad():
    xs = torch.cat([x for x, _ in crashes]); ys = torch.cat([y for _, y in crashes])
    feat, yt = (xs - Dr.MU) / Dr.SD, ys / Dr.RS
    errs = dict(faulty=float(((net(feat) - yt) ** 2).sum(-1).mean()), clean=float(((clean(feat) - yt) ** 2).sum(-1).mean()))
log("error on all crash-dive transitions", errs)
add("faulty", lambda: net, crash_err=errs["faulty"])
add("clean (bad flights removed)", lambda: clean, crash_err=errs["clean"])
plan = [(1, sorted({12, args.k})), (3, [args.k]), ("all", [args.k])]
orders = {}
for n, ks in plan:
    o, det = score(n)
    orders[n] = o
    log(f"ours from {det['n_dives']} dives:", det)
    for k in ks:
        keep = ~np.isin(Dr.FLIGHT, flights[o[:k]])
        add(f"ours from {det['n_dives']} dives, remove {k}", lambda: Dr.fit(Xf, Rf, keep, seed=args.seed),
            bad_removed=int(fbad[o[:k]].sum()), distr_removed=int(fdistr[o[:k]].sum()), **det)
for n in [1, 3, "all"]:
    Xa, Ra = augmented(n, np.ones(len(Xf), bool))
    add(f"add {len(subset(n))} crash dives, W={args.w}", lambda: Dr.fit(Xa, Ra, np.ones(len(Xa), bool), seed=args.seed))
for n in [1, 3]:
    keep = ~np.isin(Dr.FLIGHT, flights[orders[n][:args.k]])
    Xa, Ra = augmented(n, keep)
    add(f"ours from {len(subset(n))} dives, remove {args.k}, and add them (W={args.w})",
        lambda: Dr.fit(Xa, Ra, np.ones(len(Xa), bool), seed=args.seed))
# the roll-up used on every platform in the paper: each log scored by its worst 2 s
for n in [1, 3, "all"]:
    o, det = score(n, "worst 2 s")
    log(f"ours (worst 2 s) from {det['n_dives']} dives:", det)
    keep = ~np.isin(Dr.FLIGHT, flights[o[:args.k]])
    add(f"ours (worst 2 s) from {det['n_dives']} dives, remove {args.k}", lambda: Dr.fit(Xf, Rf, keep, seed=args.seed),
        bad_removed=int(fbad[o[:args.k]].sum()), distr_removed=int(fdistr[o[:args.k]].sum()), **det)
# parameter-free stopping rule: remove every flight whose mean score is positive, i.e. whose data on balance
# pushed the model toward the crash dive's error
for n in [1, 3]:
    o, det = score(n)
    k = det["n_positive"]
    keep = ~np.isin(Dr.FLIGHT, flights[o[:k]])
    add(f"ours (sign rule) from {det['n_dives']} dives", lambda: Dr.fit(Xf, Rf, keep, seed=args.seed),
        bad_removed=int(fbad[o[:k]].sum()), distr_removed=int(fdistr[o[:k]].sum()), removed=k, **det)
    if n == 1:
        Xa, Ra = augmented(n, keep)
        add(f"ours (sign rule) from {det['n_dives']} dives, and add them (W={args.w})",
            lambda: Dr.fit(Xa, Ra, np.ones(len(Xa), bool), seed=args.seed), removed=k)
    # the same count, but the logs chosen by their worst 2 s (better at ranking logs whose error is spread out)
    o2, det2 = score(n, "worst 2 s")
    keep = ~np.isin(Dr.FLIGHT, flights[o2[:k]])
    add(f"ours (worst 2 s, sign-rule count) from {det['n_dives']} dives", lambda: Dr.fit(Xf, Rf, keep, seed=args.seed),
        bad_removed=int(fbad[o2[:k]].sum()), distr_removed=int(fdistr[o2[:k]].sum()), removed=k)
log("done")
