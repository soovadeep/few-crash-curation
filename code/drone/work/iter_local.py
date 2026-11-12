"""Rounds of flight-level FACT on the within-flight fault: each round takes the first crash of the current
model on the 24 attribution dives, scores the remaining flights, removes those with a positive score and retrains;
stops when no attribution dive crashes or after --rounds failures. Same setup as local_drone.py.

Setup. Within-flight fault: the propeller fault (attr_data.prop_fault) starts partway through each faulty flight, so the
same flight holds correct transitions (before the onset) and faulty ones (after it). Faulty flights are the nper with
the most axial descent per day; each one's onset is drawn uniformly in [lo, hi] of its length (fixed rng, the same
for every seed). FACT scores every sample from the first crash of the 24 attribution dives, as in sweep_drone.py,
and removes data by the sign rule at three granularities: whole flights, windows of `win` consecutive samples, and
single samples. Flown against the unfixed model and two oracles: whole faulty flights removed, and exactly the
post-onset samples removed.

usage: python work/local_drone.py --seed 1
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.func import functional_call, grad, vmap

import attr_data as AD
import drone as Dr

p = argparse.ArgumentParser()
p.add_argument("--sev", type=int, default=50, help="extra thrust in axial descent, %")
p.add_argument("--nper", type=int, default=11, help="faulty flights per day")
p.add_argument("--lo", type=float, default=0.25, help="earliest onset, fraction of the flight")
p.add_argument("--hi", type=float, default=0.5, help="latest onset")
p.add_argument("--win", type=int, default=200, help="window length in samples (100 Hz)")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--threads", type=int, default=1)
p.add_argument("--rounds", type=int, default=3)
p.add_argument("--unit", default="flights", choices=["flights", "windows"])
p.add_argument("--nofly", action="store_true", help="removal statistics only (out *_nofly.json)")
args = p.parse_args()
torch.set_num_threads(args.threads)
t0 = time.time()
tag = f"sev{args.sev}_n{args.nper}_on{int(100 * args.lo)}-{int(100 * args.hi)}_s{args.seed}"
OUT = (f"work/localiter_{tag}" if args.unit == "flights" else f"work/localiterw_{tag}") + ("_nofly.json" if args.nofly else ".json")


def log(*a):
    print(*a, f"({time.time() - t0:.0f}s)", flush=True)


faulty = AD.prop_flights(args.nper)
rng = np.random.default_rng(0)
bad = np.zeros(len(Dr.FLIGHT), bool)
onset = {}
for f in faulty:
    idx = np.where(Dr.FLIGHT == f)[0]
    on = int(len(idx) * rng.uniform(args.lo, args.hi))
    bad[idx[on:]] = True
    onset[int(f)] = on / len(idx)
d = 1.0 / (1.0 + np.exp((Dr.V[:, 2] + 1.0) / 0.4))
Fm = Dr.F.copy()
Fm[:, 2] = np.where(bad, Dr.F[:, 2] * (1 + args.sev / 100 * d), Dr.F[:, 2])
X, R = Dr.X.copy(), Fm - np.stack([0 * Dr.C, 0 * Dr.C, Dr.C], 1)
fflight = np.isin(Dr.FLIGHT, faulty)
AD.CFG["c_hi"] = 14.0
desc = Dr.V[:, 2] < -1.5
res = dict(args=vars(args), n_faulty=len(faulty), onset=onset,
           faulty_share_of_descent=float((desc & bad).sum() / desc.sum()),
           faulty_flights_descent_before_onset=float((desc & fflight & ~bad).sum() / (desc & fflight).sum()))
mp = f"work/model_local_{tag}.pt"
if os.path.exists(mp):
    net = Dr.make_net(); net.load_state_dict(torch.load(mp)); [q.requires_grad_(False) for q in net.parameters()]
else:
    net = Dr.fit(X, R, np.ones(len(X), bool), seed=args.seed); torch.save(net.state_dict(), mp)
log("faulty model ready", {k: v for k, v in res.items() if k not in ("args", "onset")})

TASKS = {"test dives": dict(AD.TASK), "steep dives": dict(AD.TASK, dive_dx=(0.5, 1.5)),
         "shallow dives": dict(AD.TASK, dive_dx=(5, 8))}


def fly(m):
    r = {}
    for name, tk in TASKS.items():
        e = Dr.evaluate(m, c_hi=AD.CFG["c_hi"], delay=3, every=3, T=250, n_each=24, seed=7, **tk)
        r[name] = float(np.mean([x["crash"] for x in e])); r[f"{name} reach"] = float(np.mean([x["reach"] for x in e]))
    return r


def first_crash(model):
    n, rec = 0, None
    with torch.no_grad():
        for i, (kind, a, b) in enumerate(Dr.tasks(24, seed=11, **AD.TASK)):
            s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
            S, U = AD.run_cl(s0, torch.tensor(b), model, seed=i)
            clr = Dr.clearance(S[:, :3])
            if clr.min() < 0:
                n += 1
                if rec is None:
                    t_c = int(torch.nonzero(clr < 0)[0, 0])
                    rec = AD.realized_residual(S[:t_c + 1], U[:t_c], AD.CFG["c_hi"])
    return n, rec


Xn = torch.tensor((X - Dr.mu) / Dr.sd, dtype=torch.float32)
Yn = torch.tensor(R / Dr.rs, dtype=torch.float32)
if args.unit == "flights":
    unit = np.searchsorted(np.unique(Dr.FLIGHT), Dr.FLIGHT)
else:  # windows of args.win samples inside each flight, as in local_drone.py
    unit = np.zeros(len(Dr.FLIGHT), int)
    nw = 0
    for f in np.unique(Dr.FLIGHT):
        idx = np.where(Dr.FLIGHT == f)[0]
        k = max(len(idx) // args.win, 1)
        unit[idx] = nw + np.minimum(np.arange(len(idx)) // args.win, k - 1)
        nw += k
n_u = unit.max() + 1
cnt = np.bincount(unit, minlength=n_u)
removed = np.zeros(n_u, bool)
f_bad = np.bincount(unit, bad, minlength=n_u) / cnt > 0.5
model = net
res["rounds"] = []
for rnd in range(1, args.rounds + 1):
    n_crash, crash = first_crash(model)
    row = dict(round=rnd, attribution_dives_crashed=n_crash)
    if crash is None:
        res["rounds"].append(row); log("no attribution crash: stop"); break
    params = {k: v.detach() for k, v in model.named_parameters()}
    g_i = vmap(grad(lambda pd, x, y: ((functional_call(model, pd, (x[None],))[0] - y) ** 2).sum()), in_dims=(None, 0, 0))
    xs, ys = crash
    feat, yt = (xs - Dr.MU) / Dr.SD, ys / Dr.RS
    pg = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    tg = torch.autograd.grad(((functional_call(model, pg, (feat,)) - yt) ** 2).sum(-1).mean(), list(pg.values()))
    live = np.where(~removed[unit])[0]
    sc = np.zeros(len(Xn))
    for s in range(0, len(live), 2048):
        b = live[s:s + 2048]
        G = g_i(params, Xn[b], Yn[b])
        nrm = torch.sqrt(sum((G[k].flatten(1) ** 2).sum(1) for k in G))
        sc[b] = (-sum((G[k].flatten(1) * t.flatten()[None]).sum(1) for k, t in zip(G, tg)) / (nrm + 1e-12)).numpy()
    f_sc = np.bincount(unit, sc, minlength=n_u) / cnt
    new = ~removed & (f_sc > 0)
    removed |= new
    keep = ~removed[unit]
    row.update(new_removed=int(new.sum()), new_bad=int((new & f_bad).sum()), removed=int(removed.sum()),
               bad_removed=int((removed & f_bad).sum()),
               faulty_descent_removed=float((~keep & bad & desc).sum() / (bad & desc).sum()),
               correct_descent_removed=float((~keep & ~bad & desc).sum() / (~bad & desc).sum()), kept=float(keep.mean()), removed_units=np.where(removed)[0].tolist())
    model = Dr.fit(X, R, keep, seed=args.seed)
    if not args.nofly:
        row["flown"] = fly(model)
    res["rounds"].append(row); log(row)
    json.dump(res, open(OUT, "w"), indent=1)
json.dump(res, open(OUT, "w"), indent=1)
log("done")
