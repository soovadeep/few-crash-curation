"""Baselines that use no failure at all, on the same drone benchmarks as mis_drone.py:

  - ensemble MPPI: 5 models trained on the faulty logs (different seeds); the planner uses their mean, and
    optionally adds beta * (members' disagreement) to the cost of every predicted step (uncertainty-aware MPPI)
  - robust regression: one model trained with a Huber or an L1 loss instead of squared error
  - training-loss filtering: remove the 20 flights the faulty model fits worst, retrain

Models depend only on the base scenario (prop50mv), so they are trained once and shared across gust levels.
Also logs the ensemble disagreement on the faulty model's crash dives against its spread on the training logs,
to show whether the failure region looks uncertain to the ensemble.

usage: python work/base_drone.py --fault prop50j8 --seed 1 --k 40
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import attr_data as AD
import drone as Dr

p = argparse.ArgumentParser()
p.add_argument("--fault", default="prop50mv")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--betas", default="10,1,100")
p.add_argument("--smoke", action="store_true")  # tiny run to check the code path
p.add_argument("--k", type=int, default=20)  # flights removed by worst-fit filtering
p.add_argument("--cheap", action="store_true")  # robust losses and worst-fit filtering only, no ensemble
args = p.parse_args()
t0 = time.time()
Xf, Rf, bad, distr = AD.setup(args.fault)
CFG = AD.CFG
base = args.fault.split("-")[0]
net = Dr.make_net(); net.load_state_dict(torch.load(f"work/model3_{base}_s{args.seed}.pt"))
TAG = "smoke_" if args.smoke else ""
if args.smoke:
    _fit = Dr.fit
    Dr.fit = lambda *a, **kw: _fit(*a, **dict(kw, epochs=1))
TASKS = {"test dives": dict(AD.TASK), "steep dives": dict(AD.TASK, dive_dx=(0.5, 1.5)),
         "shallow dives": dict(AD.TASK, dive_dx=(5, 8))}
ALL = np.ones(len(Xf), bool)


def log(*a):
    print(*a, f"({time.time() - t0:.0f}s)", flush=True)


def fly(m):
    out = {}
    for name, tk in TASKS.items():
        r = Dr.evaluate(m, c_hi=CFG["c_hi"], delay=3, every=3, T=30 if args.smoke else 250,
                        n_each=1 if args.smoke else 24, seed=7, **tk)
        out[name] = float(np.mean([x["crash"] for x in r]))
        out[f"{name} reach"] = float(np.mean([x["reach"] for x in r]))  # goal reached (a timid planner never crashes)
    return out


def cached(path, make):
    m = Dr.make_net()
    if os.path.exists(path):
        m.load_state_dict(torch.load(path)); [q.requires_grad_(False) for q in m.parameters()]
        return m
    m = make(); torch.save(m.state_dict(), path)
    return m


def fit_loss(Xd, Rd, mask, seed, kind, epochs=None):
    """drone.fit with a robust loss: "huber" (delta = 1 in normalized units) or "l1"."""
    epochs = epochs or (1 if args.smoke else 12)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Dr.make_net()
    Xt = torch.tensor((Xd - Dr.mu) / Dr.sd, dtype=torch.float32)
    Rt = torch.tensor(Rd / Dr.rs, dtype=torch.float32)
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    idx = np.where(mask)[0]
    lf = torch.nn.HuberLoss(delta=1.0) if kind == "huber" else torch.nn.L1Loss()
    for ep in range(epochs):
        rng.shuffle(idx)
        for b in range(0, len(idx), 2048):
            bi = idx[b:b + 2048]
            loss = lf(m(Xt[bi]), Rt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
        for g in opt.param_groups:
            g['lr'] *= 0.8
    for q in m.parameters():
        q.requires_grad_(False)
    return m


class Ensemble(torch.nn.Module):
    """Mean of the members; keeps the members' disagreement (m/s^2) of the last call for the cost."""

    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)
        self.spread = None

    def forward(self, f):
        P = torch.stack([m(f) for m in self.members])
        self.spread = (P.std(0) * Dr.RS).norm(dim=-1)
        return P.mean(0)


OUT = f"work/{TAG}base_{args.fault}_s{args.seed}.json"
try:
    _old = json.load(open(OUT)); rows, spread = _old["rows"], _old.get("spread", {})
except FileNotFoundError:
    rows, spread = [], {}
done = {r["model"] for r in rows}


def add(name, make, **kw):
    if name in done:
        return
    r = dict(model=name, **kw, **fly(make()))
    rows.append(r); log(r)
    json.dump(dict(args=vars(args), spread=spread, rows=rows), open(OUT, "w"), indent=1)


add("faulty", lambda: net)
# cheap baselines first, then the ensemble (each ensemble row costs about five normal rows; skipped with --cheap)
for kind in ["huber", "l1"]:
    add(f"{kind} loss", lambda: cached(f"work/{TAG}{kind}_{base}_s{args.seed}.pt",
                                        lambda: fit_loss(Xf, Rf, ALL, args.seed, kind)))
flights = np.unique(Dr.FLIGHT)
fbad = np.array([bad[Dr.FLIGHT == f].any() for f in flights])
fdistr = np.array([distr[Dr.FLIGHT == f].any() for f in flights])
with torch.no_grad():
    Xt = torch.tensor((Xf - Dr.mu) / Dr.sd, dtype=torch.float32)
    err = ((net(Xt) - torch.tensor(Rf / Dr.rs, dtype=torch.float32)) ** 2).sum(-1).numpy()
fl = np.array([err[Dr.FLIGHT == f].mean() for f in flights])
o = np.argsort(-fl)
keep = ~np.isin(Dr.FLIGHT, flights[o[:args.k]])
add(f"remove {args.k} worst-fit flights", lambda: Dr.fit(Xf, Rf, keep, seed=args.seed),
    bad_removed=int(fbad[o[:args.k]].sum()), distr_removed=int(fdistr[o[:args.k]].sum()))
# crude rules that use where the crash happens (axial descent) but not the labels: drop the k flights with the most
# axial descent, or drop every axial-descent sample (the "just drop the regime" fix)
desc = Dr.V[:, 2] < -1.5
od = np.argsort(-np.array([desc[Dr.FLIGHT == f].sum() for f in flights]))
keep_d = ~np.isin(Dr.FLIGHT, flights[od[:args.k]])
add(f"remove {args.k} most crash-like flights (most axial descent)", lambda: Dr.fit(Xf, Rf, keep_d, seed=args.seed),
    bad_removed=int(fbad[od[:args.k]].sum()), distr_removed=int(fdistr[od[:args.k]].sum()))
add("remove all axial-descent samples", lambda: Dr.fit(Xf, Rf, ~desc, seed=args.seed),
    samples_removed=int(desc.sum()), bad_samples_removed=int((desc & bad).sum()))
if args.cheap:
    log("done"); raise SystemExit
members = [cached(f"work/{TAG}ens_{base}_s{args.seed}_{k}.pt",
                  lambda k=k: Dr.fit(Xf, Rf, ALL, seed=1000 + 10 * args.seed + k)) for k in range(5)]
ENS = Ensemble(members)
BETA = [0.0]
_cost = Dr.cost


def cost_pen(s, u, goal):
    c = _cost(s, u, goal)
    if BETA[0] and ENS.spread is not None and ENS.spread.shape == c.shape:
        c = c + BETA[0] * ENS.spread
    return c


Dr.cost = cost_pen

# crash dives of the faulty model, as in mis_drone.py, only to measure the ensemble's disagreement there
if hasattr(Dr.PLANT, "offset"):
    Dr.PLANT.offset = 50_000
crash_x = []
with torch.no_grad():
    for i, (kind, a, b) in enumerate(Dr.tasks(3 if args.smoke else 24, seed=11, **AD.TASK)):
        s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
        S, U = AD.run_cl(s0, torch.tensor(b), net, seed=i)
        clr = Dr.clearance(S[:, :3])
        if clr.min() < 0:
            t_c = int(torch.nonzero(clr < 0)[0, 0])
            crash_x.append(AD.realized_residual(S[:t_c + 1], U[:t_c], CFG["c_hi"])[0])
if hasattr(Dr.PLANT, "offset"):
    Dr.PLANT.offset = 0
with torch.no_grad():
    xc = torch.cat(crash_x) if crash_x else torch.tensor(Xf[:50], dtype=torch.float32)  # (smoke runs may not crash)
    ENS((xc - Dr.MU) / Dr.SD); sp_crash = ENS.spread.numpy()
    sub = np.random.default_rng(0).choice(len(Xf), 20000, replace=False)
    ENS(torch.tensor((Xf[sub] - Dr.mu) / Dr.sd, dtype=torch.float32)); sp_train = ENS.spread.numpy()
    sub_bad = np.where(bad)[0]
    ENS(torch.tensor((Xf[sub_bad] - Dr.mu) / Dr.sd, dtype=torch.float32)); sp_bad = ENS.spread.numpy()
spread = dict(crash_dives_median=float(np.median(sp_crash)), crash_dives_p90=float(np.percentile(sp_crash, 90)),
              train_median=float(np.median(sp_train)), train_p90=float(np.percentile(sp_train, 90)),
              bad_flights_median=float(np.median(sp_bad)), crash_above_train_p90=float((sp_crash > np.percentile(sp_train, 90)).mean()))
log(f"{len(crash_x)} crash dives; ensemble disagreement (m/s^2):", spread)
add("ensemble of 5, mean", lambda: ENS)
for b in [float(x) for x in args.betas.split(",")]:
    BETA[0] = b
    add(f"ensemble of 5, disagreement penalty beta={b:g}", lambda: ENS)
BETA[0] = 0.0
log("done")
