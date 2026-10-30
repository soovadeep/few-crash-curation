"""Sensitivity of FACT to the number of flights removed (k), from one crash dive, mean roll-up, final checkpoint.
Detection (bad flights and look-alikes in the top k) for every k from 8 to 96, and flights for a few k.

usage: python work/ksens_drone.py --fault prop50j8 --seed 1 --fly 24,32,48,64
"""
import argparse
import json
import time

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

import attr_data as AD
import drone as Dr

p = argparse.ArgumentParser()
p.add_argument("--fault", default="prop50j8")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--fly", default="24,32,48,64")
p.add_argument("--threads", type=int, default=1)
args = p.parse_args()
torch.set_num_threads(args.threads)
t0 = time.time()
Xf, Rf, bad, distr = AD.setup(args.fault)
CFG = AD.CFG
net = Dr.make_net(); net.load_state_dict(torch.load(f"work/model3_{args.fault}_s{args.seed}.pt"))
TASKS = {"test dives": dict(AD.TASK), "steep dives": dict(AD.TASK, dive_dx=(0.5, 1.5)),
         "shallow dives": dict(AD.TASK, dive_dx=(5, 8))}


def log(*a):
    print(*a, f"({time.time() - t0:.0f}s)", flush=True)


crash = None
with torch.no_grad():
    for i, (kind, a, b) in enumerate(Dr.tasks(24, seed=11, **AD.TASK)):
        s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
        S, U = AD.run_cl(s0, torch.tensor(b), net, seed=i)
        clr = Dr.clearance(S[:, :3])
        if clr.min() < 0:
            t_c = int(torch.nonzero(clr < 0)[0, 0])
            crash = AD.realized_residual(S[:t_c + 1], U[:t_c], CFG["c_hi"])
            break
params = {n: q.detach() for n, q in net.named_parameters()}
Xn = torch.tensor((Xf - Dr.mu) / Dr.sd, dtype=torch.float32)
Yn = torch.tensor(Rf / Dr.rs, dtype=torch.float32)
flights = np.unique(Dr.FLIGHT)
fbad = np.array([bad[Dr.FLIGHT == f].any() for f in flights])
fdistr = np.array([distr[Dr.FLIGHT == f].any() for f in flights])
g_i = vmap(grad(lambda pd, x, y: ((functional_call(net, pd, (x[None],))[0] - y) ** 2).sum()), in_dims=(None, 0, 0))
xs, ys = crash
feat, yt = (xs - Dr.MU) / Dr.SD, ys / Dr.RS
pg = {k: v.clone().requires_grad_(True) for k, v in params.items()}
tg = torch.autograd.grad(((functional_call(net, pg, (feat,)) - yt) ** 2).sum(-1).mean(), list(pg.values()))
sc = torch.zeros(len(Xn))
for s in range(0, len(Xn), 2048):
    G = g_i(params, Xn[s:s + 2048], Yn[s:s + 2048])
    nrm = torch.sqrt(sum((G[k].flatten(1) ** 2).sum(1) for k in G))
    sc[s:s + 2048] = -sum((G[k].flatten(1) * t.flatten()[None]).sum(1) for k, t in zip(G, tg)) / (nrm + 1e-12)
fs = np.array([sc.numpy()[Dr.FLIGHT == f].mean() for f in flights])
o = np.argsort(-fs)
det = {int(k): dict(bad=int(fbad[o[:k]].sum()), lookalike=int(fdistr[o[:k]].sum())) for k in range(8, 97, 4)}
res = dict(args=vars(args), n_bad=int(fbad.sum()), n_lookalike=int(fdistr.sum()), n_positive=int((fs > 0).sum()),
           detection=det, rows=[])
OUT = f"work/ksens_{args.fault}_s{args.seed}.json"
json.dump(res, open(OUT, "w"), indent=1)
log("detection", {k: v for k, v in det.items() if k in (24, 32, 40, 48, 64)})
for k in [int(x) for x in args.fly.split(",") if x]:
    keep = ~np.isin(Dr.FLIGHT, flights[o[:k]])
    m = Dr.fit(Xf, Rf, keep, seed=args.seed)
    r = dict(k=k, bad_removed=int(fbad[o[:k]].sum()), lookalike_removed=int(fdistr[o[:k]].sum()))
    for name, tk in TASKS.items():
        e = Dr.evaluate(m, c_hi=CFG["c_hi"], delay=3, every=3, T=250, n_each=24, seed=7, **tk)
        r[name] = float(np.mean([x["crash"] for x in e]))
        r[f"{name} reach"] = float(np.mean([x["reach"] for x in e]))
    res["rows"].append(r); log(r)
    json.dump(res, open(OUT, "w"), indent=1)
log("done")
