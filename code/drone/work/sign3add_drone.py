"""Curate with the sign rule from the first 3 crash dives, then add those dives at W=20 (as mis_drone.py does for 1
dive). Appends the row to work/mis_{fault}_s{seed}.json.

usage: python work/sign3add_drone.py --fault prop50j8 --seed 1
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
p.add_argument("--w", type=int, default=20)
args = p.parse_args()
t0 = time.time()
Xf, Rf, bad, distr = AD.setup(args.fault)
CFG = AD.CFG
base = args.fault.split("-")[0]
net = Dr.make_net(); net.load_state_dict(torch.load(f"work/model3_{base}_s{args.seed}.pt"))
TASKS = {"test dives": dict(AD.TASK), "steep dives": dict(AD.TASK, dive_dx=(0.5, 1.5)),
         "shallow dives": dict(AD.TASK, dive_dx=(5, 8))}
OUT = f"work/mis_{args.fault}_s{args.seed}.json"
NAME = f"ours (sign rule) from 3 dives, and add them (W={args.w})"


def log(*a):
    print(*a, f"({time.time() - t0:.0f}s)", flush=True)


def fly(m):
    out = {}
    for name, tk in TASKS.items():
        r = Dr.evaluate(m, c_hi=CFG["c_hi"], delay=3, every=3, T=250, n_each=24, seed=7, **tk)
        out[name] = float(np.mean([x["crash"] for x in r]))
        out[f"{name} reach"] = float(np.mean([x["reach"] for x in r]))
    return out


if any(r["model"] == NAME for r in json.load(open(OUT))["rows"]):
    log("already done"); raise SystemExit
crashes = []
with torch.no_grad():
    for i, (kind, a, b) in enumerate(Dr.tasks(24, seed=11, **AD.TASK)):
        s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
        S, U = AD.run_cl(s0, torch.tensor(b), net, seed=i)
        clr = Dr.clearance(S[:, :3])
        if clr.min() < 0:
            t_c = int(torch.nonzero(clr < 0)[0, 0])
            crashes.append(AD.realized_residual(S[:t_c + 1], U[:t_c], CFG["c_hi"]))
            if len(crashes) == 3:
                break
params = {n: q.detach() for n, q in net.named_parameters()}
Xn = torch.tensor((Xf - Dr.mu) / Dr.sd, dtype=torch.float32)
Yn = torch.tensor(Rf / Dr.rs, dtype=torch.float32)
g_i = vmap(grad(lambda pd, x, y: ((functional_call(net, pd, (x[None],))[0] - y) ** 2).sum()), in_dims=(None, 0, 0))
xs = torch.cat([x for x, _ in crashes]); ys = torch.cat([y for _, y in crashes])
pg = {k: v.clone().requires_grad_(True) for k, v in params.items()}
L = ((functional_call(net, pg, ((xs - Dr.MU) / Dr.SD,)) - ys / Dr.RS) ** 2).sum(-1).mean()
tg = torch.autograd.grad(L, list(pg.values()))
sc = torch.zeros(len(Xn))
for s in range(0, len(Xn), 2048):
    G = g_i(params, Xn[s:s + 2048], Yn[s:s + 2048])
    nrm = torch.sqrt(sum((G[k].flatten(1) ** 2).sum(1) for k in G))
    sc[s:s + 2048] = -sum((G[k].flatten(1) * t.flatten()[None]).sum(1) for k, t in zip(G, tg)) / (nrm + 1e-12)
flights = np.unique(Dr.FLIGHT)
fs = np.array([sc.numpy()[Dr.FLIGHT == f].mean() for f in flights])
fbad = np.array([bad[Dr.FLIGHT == f].any() for f in flights])
rm = fs > 0
keep = ~np.isin(Dr.FLIGHT, flights[rm])
log(f"sign rule from {len(crashes)} dives removes {rm.sum()} ({(rm & fbad).sum()} bad)")
Xa = np.concatenate([Xf[keep]] + [xs.numpy()] * args.w); Ra = np.concatenate([Rf[keep]] + [ys.numpy()] * args.w)
row = dict(model=NAME, removed=int(rm.sum()), bad_removed=int((rm & fbad).sum()),
           **fly(Dr.fit(Xa, Ra, np.ones(len(Xa), bool), seed=args.seed)))
d = json.load(open(OUT)); d["rows"].append(row); json.dump(d, open(OUT, "w"), indent=1)
log(row)
