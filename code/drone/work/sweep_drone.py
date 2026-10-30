"""Robustness of FACT's TracIn score: vary the fault severity, the number of faulty flights and the number of
correct look-alike flights. Faulty flights carry the propeller fault (attr_data.prop_fault). Per day, the M = n + l
flights with the most axial descent are split into n faulty and l look-alikes, the look-alikes spread evenly over the
ranks so that they match the faulty flights in descent. One faulty model per seed; FACT scores from its first crash
dive (of the 24 attribution dives) and, with --fly, the sign-rule model is flown on test, steep and shallow dives.

usage: python work/sweep_drone.py --sev 50 --nper 8 --look 0 --seed 1 --fly
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.func import functional_call, grad, vmap

import attr_data as AD
import drone as Dr

p = argparse.ArgumentParser()
p.add_argument("--sev", type=int, default=50, help="extra thrust in axial descent, %")
p.add_argument("--nper", type=int, default=8, help="faulty flights per day")
p.add_argument("--look", type=int, default=0, help="correct look-alike flights per day")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--fly", action="store_true")
p.add_argument("--threads", type=int, default=1)
args = p.parse_args()
torch.set_num_threads(args.threads)
t0 = time.time()
tag = f"sev{args.sev}_n{args.nper}_l{args.look}_s{args.seed}"
OUT = f"work/sweep_{tag}.json"


def log(*a):
    print(*a, f"({time.time() - t0:.0f}s)", flush=True)


M = args.nper + args.look
top = AD.prop_flights(M)
look_pos = {int((j + 0.5) * M / args.look) for j in range(args.look)} if args.look else set()
faulty, look = [], []
for d in range(len(top) // M):
    for r, f in enumerate(top[d * M:(d + 1) * M]):
        (look if r in look_pos else faulty).append(f)
X, R, bad = AD.prop_fault(args.sev / 100, faulty)
distr = np.isin(Dr.FLIGHT, look)
AD.CFG["c_hi"] = 14.0
desc = Dr.V[:, 2] < -1.5
res = dict(args=vars(args), n_faulty=len(faulty), n_look=len(look),
           faulty_share_of_descent=float((desc & bad).sum() / desc.sum()),
           look_share_of_descent=float((desc & distr).sum() / desc.sum()))
mp = f"work/model_sweep_{tag}.pt"
if os.path.exists(mp):
    net = Dr.make_net(); net.load_state_dict(torch.load(mp)); [q.requires_grad_(False) for q in net.parameters()]
else:
    net = Dr.fit(X, R, np.ones(len(X), bool), seed=args.seed); torch.save(net.state_dict(), mp)
log("faulty model ready", {k: v for k, v in res.items() if k != "args"})

n_crash, crash = 0, None
with torch.no_grad():
    for i, (kind, a, b) in enumerate(Dr.tasks(24, seed=11, **AD.TASK)):
        s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
        S, U = AD.run_cl(s0, torch.tensor(b), net, seed=i)
        clr = Dr.clearance(S[:, :3])
        if clr.min() < 0:
            n_crash += 1
            if crash is None:
                t_c = int(torch.nonzero(clr < 0)[0, 0])
                crash = AD.realized_residual(S[:t_c + 1], U[:t_c], AD.CFG["c_hi"])
res["attribution_dives_crashed"] = n_crash
log(f"{n_crash} of 24 attribution dives crash")
if crash is None:
    json.dump(res, open(OUT, "w"), indent=1); log("no crash: nothing to curate"); raise SystemExit

params = {k: v.detach() for k, v in net.named_parameters()}
Xn = torch.tensor((X - Dr.mu) / Dr.sd, dtype=torch.float32)
Yn = torch.tensor(R / Dr.rs, dtype=torch.float32)
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
flights = np.unique(Dr.FLIGHT)
fbad = np.array([bad[Dr.FLIGHT == f].any() for f in flights])
fdistr = np.array([distr[Dr.FLIGHT == f].any() for f in flights])
fs = np.array([sc.numpy()[Dr.FLIGHT == f].mean() for f in flights])
pos = fs > 0
res["detection"] = dict(auroc=float(roc_auc_score(fbad, fs)), ap=float(average_precision_score(fbad, fs)),
                        sign_removed=int(pos.sum()), sign_bad=int(fbad[pos].sum()),
                        sign_lookalike=int(fdistr[pos].sum()))
log("detection", res["detection"])
json.dump(res, open(OUT, "w"), indent=1)
if args.fly:
    m = Dr.fit(X, R, ~np.isin(Dr.FLIGHT, flights[pos]), seed=args.seed)
    TASKS = {"test dives": dict(AD.TASK), "steep dives": dict(AD.TASK, dive_dx=(0.5, 1.5)),
             "shallow dives": dict(AD.TASK, dive_dx=(5, 8))}
    r = {}
    for name, tk in TASKS.items():
        e = Dr.evaluate(m, c_hi=AD.CFG["c_hi"], delay=3, every=3, T=250, n_each=24, seed=7, **tk)
        r[name] = float(np.mean([x["crash"] for x in e])); r[f"{name} reach"] = float(np.mean([x["reach"] for x in e]))
    res["sign_rule_flown"] = r
    log("sign rule flown", r)
    json.dump(res, open(OUT, "w"), indent=1)
log("done")
