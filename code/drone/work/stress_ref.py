"""Reference rows for the robustness stress tests: fly the unfixed faulty model and the oracle (bad flights removed)
on the same test, steep and shallow dives as sweep_drone.py. Same setup as sweep_drone.py.

Setup (sweep_drone.py). Robustness of FACT's TracIn score: vary the fault severity, the number of faulty flights and the number of
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
p.add_argument("--threads", type=int, default=1)
args = p.parse_args()
torch.set_num_threads(args.threads)
t0 = time.time()
tag = f"sev{args.sev}_n{args.nper}_l{args.look}_s{args.seed}"
OUT = f"work/stressref_{tag}.json"


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
    raise SystemExit(f"missing {mp}")
log("faulty model ready", {k: v for k, v in res.items() if k != "args"})


TASKS = {"test dives": dict(AD.TASK), "steep dives": dict(AD.TASK, dive_dx=(0.5, 1.5)),
         "shallow dives": dict(AD.TASK, dive_dx=(5, 8))}


def fly(m):
    r = {}
    for name, tk in TASKS.items():
        e = Dr.evaluate(m, c_hi=AD.CFG["c_hi"], delay=3, every=3, T=250, n_each=24, seed=7, **tk)
        r[name] = float(np.mean([x["crash"] for x in e])); r[f"{name} reach"] = float(np.mean([x["reach"] for x in e]))
    return r


res["unfixed"] = fly(net); log("unfixed", res["unfixed"])
json.dump(res, open(OUT, "w"), indent=1)
res["oracle"] = fly(Dr.fit(X, R, ~bad, seed=args.seed)); log("oracle", res["oracle"])
json.dump(res, open(OUT, "w"), indent=1)
log("done")
