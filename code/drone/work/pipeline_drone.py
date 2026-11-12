"""How much FACT depends on the measurement-to-target pipeline: the same NeuroBEM logs, processed three other ways.

The training targets are the body-frame specific force from NeuroBEM's processed logs (filtered by the dataset
authors from Vicon and onboard data, 400 Hz, kept at every fourth sample). Each variant changes only how the logs'
targets are made; the plant, and so the deployed robot and its crash dive, stay the same.
  filtered     a further centred moving average over 5 samples (50 ms) within each segment
  shifted      targets one sample (10 ms) late relative to the inputs, a synchronisation error
  differenced  targets differentiated by us: central differences of the world-frame velocity, plus gravity,
               rotated into the body frame
Then the propeller fault, the faulty model, the first crash dive, FACT's sign rule and (with --fly) its model flown
on the evaluation dives, as in reach_drone.py.

usage (from code/drone/): python work/pipeline_drone.py --variant filtered --seed 1 [--fault prop50j8] [--fly]
"""
import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.func import functional_call, grad, vmap

import attr_data as AD
import drone as Dr

p = argparse.ArgumentParser()
p.add_argument("--variant", choices=["filtered", "shifted", "differenced"])
p.add_argument("--fault", default="prop50j8")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--fly", action="store_true")
args = p.parse_args()
t0 = time.time()

F0 = Dr.F.copy()
segs = [np.where(Dr.SEG == s)[0] for s in np.unique(Dr.SEG)]
F = F0.copy()
if args.variant == "filtered":
    for ii in segs:
        for j in range(3):
            F[ii, j] = np.convolve(np.pad(F0[ii, j], 2, mode="edge"), np.ones(5) / 5, mode="valid")
elif args.variant == "shifted":
    for ii in segs:
        F[ii[1:]] = F0[ii[:-1]]
else:
    q = torch.tensor(Dr.QA, dtype=torch.float64)
    vw = Dr.rotate(q, torch.tensor(Dr.V, dtype=torch.float64)).numpy()
    t = Dr.D[:, Dr.cols.index("t")]
    aw = np.zeros_like(vw)
    for ii in segs:
        aw[ii] = np.gradient(vw[ii], t[ii], axis=0)
    F = Dr.rotate_inv(q, torch.tensor(aw + np.array([0.0, 0.0, Dr.G]))).numpy()
Dr.F = F
Dr.R = F - np.stack([0 * Dr.C, 0 * Dr.C, Dr.C], 1)
change = dict(rms_change=[float(x) for x in np.sqrt(((F - F0) ** 2).mean(0))], rms_target=[float(x) for x in F0.std(0)])

Xf, Rf, bad, distr = AD.setup(args.fault)
CFG = AD.CFG
net = Dr.fit(Xf, Rf, np.ones(len(Xf), bool), seed=args.seed)
with torch.no_grad():
    for i, (kind, a, b) in enumerate(Dr.tasks(24, seed=11, **AD.TASK)):
        s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
        S, U = AD.run_cl(s0, torch.tensor(b), net, seed=i)
        clr = Dr.clearance(S[:, :3])
        if clr.min() < 0:
            t_c = int(torch.nonzero(clr < 0)[0, 0])
            xs, ys = AD.realized_residual(S[:t_c + 1], U[:t_c], CFG["c_hi"])
            break
fx, fy = (xs - Dr.MU) / Dr.SD, ys / Dr.RS
Xn = torch.tensor((Xf - Dr.mu) / Dr.sd, dtype=torch.float32)
Yn = torch.tensor(Rf / Dr.rs, dtype=torch.float32)
params = {k: v.detach() for k, v in net.named_parameters()}
pg = {k: v.clone().requires_grad_(True) for k, v in params.items()}
L = ((functional_call(net, pg, (fx,)) - fy) ** 2).sum(-1).mean()
tg = torch.cat([g.flatten() for g in torch.autograd.grad(L, list(pg.values()))])
f = lambda q_, a_, b_: ((functional_call(net, q_, (a_[None],))[0] - b_) ** 2).sum()
sc = torch.zeros(len(Xn))
for s in range(0, len(Xn), 2048):
    G = vmap(grad(f), in_dims=(None, 0, 0))(params, Xn[s:s + 2048], Yn[s:s + 2048])
    G = torch.cat([G[k].flatten(1) for k in params], 1)
    sc[s:s + 2048] = -(G @ tg) / (G.norm(dim=1) + 1e-12)
flights = np.unique(Dr.FLIGHT)
fidx = np.searchsorted(flights, Dr.FLIGHT)
fs = np.bincount(fidx, sc.numpy(), minlength=len(flights)) / np.bincount(fidx, minlength=len(flights))
fbad = np.bincount(fidx, bad, minlength=len(flights)) > 0
rm = fs > 0
info = dict(change, crash_dive=i, crash_transitions=len(xs), removed=int(rm.sum()), bad_removed=int((rm & fbad).sum()),
            bad_flights=int(fbad.sum()), auroc=float(roc_auc_score(fbad, fs)))
print(args.variant, args.seed, info, f"({time.time() - t0:.0f}s)", flush=True)

summ = {}
if args.fly:
    m = Dr.fit(Xf, Rf, ~rm[fidx], seed=args.seed)
    for name, tk in {"test dives": dict(AD.TASK), "steep dives": dict(AD.TASK, dive_dx=(0.5, 1.5)),
                     "shallow dives": dict(AD.TASK, dive_dx=(5, 8))}.items():
        r = Dr.evaluate(m, c_hi=CFG["c_hi"], delay=3, every=3, T=250, n_each=24, seed=7, **tk)
        summ[name] = dict(crash=float(np.mean([x["crash"] for x in r])), reach=float(np.mean([x["reach"] for x in r])),
                          safe=float(np.mean([x["reach"] and not x["crash"] for x in r])))
    print(json.dumps(summ), f"({time.time() - t0:.0f}s)", flush=True)
json.dump(dict(args=vars(args), info=info, summary=summ),
          open(f"work/pipeline_{args.variant}_{args.fault}_s{args.seed}.json", "w"), indent=1)
