"""How much does rolling scores up per log help? Per-sample TracIn-cos scores from the first crash dive, and the share of
wrong signs per sample, per 2-s window and per flight, with the margin, spread and effective sample size that enter the
concentration bound of the aggregation proposition. Writes work/agg_{fault}_s{seed}.json.

usage: python work/agg_check.py --fault prop50j8 --seed 1
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
p.add_argument("--win", type=int, default=200)  # 2 s at 100 Hz
args = p.parse_args()
t0 = time.time()
Xf, Rf, bad, distr = AD.setup(args.fault)
CFG = AD.CFG
net = Dr.make_net(); net.load_state_dict(torch.load(f"work/model3_{args.fault.split('-')[0]}_s{args.seed}.pt"))
with torch.no_grad():
    for i, (kind, a, b) in enumerate(Dr.tasks(24, seed=11, **AD.TASK)):
        s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
        S, U = AD.run_cl(s0, torch.tensor(b), net, seed=i)
        clr = Dr.clearance(S[:, :3])
        if clr.min() < 0:
            t_c = int(torch.nonzero(clr < 0)[0, 0])
            xs, ys = AD.realized_residual(S[:t_c + 1], U[:t_c], CFG["c_hi"])
            break
params = {n: q.detach() for n, q in net.named_parameters()}
Xn = torch.tensor((Xf - Dr.mu) / Dr.sd, dtype=torch.float32)
Yn = torch.tensor(Rf / Dr.rs, dtype=torch.float32)
g_i = vmap(grad(lambda pd, x, y: ((functional_call(net, pd, (x[None],))[0] - y) ** 2).sum()), in_dims=(None, 0, 0))
pg = {k: v.clone().requires_grad_(True) for k, v in params.items()}
L = ((functional_call(net, pg, ((xs - Dr.MU) / Dr.SD,)) - ys / Dr.RS) ** 2).sum(-1).mean()
tg = torch.autograd.grad(L, list(pg.values()))
tn = torch.sqrt(sum((t ** 2).sum() for t in tg))
sc = torch.zeros(len(Xn))
for s in range(0, len(Xn), 2048):
    G = g_i(params, Xn[s:s + 2048], Yn[s:s + 2048])
    nrm = torch.sqrt(sum((G[k].flatten(1) ** 2).sum(1) for k in G))
    # FACT's score divided by the reference gradient's norm: a cosine in [-1, 1], same sign
    sc[s:s + 2048] = -sum((G[k].flatten(1) * t.flatten()[None]).sum(1) for k, t in zip(G, tg)) / (nrm + 1e-12) / tn
sc = sc.numpy()
F = Dr.FLIGHT
flights = np.unique(F)
out = dict(fault=args.fault, seed=args.seed, crash_transitions=len(xs), win=args.win)
# sign errors: a faulty unit should score > 0, a correct one < 0
out["sample_err_bad"] = float((sc[bad] <= 0).mean()); out["sample_err_good"] = float((sc[~bad] >= 0).mean())
wm, wb = [], []
for f in flights:
    ii = np.where(F == f)[0]
    for s in range(0, len(ii) - args.win + 1, args.win):
        wm.append(sc[ii[s:s + args.win]].mean()); wb.append(bad[ii[s]])
wm, wb = np.array(wm), np.array(wb)
out["window_err_bad"] = float((wm[wb] <= 0).mean()); out["window_err_good"] = float((wm[~wb] >= 0).mean())
fm = np.array([sc[F == f].mean() for f in flights]); fb = np.array([bad[F == f].any() for f in flights])
out["flight_err_bad"] = float((fm[fb] <= 0).mean()); out["flight_err_good"] = float((fm[~fb] >= 0).mean())
# per flight: n samples, |mean| (margin), sample s.d., integrated autocorrelation time from window means (batch means)
per = []
for f, b in zip(flights, fb):
    v = sc[F == f]; n = len(v)
    k = n // args.win
    bm = v[:k * args.win].reshape(k, args.win).mean(1)
    tau = max(1.0, args.win * bm.var(ddof=1) / v.var(ddof=1)) if k > 1 else float("nan")
    neff = n / tau
    m, sd = abs(v.mean()), v.std(ddof=1)
    per.append(dict(flight=int(f), bad=bool(b), n=n, mean=float(v.mean()), sd=float(sd), tau=float(tau), n_eff=float(neff),
                    bound=float(np.exp(-neff * m ** 2 / (2 * sd ** 2))) if np.isfinite(neff) else None))
out["flights"] = per
tau = np.array([q["tau"] for q in per if np.isfinite(q["tau"])])
bnd = np.array([q["bound"] for q in per if q["bound"] is not None])
out["summary"] = dict(tau_median=float(np.median(tau)), n_median=float(np.median([q["n"] for q in per])),
                      neff_median=float(np.median([q["n_eff"] for q in per if np.isfinite(q["n_eff"])])),
                      bound_max=float(bnd.max()), bound_sum=float(bnd.sum()), flights_bound_over_1pct=int((bnd > 0.01).sum()))
json.dump(out, open(f"work/agg_{args.fault}_s{args.seed}.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "flights"}, f"({time.time() - t0:.0f}s)", flush=True)
