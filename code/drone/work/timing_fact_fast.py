"""FACT's scoring pass without per-sample gradients, for the cost comparison in timing_l2rw.py.

For a linear layer z = W a + b the per-sample gradient is (delta_i a_i^T, delta_i), with delta_i = dl_i/dz_i, so its
squared norm is |delta_i|^2 (|a_i|^2 + 1) and its inner product with a direction (V, c) is delta_i^T V a_i + delta_i^T c.
One forward and one backward pass per batch give every sample's normalised score. The script checks the result
against the per-sample (vmap) scores of timing_l2rw.py on the same model and adds the timing to work/timing_l2rw.json.

usage (from code/drone/, idle machine): NT=1 python work/timing_fact_fast.py
"""
import json
import time

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

import attr_data as AD
import drone as Dr

FAULT, SEED = "prop50j8", 1
Xf, Rf, bad, distr = AD.setup(FAULT)
CFG = AD.CFG
net = Dr.make_net(); net.load_state_dict(torch.load(f"work/model3_{FAULT}_s{SEED}.pt"))
[q.requires_grad_(False) for q in net.parameters()]
BATCH = 2048
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
N = len(Xn)
LIN = [m for m in net if isinstance(m, torch.nn.Linear)]


def val_grad():
    pg = {k: v.clone().requires_grad_(True) for k, v in net.named_parameters()}
    L = ((functional_call(net, pg, (fx,)) - fy) ** 2).sum(-1).mean()
    g = torch.autograd.grad(L, list(pg.values()))
    return [(g[2 * j], g[2 * j + 1]) for j in range(len(LIN))]  # (V, c) per linear layer


def fast_scores(x, y, tg):
    acts, zs, h = [], [], x
    for m in net:
        if isinstance(m, torch.nn.Linear):
            acts.append(h)
            h = (h @ m.weight.T + m.bias).requires_grad_(True) if not zs else h @ m.weight.T + m.bias
            zs.append(h)
        else:
            h = m(h)
    deltas = torch.autograd.grad(((h - y) ** 2).sum(), zs)
    dot = sum(((d @ V) * a).sum(1) + d @ c for d, a, (V, c) in zip(deltas, acts, tg))
    nrm2 = sum((d ** 2).sum(1) * ((a ** 2).sum(1) + 1) for d, a in zip(deltas, acts))
    return -dot / (nrm2.sqrt() + 1e-12)


def vmap_scores(x, y, tgf):
    pd = {k: v.detach() for k, v in net.named_parameters()}
    f = lambda q, a, b: ((functional_call(net, q, (a[None],))[0] - b) ** 2).sum()
    G = vmap(grad(f), in_dims=(None, 0, 0))(pd, x, y)
    G = torch.cat([G[k].flatten(1) for k in pd], 1)
    return -(G @ tgf) / (G.norm(dim=1) + 1e-12)


tg = val_grad()
tgf = torch.cat([t.flatten() for V, c in tg for t in (V, c)])
err = 0.0
for s in range(0, 8 * BATCH, BATCH):
    a, b = fast_scores(Xn[s:s + BATCH], Yn[s:s + BATCH], tg).detach(), vmap_scores(Xn[s:s + BATCH], Yn[s:s + BATCH], tgf)
    err = max(err, float((a - b).abs().max()))
print("max |fast - vmap| score difference:", err, flush=True)

T = []
for _ in range(3):
    t0 = time.perf_counter()
    tg = val_grad()
    sc = torch.cat([fast_scores(Xn[s:s + BATCH], Yn[s:s + BATCH], tg).detach() for s in range(0, N, BATCH)])
    T.append(time.perf_counter() - t0)
flights = np.unique(Dr.FLIGHT)
fidx = np.searchsorted(flights, Dr.FLIGHT)
rm = np.bincount(fidx, sc.numpy(), minlength=len(flights)) / np.bincount(fidx, minlength=len(flights)) > 0
res = json.load(open("work/timing_l2rw.json"))
res["fact_scoring_fast_s"] = float(np.median(T))
res["fact_scoring_fast_max_abs_diff"] = err
res["fact_fast_kept_samples"] = int((~rm[fidx]).sum())
res["fact_fast_total_s"] = res["fact_scoring_fast_s"] + res["fact_retrain_s"]
U = res["updates_per_training"]
res["ratios"].update({f"{k}_training/fact_fast_total": res["s_per_update"][k] * U / res["fact_fast_total_s"]
                      for k in res["s_per_update"] if k != "plain"})
res["ratios"]["fact_fast_total/full_fit"] = res["fact_fast_total_s"] / res["full_fit_s"]
print(json.dumps(res, indent=1))
json.dump(res, open("work/timing_l2rw.json", "w"), indent=1)
