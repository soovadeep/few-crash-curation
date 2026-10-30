"""Wall-clock cost of FACT and of learning to reweight (Ren et al. 2018) on the drone propeller benchmark, seed 1.

Learning to reweight weights each sample of a minibatch by max(<grad L_val, grad l_i>, 0), normalised. Three ways to
get those weights, all exact:
  per-sample   per-sample gradients by torch.func.vmap, dotted with grad L_val (the implementation in controls_drone.py)
  look-ahead   Ren et al.'s implementation: perturb each sample's loss by eps_i, take one SGD step, differentiate the
               validation loss after the step with respect to eps at eps = 0 (double backpropagation)
  jvp          per-sample losses differentiated along grad L_val in one forward-mode pass
Each update then takes a step on the weighted loss. The script checks that the three give the same weights, and
times them against a plain update, FACT's scoring pass and FACT's retraining (Dr.fit, 12 epochs, kept data).

usage (from code/drone/, idle machine): NT=1 python work/timing_l2rw.py
"""
import json
import os
import platform
import time

import numpy as np
import torch
from torch.func import functional_call, grad, jvp, vmap

import attr_data as AD
import drone as Dr

FAULT, SEED = "prop50j8", 1
Xf, Rf, bad, distr = AD.setup(FAULT)
CFG = AD.CFG
net = Dr.make_net(); net.load_state_dict(torch.load(f"work/model3_{FAULT}_s{SEED}.pt"))
[q.requires_grad_(False) for q in net.parameters()]
BATCH, EPOCH = 2048, -(-len(Xf) // 2048)

with torch.no_grad():  # the first crash dive of the faulty model, as in controls_drone.py
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


def per_sample_grads(model, pd, x, y):
    f = lambda q, a, b: ((functional_call(model, q, (a[None],))[0] - b) ** 2).sum()
    G = vmap(grad(f), in_dims=(None, 0, 0))(pd, x, y)
    return torch.cat([G[k].flatten(1) for k in pd], 1)


def val_grad_dict(model, pd):
    pg = {k: v.detach().clone().requires_grad_(True) for k, v in pd.items()}
    L = ((functional_call(model, pg, (fx,)) - fy) ** 2).sum(-1).mean()
    return dict(zip(pg, torch.autograd.grad(L, list(pg.values()))))


def w_per_sample(m, x, y):
    pd = {k: v.detach() for k, v in m.named_parameters()}
    v = torch.cat([g.flatten() for g in val_grad_dict(m, pd).values()])
    return (per_sample_grads(m, pd, x, y) @ v).clamp(min=0)


def w_lookahead(m, x, y):
    pd = dict(m.named_parameters())
    eps = torch.zeros(len(x), requires_grad=True)
    li = ((functional_call(m, pd, (x,)) - y) ** 2).sum(1)
    g = torch.autograd.grad((eps * li).sum(), list(pd.values()), create_graph=True)
    step = {k: v - gk for (k, v), gk in zip(pd.items(), g)}  # step size 1: it cancels in the normalisation
    Lv = ((functional_call(m, step, (fx,)) - fy) ** 2).sum(-1).mean()
    return (-torch.autograd.grad(Lv, eps)[0]).clamp(min=0)


def w_jvp(m, x, y):
    pd = {k: v.detach() for k, v in m.named_parameters()}
    v = val_grad_dict(m, pd)
    _, d = jvp(lambda q: ((functional_call(m, q, (x,)) - y) ** 2).sum(1), (pd,), (v,))
    return d.clamp(min=0)


def make():
    torch.manual_seed(SEED)
    m = Dr.make_net()
    return m, torch.optim.Adam(m.parameters(), 2e-3)


g = np.random.default_rng(SEED)
BI = [torch.as_tensor(g.choice(N, BATCH, replace=False)) for _ in range(64)]


def time_updates(weigher, n):
    """Seconds per update over n updates (after 3 warm-up updates)."""
    m, opt = make()
    for t in range(n + 3):
        if t == 3:
            t0 = time.perf_counter()
        x, y = Xn[BI[t % len(BI)]], Yn[BI[t % len(BI)]]
        if weigher is None:
            loss = ((m(x) - y) ** 2).mean()
        else:
            w = weigher(m, x, y)
            loss = (w / w.sum().clamp(min=1e-12) * ((m(x) - y) ** 2).sum(1)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return (time.perf_counter() - t0) / n


res = dict(hardware=dict(cpu=open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t"),
                         threads=torch.get_num_threads(), torch=torch.__version__, python=platform.python_version()),
           samples=N, batch=BATCH, updates_per_training=12 * EPOCH, crash_transitions=len(fx),
           params=sum(q.numel() for q in net.parameters()))

# --- the three weightings agree (on a partly trained model and a few batches)
m, opt = make()
for t in range(50):
    x, y = Xn[BI[t]], Yn[BI[t]]
    loss = ((m(x) - y) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
err = {"lookahead": 0.0, "jvp": 0.0}
for t in range(4):
    x, y = Xn[BI[t]], Yn[BI[t]]
    w0 = w_per_sample(m, x, y)
    for k, f in (("lookahead", w_lookahead), ("jvp", w_jvp)):
        w = f(m, x, y).detach()
        err[k] = max(err[k], float((w / w.sum() - w0 / w0.sum()).abs().max() / (w0 / w0.sum()).max()))
res["max_rel_weight_diff"] = err
print("weights agree:", err, flush=True)

# --- seconds per update (median of 3 repeats)
per = {}
for name, f, n in (("plain", None, 200), ("l2rw_per_sample", w_per_sample, 20), ("l2rw_lookahead", w_lookahead, 100),
                   ("l2rw_jvp", w_jvp, 100)):
    per[name] = float(np.median([time_updates(f, n) for _ in range(3)]))
    print(name, f"{1e3 * per[name]:.1f} ms/update", flush=True)
res["s_per_update"] = per

# --- FACT: scoring pass over all samples, then Dr.fit on the kept flights
params = {k: v.detach() for k, v in net.named_parameters()}
T = []
for _ in range(3):
    t0 = time.perf_counter()
    tg = torch.cat([g.flatten() for g in val_grad_dict(net, params).values()])
    sc = torch.zeros(N)
    for s in range(0, N, BATCH):
        G = per_sample_grads(net, params, Xn[s:s + BATCH], Yn[s:s + BATCH])
        sc[s:s + BATCH] = -(G @ tg) / (G.norm(dim=1) + 1e-12)
    T.append(time.perf_counter() - t0)
res["fact_scoring_s"] = float(np.median(T))
flights = np.unique(Dr.FLIGHT)
fidx = np.searchsorted(flights, Dr.FLIGHT)
rm = np.bincount(fidx, sc.numpy(), minlength=len(flights)) / np.bincount(fidx, minlength=len(flights)) > 0
keep = ~rm[fidx]
res["fact_kept_samples"] = int(keep.sum())
t0 = time.perf_counter(); Dr.fit(Xf, Rf, keep, seed=SEED); res["fact_retrain_s"] = time.perf_counter() - t0
t0 = time.perf_counter(); Dr.fit(Xf, Rf, np.ones(N, bool), seed=SEED); res["full_fit_s"] = time.perf_counter() - t0
print({k: res[k] for k in ("fact_scoring_s", "fact_retrain_s", "full_fit_s")}, flush=True)

U = res["updates_per_training"]
res["training_s"] = {k: v * U for k, v in per.items()}
res["fact_total_s"] = res["fact_scoring_s"] + res["fact_retrain_s"]
res["ratios"] = {**{f"{k}/plain": v / per["plain"] for k, v in per.items() if k != "plain"},
                 **{f"{k}_training/fact_total": v * U / res["fact_total_s"] for k, v in per.items() if k != "plain"},
                 "fact_total/full_fit": res["fact_total_s"] / res["full_fit_s"]}
print(json.dumps(res, indent=1))
json.dump(res, open("work/timing_l2rw.json", "w"), indent=1)
