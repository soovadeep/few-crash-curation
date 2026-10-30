"""Wall-clock cost of complete fixes, from the logs and the failure record to a trained model, on one CPU thread.

Each timed run starts from the fault's logs (attr_data.setup) and the recorded crash dive (states and commands), and
ends with the trained model: it includes building the training tensors, the failure's labels, any scoring, and the
whole training. Flying the faulty model (which produced the crash) and evaluating the result are not included.
  plain         a fit on all logs (12 epochs, the standard routine)
  add           the crash dive added to the logs with weight W=20, then the standard routine
  fact          FACT: scoring by one backward pass per batch (timing_fact_fast.py), sign rule, refit on the kept
                logs with the standard routine (12 epochs over the kept logs: the default schedule)
  fact_matched  the same, retrained for as many updates as a fit on all logs (matched updates)
  l2rw_la       learning to reweight with Ren et al.'s look-ahead weights, as many updates as a fit on all logs
  l2rw_jvp      the same weights from one forward-mode product per batch
  l2rw          the same weights from per-sample gradients (propeller only, one repeat: about 12 minutes)
Models and update loops are those of reach_drone.py, which flies them. Median of 3 repeats per fault (seed 1).

usage (from code/drone/, idle machine): NT=1 python work/timing_e2e.py
"""
import json
import platform
import time

import numpy as np
import torch
from torch.func import functional_call, grad, jvp, vmap

import attr_data as AD
import drone as Dr

SEED, BATCH = 1, 2048
FAULTS = ["prop50j8", "gain40j8", "prop50j8m"]


def crash_record(fault):
    AD.setup(fault)
    net = Dr.make_net(); net.load_state_dict(torch.load(f"work/model3_{fault}_s{SEED}.pt"))
    [q.requires_grad_(False) for q in net.parameters()]
    with torch.no_grad():
        for i, (kind, a, b) in enumerate(Dr.tasks(24, seed=11, **AD.TASK)):
            s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
            S, U = AD.run_cl(s0, torch.tensor(b), net, seed=i)
            clr = Dr.clearance(S[:, :3])
            if clr.min() < 0:
                t_c = int(torch.nonzero(clr < 0)[0, 0])
                return net, S[:t_c + 1], U[:t_c]


def prepare(fault, S, U):
    Xf, Rf, bad, distr = AD.setup(fault)
    xs, ys = AD.realized_residual(S, U, AD.CFG["c_hi"])
    Xn = torch.tensor((Xf - Dr.mu) / Dr.sd, dtype=torch.float32)
    Yn = torch.tensor(Rf / Dr.rs, dtype=torch.float32)
    return Xf, Rf, Xn, Yn, (xs - Dr.MU) / Dr.SD, ys / Dr.RS


def fact_keep(net, Xn, Yn, fx, fy):
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    pg = {k: v.clone().requires_grad_(True) for k, v in net.named_parameters()}
    g = torch.autograd.grad(((functional_call(net, pg, (fx,)) - fy) ** 2).sum(-1).mean(), list(pg.values()))
    tg = [(g[2 * j], g[2 * j + 1]) for j in range(len(lin))]
    sc = torch.zeros(len(Xn))
    for s in range(0, len(Xn), BATCH):
        acts, zs, h = [], [], Xn[s:s + BATCH]
        for m in net:
            if isinstance(m, torch.nn.Linear):
                acts.append(h)
                h = (h @ m.weight.T + m.bias).requires_grad_(True) if not zs else h @ m.weight.T + m.bias
                zs.append(h)
            else:
                h = m(h)
        deltas = torch.autograd.grad(((h - Yn[s:s + BATCH]) ** 2).sum(), zs)
        dot = sum(((d @ V) * a).sum(1) + d @ c for d, a, (V, c) in zip(deltas, acts, tg))
        nrm2 = sum((d ** 2).sum(1) * ((a ** 2).sum(1) + 1) for d, a in zip(deltas, acts))
        sc[s:s + BATCH] = (-dot / (nrm2.sqrt() + 1e-12)).detach()
    flights = np.unique(Dr.FLIGHT)
    fidx = np.searchsorted(flights, Dr.FLIGHT)
    rm = np.bincount(fidx, sc.numpy(), minlength=len(flights)) / np.bincount(fidx, minlength=len(flights)) > 0
    return ~rm[fidx]


def train(Xn, Yn, pool, weigher=None):
    """reach_drone.train: 12 full-data epochs of updates over `pool`, lr x0.8 every full-data epoch."""
    epoch = -(-len(Xn) // BATCH)
    torch.manual_seed(SEED)
    g = np.random.default_rng(SEED)
    m = Dr.make_net()
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    idx, pos = np.array(pool), len(pool)
    for t in range(12 * epoch):
        if pos + BATCH > len(idx):
            g.shuffle(idx); pos = 0
        bi = idx[pos:pos + BATCH]; pos += BATCH
        x, y = Xn[bi], Yn[bi]
        if weigher is None:
            loss = ((m(x) - y) ** 2).mean()
        else:
            w = weigher(m, x, y)
            loss = (w / w.sum() * ((m(x) - y) ** 2).sum(1)).sum() if w.sum() > 0 else None
        if loss is not None:
            opt.zero_grad(); loss.backward(); opt.step()
        if (t + 1) % epoch == 0:
            for pgr in opt.param_groups:
                pgr["lr"] *= 0.8
    return m


def run(method, fault, net, S, U):
    Xf, Rf, Xn, Yn, fx, fy = prepare(fault, S, U)

    def w_lookahead(m, x, y):
        pd = dict(m.named_parameters())
        eps = torch.zeros(len(x), requires_grad=True)
        li = ((functional_call(m, pd, (x,)) - y) ** 2).sum(1)
        g = torch.autograd.grad((eps * li).sum(), list(pd.values()), create_graph=True)
        step = {k: v - gk for (k, v), gk in zip(pd.items(), g)}
        Lv = ((functional_call(m, step, (fx,)) - fy) ** 2).sum(-1).mean()
        return (-torch.autograd.grad(Lv, eps)[0]).clamp(min=0)

    def w_jvp(m, x, y):
        pd = {k: v.detach() for k, v in m.named_parameters()}
        pg = {k: v.clone().requires_grad_(True) for k, v in pd.items()}
        v = torch.autograd.grad(((functional_call(m, pg, (fx,)) - fy) ** 2).sum(-1).mean(), list(pg.values()))
        _, d = jvp(lambda q: ((functional_call(m, q, (x,)) - y) ** 2).sum(1), (pd,), (dict(zip(pd, v)),))
        return d.clamp(min=0)

    def w_per_sample(m, x, y):
        pd = {k: v.detach() for k, v in m.named_parameters()}
        pg = {k: v.clone().requires_grad_(True) for k, v in pd.items()}
        v = torch.cat([g.flatten() for g in torch.autograd.grad(((functional_call(m, pg, (fx,)) - fy) ** 2).sum(-1).mean(),
                                                                 list(pg.values()))])
        f = lambda q, a, b: ((functional_call(m, q, (a[None],))[0] - b) ** 2).sum()
        G = vmap(grad(f), in_dims=(None, 0, 0))(pd, x, y)
        return (torch.cat([G[k].flatten(1) for k in pd], 1) @ v).clamp(min=0)

    if method == "plain":
        return Dr.fit(Xf, Rf, np.ones(len(Xf), bool), seed=SEED)
    if method == "add":
        xs, ys = fx * Dr.SD + Dr.MU, fy * Dr.RS
        Xa = np.concatenate([Xf] + [xs.numpy()] * 20); Ra = np.concatenate([Rf] + [ys.numpy()] * 20)
        return Dr.fit(Xa, Ra, np.ones(len(Xa), bool), seed=SEED)
    if method == "fact":
        return Dr.fit(Xf, Rf, fact_keep(net, Xn, Yn, fx, fy), seed=SEED)
    if method == "fact_matched":
        return train(Xn, Yn, np.where(fact_keep(net, Xn, Yn, fx, fy))[0])
    weigher = {"l2rw_la": w_lookahead, "l2rw_jvp": w_jvp, "l2rw": w_per_sample}[method]
    return train(Xn, Yn, np.arange(len(Xn)), weigher)


res = dict(hardware=dict(cpu=open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t"),
                         threads=torch.get_num_threads(), torch=torch.__version__, python=platform.python_version()),
           seed=SEED, repeats=3, seconds={})
for fault in FAULTS:
    net, S, U = crash_record(fault)
    res["seconds"][fault] = {}
    for method in ["plain", "add", "fact", "fact_matched", "l2rw_la", "l2rw_jvp"] + (["l2rw"] if fault == "prop50j8" else []):
        T = []
        for _ in range(1 if method == "l2rw" else 3):
            t0 = time.perf_counter()
            run(method, fault, net, S, U)
            T.append(time.perf_counter() - t0)
        res["seconds"][fault][method] = float(np.median(T))
        print(fault, method, f"{np.median(T):.1f} s (repeats {[round(t, 1) for t in T]})", flush=True)
json.dump(res, open("work/timing_e2e.json", "w"), indent=1)
