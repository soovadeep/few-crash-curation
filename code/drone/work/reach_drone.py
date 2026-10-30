"""Re-fly three propeller-benchmark models with per-dive records, for safe completion and terminal error.

  fact     sign rule from the first crash dive (as in mis_drone.py), refit with Dr.fit
  oracle   bad flights removed (work/model3_prop50j8_dropharm_s{seed}.pt)
  l2rw     learning to reweight with the crash dive as clean set, trained as in controls_drone.py (per-sample gradients)

Each dive records crash, reach (within 0.5 m of the goal at any step, as in Dr.evaluate), safe completion (reach and
no crash) and the distance to the goal at the last step. Dives, plant and planner are those of mis_drone.py.

usage (from code/drone/): python work/reach_drone.py --model fact --seed 1
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
p.add_argument("--model", choices=["fact", "oracle", "l2rw"])
p.add_argument("--seed", type=int, default=1)
p.add_argument("--smoke", action="store_true", help="oracle model, one dive per task, no output")
args = p.parse_args()
t0 = time.time()
FAULT = "prop50j8"
Xf, Rf, bad, distr = AD.setup(FAULT)
CFG = AD.CFG
net = Dr.make_net(); net.load_state_dict(torch.load(f"work/model3_{FAULT}_s{args.seed}.pt"))
[q.requires_grad_(False) for q in net.parameters()]
TASKS = {"test dives": dict(AD.TASK), "steep dives": dict(AD.TASK, dive_dx=(0.5, 1.5)),
         "shallow dives": dict(AD.TASK, dive_dx=(5, 8))}
BATCH, EPOCH = 2048, -(-len(Xf) // 2048)

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


def per_sample_grads(model, pd, x, y):
    f = lambda q, a, b: ((functional_call(model, q, (a[None],))[0] - b) ** 2).sum()
    G = vmap(grad(f), in_dims=(None, 0, 0))(pd, x, y)
    return torch.cat([G[k].flatten(1) for k in pd], 1)


def val_grad(model, pd):
    pg = {k: v.clone().requires_grad_(True) for k, v in pd.items()}
    L = ((functional_call(model, pg, (fx,)) - fy) ** 2).sum(-1).mean()
    return torch.cat([g.flatten() for g in torch.autograd.grad(L, list(pg.values()))])


info = {}
if args.model == "oracle":
    m = Dr.make_net(); m.load_state_dict(torch.load(f"work/model3_{FAULT}_dropharm_s{args.seed}.pt"))
elif args.model == "fact":
    params = {k: v.detach() for k, v in net.named_parameters()}
    tg = val_grad(net, params)
    sc = torch.zeros(N)
    for s in range(0, N, BATCH):
        G = per_sample_grads(net, params, Xn[s:s + BATCH], Yn[s:s + BATCH])
        sc[s:s + BATCH] = -(G @ tg) / (G.norm(dim=1) + 1e-12)
    flights = np.unique(Dr.FLIGHT)
    fidx = np.searchsorted(flights, Dr.FLIGHT)
    rm = np.bincount(fidx, sc.numpy(), minlength=len(flights)) / np.bincount(fidx, minlength=len(flights)) > 0
    fbad = np.bincount(fidx, bad, minlength=len(flights)) > 0
    info = dict(removed=int(rm.sum()), bad_removed=int((rm & fbad).sum()))
    m = Dr.fit(Xf, Rf, ~rm[fidx], seed=args.seed)
else:  # controls_drone.train(ALL, l2rw=True)
    torch.manual_seed(args.seed)
    g = np.random.default_rng(args.seed)
    m = Dr.make_net()
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    idx, pos = np.arange(N), N
    for t in range(12 * EPOCH):
        if pos + BATCH > len(idx):
            g.shuffle(idx); pos = 0
        bi = idx[pos:pos + BATCH]; pos += BATCH
        x, y = Xn[bi], Yn[bi]
        pd = {k: v.detach() for k, v in m.named_parameters()}
        w = (per_sample_grads(m, pd, x, y) @ val_grad(m, pd)).clamp(min=0)
        if w.sum() > 0:
            loss = (w / w.sum() * ((m(x) - y) ** 2).sum(1)).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        if (t + 1) % EPOCH == 0:
            for pg in opt.param_groups:
                pg["lr"] *= 0.8
    for q in m.parameters():
        q.requires_grad_(False)
print(args.model, args.seed, info, f"model ready ({time.time() - t0:.0f}s)", flush=True)

runs = {}
for name, tk in TASKS.items():
    out = []
    kw = dict(tk); kinds = kw.pop("kinds", ("dive", "wall"))  # as Dr.evaluate(..., n_each=24, seed=7, **tk)
    for i, (kind, a, b) in enumerate(Dr.tasks(24, 7, kinds, **kw)[:1 if args.smoke else None]):
        s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
        S = Dr.run(s0, torch.tensor(b), m, CFG["c_hi"], 3, 3, 250, seed=i)
        pos_ = S[:, :3]
        d = (pos_ - torch.tensor(b)).norm(dim=-1)
        crash = bool(Dr.clearance(pos_).min() < 0)
        reach = bool((d < 0.5).any())
        out.append(dict(kind=kind, crash=crash, reach=reach, safe=reach and not crash, final_dist=float(d[-1]), min_dist=float(d.min()),
                        steps=len(S), first_reach=int(torch.nonzero(d < 0.5)[0, 0]) if reach else None))
    runs[name] = out
summ = {name: dict(crash=float(np.mean([r["crash"] for r in v])), reach=float(np.mean([r["reach"] for r in v])),
                   safe=float(np.mean([r["safe"] for r in v])),
                   final_dist_nocrash=float(np.median([r["final_dist"] for r in v if not r["crash"]] or [np.nan])))
        for name, v in runs.items()}
print(json.dumps(summ), f"({time.time() - t0:.0f}s)", flush=True)
if not args.smoke:
    json.dump(dict(args=vars(args), info=info, summary=summ, runs=runs),
              open(f"work/reach_{args.model}_{FAULT}_s{args.seed}.json", "w"), indent=1)
