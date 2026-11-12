"""Fly one fix of a quadrotor benchmark with per-dive records, for crash rate, safe completion and terminal error.

Models (one crash dive is the failure run, as in mis_drone.py):
  unfixed       the model trained on all logs
  oracle        bad flights removed (work/model3_{fault}_dropharm_s{seed}.pt)
  add           the crash dive added to training with weight W=20, 12 epochs (Dr.fit)
  fact          sign rule, refit with Dr.fit (12 epochs over the kept logs)
  fact_matched  sign rule, retrained for the full-data number of updates (as controls_drone.py)
  gradfree      sign rule with the gradient-free score -<mean failure error, e_i> (the identity in place of the
                tangent kernel), refit with Dr.fit as fact
  l2rw          learning to reweight with the crash dive as clean set, weights from per-sample gradients
  l2rw_la       the same, weights from Ren et al.'s look-ahead (one perturbed SGD step, double backpropagation)
Both reweighting models take the full-data number of updates; timing_e2e.py times complete runs of each fix.

--ref picks the reference FACT scores against (all with the same number of transitions, labelled with the residuals
the plant produced):
  failure   the crash dive, up to impact
  routine   ordinary successful operation: the faulty model's first wall-stop run that does not crash (the full run,
            thinned evenly to the crash dive's length)
  hover     the faulty model's planner holding its start position (no axial descent at all)
  expert    the crash dive's start and goal flown by a planner that uses the plant as its model (standing in for a
            pilot or a conservative controller), over the crash dive's duration
Coverage is the share of reference transitions descending along the rotor axis faster than 1 m/s, where the
propeller fault acts.

Each dive records crash, reach (within 0.5 m of the goal at any step, as in Dr.evaluate), safe completion (reach and
no crash) and the distance to the goal at the last step. Dives, plant and planner are those of mis_drone.py.

usage (from code/drone/): python work/reach_drone.py --fault prop50j8 --model fact --seed 1 [--ref expert]
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
p.add_argument("--fault", default="prop50j8")
p.add_argument("--model", choices=["unfixed", "oracle", "add", "fact", "fact_matched", "gradfree", "l2rw", "l2rw_la"])
p.add_argument("--ref", choices=["failure", "routine", "hover", "expert"], default="failure")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--smoke", action="store_true", help="one dive per task, no output")
args = p.parse_args()
t0 = time.time()
FAULT = args.fault
Xf, Rf, bad, distr = AD.setup(FAULT)
CFG = AD.CFG
net = Dr.make_net(); net.load_state_dict(torch.load(f"work/model3_{FAULT}_s{args.seed}.pt"))
[q.requires_grad_(False) for q in net.parameters()]
TASKS = {"test dives": dict(AD.TASK), "steep dives": dict(AD.TASK, dive_dx=(0.5, 1.5)),
         "shallow dives": dict(AD.TASK, dive_dx=(5, 8))}
BATCH, EPOCH = 2048, -(-len(Xf) // 2048)

with torch.no_grad():
    for i_c, (kind, a, b) in enumerate(Dr.tasks(24, seed=11, **AD.TASK)):
        s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
        S, U = AD.run_cl(s0, torch.tensor(b), net, seed=i_c)
        clr = Dr.clearance(S[:, :3])
        if clr.min() < 0:
            t_c = int(torch.nonzero(clr < 0)[0, 0])
            xs, ys = AD.realized_residual(S[:t_c + 1], U[:t_c], CFG["c_hi"])
            break
    ref_info = dict(ref=args.ref, transitions=len(xs))
    if args.ref in ("routine", "hover"):
        for i, (kind, a, b) in enumerate(Dr.tasks(24, seed=11, kinds=("wall",))):
            b = {"routine": b, "hover": a}[args.ref]
            s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
            S, U = AD.run_cl(s0, torch.tensor(b), net, seed=i)
            if Dr.clearance(S[:, :3]).min() >= 0:
                break
        x_all, y_all = AD.realized_residual(S, U, CFG["c_hi"])
        sel = np.linspace(0, len(x_all) - 1, len(xs)).round().astype(int)
        xs, ys = x_all[sel], y_all[sel]
        ref_info.update(task=i, reached=bool(((S[:, :3] - torch.tensor(b)).norm(dim=-1) < 0.5).any()))
    elif args.ref == "expert":
        S, U = AD.run_cl(s0, torch.tensor(b), Dr.PLANT, seed=i_c)
        xs, ys = AD.realized_residual(S[:t_c + 1], U[:t_c], CFG["c_hi"])
        ref_info.update(task=i_c, crashed=bool(Dr.clearance(S[:, :3]).min() < 0))
    ref_info.update(axial_descent_share=float((xs[:, 2] < -1.0).float().mean()),
                    faulty_err=float(((net((xs - Dr.MU) / Dr.SD) - ys / Dr.RS) ** 2).sum(-1).mean()))
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


def w_per_sample(m, x, y):
    pd = {k: v.detach() for k, v in m.named_parameters()}
    return (per_sample_grads(m, pd, x, y) @ val_grad(m, pd)).clamp(min=0)


def w_lookahead(m, x, y):
    pd = dict(m.named_parameters())
    eps = torch.zeros(len(x), requires_grad=True)
    li = ((functional_call(m, pd, (x,)) - y) ** 2).sum(1)
    g = torch.autograd.grad((eps * li).sum(), list(pd.values()), create_graph=True)
    step = {k: v - gk for (k, v), gk in zip(pd.items(), g)}  # step size 1: it cancels in the normalisation
    Lv = ((functional_call(m, step, (fx,)) - fy) ** 2).sum(-1).mean()
    return (-torch.autograd.grad(Lv, eps)[0]).clamp(min=0)


def train(pool, weigher=None):
    """Dr.fit's loop by update count (controls_drone.train): 12 full-data epochs of updates over `pool` in shuffled
    order, lr x0.8 every EPOCH updates; with a weigher each batch is reweighted."""
    torch.manual_seed(args.seed)
    g = np.random.default_rng(args.seed)
    m = Dr.make_net()
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    idx, pos = np.array(pool), len(pool)
    for t in range(12 * EPOCH):
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
        if (t + 1) % EPOCH == 0:
            for pg in opt.param_groups:
                pg["lr"] *= 0.8
    for q in m.parameters():
        q.requires_grad_(False)
    return m


def sign_rule(gradient_free=False):
    params = {k: v.detach() for k, v in net.named_parameters()}
    sc = torch.zeros(N)
    if gradient_free:
        with torch.no_grad():
            ebar = (net(fx) - fy).mean(0)
            for s in range(0, N, BATCH):
                sc[s:s + BATCH] = -((net(Xn[s:s + BATCH]) - Yn[s:s + BATCH]) @ ebar)
    else:
        tg = val_grad(net, params)
        for s in range(0, N, BATCH):
            G = per_sample_grads(net, params, Xn[s:s + BATCH], Yn[s:s + BATCH])
            sc[s:s + BATCH] = -(G @ tg) / (G.norm(dim=1) + 1e-12)
    flights = np.unique(Dr.FLIGHT)
    fidx = np.searchsorted(flights, Dr.FLIGHT)
    fs = np.bincount(fidx, sc.numpy(), minlength=len(flights)) / np.bincount(fidx, minlength=len(flights))
    rm = fs > 0
    fbad = np.bincount(fidx, bad, minlength=len(flights)) > 0
    return ~rm[fidx], dict(removed=int(rm.sum()), bad_removed=int((rm & fbad).sum()), bad_flights=int(fbad.sum()),
                           auroc=float(roc_auc_score(fbad, fs)))


info = {}
t1 = time.time()
if args.model == "unfixed":
    m = net
elif args.model == "oracle":
    m = Dr.make_net(); m.load_state_dict(torch.load(f"work/model3_{FAULT}_dropharm_s{args.seed}.pt"))
elif args.model == "add":
    Xa = np.concatenate([Xf] + [xs.numpy()] * 20); Ra = np.concatenate([Rf] + [ys.numpy()] * 20)
    m = Dr.fit(Xa, Ra, np.ones(len(Xa), bool), seed=args.seed)
elif args.model in ("fact", "fact_matched", "gradfree"):
    keep, info = sign_rule(gradient_free=args.model == "gradfree")
    info["reference"] = ref_info
    m = train(np.where(keep)[0]) if args.model == "fact_matched" else Dr.fit(Xf, Rf, keep, seed=args.seed)
else:
    m = train(np.arange(N), w_per_sample if args.model == "l2rw" else w_lookahead)
info["build_s_shared_machine"] = round(time.time() - t1, 1)
print(args.model, args.ref, args.seed, info, f"model ready ({time.time() - t0:.0f}s)", flush=True)

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
    tag = args.model if args.ref == "failure" else f"{args.model}-{args.ref}"
    json.dump(dict(args=vars(args), info=info, summary=summ, runs=runs),
              open(f"work/reach_{tag}_{FAULT}_s{args.seed}.json", "w"), indent=1)
