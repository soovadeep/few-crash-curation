"""CUPID (Agia et al., CoRL 2025) adapted to MPPI + learned dynamics, on the NeuroBEM drone.

CUPID scores a training demo xi by its influence on the closed-loop return J:
    Psi(xi) = -grad J^T H^-1 grad l(xi),   grad J ~ (1/m) sum_tau R(tau) sum_t grad log pi(a_t | s_t)
with R = +1 for a success and -1 for a failure, m rollouts, no baseline, demo score = sum over its pairs.
Here H = I (as for every other score in this study) and the "policy" is MPPI on the learned model.

Three adaptations, all from the same m rollouts:
  PG   : the faithful policy-gradient form. Rollouts add Gaussian exploration noise (sigma = SIGF * MPPI
         noise) to each new command, so log pi(a|s) = -|a - mu_theta(s)|^2 / 2 sigma^2, where mu_theta is the
         MPPI command (one planning step differentiated, states held fixed: no backprop through the closed
         loop). PGb adds the mean-return baseline, which CUPID does not use.
  sq   : CUPID's practical surrogate: grad log pi replaced by grad of l_sq = |f_theta(x)|^2 (no labels), on
         both the rollout transitions and the training samples.
  pred : grad log pi replaced by the model's prediction loss on the rollout transitions (plant labels).
Harmful = -Psi (high = remove first). Flights are scored by the sum over their samples (CUPID) and the mean.
Then CUPID's own curation step: remove the top-k flights once, retrain, fly the test, steep and shallow dives.

usage: python work/cupid_drone.py --fault prop50v --seed 1
"""
import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.func import functional_call, grad, vmap

import attr_data as AD
import drone as Dr

p = argparse.ArgumentParser()
p.add_argument("--fault", default="prop50v")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--m", type=int, default=100)
p.add_argument("--sigf", type=float, default=0.5)
p.add_argument("--ks", default="10,20,30")
args = p.parse_args()
t0 = time.time()
Xf, Rf, bad, distr = AD.setup(args.fault)
CFG = AD.CFG
TASKS = {"test dives": dict(AD.TASK), "steep dives": dict(AD.TASK, dive_dx=(0.5, 1.5)),
         "shallow dives": dict(AD.TASK, dive_dx=(5, 8))}
net = Dr.make_net()
net.load_state_dict(torch.load(f"work/model3_{args.fault}_s{args.seed}.pt"))
params = {n: q.detach().clone().requires_grad_(True) for n, q in net.named_parameters()}
model = lambda x: functional_call(net, params, (x,))
SIGMA = Dr.SIG * args.sigf


def log(*a):
    print(*a, f"({time.time() - t0:.0f}s)", flush=True)


def rollout_pg(s0, goal, seed, T=250, topm=64, lam_att=20.0):
    """One noisy MPPI rollout. Returns R, per-rollout grad of sum_t (eps_t / sigma) . mu_theta(s_t),
    and the executed transitions (states, applied commands)."""
    c_hi, delay, every = CFG["c_hi"], CFG["delay"], CFG["every"]
    gen = torch.Generator().manual_seed(seed)
    gx = torch.Generator().manual_seed(10_000 + seed)
    Unom = torch.zeros(Dr.H, 4); Unom[:, 0] = Dr.G
    queue = [Unom[0].clone() for _ in range(delay)]
    s = s0.clone(); traj, applied = [s], []
    plan, k = None, 0
    obj = torch.zeros(())
    crashed = False
    for t in range(T):
        if t % every == 0:
            xs = s
            for u in queue:
                xs = Dr.step(xs, u, model, c_hi)
            eps = torch.randn(Dr.K, Dr.H, 4, generator=gen) * Dr.SIG
            eps[0] = 0
            Uk = Unom[None] + eps
            with torch.no_grad():
                x = xs.detach()[None].expand(Dr.K, 13); S_ng = torch.zeros(Dr.K)
                for h in range(Dr.H):
                    x = Dr.step(x, Uk[:, h], model, c_hi)
                    S_ng = S_ng + Dr.cost(x, Uk[:, h], goal)
            idx = torch.topk(-S_ng, topm).indices
            Ut = Uk[idx]
            x = xs[None].expand(topm, 13); St = torch.zeros(topm)
            for h in range(Dr.H):
                x = Dr.step(x, Ut[:, h], model, c_hi)
                St = St + Dr.cost(x, Ut[:, h], goal)
            Sc = S_ng.scatter(0, idx, St)
            w_f = torch.softmax(-(Sc.detach() - Sc.detach().min()) / Dr.LAM, 0)
            w_a = torch.softmax(-(Sc - Sc.detach().min()) / lam_att, 0)
            U_f = (w_f[:, None, None] * Uk).sum(0); U_a = (w_a[:, None, None] * Uk).sum(0)
            plan = U_f + (U_a - U_a.detach())
            k = 0
        mu = plan[min(k, Dr.H - 1)]; k += 1
        z = torch.randn(4, generator=gx)
        a = mu.detach() + SIGMA * z
        obj = obj + ((z / SIGMA) * mu).sum()        # d/dtheta log N(a; mu_theta, sigma^2)
        queue.append(a)
        u = queue.pop(0)
        applied.append(u)
        with torch.no_grad():
            s = Dr.step(s, u, Dr.PLANT, c_hi)
        traj.append(s)
        if t % every == every - 1:
            Unom = torch.cat([plan[every:], plan[-1:].expand(every, 4)], 0).detach()
        if Dr.clearance(s[:3]) < 0:
            crashed = True
            break
    g = torch.autograd.grad(obj, list(params.values()))
    return (-1.0 if crashed else 1.0), torch.cat([x.flatten() for x in g]), torch.stack(traj[:-1]), torch.stack(applied)


# ------------------------------------------------------------------ rollouts
tasks = Dr.tasks(args.m, seed=11, **AD.TASK)
Rs, Gs, Sx, Ux, Rx = [], [], [], [], []
for i, (kind, a, b) in enumerate(tasks):
    s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
    R, g, S, U = rollout_pg(s0, torch.tensor(b), seed=i)
    Rs.append(R); Gs.append(g); Sx.append(S); Ux.append(U); Rx.append(torch.full((len(U),), R))
    if i % 10 == 9:
        log(f"rollout {i + 1}/{args.m}: {sum(r < 0 for r in Rs)} crashed so far")
Rs = torch.tensor(Rs); Gs = torch.stack(Gs)
Sx, Ux, Rx = torch.cat(Sx), torch.cat(Ux), torch.cat(Rx)
n_crash = int((Rs < 0).sum())
log(f"{n_crash}/{args.m} noisy rollouts crashed")

# rollout-side features and labels
cc = Ux[:, 0].clamp(Dr.C_LO, CFG["c_hi"])
feat = (torch.cat([Dr.rotate_inv(Sx[:, 6:10], Sx[:, 3:6]), Sx[:, 10:13], cc[:, None]], -1) - Dr.MU) / Dr.SD
with torch.no_grad():
    yt = Dr.PLANT(feat)


def unflat(v):
    out, o = [], 0
    for q in params.values():
        out.append(v[o:o + q.numel()].view_as(q)); o += q.numel()
    return out


pdet = {n: q.detach() for n, q in params.items()}
g_pred = vmap(grad(lambda pd, x, y: ((functional_call(net, pd, (x[None],))[0] - y) ** 2).sum()), in_dims=(None, 0, 0))
g_sq = vmap(grad(lambda pd, x: (functional_call(net, pd, (x[None],))[0] ** 2).sum()), in_dims=(None, 0))


def flat(G):
    return torch.cat([G[n].flatten(1) for n in G], 1)


v_pred = torch.zeros(Gs.shape[1]); v_sq = torch.zeros(Gs.shape[1])
for s in range(0, len(feat), 2048):
    sl = slice(s, s + 2048)
    v_pred += (Rx[sl, None] * flat(g_pred(pdet, feat[sl], yt[sl]))).sum(0)
    v_sq += (Rx[sl, None] * flat(g_sq(pdet, feat[sl]))).sum(0)
v_pred /= args.m; v_sq /= args.m
gJ = (Rs[:, None] * Gs).mean(0)
gJb = ((Rs - Rs.mean())[:, None] * Gs).mean(0)

# ------------------------------------------------------------------ per-sample scores (harmful = -Psi)
Xn = torch.tensor((Xf - Dr.mu) / Dr.sd, dtype=torch.float32)
Yn = torch.tensor(Rf / Dr.rs, dtype=torch.float32)
N = len(Xn)
sc = {k: np.zeros(N) for k in ["PG", "PGb", "pred", "sq"]}
for s in range(0, N, 2048):
    sl = slice(s, s + 2048)
    Gp = flat(g_pred(pdet, Xn[sl], Yn[sl]))
    Gq = flat(g_sq(pdet, Xn[sl]))
    # Psi_i = -gradJ . grad l_i  (PG: grad J estimated directly); harmful = gradJ . grad l_i
    sc["PG"][sl] = (Gp @ gJ).numpy()
    sc["PGb"][sl] = (Gp @ gJb).numpy()
    # Psi_i = sum_tau R sum_t grad l'(x'_t) . grad l(x_i); harmful = -Psi_i
    sc["pred"][sl] = (-(Gp @ v_pred)).numpy()
    sc["sq"][sl] = (-(Gq @ v_sq)).numpy()
log("per-sample scores done")

flights = np.unique(Dr.FLIGHT)
fidx = np.searchsorted(flights, Dr.FLIGHT)
fbad = np.array([bad[Dr.FLIGHT == f].any() for f in flights])
fdistr = np.array([distr[Dr.FLIGHT == f].any() for f in flights])
cnt = np.bincount(fidx, minlength=len(flights))
fs = {}
for k, v in sc.items():
    s_ = np.bincount(fidx, weights=v, minlength=len(flights))
    fs[f"CUPID-{k} sum"] = s_
    fs[f"CUPID-{k} mean"] = s_ / cnt
try:
    z = np.load(f"work/attr_{args.fault}_s{args.seed}.npz")
    fs["ours: prediction loss on crashes"] = z["Traj__TracIn-cos"]
except FileNotFoundError:
    pass  # newer benchmarks score ours in mis_drone.py
det = {}
for k, v in fs.items():
    o = np.argsort(-v)
    det[k] = dict(auroc=float(roc_auc_score(fbad, v)), ap=float(average_precision_score(fbad, v)),
                  bad_top12=int(fbad[o[:12]].sum()), decoys_top20=int(fdistr[o[:20]].sum()),
                  bad_in_top_nbad=int(fbad[o[:fbad.sum()]].sum()))
    log(f"  {k:36s} AUROC {det[k]['auroc']:.2f} AP {det[k]['ap']:.2f} bad in top 12 {det[k]['bad_top12']:2d} decoys in top 20 {det[k]['decoys_top20']}")
tag = f"{args.fault}_s{args.seed}"
np.savez(f"work/cupid_{tag}.npz", flights=flights, fbad=fbad, fdistr=fdistr, **{k.replace(" ", "_"): v for k, v in fs.items()})
out = dict(args=vars(args), n_rollouts=args.m, n_crash=n_crash, detection=det, removal=[])
json.dump(out, open(f"work/cupid_{tag}.json", "w"), indent=1)

# ------------------------------------------------------------------ CUPID's curation: filter top-k once
for m in ["CUPID-PG sum", "CUPID-sq sum"]:
    for k in [int(x) for x in args.ks.split(",") if x]:
        drop = flights[np.argsort(-fs[m])[:k]]
        keep = ~np.isin(Dr.FLIGHT, drop)
        net_k = Dr.fit(Xf, Rf, keep, seed=args.seed)
        rec = dict(method=m, k=k, bad_removed=int(np.isin(drop, flights[fbad]).sum()),
                   decoys_removed=int(np.isin(drop, flights[fdistr]).sum()))
        # test dives as before (crash, reach), plus steep and shallow dives as in mis_drone.py
        for name, tk in TASKS.items():
            r = Dr.evaluate(net_k, c_hi=CFG["c_hi"], delay=3, every=3, T=250, n_each=24, seed=7, **tk)
            c, rc = float(np.mean([x["crash"] for x in r])), float(np.mean([x["reach"] for x in r]))
            if name == "test dives":
                rec.update(crash=c, reach=rc)
            else:
                rec[name], rec[f"{name} reach"] = c, rc
        out["removal"].append(rec)
        log(rec)
        json.dump(out, open(f"work/cupid_{tag}.json", "w"), indent=1)
log("done")
