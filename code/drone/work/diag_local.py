"""Diagnostics for local_drone.py (same setup): why sub-flight sign rules fail.

Setup. Within-flight fault: the propeller fault (attr_data.prop_fault) starts partway through each faulty flight, so the
same flight holds correct transitions (before the onset) and faulty ones (after it). Faulty flights are the nper with
the most axial descent per day; each one's onset is drawn uniformly in [lo, hi] of its length (fixed rng, the same
for every seed). FACT scores every sample from the first crash of the 24 attribution dives, as in sweep_drone.py,
and removes data by the sign rule at three granularities: whole flights, windows of `win` consecutive samples, and
single samples. Flown against the unfixed model and two oracles: whole faulty flights removed, and exactly the
post-onset samples removed.

usage: python work/local_drone.py --seed 1
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.func import functional_call, grad, vmap

import attr_data as AD
import drone as Dr

p = argparse.ArgumentParser()
p.add_argument("--sev", type=int, default=50, help="extra thrust in axial descent, %")
p.add_argument("--nper", type=int, default=11, help="faulty flights per day")
p.add_argument("--lo", type=float, default=0.25, help="earliest onset, fraction of the flight")
p.add_argument("--hi", type=float, default=0.5, help="latest onset")
p.add_argument("--win", type=int, default=200, help="window length in samples (100 Hz)")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--threads", type=int, default=1)
args = p.parse_args()
torch.set_num_threads(args.threads)
t0 = time.time()
tag = f"sev{args.sev}_n{args.nper}_on{int(100 * args.lo)}-{int(100 * args.hi)}_s{args.seed}"
OUT = f"work/localdiag_{tag}.json"


def log(*a):
    print(*a, f"({time.time() - t0:.0f}s)", flush=True)


faulty = AD.prop_flights(args.nper)
rng = np.random.default_rng(0)
bad = np.zeros(len(Dr.FLIGHT), bool)
onset = {}
for f in faulty:
    idx = np.where(Dr.FLIGHT == f)[0]
    on = int(len(idx) * rng.uniform(args.lo, args.hi))
    bad[idx[on:]] = True
    onset[int(f)] = on / len(idx)
d = 1.0 / (1.0 + np.exp((Dr.V[:, 2] + 1.0) / 0.4))
Fm = Dr.F.copy()
Fm[:, 2] = np.where(bad, Dr.F[:, 2] * (1 + args.sev / 100 * d), Dr.F[:, 2])
X, R = Dr.X.copy(), Fm - np.stack([0 * Dr.C, 0 * Dr.C, Dr.C], 1)
fflight = np.isin(Dr.FLIGHT, faulty)
AD.CFG["c_hi"] = 14.0
desc = Dr.V[:, 2] < -1.5
res = dict(args=vars(args), n_faulty=len(faulty), onset=onset,
           faulty_share_of_descent=float((desc & bad).sum() / desc.sum()),
           faulty_flights_descent_before_onset=float((desc & fflight & ~bad).sum() / (desc & fflight).sum()))
mp = f"work/model_local_{tag}.pt"
if os.path.exists(mp):
    net = Dr.make_net(); net.load_state_dict(torch.load(mp)); [q.requires_grad_(False) for q in net.parameters()]
else:
    net = Dr.fit(X, R, np.ones(len(X), bool), seed=args.seed); torch.save(net.state_dict(), mp)
log("faulty model ready", {k: v for k, v in res.items() if k not in ("args", "onset")})

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
sc = sc.numpy()
np.savez(f"work/local_scores_{tag}.npz", sc=sc)
log("scores ready")

# windows: as in local_drone.py
win_id = np.zeros(len(sc), int)
nw = 0
for f in np.unique(Dr.FLIGHT):
    idx = np.where(Dr.FLIGHT == f)[0]
    k = max(len(idx) // args.win, 1)
    win_id[idx] = nw + np.minimum(np.arange(len(idx)) // args.win, k - 1)
    nw += k
cnt = np.bincount(win_id)
w_sc = np.bincount(win_id, sc) / cnt
w_bad = np.bincount(win_id, bad) / cnt > 0.5
w_desc = np.bincount(win_id, desc) > 0
f_ids = np.unique(Dr.FLIGHT)
f_sc = np.array([sc[Dr.FLIGHT == f].mean() for f in f_ids])
f_flag = np.isin(Dr.FLIGHT, f_ids[f_sc > 0])
pos_w = w_sc[win_id] > 0
keep = {
    # oracle removal of the faulty part, plus FACT's positive windows that hold no descent: is it selection on noise?
    "Oracle + FACT non-descent windows": ~bad & ~(pos_w & ~w_desc[win_id]),
    # sign rule only among windows that hold descent data
    "FACT descent windows": ~(pos_w & w_desc[win_id]),
    # sign rule on windows inside the flights the flight rule flags
    "FACT windows in flagged flights": ~(pos_w & f_flag),
    # window oracle: windows mostly after the onset
    "Oracle windows": ~w_bad[win_id],
}
# label bias of what each rule keeps: mean thrust residual of kept correct hover-ish samples vs all of them
hov = ~bad & (np.abs(Dr.V[:, 2]) < 0.5)
res["removal"] = {}
for k, m in keep.items():
    rm = ~m
    res["removal"][k] = dict(kept=float(m.mean()), faulty_descent_removed=float((rm & bad & desc).sum() / (bad & desc).sum()),
                             correct_descent_removed=float((rm & ~bad & desc).sum() / (~bad & desc).sum()),
                             correct_removed=float((rm & ~bad).sum() / (~bad).sum()),
                             hover_resid_shift=float(R[m & hov, 2].mean() - R[hov, 2].mean()),
                             faulty_label_mass_left=float((d * (m & bad)).sum() / (d * bad).sum()))
    log(k, res["removal"][k])
for k, m in {"FACT windows": ~pos_w, "FACT samples": sc <= 0, "FACT flights": ~f_flag, "Oracle samples": ~bad}.items():
    res["removal"][k] = dict(hover_resid_shift=float(R[m & hov, 2].mean() - R[hov, 2].mean()),
                             faulty_label_mass_left=float((d * (m & bad)).sum() / (d * bad).sum()))
    log(k, res["removal"][k])
json.dump(res, open(OUT, "w"), indent=1)

TASKS = {"test dives": dict(AD.TASK), "steep dives": dict(AD.TASK, dive_dx=(0.5, 1.5)),
         "shallow dives": dict(AD.TASK, dive_dx=(5, 8))}


def fly(m):
    r = {}
    for name, tk in TASKS.items():
        e = Dr.evaluate(m, c_hi=AD.CFG["c_hi"], delay=3, every=3, T=250, n_each=24, seed=7, **tk)
        r[name] = float(np.mean([x["crash"] for x in e])); r[f"{name} reach"] = float(np.mean([x["reach"] for x in e]))
    return r


res["flown"] = {}
for k, m in keep.items():
    res["flown"][k] = fly(Dr.fit(X, R, m, seed=args.seed))
    log(k, res["flown"][k])
    json.dump(res, open(OUT, "w"), indent=1)
log("done")
