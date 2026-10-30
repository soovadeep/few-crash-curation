"""Failure runs that never touch the floor: a safety supervisor takes over before the crash.

The faulty model flies the 24 attribution dives as in mis_drone.py. A supervisor watching the true state takes over
the first time the drone could no longer stop above the floor with margin:
    h < v_down^2 / (2 a_brake) + v_down * t_react + margin,   a_brake = c_hi - g
where h is the floor clearance. It then commands full thrust and levels the attitude; commands already sent still
land first (60 ms delay). We check on the true plant that the takeover avoids the floor. A failure run is any run
the supervisor took over, and FACT scores the logs with the transitions BEFORE the takeover only.

usage: python work/sup_drone.py --fault prop50j8 --seed 1 [--fly]
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
p.add_argument("--fault", default="prop50j8")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--k", type=int, default=40)
p.add_argument("--t_react", type=float, default=0.1)
p.add_argument("--margin", type=float, default=0.1)
p.add_argument("--fly", action="store_true")
p.add_argument("--threads", type=int, default=1)
args = p.parse_args()
torch.set_num_threads(args.threads)
t0 = time.time()
Xf, Rf, bad, distr = AD.setup(args.fault)
CFG = AD.CFG
net = Dr.make_net(); net.load_state_dict(torch.load(f"work/model3_{args.fault}_s{args.seed}.pt"))
A_BRAKE = CFG["c_hi"] - Dr.G
TASKS = {"test dives": dict(AD.TASK), "steep dives": dict(AD.TASK, dive_dx=(0.5, 1.5)),
         "shallow dives": dict(AD.TASK, dive_dx=(5, 8))}


def log(*a):
    print(*a, f"({time.time() - t0:.0f}s)", flush=True)


def floor_h(s):
    return s[..., 2] - Dr.BOX[2][0] - Dr.R_DRONE


def trigger(S):
    """First step at which the supervisor takes over, or None."""
    h, vd = floor_h(S), -S[:, 5]
    need = vd.clamp(min=0) ** 2 / (2 * A_BRAKE) + vd.clamp(min=0) * args.t_react + args.margin
    hit = torch.nonzero((vd > 0) & (h < need))
    return None if len(hit) == 0 else int(hit[0, 0])


def recover(s, queued, T=100):
    """Full thrust, level the attitude; the commands already sent land first. Returns the min floor clearance and
    the takeover's own states and commands until the drone stops descending."""
    z = torch.tensor([0.0, 0.0, 1.0])
    hmin = float(floor_h(s))
    S, U, t_stop = [s], [], None
    with torch.no_grad():
        for t in range(T):
            if t < len(queued):
                u = queued[t]
            else:
                zb = Dr.rotate(s[6:10], z)
                w_cmd = Dr.rotate_inv(s[6:10], torch.linalg.cross(zb, z)) * 10.0
                w_cmd[2] = -2.0 * s[12]
                u = torch.cat([torch.tensor([CFG["c_hi"]]), w_cmd])
            s = Dr.step(s, u, Dr.PLANT, CFG["c_hi"])
            S.append(s); U.append(u)
            hmin = min(hmin, float(floor_h(s)))
            if t_stop is None and s[5] >= 0:
                t_stop = t + 1
    t_stop = t_stop or T
    return hmin, torch.stack(S[:t_stop + 1]), torch.stack(U[:t_stop])


runs, info = [], []
with torch.no_grad():
    for i, (kind, a, b) in enumerate(Dr.tasks(24, seed=11, **AD.TASK)):
        s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
        S, U = AD.run_cl(s0, torch.tensor(b), net, seed=i)
        clr = Dr.clearance(S[:, :3])
        t_c = int(torch.nonzero(clr < 0)[0, 0]) if clr.min() < 0 else None
        t_s = trigger(S)
        if t_s is None or (t_c is not None and t_s >= t_c):
            info.append(dict(i=i, crash=t_c is not None, t_crash=t_c, t_sup=t_s, saved=None))
            continue
        hmin, Sr, Ur = recover(S[t_s], [U[t] for t in range(t_s, min(t_s + CFG["delay"], len(U)))])
        info.append(dict(i=i, crash=t_c is not None, t_crash=t_c, t_sup=t_s, saved=hmin > 0, hmin=hmin,
                         kept_s=t_s * Dr.DT, cut_s=None if t_c is None else (t_c - t_s) * Dr.DT,
                         v_down=float(-S[t_s, 5]), h=float(floor_h(S[t_s])), takeover_s=len(Ur) * Dr.DT))
        runs.append(dict(i=i, sup=AD.realized_residual(S[:t_s + 1], U[:t_s], CFG["c_hi"]),
                         rec=AD.realized_residual(torch.cat([S[:t_s + 1], Sr[1:]]), torch.cat([U[:t_s], Ur]),
                                                  CFG["c_hi"]),
                         full=None if t_c is None else AD.realized_residual(S[:t_c + 1], U[:t_c], CFG["c_hi"])))
sup = [r for r in info if r["saved"] is not None]
log(f"{sum(r['crash'] for r in info)} of 24 dives crash unsupervised; supervisor takes over on {len(sup)}, "
    f"saves {sum(r['saved'] for r in sup)}; takeover {np.mean([r['cut_s'] for r in sup if r['cut_s'] is not None]):.2f} s "
    f"before contact on average")
for r in sup[:5]:
    log(r)

params = {n: q.detach() for n, q in net.named_parameters()}
Xn = torch.tensor((Xf - Dr.mu) / Dr.sd, dtype=torch.float32)
Yn = torch.tensor(Rf / Dr.rs, dtype=torch.float32)
flights = np.unique(Dr.FLIGHT)
fbad = np.array([bad[Dr.FLIGHT == f].any() for f in flights])
fdistr = np.array([distr[Dr.FLIGHT == f].any() for f in flights])
N = len(Xn)


def loss_i(pd, x, y):
    return ((functional_call(net, pd, (x[None],))[0] - y) ** 2).sum()


g_i = vmap(grad(loss_i), in_dims=(None, 0, 0))
# flight means of the normalised per-sample gradients: a flight's score for any target is a dot product with its row
fidx = torch.tensor(np.searchsorted(flights, Dr.FLIGHT))
nf = torch.bincount(fidx).float()
A = torch.zeros(len(flights), sum(q.numel() for q in params.values()))
for s in range(0, N, 2048):
    G = g_i(params, Xn[s:s + 2048], Yn[s:s + 2048])
    Gf = torch.cat([G[k].flatten(1) for k in G], 1)
    A.index_add_(0, fidx[s:s + 2048], -Gf / (Gf.norm(dim=1, keepdim=True) + 1e-12))
A = A / nf[:, None]
log("flight matrix done")


def score(pairs):
    xs = torch.cat([x for x, _ in pairs]); ys = torch.cat([y for _, y in pairs])
    feat, yt = (xs - Dr.MU) / Dr.SD, ys / Dr.RS
    pg = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    L = ((functional_call(net, pg, (feat,)) - yt) ** 2).sum(-1).mean()
    tg = torch.cat([g.flatten() for g in torch.autograd.grad(L, list(pg.values()))])
    fs = (A @ tg).numpy()
    o = np.argsort(-fs)
    return o, dict(transitions=len(xs), auroc=float(roc_auc_score(fbad, fs)),
                   ap=float(average_precision_score(fbad, fs)), bad_topk=int(fbad[o[:args.k]].sum()),
                   distr_topk=int(fdistr[o[:args.k]].sum()), n_positive=int((fs > 0).sum()),
                   bad_positive=int(fbad[fs > 0].sum()), distr_positive=int(fdistr[fs > 0].sum()))


det = {}
for n in [1, 3]:
    det[f"sup{n}"] = score([r["sup"] for r in runs[:n]])
    det[f"rec{n}"] = score([r["rec"] for r in runs[:n]])
    full = [r["full"] for r in runs[:n] if r["full"] is not None]
    if len(full) == n:
        det[f"full{n}"] = score(full)
    for key in (f"sup{n}", f"rec{n}", f"full{n}"):
        if key in det:
            log(key, det[key][1])

OUT = f"work/sup_{args.fault}_s{args.seed}.json"
res = dict(args=vars(args), dives=info, detection={k: v[1] for k, v in det.items()}, rows=[])
try:  # keep rows already flown
    res["rows"] = json.load(open(OUT))["rows"]
except FileNotFoundError:
    pass
json.dump(res, open(OUT, "w"), indent=1)
if not args.fly:
    raise SystemExit


def fly(m):
    out = {}
    for name, tk in TASKS.items():
        r = Dr.evaluate(m, c_hi=CFG["c_hi"], delay=3, every=3, T=250, n_each=24, seed=7, **tk)
        out[name] = float(np.mean([x["crash"] for x in r]))
        out[f"{name} reach"] = float(np.mean([x["reach"] for x in r]))
    return out


for name, key, k in [("FACT from 1 supervised run with takeover, remove k", "rec1", args.k),
                     ("FACT from 1 supervised run with takeover, sign rule", "rec1", None),
                     ("FACT from 1 supervised run before takeover, remove k", "sup1", args.k),
                     ("FACT from 1 supervised run before takeover, sign rule", "sup1", None)]:
    if any(r["model"] == name for r in res["rows"]):
        continue
    o, d = det[key]
    k = d["n_positive"] if k is None else k
    keep = ~np.isin(Dr.FLIGHT, flights[o[:k]])
    r = dict(model=name, removed=k, bad_removed=int(fbad[o[:k]].sum()), distr_removed=int(fdistr[o[:k]].sum()),
             **fly(Dr.fit(Xf, Rf, keep, seed=args.seed)))
    res["rows"].append(r); log(r)
    json.dump(res, open(OUT, "w"), indent=1)
log("done")
