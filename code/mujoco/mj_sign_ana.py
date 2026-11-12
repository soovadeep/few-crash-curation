"""Which logs does the sign rule remove on the car, and why? Breakdown of positive-score logs by kind and by ice
coverage, and the clean (oracle) model's error on the crash window (representation error).

usage: python mj_sign_ana.py --seed 1
"""
import argparse
import json

import numpy as np
import torch
from torch.func import functional_call

import attrlib as A
import core as C
import env as E

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=1)
p.add_argument("--tf", type=int, default=40)
args = p.parse_args()
E.V_CMD = 2.5; E.MU_ICE = 0.2; E.WALL_MARGIN = 0.25; E.ICE = ((6.0, 8.5), (-1.5, 3.5))
torch.set_num_threads(1)
tag = "V2.5_dry300_m100"
D = dict(np.load(f"logs_{tag}.npz")); bad = D["bad"]; kind = D["kind"]; n_logs = len(bad); Lg = D["log"]
net = C.Dyn(torch.ones(8)); net.load_state_dict(torch.load(f"m_{tag}_faulty_s{args.seed}.pt"))
cr = A.crash_runs(net, C.starts(24, 11), T=300)
params = {n: q.detach().clone().requires_grad_(True) for n, q in net.named_parameters()}
c = cr[0]
S, Sn, U = c["S"][:-1][-args.tf:], c["S"][1:][-args.tf:], c["U"][-args.tf:]
F, Y = C.features(S, U), C.deltas(S, Sn) / net.out_scale
L = ((functional_call(net, params, (F,)) - Y) ** 2).sum(-1).mean()
sc = A.per_sample(net, D, {1: torch.autograd.grad(L, list(params.values()))})
lm = A.log_mean(sc[1], Lg, n_logs)
pos = lm > 0
xy = D["S"][:, :2]
onice = (xy[:, 0] > E.ICE[0][0]) & (xy[:, 0] < E.ICE[0][1]) & (xy[:, 1] > E.ICE[1][0]) & (xy[:, 1] < E.ICE[1][1])
ice_frac = np.bincount(Lg, weights=onice, minlength=n_logs) / np.maximum(np.bincount(Lg, minlength=n_logs), 1)
speed = np.linalg.norm(D["S"][:, 3:5], axis=1)
fast_ice = np.bincount(Lg, weights=onice & (speed > 2.0), minlength=n_logs)
grp = np.where(bad, "dry-day (bad)", np.where(kind == 3, "look-alike", np.where(kind == 1, "slow route",
               np.where(ice_frac > 0, "explore, touches ice", "explore, off ice"))))
out = {}
for g in ["dry-day (bad)", "look-alike", "slow route", "explore, touches ice", "explore, off ice"]:
    m = grp == g
    out[g] = dict(n=int(m.sum()), removed=int((pos & m).sum()), mean_score=float(lm[m].mean()),
                  fast_ice_samples=int(fast_ice[m].sum()), fast_ice_kept=int(fast_ice[m & ~pos].sum()))
    print(g, out[g], flush=True)
# the logs the paper's "sign-rule count" row removes: as many as have a positive mean score, chosen by the
# worst-T rating (mean of each log's top 40 transition scores), as in mj_major.py
lt = A.log_top(sc[1], Lg, n_logs, k=40)
o = np.argsort(-lt)
rm = np.zeros(n_logs, bool); rm[o[:int(pos.sum())]] = True
for g in ["dry-day (bad)", "look-alike", "slow route", "explore, touches ice", "explore, off ice"]:
    m = grp == g
    out[g].update(row_removed=int((rm & m).sum()), row_fast_ice_kept=int(fast_ice[m & ~rm].sum()),
                  mean_top40=float(lt[m].mean()))
    print(g, "row:", int((rm & m).sum()), "removed, fast-ice samples kept", int(fast_ice[m & ~rm].sum()),
          "mean top-40 score", round(float(lt[m].mean()), 3), flush=True)
np.savez(f"signana_{tag}_s{args.seed}.npz", sc=sc[1], lm=lm, lt=lt, grp=grp)
# representation error: the oracle-trained model's error on the crash window vs the faulty model's
clean = C.fit({k: D[k][np.isin(Lg, np.flatnonzero(~bad))] for k in ["S", "U", "Sn", "log"]}, None, seed=args.seed,
              epochs=60)
with torch.no_grad():
    for nm, m in [("faulty", net), ("clean", clean)]:
        Yc = C.deltas(S, Sn) / m.out_scale
        out[f"crash_err_{nm}"] = float(((m(F) - Yc) ** 2).sum(-1).mean())
        Ft, Yt = C.tensors(D)
        ice_fast = torch.tensor(onice & (speed > 2.0) & ~bad[Lg])
        out[f"train_err_correct_fast_ice_{nm}"] = float(((m(Ft[ice_fast]) - Yt[ice_fast] / m.out_scale) ** 2).sum(-1).mean())
print({k: v for k, v in out.items() if k.startswith(("crash", "train"))}, flush=True)
json.dump(out, open(f"signana_{tag}_s{args.seed}.json", "w"), indent=1)
