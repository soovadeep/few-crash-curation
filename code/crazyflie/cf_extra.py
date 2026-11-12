"""Crazyflie controls for cf_table.py: (1) detection of the high-voltage flights by our score under other
roll-ups (worst 2 s, sum), and against voltage directly; (2) removing 76 random flights (3 draws), and removing
the top 76 by the worst-2-s roll-up.  usage: CF_VOLTS=3.1 python cf_extra.py --seed 1"""
import json, sys, time
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.argv += [] 
import cf_table as T  # runs the table's setup (fit, crash dives) and resumes its rows (all done, so nothing re-flown)
import cf
fs_all = {}
for n in [1, 3]:
    xs = torch.cat([x for x, _ in T.subset(n)]); ys = torch.cat([y for _, y in T.subset(n)])
    feat, yt = (xs - cf.MU) / cf.SD, ys / cf.RS
    pg = {k: v.clone().requires_grad_(True) for k, v in T.params.items()}
    L = ((T.functional_call(T.net, pg, (feat,)) - yt) ** 2).sum(-1).mean()
    tg = torch.autograd.grad(L, list(pg.values()))
    sc = torch.zeros(T.N)
    for s in range(0, T.N, 2048):
        G = T.g_i(T.params, T.Xn[s:s + 2048], T.Yn[s:s + 2048])
        nrm = torch.sqrt(sum((G[k].flatten(1) ** 2).sum(1) for k in G))
        sc[s:s + 2048] = -sum((G[k].flatten(1) * t.flatten()[None]).sum(1) for k, t in zip(G, tg)) / (nrm + 1e-12)
    sc = sc.numpy()
    for agg, f in [("mean", np.mean), ("worst 2 s", lambda v: np.sort(v)[-200:].mean()), ("sum", np.sum)]:
        fs = np.array([f(sc[cf.FLIGHT == i]) for i in T.flights])
        fs_all[(n, agg)] = fs
        T.log(f"{n} dives, {agg}: AUROC {roc_auc_score(T.fbad, fs):.2f}  Spearman with voltage {spearmanr(fs, cf.FLIGHT_V)[0]:.2f}")
k = T.args.k
o2 = np.argsort(-fs_all[(1, "worst 2 s")])
T.add(f"ours from 1 dives (worst 2 s), remove {k}", lambda: cf.fit(T.Xf, T.Rf, ~np.isin(cf.FLIGHT, o2[:k]), seed=T.args.seed), **T.info(o2, k))
for d in range(3):
    o = np.random.default_rng(100 + d).permutation(len(T.flights))
    T.add(f"remove {k} random flights (draw {d + 1})", lambda: cf.fit(T.Xf, T.Rf, ~np.isin(cf.FLIGHT, o[:k]), seed=T.args.seed), **T.info(o, k))
T.log("extra done")
