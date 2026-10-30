"""Per-flight scores for the paper's Fig. 2: training loss (what worst-fit filtering ranks by), our TracIn score and
the gradient-free score from one crash dive, with the flight's group (bad / look-alike / other).
usage: python work/fig_scores.py prop50j8m 1   -> work/fig_scores_prop50j8m_s1.npz and .json (the .json is what
paper/make_figs.py reads, from results/drone/)"""
import json
import sys
import numpy as np, torch
from torch.func import functional_call, grad, vmap
import attr_data as AD
import drone as Dr
fault, seed = sys.argv[1], int(sys.argv[2])
Xf, Rf, bad, distr = AD.setup(fault)
net = Dr.make_net(); net.load_state_dict(torch.load(f"work/model3_{fault.split('-')[0]}_s{seed}.pt"))
crash = None
with torch.no_grad():
    for i, (kind, a, b) in enumerate(Dr.tasks(24, seed=11, **AD.TASK)):
        S, U = AD.run_cl(torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0]), torch.tensor(b), net, seed=i)
        clr = Dr.clearance(S[:, :3])
        if clr.min() < 0:
            t_c = int(torch.nonzero(clr < 0)[0, 0]); crash = AD.realized_residual(S[:t_c + 1], U[:t_c], AD.CFG["c_hi"]); break
params = {n: q.detach() for n, q in net.named_parameters()}
Xn = torch.tensor((Xf - Dr.mu) / Dr.sd, dtype=torch.float32); Yn = torch.tensor(Rf / Dr.rs, dtype=torch.float32)
g_i = vmap(grad(lambda pd, x, y: ((functional_call(net, pd, (x[None],))[0] - y) ** 2).sum()), in_dims=(None, 0, 0))
feat, yt = (crash[0] - Dr.MU) / Dr.SD, crash[1] / Dr.RS
pg = {k: v.clone().requires_grad_(True) for k, v in params.items()}
tg = torch.autograd.grad(((functional_call(net, pg, (feat,)) - yt) ** 2).sum(-1).mean(), list(pg.values()))
N = len(Xn); sc = torch.zeros(N)
for s in range(0, N, 2048):
    G = g_i(params, Xn[s:s + 2048], Yn[s:s + 2048])
    nrm = torch.sqrt(sum((G[k].flatten(1) ** 2).sum(1) for k in G))
    sc[s:s + 2048] = -sum((G[k].flatten(1) * t.flatten()[None]).sum(1) for k, t in zip(G, tg)) / (nrm + 1e-12)
with torch.no_grad():
    E = (net(Xn) - Yn).numpy(); ec = (net(feat) - yt).numpy().mean(0)
flights = np.unique(Dr.FLIGHT); fidx = np.searchsorted(flights, Dr.FLIGHT); cnt = np.bincount(fidx)
m = lambda v: np.bincount(fidx, weights=v) / cnt
out = dict(train_loss=m((E ** 2).sum(1)), tracin=m(sc.numpy()), output=m(-(E @ ec)),
           bad=np.array([bad[Dr.FLIGHT == f].any() for f in flights]), lookalike=np.array([distr[Dr.FLIGHT == f].any() for f in flights]),
           descent=m((Dr.V[:, 2] < -1.0).astype(float)))
np.savez(f"work/fig_scores_{fault}_s{seed}.npz", **out)
json.dump({k: v.tolist() for k, v in out.items()}, open(f"work/fig_scores_{fault}_s{seed}.json", "w"))
print("saved")
