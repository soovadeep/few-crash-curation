"""Sensitivity of FACT's score to the checkpoint: retrain the faulty model with the same seed, keep the weights after
epochs 3, 6, 9 and 12, and score the logs from one crash dive (of the original faulty model) at each checkpoint, and
with standard multi-checkpoint TracIn (learning-rate-weighted sum over the four). Detection only: a removal set
that is unchanged gives the same retrained model.

usage: python work/ckpt_drone.py --fault prop50j8 --seed 1
"""
import argparse
import copy
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
p.add_argument("--threads", type=int, default=1)
args = p.parse_args()
torch.set_num_threads(args.threads)
t0 = time.time()
Xf, Rf, bad, distr = AD.setup(args.fault)
CFG = AD.CFG
net0 = Dr.make_net(); net0.load_state_dict(torch.load(f"work/model3_{args.fault}_s{args.seed}.pt"))


def log(*a):
    print(*a, f"({time.time() - t0:.0f}s)", flush=True)


crash = None
with torch.no_grad():
    for i, (kind, a, b) in enumerate(Dr.tasks(24, seed=11, **AD.TASK)):
        s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
        S, U = AD.run_cl(s0, torch.tensor(b), net0, seed=i)
        clr = Dr.clearance(S[:, :3])
        if clr.min() < 0:
            t_c = int(torch.nonzero(clr < 0)[0, 0])
            crash = AD.realized_residual(S[:t_c + 1], U[:t_c], CFG["c_hi"])
            break

# Dr.fit with snapshots
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
net = Dr.make_net()
Xt = torch.tensor((Xf - Dr.mu) / Dr.sd, dtype=torch.float32)
Rt = torch.tensor(Rf / Dr.rs, dtype=torch.float32)
opt = torch.optim.Adam(net.parameters(), 2e-3)
idx = np.arange(len(Xf))
snaps = {}
for ep in range(12):
    rng.shuffle(idx)
    for b in range(0, len(idx), 2048):
        bi = idx[b:b + 2048]
        loss = ((net(Xt[bi]) - Rt[bi]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    lr = opt.param_groups[0]["lr"]
    for g in opt.param_groups:
        g['lr'] *= 0.8
    if ep + 1 in (3, 6, 9, 12):
        snaps[ep + 1] = (copy.deepcopy(net.state_dict()), lr)
d12 = max(float((snaps[12][0][k] - v).abs().max()) for k, v in net0.state_dict().items())
log(f"retrained; max weight difference to the saved faulty model at epoch 12: {d12:.2e}")

flights = np.unique(Dr.FLIGHT)
fbad = np.array([bad[Dr.FLIGHT == f].any() for f in flights])
fdistr = np.array([distr[Dr.FLIGHT == f].any() for f in flights])
xs, ys = crash
feat, yt = (xs - Dr.MU) / Dr.SD, ys / Dr.RS
m = Dr.make_net()
g_i = vmap(grad(lambda pd, x, y: ((functional_call(m, pd, (x[None],))[0] - y) ** 2).sum()), in_dims=(None, 0, 0))


def sample_scores(sd):
    params = {k: v.clone() for k, v in sd.items()}
    pg = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    tg = torch.autograd.grad(((functional_call(m, pg, (feat,)) - yt) ** 2).sum(-1).mean(), list(pg.values()))
    sc = torch.zeros(len(Xt))
    for s in range(0, len(Xt), 2048):
        G = g_i(params, Xt[s:s + 2048], Rt[s:s + 2048])
        nrm = torch.sqrt(sum((G[k].flatten(1) ** 2).sum(1) for k in G))
        sc[s:s + 2048] = -sum((G[k].flatten(1) * t.flatten()[None]).sum(1) for k, t in zip(G, tg)) / (nrm + 1e-12)
    return sc.numpy()


def detect(sc):
    fs = np.array([sc[Dr.FLIGHT == f].mean() for f in flights])
    o = np.argsort(-fs)
    pos = fs > 0
    return dict(auroc=float(roc_auc_score(fbad, fs)), ap=float(average_precision_score(fbad, fs)),
                bad_top40=int(fbad[o[:40]].sum()), lookalike_top40=int(fdistr[o[:40]].sum()),
                sign_removed=int(pos.sum()), sign_bad=int(fbad[pos].sum()), sign_lookalike=int(fdistr[pos].sum()),
                top40=[int(x) for x in o[:40]], sign=[int(x) for x in np.flatnonzero(pos)])


out, tot = {}, 0
for ep, (sd, lr) in snaps.items():
    sc = sample_scores(sd)
    tot = tot + lr * sc
    out[f"epoch {ep}"] = detect(sc)
    log(ep, {k: v for k, v in out[f"epoch {ep}"].items() if k not in ("top40", "sign")})
out["multi-checkpoint"] = detect(tot)
out["saved model"] = detect(sample_scores(net0.state_dict()))
for k in ("multi-checkpoint", "saved model"):
    log(k, {kk: v for kk, v in out[k].items() if kk not in ("top40", "sign")})
ref = out["saved model"]
for k, v in out.items():
    v["same_top40_as_saved"] = set(v["top40"]) == set(ref["top40"])
    v["same_sign_set_as_saved"] = set(v["sign"]) == set(ref["sign"])
json.dump(dict(args=vars(args), weight_diff_epoch12=d12, results=out), open(f"work/ckpt_{args.fault}_s{args.seed}.json", "w"), indent=1)
log("done", {k: (v["same_top40_as_saved"], v["same_sign_set_as_saved"]) for k, v in out.items()})
