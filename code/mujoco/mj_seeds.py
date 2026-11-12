"""Curation from few failures on the MuJoCo car with a minority fault (100 dry-day logs), with the windowed score
(the failure's last T_f transitions, each log's top T_l per-sample scores), against adding those failure runs to the
training data (W=20). Flown from the standard test starts and two corner entries the crash runs never covered.

usage: python mj_seeds.py --logs logs_V2.5_dry100.npz --seed 1 --tf 40 --tl 10 --rs 1
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.func import functional_call

import attrlib as A
import core as C
import env as E

p = argparse.ArgumentParser()
p.add_argument("--logs", default="logs_V2.5_dry100.npz")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--tf", type=int, default=40, help="failure window, transitions before contact (dt 0.05 s)")
p.add_argument("--tl", type=int, default=10, help="log roll-up: mean of the log's top tl per-sample scores")
p.add_argument("--rs", default="0,1")
p.add_argument("--w", type=int, default=20)
args = p.parse_args()
E.V_CMD = 2.5; E.MU_ICE = 0.2; E.WALL_MARGIN = 0.25; E.ICE = ((6.0, 8.5), (-1.5, 3.5))
torch.set_num_threads(1)
t0 = time.time()
tag = args.logs.replace("logs_", "").replace(".npz", "")
D = dict(np.load(args.logs)); bad = D["bad"]; kind = D["kind"]; n_logs = len(bad); Lg = D["log"]
mp = f"m_{tag}_faulty_s{args.seed}.pt"
if os.path.exists(mp):
    net = C.Dyn(torch.ones(8)); net.load_state_dict(torch.load(mp))
else:
    net = C.fit(D, np.arange(n_logs), seed=args.seed, epochs=60); torch.save(net.state_dict(), mp)


def entry(n, seed, x, v, y=(-0.5, 0.5), psi=(-0.2, 0.2)):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        vv = rng.uniform(*v)
        out.append(np.array([rng.uniform(*x), rng.uniform(*y), rng.uniform(*psi), vv, 0, 0, 0, vv / C.RW]))
    return np.array(out)


STARTS = {"test": C.starts(24, 7),
          "late entry": entry(24, 7, x=(3.0, 4.5), v=(2.2, 2.5)),
          "ice-edge entry": entry(24, 7, x=(5.0, 5.8), v=(1.8, 2.4), y=(-0.4, 0.4), psi=(-0.15, 0.15))}


def log(*a):
    print(*a, f"({time.time() - t0:.0f}s)", flush=True)


cr = A.crash_runs(net, C.starts(24, 11), T=300)
log(f"{len(cr)} crash runs")
params = {n: q.detach().clone().requires_grad_(True) for n, q in net.named_parameters()}


def subset(n):
    return cr if n == "all" else cr[:n]


def target(n):
    sl = lambda a: a[-args.tf:] if args.tf else a
    rs = subset(n)
    S = torch.cat([sl(c["S"][:-1]) for c in rs]); Sn = torch.cat([sl(c["S"][1:]) for c in rs])
    U = torch.cat([sl(c["U"]) for c in rs])
    F, Y = C.features(S, U), C.deltas(S, Sn) / net.out_scale
    L = ((functional_call(net, params, (F,)) - Y) ** 2).sum(-1).mean()
    return torch.autograd.grad(L, list(params.values()))


NS = [1, 3]
sc = A.per_sample(net, D, {n: target(n) for n in NS})
log("scores done")
OUT = f"seeds_{tag}_s{args.seed}.json"
try:
    rows = json.load(open(OUT))["rows"]
except FileNotFoundError:
    rows = []
done = {r["model"] for r in rows}


def add(name, make, **kw):
    if name in done:
        return
    m = make()
    r = dict(model=name, **kw)
    for sn, S0 in STARTS.items():
        s_ = C.summary(C.run(m, S0, seed=0, T=300))
        r[sn] = dict(crash=s_["crash"], reach=s_["reach"])
    rows.append(r); log(r)
    json.dump(dict(args=vars(args), n_crash=len(cr), rows=rows), open(OUT, "w"), indent=1)


def data_with(keep, n=None):
    m = np.isin(D["log"], keep)
    if n is None:
        return {k: D[k][m] for k in ["S", "U", "Sn", "log"]}
    rs = subset(n)
    Sc = torch.cat([c["S"][:-1] for c in rs]).numpy(); Snc = torch.cat([c["S"][1:] for c in rs]).numpy()
    Uc = torch.cat([c["U"] for c in rs]).numpy(); w = args.w
    return dict(S=np.concatenate([D["S"][m]] + [Sc] * w), U=np.concatenate([D["U"][m]] + [Uc] * w),
                Sn=np.concatenate([D["Sn"][m]] + [Snc] * w),
                log=np.concatenate([D["log"][m]] + [np.full(len(Uc), n_logs)] * w))


add("faulty", lambda: net)
ALL = np.arange(n_logs)
for r in [int(x) for x in args.rs.split(",")]:
    fs = args.seed + 1000 * r  # retrain r: r = 0 uses the data seed itself
    fit = lambda d, scale=None: C.fit(d, None, seed=fs, epochs=60, out_scale=scale)
    add(f"clean (bad logs removed) | retrain {r}", lambda: fit(data_with(np.flatnonzero(~bad))))
    for n in NS:
        o = np.argsort(-A.log_top(sc[n], Lg, n_logs, k=args.tl))
        keep = np.setdiff1d(ALL, o[:100])
        info = dict(bad_removed=int(bad[o[:100]].sum()), lookalike_removed=int((kind[o[:100]] == 3).sum()))
        add(f"ours from {n} runs, remove 100 | retrain {r}", lambda: fit(data_with(keep)), **info)
        add(f"add {n} runs, W={args.w} | retrain {r}", lambda: fit(data_with(ALL, n), net.out_scale))
        add(f"ours from {n} runs, remove 100, and add them | retrain {r}", lambda: fit(data_with(keep, n), net.out_scale), **info)
log("done")
