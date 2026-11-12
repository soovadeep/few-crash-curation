"""Label-blind crude rules on the car's majority fault (logs_V2.5_dry300_m100), the analogue of the drone's
"drop the k most crash-like logs" and "drop all descent data": they use where the crashes happen (fast on the ice)
but no crash labels.

usage: python mj_crude.py --seed 1 --k 360
"""
import argparse
import json
import time

import numpy as np
import torch

import core as C
import env as E

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=1)
p.add_argument("--k", type=int, default=360)
args = p.parse_args()
E.V_CMD = 2.5; E.MU_ICE = 0.2; E.WALL_MARGIN = 0.25; E.ICE = ((6.0, 8.5), (-1.5, 3.5))
torch.set_num_threads(1)
t0 = time.time()
tag = "V2.5_dry300_m100"
D = dict(np.load(f"logs_{tag}.npz")); bad = D["bad"]; kind = D["kind"]; n_logs = len(bad); Lg = D["log"]
xy, speed = D["S"][:, :2], np.linalg.norm(D["S"][:, 3:5], axis=1)
onice = (xy[:, 0] > E.ICE[0][0]) & (xy[:, 0] < E.ICE[0][1]) & (xy[:, 1] > E.ICE[1][0]) & (xy[:, 1] < E.ICE[1][1])
fast_ice = onice & (speed > 2.0)
STARTS = {"test": C.starts(24, 7)}


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
OUT = f"crude_{tag}_s{args.seed}.json"
rows = []


def add(name, keep_samples, **kw):
    d = {k: D[k][keep_samples] for k in ["S", "U", "Sn", "log"]}
    m = C.fit(d, None, seed=args.seed, epochs=60)
    r = dict(model=name, **kw)
    for sn, S0 in STARTS.items():
        s_ = C.summary(C.run(m, S0, seed=0, T=300))
        r[sn] = dict(crash=s_["crash"], reach=s_["reach"])
    rows.append(r); print(r, f"({time.time() - t0:.0f}s)", flush=True)
    json.dump(dict(args=vars(args), rows=rows), open(OUT, "w"), indent=1)


per_log = np.bincount(Lg, weights=fast_ice, minlength=n_logs)
o = np.argsort(-per_log)[:args.k]
add(f"drop {args.k} most crash-like logs", ~np.isin(Lg, o), removed=args.k, bad_removed=int(bad[o].sum()),
    lookalike_removed=int((kind[o] == 3).sum()))
add("drop all fast data on the ice", ~fast_ice, samples_removed=int(fast_ice.sum()))
