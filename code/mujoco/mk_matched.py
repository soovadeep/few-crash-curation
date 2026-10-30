"""Input-matched clean logs for the dry-day fault: fast route runs (same pursuit controller, same speeds as
the dry-day runs) driven on the icy day. Their labels are right; only the labels tell them apart from the
dry-day logs. A run ends when it reaches the goal, touches a wall, or after 300 steps.

usage: python mk_matched.py 100 [n_extra_dry]   # appends 100 such logs (kind 3), and optionally more
                                                # dry-day runs (kind 2), to logs_V2.5_dry100.npz
"""
import sys

import numpy as np
import torch

import core as C
import env as E

E.V_CMD = 2.5; E.MU_ICE = 0.2; E.WALL_MARGIN = 0.25; E.ICE = ((6.0, 8.5), (-1.5, 3.5))
n = int(sys.argv[1])
n_dry = int(sys.argv[2]) if len(sys.argv) > 2 else 0
D = dict(np.load("logs_V2.5_dry100.npz"))
rng = np.random.default_rng(1)
ice = E.Sim(E.MU_ICE)
S, U, Sn, LOG = [D["S"]], [D["U"]], [D["Sn"]], [D["log"]]
kinds = list(D["kind"])
lens, walls = [], 0
dry = E.Sim(E.MU_DRY)
for _ in range(n_dry):
    v = rng.uniform(0.75 * E.V_CMD, E.V_CMD)
    s = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4), rng.uniform(-0.1, 0.1), v, 0, 0, 0, v / C.RW])
    dry.set_state(s)
    st, us = [dry.get_state()], []
    for t in range(300):
        u = C.pursuit(st[-1], v) + np.array([rng.normal(0, 0.05), 0])
        us.append(u); st.append(dry.step(u))
        if st[-1][1] > 8.5:
            break
    st = np.array(st)
    S.append(st[:-1]); Sn.append(st[1:]); U.append(np.array(us))
    LOG.append(np.full(len(us), len(kinds))); kinds.append(2)
for _ in range(n):
    v = rng.uniform(0.75 * E.V_CMD, E.V_CMD)
    s = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4), rng.uniform(-0.1, 0.1), v, 0, 0, 0, v / C.RW])
    ice.set_state(s)
    st, us = [ice.get_state()], []
    for t in range(300):
        u = C.pursuit(st[-1], v) + np.array([rng.normal(0, 0.05), 0])
        us.append(u); st.append(ice.step(u))
        if st[-1][1] > 8.5:
            break
        if E.clearance(torch.tensor(st[-1][:2])).item() < 0:
            walls += 1
            break
    st = np.array(st)
    S.append(st[:-1]); Sn.append(st[1:]); U.append(np.array(us))
    LOG.append(np.full(len(us), len(kinds))); kinds.append(3)
    lens.append(len(us))
kinds = np.array(kinds)
out = dict(S=np.concatenate(S), U=np.concatenate(U), Sn=np.concatenate(Sn), log=np.concatenate(LOG),
           kind=kinds, bad=kinds == 2)
np.savez(f"logs_V2.5_dry{100 + n_dry}_m{n}.npz", **out)
dl = np.bincount(D["log"])[D["kind"] == 2]
print(f"{n} icy fast runs: {walls} ended at a wall; length median {np.median(lens):.0f} "
      f"(range {min(lens)}-{max(lens)}); dry-day runs median {np.median(dl):.0f} ({dl.min()}-{dl.max()})")
