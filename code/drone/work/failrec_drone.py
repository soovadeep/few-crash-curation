"""Does FACT still work when the failure record is imperfect? One faulty model per seed; the failure run is the first
attribution dive that crashes (or, for the supervisor variants, the first dive the supervisor takes over). Variants
of what the robot logged:

  clean           the crash run as simulated (the default everywhere else)
  noise1, noise3  logged from a noisy state estimate: correlated noise (0.2 s) of 2 cm, 0.05 m/s, 0.6 deg, 0.05 rad/s,
                  and 3x that; the targets are then differences of noisy velocities
  first75, first50  only the first 75% / 50% of the transitions before contact are logged
  sup_m0.1        supervised run, takeover with 0.1 m margin and 0.1 s reaction (as in sup_drone.py), logged through
                  the takeover until the descent stops
  sup_m0.5, sup_m1.0  earlier takeovers (0.5 m / 1.0 m margin): less of the fast descent is recorded
  sup_late        reaction time 0.3 s instead of 0.1 s (a slower supervisor; if it fails to save the drone the
                  run is a crash run)
  combined        sup_m0.5 logged with noise1

Each variant: sign-rule detection (flights removed / bad among them) and the sign-rule model flown on the test,
steep and shallow dives (models with the same removal set are flown once).
With --sweep sev,nper,look the faulty logs come from sweep_drone.py's setup instead of --fault.

usage: python work/failrec_drone.py --fault prop50j8 --seed 1
       python work/failrec_drone.py --sweep 20,8,4 --seed 1
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

import attr_data as AD
import drone as Dr

p = argparse.ArgumentParser()
p.add_argument("--fault", default="prop50j8")
p.add_argument("--sweep", default=None)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--only", default=None, help="comma-separated variants")
p.add_argument("--threads", type=int, default=1)
args = p.parse_args()
torch.set_num_threads(args.threads)
t0 = time.time()


def log(*a):
    print(*a, f"({time.time() - t0:.0f}s)", flush=True)


if args.sweep:
    sev, nper, look = (int(v) for v in args.sweep.split(","))
    tag = f"sweep_sev{sev}_n{nper}_l{look}_s{args.seed}"
    M = nper + look
    top = AD.prop_flights(M)
    look_pos = {int((j + 0.5) * M / look) for j in range(look)} if look else set()
    faulty, lk = [], []
    for d in range(len(top) // M):
        for r, f in enumerate(top[d * M:(d + 1) * M]):
            (lk if r in look_pos else faulty).append(f)
    X, R, bad = AD.prop_fault(sev / 100, faulty)
    distr = np.isin(Dr.FLIGHT, lk)
    AD.CFG["c_hi"] = 14.0
    mp = f"work/model_{tag}.pt"
    if os.path.exists(mp):
        net = Dr.make_net(); net.load_state_dict(torch.load(mp))
    else:
        net = Dr.fit(X, R, np.ones(len(X), bool), seed=args.seed); torch.save(net.state_dict(), mp)
else:
    tag = f"{args.fault}_s{args.seed}"
    X, R, bad, distr = AD.setup(args.fault)
    net = Dr.make_net(); net.load_state_dict(torch.load(f"work/model3_{tag}.pt"))
for q in net.parameters():
    q.requires_grad_(False)
CFG = AD.CFG
c_hi = CFG["c_hi"]
OUT = f"work/failrec_{tag}.json"
A_BRAKE = c_hi - Dr.G
NOISE = torch.tensor([0.02] * 3 + [0.05] * 3 + [0.005] * 4 + [0.05] * 3)


def noisy(S, scale, seed):
    g = torch.Generator().manual_seed(300_000 + seed)
    a = Dr.DT / 0.2
    e = torch.zeros(13); out = []
    for s in S:
        e = (1 - a) * e + scale * NOISE * np.sqrt(2 * a) * torch.randn(13, generator=g)
        sh = s + e
        out.append(torch.cat([sh[:6], sh[6:10] / sh[6:10].norm(), sh[10:]]))
    return torch.stack(out)


def floor_h(s):
    return s[..., 2] - Dr.BOX[2][0] - Dr.R_DRONE


def trigger(S, margin, t_react):
    h, vd = floor_h(S), -S[:, 5]
    need = vd.clamp(min=0) ** 2 / (2 * A_BRAKE) + vd.clamp(min=0) * t_react + margin
    hit = torch.nonzero((vd > 0) & (h < need))
    return None if len(hit) == 0 else int(hit[0, 0])


def recover(s, queued, T=100):
    """Full thrust and level attitude after the queued commands; stops logging when the descent stops."""
    z = torch.tensor([0.0, 0.0, 1.0])
    S, U, t_stop, hmin = [s], [], None, float(floor_h(s))
    with torch.no_grad():
        for t in range(T):
            if t < len(queued):
                u = queued[t]
            else:
                zb = Dr.rotate(s[6:10], z)
                w_cmd = Dr.rotate_inv(s[6:10], torch.linalg.cross(zb, z)) * 10.0
                w_cmd[2] = -2.0 * s[12]
                u = torch.cat([torch.tensor([c_hi]), w_cmd])
            s = Dr.step(s, u, Dr.PLANT, c_hi)
            S.append(s); U.append(u); hmin = min(hmin, float(floor_h(s)))
            if t_stop is None and s[5] >= 0:
                t_stop = t + 1
    t_stop = t_stop or T
    return hmin, torch.stack(S[:t_stop + 1]), torch.stack(U[:t_stop])


# simulate the 24 attribution dives once
runs = []
with torch.no_grad():
    for i, (kind, a, b) in enumerate(Dr.tasks(24, seed=11, **AD.TASK)):
        s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
        S, U = AD.run_cl(s0, torch.tensor(b), net, seed=i)
        clr = Dr.clearance(S[:, :3])
        t_c = int(torch.nonzero(clr < 0)[0, 0]) if clr.min() < 0 else None
        runs.append((i, S, U, t_c))
crash_runs = [r for r in runs if r[3] is not None]
log(f"{len(crash_runs)} of 24 attribution dives crash")
records, info = {}, {}
i, S, U, t_c = crash_runs[0]
records["clean"] = AD.realized_residual(S[:t_c + 1], U[:t_c], c_hi)
for sc in (1, 3):
    records[f"noise{sc}"] = AD.realized_residual(noisy(S[:t_c + 1], sc, i), U[:t_c], c_hi)
for frac in (75, 50):
    n = max(2, int(t_c * frac / 100))
    records[f"first{frac}"] = AD.realized_residual(S[:n + 1], U[:n], c_hi)


def supervised(margin, t_react):
    """First dive the supervisor takes over (before contact): its record through the takeover, and whether it saved it."""
    for i, S, U, t_c in runs:
        t_s = trigger(S, margin, t_react)
        if t_s is None or (t_c is not None and t_s >= t_c):
            continue
        hmin, Sr, Ur = recover(S[t_s], [U[t] for t in range(t_s, min(t_s + CFG["delay"], len(U)))])
        Sa, Ua = torch.cat([S[:t_s + 1], Sr[1:]]), torch.cat([U[:t_s], Ur])
        if hmin <= 0:  # the takeover came too late: the drone hits the floor, log up to contact
            k = int(torch.nonzero(Dr.clearance(Sa[:, :3]) < 0)[0, 0])
            Sa, Ua = Sa[:k + 1], Ua[:k]
        return i, Sa, Ua, dict(dive=i, saved=hmin > 0, takeover_h=float(floor_h(S[t_s])), v_down=float(-S[t_s, 5]))
    return None


for name, (m, tr) in {"sup_m0.1": (0.1, 0.1), "sup_m0.5": (0.5, 0.1), "sup_m1.0": (1.0, 0.1),
                      "sup_late": (0.1, 0.3)}.items():
    r = supervised(m, tr)
    if r is None:
        continue
    i, Sa, Ua, inf = r
    records[name] = AD.realized_residual(Sa, Ua, c_hi); info[name] = inf
    if name == "sup_m0.5":
        records["combined"] = AD.realized_residual(noisy(Sa, 1, i), Ua, c_hi)
if args.only:
    records = {k: v for k, v in records.items() if k in args.only.split(",")}
log("records", {k: len(v[0]) for k, v in records.items()}, info)

# flight means of normalised per-sample gradients: a flight's score for any target is a dot product with its row
params = {k: v.detach() for k, v in net.named_parameters()}
Xn = torch.tensor((X - Dr.mu) / Dr.sd, dtype=torch.float32)
Yn = torch.tensor(R / Dr.rs, dtype=torch.float32)
flights = np.unique(Dr.FLIGHT)
fbad = np.array([bad[Dr.FLIGHT == f].any() for f in flights])
fdistr = np.array([distr[Dr.FLIGHT == f].any() for f in flights])
fidx = torch.tensor(np.searchsorted(flights, Dr.FLIGHT))
g_i = vmap(grad(lambda pd, x, y: ((functional_call(net, pd, (x[None],))[0] - y) ** 2).sum()), in_dims=(None, 0, 0))
A = torch.zeros(len(flights), sum(q.numel() for q in params.values()))
for s in range(0, len(Xn), 2048):
    G = g_i(params, Xn[s:s + 2048], Yn[s:s + 2048])
    Gf = torch.cat([G[k].flatten(1) for k in G], 1)
    A.index_add_(0, fidx[s:s + 2048], -Gf / (Gf.norm(dim=1, keepdim=True) + 1e-12))
A = A / torch.bincount(fidx).float()[:, None]
log("flight matrix done")

TASKS = {"test dives": dict(AD.TASK), "steep dives": dict(AD.TASK, dive_dx=(0.5, 1.5)),
         "shallow dives": dict(AD.TASK, dive_dx=(5, 8))}


def fly(m):
    r = {}
    for name, tk in TASKS.items():
        e = Dr.evaluate(m, c_hi=c_hi, delay=3, every=3, T=250, n_each=24, seed=7, **tk)
        r[name] = float(np.mean([x["crash"] for x in e]))
    r["crash"] = float(np.mean([r[n] for n in TASKS]))
    return r


res = dict(args=vars(args), tag=tag, n_bad=int(fbad.sum()), n_look=int(fdistr.sum()),
           attribution_dives_crashed=len(crash_runs), supervisor=info, variants={})
flown = {}
for name, (xs, ys) in records.items():
    pg = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    L = ((functional_call(net, pg, ((xs - Dr.MU) / Dr.SD,)) - ys / Dr.RS) ** 2).sum(-1).mean()
    tg = torch.cat([g.flatten() for g in torch.autograd.grad(L, list(pg.values()))])
    fs = (A @ tg).numpy()
    pos = fs > 0
    key = tuple(np.flatnonzero(pos))
    if key not in flown:
        flown[key] = fly(Dr.fit(X, R, ~np.isin(Dr.FLIGHT, flights[pos]), seed=args.seed))
    res["variants"][name] = dict(transitions=len(xs), removed=int(pos.sum()), bad_removed=int(fbad[pos].sum()),
                                 lookalike_removed=int(fdistr[pos].sum()), **flown[key])
    log(name, res["variants"][name])
    json.dump(res, open(OUT, "w"), indent=1)
if args.sweep:  # the benchmark's unfixed rate is known (Table I); a sweep setting needs its own
    res["unfixed"] = fly(net); log("unfixed", res["unfixed"])
    json.dump(res, open(OUT, "w"), indent=1)
log("done")
