"""Logs, learned dynamics, MPPI and evaluation for the MuJoCo icy-corner car.

Logs (each one is a unit for whole-log removal):
  explore : 1 s of random smooth commands from a random state anywhere in the corridor
            (icy-corner starts thinned to P_ICE), on the icy day
  route   : the route driven slowly (pure pursuit, ~1 m/s) on the icy day
  dry     : the route driven fast (pure pursuit, 2-3 m/s) on a day the corner was dry, filed with
            the icy-day logs (the fault). Only the part over the ice patch is wrong for today.
"""
import math

import numpy as np
import torch
import torch.nn as nn

import env as E

RW = 0.0488          # wheel radius: omega = v / RW
L_WB = 0.2965        # wheelbase
torch.set_num_threads(1)


# ------------------------------------------------------------------ logs
def _centerline():
    a = np.stack([np.linspace(-1, 7, 200), np.zeros(200)], 1)
    b = np.stack([np.full(200, 7.0), np.linspace(0, 9, 200)], 1)
    return np.concatenate([a, b])


CL = _centerline()


def pursuit(s, v_t, look=None):
    look = 0.6 + 0.5 * v_t if look is None else look
    p = s[:2]
    i = np.argmin(np.linalg.norm(CL - p, axis=1))
    j = min(i + 1, len(CL) - 1)
    while j < len(CL) - 1 and np.linalg.norm(CL[j] - p) < look:
        j += 1
    dx, dy = CL[j] - p
    alpha = E.wrap(np.arctan2(dy, dx) - s[2])
    delta = np.arctan(2 * L_WB * np.sin(alpha) / look)
    return np.array([delta / E.STEER_MAX, v_t])


def _knots(rng, L, lo, hi, n=3):
    k = rng.uniform(lo, hi, n)
    t = np.linspace(0, n - 1, L)
    return np.interp(t, np.arange(n), k)


def _in_track(x, y):
    def inb(b):
        (x0, x1), (y0, y1) = b
        return x0 < x < x1 and y0 < y < y1
    return inb(E.BOX_A) or inb(E.BOX_B)


def _random_start(rng, p_ice):
    while True:
        x, y = rng.uniform(*E.ARENA_X), rng.uniform(*E.ARENA_Y)
        if not _in_track(x, y):
            continue
        if E.ICE[0][0] < x < E.ICE[0][1] and E.ICE[1][0] < y < E.ICE[1][1] and rng.random() > p_ice:
            continue
        v = rng.uniform(0, min(3.5, E.V_CMD))
        return np.array([x, y, rng.uniform(-np.pi, np.pi), v, 0.0, 0.0, rng.uniform(-0.3, 0.3), v / RW])


def _ice_start(rng):
    """A start on the icy part of the track, heading roughly along the route, at speed."""
    (x0, x1), (y0, y1) = E.ICE
    while True:
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        if _in_track(x, y):
            break
    psi = rng.choice([0.0, np.pi / 2]) + rng.uniform(-0.6, 0.6)
    v = rng.uniform(1.0, min(3.5, E.V_CMD))
    return np.array([x, y, psi, v, 0.0, 0.0, rng.uniform(-0.3, 0.3), v / RW])


def make_logs(seed, n_explore=400, seg_len=20, p_ice=0.3, n_route=10, n_dry=10, dry_v=(2.0, 3.0), n_ice=0):
    """Returns dict of arrays: S (N,8) state, U (N,2), Sn (N,8) next state, log (N,) log id,
    kind (n_logs,) 0 explore / 1 route / 2 dry, plus `bad` (n_logs,) bool."""
    rng = np.random.default_rng(seed)
    ice, dry = E.Sim(E.MU_ICE), E.Sim(E.MU_DRY)
    S, U, Sn, LOG, kinds = [], [], [], [], []

    def add(states, controls, kind):
        st = np.array(states)
        S.append(st[:-1]); Sn.append(st[1:]); U.append(np.array(controls))
        LOG.append(np.full(len(controls), len(kinds))); kinds.append(kind)

    for j in range(n_explore + n_ice):
        s = _random_start(rng, p_ice) if j < n_explore else _ice_start(rng)
        ice.set_state(s)
        nk = int(rng.integers(2, 7))
        us = np.stack([_knots(rng, seg_len, -1, 1, nk) + rng.normal(0, 0.1, seg_len),
                       _knots(rng, seg_len, 0, E.V_CMD, nk) + rng.normal(0, 0.3, seg_len)], 1)
        us = np.stack([us[:, 0].clip(-1, 1), us[:, 1].clip(0, E.V_CMD)], 1)
        st = [ice.get_state()]
        for u in us:
            st.append(ice.step(u))
        add(st, us, 0)
    for kind, sim, n, vr in [(1, ice, n_route, (0.8, 1.1)), (2, dry, n_dry, dry_v)]:
        for _ in range(n):
            v = rng.uniform(*vr)
            s = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4), rng.uniform(-0.1, 0.1), v, 0, 0, 0, v / RW])
            sim.set_state(s)
            st, us = [sim.get_state()], []
            for t in range(300):
                u = pursuit(st[-1], v) + np.array([rng.normal(0, 0.05), 0])
                us.append(u); st.append(sim.step(u))
                if st[-1][1] > 8.5:
                    break
            add(st, us, kind)
    return dict(S=np.concatenate(S), U=np.concatenate(U), Sn=np.concatenate(Sn), log=np.concatenate(LOG),
                kind=np.array(kinds), bad=np.array(kinds) == 2)


# ------------------------------------------------------------------ learned model
FEAT_SCALE = torch.tensor([4.5, 5.0, 3.0, 1.0, 2.0, 0.3, 60.0, 1.0, 4.0])
FEAT_SHIFT = torch.tensor([3.5, 4.0, 1.5, 0.0, 0.0, 0.0, 30.0, 0.0, 2.0])


def features(s, u):
    """position (for the surface), body velocity, yaw rate, steering, wheel speed, commands."""
    uu = torch.stack([u[..., 0].clamp(-1, 1), u[..., 1].clamp(0, E.V_CMD)], -1)
    f = torch.cat([s[..., :2], s[..., 3:8], uu], -1)
    return (f - FEAT_SHIFT) / FEAT_SCALE


def deltas(s, sn):
    """Targets: body-frame displacement, heading change, change of the rest."""
    c, sn_ = torch.cos(s[..., 2]), torch.sin(s[..., 2])
    dx, dy = sn[..., 0] - s[..., 0], sn[..., 1] - s[..., 1]
    dpsi = torch.remainder(sn[..., 2] - s[..., 2] + math.pi, 2 * math.pi) - math.pi
    pose = torch.stack([c * dx + sn_ * dy, -sn_ * dx + c * dy, dpsi], -1)
    return torch.cat([pose, sn[..., 3:] - s[..., 3:]], -1)


class Dyn(nn.Module):
    def __init__(self, out_scale, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 8))
        self.register_buffer("out_scale", torch.as_tensor(out_scale, dtype=torch.float32))

    def forward(self, f):
        return self.net(f)

    def step(self, s, u):
        d = self.net(features(s, u)) * self.out_scale
        c, sn_ = torch.cos(s[..., 2]), torch.sin(s[..., 2])
        x = s[..., 0] + c * d[..., 0] - sn_ * d[..., 1]
        y = s[..., 1] + sn_ * d[..., 0] + c * d[..., 1]
        return torch.cat([torch.stack([x, y, s[..., 2] + d[..., 2]], -1), s[..., 3:] + d[..., 3:]], -1)


def tensors(D):
    S, U, Sn = (torch.tensor(D[k], dtype=torch.float32) for k in ("S", "U", "Sn"))
    return features(S, U), deltas(S, Sn)


def fit(D, keep_logs=None, seed=0, epochs=60, bs=512, lr=2e-3, out_scale=None):
    F, Y = tensors(D)
    m = np.ones(len(F), bool) if keep_logs is None else np.isin(D["log"], keep_logs)
    F, Y = F[m], Y[m]
    if out_scale is None:
        out_scale = Y.std(0) + 1e-6
    torch.manual_seed(seed)
    net = Dyn(out_scale)
    Yn = Y / net.out_scale
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    n = len(F)
    steps = epochs * max(1, n // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.02)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            loss = ((net(F[idx]) - Yn[idx]) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net


# ------------------------------------------------------------------ MPPI (batched over E runs)
K, H, LAM = 256, 30, 1.0
SIG = torch.tensor([0.5, 1.0])


KNOT = 5   # MPPI perturbations are drawn every KNOT steps and linearly interpolated (smooth commands)


def sample_eps(Ecount, gen):
    nk = H // KNOT + 1
    e = torch.randn(Ecount, K, nk, 2, generator=gen) * SIG
    t = torch.arange(H) / KNOT
    i0 = t.floor().long().clamp(max=nk - 2)
    fr = (t - i0)[None, None, :, None]
    eps = e[:, :, i0] * (1 - fr) + e[:, :, i0 + 1] * fr
    eps[:, 0] = 0
    return eps


def mppi_plan(model, s, Unom, gen):
    """s (E,8), Unom (E,H,2) -> new plan (E,H,2)."""
    Ecount = s.shape[0]
    eps = sample_eps(Ecount, gen)
    V = Unom[:, None] + eps
    V = torch.stack([V[..., 0].clamp(-1, 1), V[..., 1].clamp(0, E.V_CMD)], -1)
    x = s[:, None].expand(Ecount, K, 8)
    cost = torch.zeros(Ecount, K)
    with torch.no_grad():
        for h in range(H):
            x = model.step(x, V[:, :, h])
            cost = cost + E.stage_cost(x, V[:, :, h])
        cost = cost + E.terminal_cost(x)
        w = torch.softmax(-(cost - cost.min(1, keepdim=True).values) / LAM, 1)
        return (w[..., None, None] * V).sum(1)


def starts(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        v = rng.uniform(1.5, 2.5)
        out.append(np.array([0.0, rng.uniform(-0.4, 0.4), rng.uniform(-0.1, 0.1), v, 0, 0, 0, v / RW]))
    return np.array(out)


def run(model, S0, seed=0, T=200, mu_ice=None, record=False):
    """Closed loop on MuJoCo for each start in S0 (batched planning). Returns per-run dicts."""
    n = len(S0)
    sims = [E.Sim(E.MU_ICE if mu_ice is None else mu_ice) for _ in range(n)]
    for sim, s0 in zip(sims, S0):
        sim.set_state(s0)
    gen = torch.Generator().manual_seed(seed)
    s = torch.tensor(np.array([sim.get_state() for sim in sims]), dtype=torch.float32)
    Unom = torch.zeros(n, H, 2); Unom[..., 1] = 1.5
    alive = np.ones(n, bool)
    traj = [s.clone()]; acts = []
    t_end = np.full(n, T)
    for t in range(T):
        plan = mppi_plan(model, s, Unom, gen)
        u = plan[:, 0]
        nxt = s.clone()
        for i in range(n):
            if alive[i]:
                nxt[i] = torch.tensor(sims[i].step(u[i].numpy()), dtype=torch.float32)
        s = nxt
        traj.append(s.clone()); acts.append(u.clone())
        Unom = torch.cat([plan[:, 1:], plan[:, -1:]], 1)
        clr = E.clearance(s[:, :2]).numpy()
        done = alive & ((clr < -0.3) | (s[:, 1].numpy() > 7.5))
        t_end[done] = t + 1
        alive &= ~done
        if not alive.any():
            break
    X = torch.stack(traj, 1); A = torch.stack(acts, 1)
    out = []
    for i in range(n):
        Xi = X[i, :t_end[i] + 1]
        clr = E.clearance(Xi[:, :2])
        r = dict(crash=bool(clr.min() < 0), reach=bool(Xi[-1, 1] > 7.5 and clr.min() >= 0), min_clr=float(clr.min()),
                 vmax=float(torch.hypot(Xi[:, 3], Xi[:, 4]).max()), steps=int(t_end[i]))
        if record:
            r["S"], r["U"] = Xi, A[i, :t_end[i]]
        out.append(r)
    return out


def summary(res):
    return dict(crash=float(np.mean([r["crash"] for r in res])), reach=float(np.mean([r["reach"] for r in res])),
                vmax=float(np.mean([r["vmax"] for r in res])))
