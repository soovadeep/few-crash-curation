"""Crazyflie 2.1 from NanoBench logs, in the same shape as code/drone/work/drone.py.

Learned models see (body velocity, body rates, collective command c) and predict the body specific force minus
[0, 0, c], where c = KF * sum(pwm^2) with one KF fit on all flights. They do not see the battery voltage.
The plant is a net of the same form that also takes the voltage, fit on all flights, and is flown at a fixed
deployment voltage (VPlant). At low voltage the same PWM gives less thrust, so a model that ignores voltage and
learns mostly from fuller batteries is optimistic: a natural, consistent majority fault.
"""
import os

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("NT", "1")))
G = 9.81
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "cf100.npz"), allow_pickle=True)
D, cols, test = z["data"].astype(np.float64), list(z["cols"]), z["test"]
c_ = {k: i for i, k in enumerate(cols)}
V = D[:, [c_["vel x"], c_["vel y"], c_["vel z"]]]  # body frame
W = D[:, [c_["ang vel x"], c_["ang vel y"], c_["ang vel z"]]]
QA = D[:, [c_["quat x"], c_["quat y"], c_["quat z"], c_["quat w"]]]
M2 = (D[:, [c_["pwm 1"], c_["pwm 2"], c_["pwm 3"], c_["pwm 4"]]] ** 2).sum(1)
F = D[:, [c_["acc x"], c_["acc y"], c_["acc z"]]]
VBAT = D[:, c_["vbat"]]
KF = (M2[~test] @ F[~test, 2]) / (M2[~test] @ M2[~test])
C = KF * M2
X = np.concatenate([V, W, C[:, None]], 1)
R = F - np.stack([0 * C, 0 * C, C], 1)
mu, sd, rs = X.mean(0), X.std(0), R.std(0)
MU, SD, RS = (torch.tensor(a, dtype=torch.float32) for a in (mu, sd, rs))
VMU, VSD = float(VBAT.mean()), float(VBAT.std())
FLIGHT, SEG = z["flight"], z["seg"]
FLIGHTS = z["flights"]
DAY = np.array(["cf"] * len(FLIGHT))
FLIGHT_V = np.array([np.median(VBAT[FLIGHT == f]) for f in range(len(FLIGHTS))])  # median in-flight voltage
HID = 128
EQ_STEPS = os.environ.get("CF_EQ", "") == "1"  # every fit takes as many gradient steps as one on all flights


def make_net(h=None, n_in=7):
    h = HID if h is None else h
    return torch.nn.Sequential(torch.nn.Linear(n_in, h), torch.nn.SiLU(), torch.nn.Linear(h, h), torch.nn.SiLU(),
                               torch.nn.Linear(h, 3))


def fit(Xd, Rd, mask, seed=0, epochs=12, weights=None, vbat=None):
    """vbat given: an 8-input net that also sees the battery voltage (the plant)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xn = (Xd - mu) / sd
    if vbat is not None:
        Xn = np.c_[Xn, (vbat - VMU) / VSD]
    net = make_net(n_in=Xn.shape[1])
    Xt = torch.tensor(Xn, dtype=torch.float32)
    Rt = torch.tensor(Rd / rs, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    idx = np.where(mask)[0]
    if EQ_STEPS:
        # as many gradient steps, on the same learning-rate schedule, as `epochs` epochs over all flights
        per = -(-len(X) // 2048); n_step = 0
        while n_step < epochs * per:
            rng.shuffle(idx)
            for b in range(0, len(idx), 2048):
                bi = idx[b:b + 2048]
                loss = ((net(Xt[bi]) - Rt[bi]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                n_step += 1
                if n_step % per == 0:
                    for g in opt.param_groups:
                        g["lr"] *= 0.8
                if n_step >= epochs * per:
                    break
        epochs = 0
    for ep in range(epochs):
        rng.shuffle(idx)
        for b in range(0, len(idx), 2048):
            bi = idx[b:b + 2048]
            loss = ((net(Xt[bi]) - Rt[bi]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        for g in opt.param_groups:
            g["lr"] *= 0.8
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class VPlant(torch.nn.Module):
    """The voltage-aware net flown at a fixed battery voltage."""

    def __init__(self, net8, volts):
        super().__init__()
        self.net8, self.volts = net8, volts

    def forward(self, f):
        v = torch.full((*f.shape[:-1], 1), (self.volts - VMU) / VSD)
        return self.net8(torch.cat([f, v], -1))


SPLIT = os.environ.get("CF_SPLIT", "") == "1"  # plant fit on every third flight by voltage; the logs are the rest
if SPLIT:
    HOLD = np.zeros(len(FLIGHTS), bool); HOLD[np.argsort(FLIGHT_V)[1::3]] = True
    _pp, _rows = os.path.join(HERE, "plant8_split.pt"), HOLD[FLIGHT]
else:
    _pp, _rows = os.path.join(HERE, "plant8.pt"), np.ones(len(X), bool)
if os.path.exists(_pp):
    PLANT8 = make_net(128, 8); PLANT8.load_state_dict(torch.load(_pp)); [p.requires_grad_(False) for p in PLANT8.parameters()]
else:
    _eq, EQ_STEPS = EQ_STEPS, False
    PLANT8 = fit(X[_rows], R[_rows], np.ones(int(_rows.sum()), bool), seed=0, epochs=40 if SPLIT else 20, vbat=VBAT[_rows])
    EQ_STEPS = _eq
    torch.save(PLANT8.state_dict(), _pp)
if SPLIT:  # the held-out flights never reach the learned models: drop them and renumber the rest
    _keep = ~HOLD[FLIGHT]
    _new = np.cumsum(~HOLD) - 1
    D, V, W, QA, M2, F, VBAT, C, X, R, SEG, DAY, test = (a[_keep] for a in (D, V, W, QA, M2, F, VBAT, C, X, R, SEG, DAY, test))
    FLIGHT, FLIGHTS, FLIGHT_V = _new[FLIGHT[_keep]], FLIGHTS[~HOLD], FLIGHT_V[~HOLD]
V_DEPLOY = float(os.environ.get("CF_VOLTS", "3.2"))
PLANT = VPlant(PLANT8, V_DEPLOY)


# ------------------------------------------------------------------ dynamics (as drone.py)
def quat_mul(a, b):
    ax, ay, az, aw = a.unbind(-1); bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


_CONJ = torch.tensor([-1.0, -1.0, -1.0, 1.0])


def rotate(q, v):
    qv = torch.cat([v, torch.zeros_like(v[..., :1])], -1)
    return quat_mul(quat_mul(q, qv), q * _CONJ)[..., :3]


def rotate_inv(q, v):
    return rotate(q * _CONJ, v)


DT, TAU = 0.02, 0.04
C_LO = float(np.percentile(C, 0.5))
C_MAX = float(KF * 4.0)  # every motor at full PWM, as the voltage-blind model sees it


def step(s, u, model, c_hi):
    p, v, q, w = s[..., :3], s[..., 3:6], s[..., 6:10], s[..., 10:13]
    c = u[..., 0].clamp(C_LO, c_hi)
    vb = rotate_inv(q, v)
    feat = (torch.cat([vb, w, c[..., None]], -1) - MU) / SD
    fb = model(feat) * RS + torch.stack([0 * c, 0 * c, c], -1)
    a = rotate(q, fb) - torch.tensor([0.0, 0.0, G])
    dq = torch.cat([w * DT / 2, torch.ones_like(w[..., :1])], -1)
    q2 = quat_mul(q, dq)
    return torch.cat([p + DT * v, v + DT * a, q2 / q2.norm(dim=-1, keepdim=True),
                      w + DT / TAU * (u[..., 1:].clamp(-3, 3) - w)], -1)


# ------------------------------------------------------------------ arena and cost (NanoBench flew in about 3 x 2.2 x 1.5 m)
BOX = ((-1.6, 1.6), (-1.3, 1.3), (0.2, 1.8))
R_DRONE = 0.05


def clearance(p):
    d = [p[..., 0] - BOX[0][0], BOX[0][1] - p[..., 0], p[..., 1] - BOX[1][0], BOX[1][1] - p[..., 1],
         p[..., 2] - BOX[2][0], BOX[2][1] - p[..., 2]]
    return torch.stack(d, -1).min(-1).values - R_DRONE


BAR_K, BAR_M = float(os.environ.get("CF_BARK", "30")), float(os.environ.get("CF_BARM", "0.02"))  # wall barrier steepness, margin


def cost(s, u, goal):
    p = s[..., :3]
    cl = clearance(p)
    return ((p - goal).norm(dim=-1) + 50.0 * torch.nn.functional.softplus(-BAR_K * (cl - BAR_M)) / BAR_K
            + 0.02 * (u[..., 1:] ** 2).sum(-1) + 0.01 * (u[..., 0] - G) ** 2)


K, H, LAM = 512, 40, 1.0
SIG = torch.tensor([2.0, 1.0, 1.0, 0.5])
Q0 = [0.0, 0.0, 0.0, 1.0]


def run(s0, goal, model, c_hi=None, delay=0, every=1, T=200, seed=0, plant=None):
    """MPPI replans every `every` steps; commands reach the motors `delay` steps late."""
    c_hi = C_MAX if c_hi is None else c_hi
    plant = PLANT if plant is None else plant
    gen = torch.Generator().manual_seed(seed)
    if hasattr(plant, "reset"):
        plant.reset(seed)
    Unom = torch.zeros(H, 4); Unom[:, 0] = G
    queue = [Unom[0].clone() for _ in range(delay)]
    s = s0.clone(); traj = [s]
    plan, k = None, 0
    with torch.no_grad():
        for t in range(T):
            if t % every == 0:
                xs = s
                for u in queue:
                    xs = step(xs, u, model, c_hi)
                eps = torch.randn(K, H, 4, generator=gen) * SIG
                eps[0] = 0
                Uk = Unom[None] + eps
                x = xs[None].expand(K, 13); Sc = torch.zeros(K)
                for h in range(H):
                    x = step(x, Uk[:, h], model, c_hi)
                    Sc = Sc + cost(x, Uk[:, h], goal)
                wts = torch.softmax(-(Sc - Sc.min()) / LAM, 0)
                Unom = (wts[:, None, None] * Uk).sum(0)
                plan, k = Unom.clone(), 0
            u_new = plan[min(k, H - 1)]; k += 1
            queue.append(u_new)
            u = queue.pop(0)
            s = step(s, u, plant, c_hi)
            traj.append(s)
            if t % every == every - 1:
                Unom = torch.cat([plan[every:], plan[-1:].expand(every, 4)], 0)
            if clearance(s[:3]) < -0.1:
                break
    return torch.stack(traj)


def tasks(n_each=4, seed=0, kinds=("dive",), dive_dx=(0.6, 1.8), dive_z=0.32, z0=1.6):
    """dive: start high on one side of the arena and stop low near the floor on the other."""
    g = np.random.default_rng(seed)
    out = []
    for i in range(n_each):
        x0 = g.uniform(-1.3, -0.6); y0 = g.uniform(-0.9, 0.9); y1 = g.uniform(-0.9, 0.9)
        out.append(("dive", (x0, y0, z0), (min(x0 + g.uniform(*dive_dx), 1.3), y1, dive_z)))
    return out


def evaluate(model, c_hi=None, delay=0, every=1, T=200, n_each=4, seed=0, kinds=("dive",), **task_kw):
    res = []
    for i, (kind, a, b) in enumerate(tasks(n_each, seed, kinds, **task_kw)):
        s0 = torch.tensor([*a, 0, 0, 0, *Q0, 0, 0, 0])
        S = run(s0, torch.tensor(b), model, c_hi, delay, every, T, seed=i)
        p = S[:, :3]
        res.append(dict(kind=kind, crash=bool(clearance(p).min() < 0), min_clr=clearance(p).min().item(),
                        reach=bool(((p - torch.tensor(b)).norm(dim=-1) < 0.15).any()),
                        vmax=S[:, 3:6].norm(dim=-1).max().item(), vdown=(-S[:, 5]).max().item()))
    return res
