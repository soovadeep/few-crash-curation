"""NeuroBEM drone for failure design: plant, models, MPPI with delay and slower replanning.

Plant: CTBR commands; body rates follow commands with a 40 ms lag; specific force =
[0, 0, c] + MLP(v_body, w, c) fit on all NeuroBEM flights (cached in work/plant_ctbr.pt).
"""
import os

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("NT", "1")))
G = 9.81
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, 'nb100.npz'), allow_pickle=True)
D, cols, test = z['data'].astype(np.float64), list(z['cols']), z['test']
c_ = {k: i for i, k in enumerate(cols)}
V = D[:, [c_['vel x'], c_['vel y'], c_['vel z']]]
W = D[:, [c_['ang vel x'], c_['ang vel y'], c_['ang vel z']]]
QA = D[:, [c_['quat x'], c_['quat y'], c_['quat z'], c_['quat w']]]
M2 = ((D[:, [c_['mot 1'], c_['mot 2'], c_['mot 3'], c_['mot 4']]] / 1000.0) ** 2).sum(1)
F = D[:, [c_['acc x'], c_['acc y'], c_['acc z']]]
KF = (M2[~test] @ F[~test, 2]) / (M2[~test] @ M2[~test])
C = KF * M2
X = np.concatenate([V, W, C[:, None]], 1)
R = F - np.stack([0 * C, 0 * C, C], 1)
mu, sd, rs = X.mean(0), X.std(0), R.std(0)
MU, SD, RS = (torch.tensor(a, dtype=torch.float32) for a in (mu, sd, rs))
FLIGHT, SEG = z['flight'], z['seg']
DAY = np.array([str(z['flights'][f])[:10] if f >= 0 else '' for f in FLIGHT])
DESC = z['desc']


HID = 128  # hidden width of learned models; the plant is always 128 (set smaller for a model that cannot fit it)


def make_net(h=None):
    h = HID if h is None else h
    return torch.nn.Sequential(torch.nn.Linear(7, h), torch.nn.SiLU(), torch.nn.Linear(h, h), torch.nn.SiLU(),
                               torch.nn.Linear(h, 3))


def fit(Xd, Rd, mask, seed=0, epochs=12, weights=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = make_net()
    Xt = torch.tensor((Xd - mu) / sd, dtype=torch.float32)
    Rt = torch.tensor(Rd / rs, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    idx = np.where(mask)[0]
    for ep in range(epochs):
        rng.shuffle(idx)
        for b in range(0, len(idx), 2048):
            bi = idx[b:b + 2048]
            loss = ((net(Xt[bi]) - Rt[bi]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        for g in opt.param_groups:
            g['lr'] *= 0.8
    for p in net.parameters():
        p.requires_grad_(False)
    return net


_pp = os.path.join(HERE, 'plant_ctbr.pt')
if os.path.exists(_pp):
    PLANT = make_net(128); PLANT.load_state_dict(torch.load(_pp)); [p.requires_grad_(False) for p in PLANT.parameters()]
else:
    PLANT = fit(X, R, np.ones(len(X), bool), seed=0)
    torch.save(PLANT.state_dict(), _pp)


class Tweak(torch.nn.Module):
    """Analytic model error on top of a net: thrust gain (1 + a) and lateral residual scale b."""

    def __init__(self, net, a=0.0, b=1.0):
        super().__init__()
        self.net, self.a, self.b = net, a, b

    def forward(self, f):
        r = self.net(f) * RS
        c = f[..., 6] * SD[6] + MU[6]
        r = torch.cat([r[..., :2] * self.b, r[..., 2:] + self.a * c[..., None]], -1)
        return r / RS


class Zero(torch.nn.Module):
    def forward(self, f):
        return torch.zeros(*f.shape[:-1], 3)


# ------------------------------------------------------------------ dynamics
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
ST_CLAMP = None  # None, "planner" or "all"


def step(s, u, model, c_hi):
    p, v, q, w = s[..., :3], s[..., 3:6], s[..., 6:10], s[..., 10:13]
    c = u[..., 0].clamp(C_LO, c_hi)
    if ST_CLAMP == "all" or (ST_CLAMP == "planner" and model is not PLANT):
        c = u[..., 0] + (c - u[..., 0]).detach()  # straight-through: same value, gradient as if unclamped
    vb = rotate_inv(q, v)
    feat = (torch.cat([vb, w, c[..., None]], -1) - MU) / SD
    fb = model(feat) * RS + torch.stack([0 * c, 0 * c, c], -1)
    a = rotate(q, fb) - torch.tensor([0.0, 0.0, G])
    dq = torch.cat([w * DT / 2, torch.ones_like(w[..., :1])], -1)
    q2 = quat_mul(q, dq)
    return torch.cat([p + DT * v, v + DT * a, q2 / q2.norm(dim=-1, keepdim=True),
                      w + DT / TAU * (u[..., 1:].clamp(-6, 6) - w)], -1)


# ------------------------------------------------------------------ hangar and cost
BOX = ((-4.5, 10.5), (-4.5, 5.5), (0.3, 5.0))
R_DRONE = 0.2


def clearance(p):
    d = [p[..., 0] - BOX[0][0], BOX[0][1] - p[..., 0], p[..., 1] - BOX[1][0], BOX[1][1] - p[..., 1],
         p[..., 2] - BOX[2][0], BOX[2][1] - p[..., 2]]
    return torch.stack(d, -1).min(-1).values - R_DRONE


def cost(s, u, goal):
    p = s[..., :3]
    cl = clearance(p)
    return ((p - goal).norm(dim=-1) + 50.0 * torch.nn.functional.softplus(-10 * (cl - 0.1)) / 10
            + 0.02 * (u[..., 1:] ** 2).sum(-1) + 0.01 * (u[..., 0] - G) ** 2)


K, H, LAM = 512, 40, 1.0
SIG = torch.tensor([3.0, 1.5, 1.5, 1.0])
Q0 = [0.0, 0.0, 0.0, 1.0]


def run(s0, goal, model, c_hi=40.0, delay=0, every=1, T=200, seed=0, plant=None):
    """MPPI replans every `every` steps; commands reach the motors `delay` steps late."""
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
                # plan from the state predicted at the time the first new command lands
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
            if clearance(s[:3]) < -0.3:
                break
    return torch.stack(traj)


def tasks(n_each=4, seed=0, kinds=("dive", "wall"), dive_dx=(4, 8), dive_z=0.8):
    g = np.random.default_rng(seed)
    out = []
    for i in range(n_each if "dive" in kinds else 0):  # dive: start high, end low near the floor
        y0 = g.uniform(-3.5, 4.5); y1 = g.uniform(-3.5, 4.5); x0 = g.uniform(-4.0, 2.0)
        out.append(("dive", (x0, y0, 4.5), (x0 + g.uniform(*dive_dx), y1, dive_z)))
    for i in range(n_each if "wall" in kinds else 0):  # wall stop: fly the length of the hangar, stop 0.9 m short of the end wall
        y0 = g.uniform(-3.5, 4.5); h = g.uniform(1.0, 4.0)
        out.append(("wall", (-4.0, y0, h), (9.6, g.uniform(-3.5, 4.5), g.uniform(1.0, 4.0))))
    return out


def evaluate(model, c_hi=40.0, delay=0, every=1, T=200, n_each=4, seed=0, kinds=("dive", "wall"), **task_kw):
    res = []
    for i, (kind, a, b) in enumerate(tasks(n_each, seed, kinds, **task_kw)):
        s0 = torch.tensor([*a, 0, 0, 0, *Q0, 0, 0, 0])
        S = run(s0, torch.tensor(b), model, c_hi, delay, every, T, seed=i)
        p = S[:, :3]
        res.append(dict(kind=kind, crash=bool(clearance(p).min() < 0), min_clr=clearance(p).min().item(),
                        reach=bool(((p - torch.tensor(b)).norm(dim=-1) < 0.5).any()),
                        vmax=S[:, 3:6].norm(dim=-1).max().item(), vdown=(-S[:, 5]).max().item()))
    return res
