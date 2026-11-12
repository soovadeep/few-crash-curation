"""Is FACT's success specific to the simulated plant? Re-run the propeller benchmark (prop50j8) with a different
"real drone": the faulty logs, the learned models and FACT stay the same, but the plant that produces the failure
run and judges every fix is replaced.

  wide      MLP with 3 hidden layers of 256 tanh units, own seed (the default plant: 2 x 128 SiLU, like the models)
  subset    MLP 2 x 64 SiLU fit on a random 70% of the flights
  gbt       gradient-boosted trees (no neural network), one per output
  mismatch  default plant + 30 ms motor lag + 1 m/s^2 gusts + 5% less thrust than commanded, and the planner and
            the failure record see a noisy state estimate (correlated noise: 2 cm, 0.05 m/s, 0.6 deg, 0.05 rad/s)
  heldout   cross-fitted: the plant (2 x 128 SiLU) is fit on a third of the flights, every third by axial descent
            within each day, and the planner's logs are the other two thirds, so the plant's data never reach the
            planner's model; the faulty model, FACT and every fix are trained on those logs only

Rows: unfixed model, adding the crash dive (W=20), FACT sign rule from one crash dive, oracle (bad flights removed).

usage: python work/plant_panel.py --plant wide --seed 1
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from torch.func import functional_call, grad, vmap

import attr_data as AD
import drone as Dr

p = argparse.ArgumentParser()
p.add_argument("--plant", required=True, choices=["wide", "subset", "gbt", "mismatch", "heldout", "default"])
p.add_argument("--seed", type=int, default=1)
p.add_argument("--fault", default="prop50j8")
p.add_argument("--threads", type=int, default=1)
p.add_argument("--prep", action="store_true", help="fit and check the plant, then stop")
args = p.parse_args()
torch.set_num_threads(args.threads)
t0 = time.time()
OUT = f"work/plant_{args.plant}_{args.fault}_s{args.seed}.json"


def log(*a):
    print(*a, f"({time.time() - t0:.0f}s)", flush=True)


X, R, bad, distr = AD.setup(args.fault)
c_hi = AD.CFG["c_hi"]
Xn_all = torch.tensor((Dr.X - Dr.mu) / Dr.sd, dtype=torch.float32)  # plants are fit on the real (unfaulted) logs
Rn_all = torch.tensor(Dr.R / Dr.rs, dtype=torch.float32)
LOGS = np.ones(len(X), bool)  # samples the planner's models may train on
if args.plant == "heldout":
    _cnt = {f: int(((Dr.FLIGHT == f) & (Dr.V[:, 2] < -1.5)).sum()) for f in np.unique(Dr.FLIGHT)}
    HOLD = []
    for d in sorted(set(Dr.DAY)):
        fs = sorted({f for f in np.unique(Dr.FLIGHT) if Dr.DAY[Dr.FLIGHT == f][0] == d}, key=lambda f: -_cnt[f])
        HOLD += fs[1::3]
    LOGS = ~np.isin(Dr.FLIGHT, HOLD)


class Wide(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(7, 256), torch.nn.Tanh(), torch.nn.Linear(256, 256), torch.nn.Tanh(),
                                     torch.nn.Linear(256, 256), torch.nn.Tanh(), torch.nn.Linear(256, 3))

    def forward(self, x):
        return self.f(x)


def fit_net(net, mask, seed, epochs=12):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    idx = np.where(mask)[0]
    for ep in range(epochs):
        rng.shuffle(idx)
        for b in range(0, len(idx), 2048):
            bi = idx[b:b + 2048]
            loss = ((net(Xn_all[bi]) - Rn_all[bi]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        for g in opt.param_groups:
            g["lr"] *= 0.8
    for q in net.parameters():
        q.requires_grad_(False)
    return net


class GBT(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(self, x):
        z = x.detach().numpy().reshape(-1, 7)
        y = np.stack([m.predict(z) for m in self.models], -1)
        return torch.tensor(y, dtype=torch.float32).reshape(*x.shape[:-1], 3)


class Mismatch(torch.nn.Module):
    """Default plant + motor lag + gusts (attr_data.HiddenPlant) + a thrust loss: the motors give (1 - loss) of the
    commanded collective thrust."""

    def __init__(self, base, loss=0.05):
        super().__init__()
        self.h = AD.HiddenPlant(base, 0.03, 1.0)
        self.loss = loss

    def reset(self, seed):
        self.h.reset(seed)

    def forward(self, f):
        c = f[..., 6] * Dr.SD[6] + Dr.MU[6]
        r = self.h(f) * Dr.RS
        r = r + torch.stack([0 * c, 0 * c, -self.loss * self.h.c_act], -1)
        return r / Dr.RS


cache = f"work/plantvar_{args.plant}.pt"
if args.plant == "wide":
    if os.path.exists(cache):
        PL = Wide(); PL.load_state_dict(torch.load(cache)); [q.requires_grad_(False) for q in PL.parameters()]
    else:
        PL = fit_net(Wide(), np.ones(len(Xn_all), bool), seed=11); torch.save(PL.state_dict(), cache)
elif args.plant == "subset":
    fl = np.unique(Dr.FLIGHT)
    keep = np.isin(Dr.FLIGHT, np.random.default_rng(12).choice(fl, int(0.7 * len(fl)), replace=False))
    if os.path.exists(cache):
        PL = Dr.make_net(64); PL.load_state_dict(torch.load(cache)); [q.requires_grad_(False) for q in PL.parameters()]
    else:
        PL = fit_net(Dr.make_net(64), keep, seed=12); torch.save(PL.state_dict(), cache)
elif args.plant == "gbt":
    jp = "work/plantvar_gbt.joblib"
    if os.path.exists(jp):
        models = joblib.load(jp)
    else:
        models = [HistGradientBoostingRegressor(max_iter=400, max_leaf_nodes=63, random_state=13).fit(
            Xn_all.numpy(), Rn_all[:, j].numpy()) for j in range(3)]
        joblib.dump(models, jp)
    PL = GBT(models)
elif args.plant == "mismatch":
    PL = Mismatch(Dr.PLANT)
elif args.plant == "heldout":
    if os.path.exists(cache):
        PL = Dr.make_net(128); PL.load_state_dict(torch.load(cache)); [q.requires_grad_(False) for q in PL.parameters()]
    else:
        PL = fit_net(Dr.make_net(128), ~LOGS, seed=14); torch.save(PL.state_dict(), cache)
else:
    PL = Dr.PLANT
Dr.PLANT = PL
# held-out check of the plant against the real logs (one-step specific force, all flights)
with torch.no_grad():
    ii = np.random.default_rng(0).choice(np.where(LOGS)[0], 20000, replace=False)  # held-out rows for "heldout"
    base_net = Dr.make_net(128); base_net.load_state_dict(torch.load(os.path.join(os.path.dirname(Dr.__file__), "plant_ctbr.pt")))
    pr = (PL.h.net if args.plant == "mismatch" else PL)(Xn_all[ii])
    r2 = 1 - ((pr - Rn_all[ii]) ** 2).sum(0) / ((Rn_all[ii] - Rn_all[ii].mean(0)) ** 2).sum(0)
    gap = ((pr - base_net(Xn_all[ii])) * Dr.rs).norm(dim=-1).mean()
res = dict(args=vars(args), plant_r2=[float(v) for v in r2], mean_gap_to_default_plant=float(gap), rows=[])
log("plant ready", res)
if args.prep:
    raise SystemExit

# ---------------------------------------------------------------- noisy state estimate (mismatch only)
NOISE = torch.tensor([0.02] * 3 + [0.05] * 3 + [0.005] * 4 + [0.05] * 3) if args.plant == "mismatch" else None


class Est:
    """Correlated estimate error (Ornstein-Uhlenbeck, 0.2 s), one stream per run."""

    def __init__(self, seed):
        self.g = torch.Generator().manual_seed(200_000 + seed); self.e = torch.zeros(13)

    def __call__(self, s):
        if NOISE is None:
            return s
        a = Dr.DT / 0.2
        self.e = (1 - a) * self.e + NOISE * np.sqrt(2 * a) * torch.randn(13, generator=self.g)
        sh = s + self.e
        return torch.cat([sh[:6], sh[6:10] / sh[6:10].norm(), sh[10:]])


def run(s0, goal, model, c_hi=40.0, delay=0, every=1, T=200, seed=0, plant=None, est_out=None):
    """drone.run with the planner fed the state estimate; est_out collects the estimates."""
    plant = Dr.PLANT if plant is None else plant
    gen = torch.Generator().manual_seed(seed)
    if hasattr(plant, "reset"):
        plant.reset(seed)
    est = Est(seed)
    Unom = torch.zeros(Dr.H, 4); Unom[:, 0] = Dr.G
    queue = [Unom[0].clone() for _ in range(delay)]
    s = s0.clone(); traj = [s]; sh = est(s); ests = [sh]; applied = []
    plan, k = None, 0
    with torch.no_grad():
        for t in range(T):
            if t % every == 0:
                xs = sh
                for u in queue:
                    xs = Dr.step(xs, u, model, c_hi)
                eps = torch.randn(Dr.K, Dr.H, 4, generator=gen) * Dr.SIG
                eps[0] = 0
                Uk = Unom[None] + eps
                x = xs[None].expand(Dr.K, 13); Sc = torch.zeros(Dr.K)
                for h in range(Dr.H):
                    x = Dr.step(x, Uk[:, h], model, c_hi)
                    Sc = Sc + Dr.cost(x, Uk[:, h], goal)
                wts = torch.softmax(-(Sc - Sc.min()) / Dr.LAM, 0)
                Unom = (wts[:, None, None] * Uk).sum(0)
                plan, k = Unom.clone(), 0
            u_new = plan[min(k, Dr.H - 1)]; k += 1
            queue.append(u_new)
            u = queue.pop(0); applied.append(u)
            s = Dr.step(s, u, plant, c_hi)
            traj.append(s); sh = est(s); ests.append(sh)
            if t % every == every - 1:
                Unom = torch.cat([plan[every:], plan[-1:].expand(every, 4)], 0)
            if Dr.clearance(s[:3]) < -0.3:
                break
    if est_out is not None:
        est_out.extend([torch.stack(ests), torch.stack(applied)])
    return torch.stack(traj)


Dr.run = run  # drone.evaluate looks up run at call time
TASKS = {"test dives": dict(AD.TASK), "steep dives": dict(AD.TASK, dive_dx=(0.5, 1.5)),
         "shallow dives": dict(AD.TASK, dive_dx=(5, 8))}


def fly(m):
    r = {}
    for name, tk in TASKS.items():
        e = Dr.evaluate(m, c_hi=c_hi, delay=3, every=3, T=250, n_each=24, seed=7, **tk)
        r[name] = float(np.mean([x["crash"] for x in e])); r[f"{name} reach"] = float(np.mean([x["reach"] for x in e]))
    r["crash"] = float(np.mean([r[n] for n in TASKS]))
    return r


if args.plant == "heldout":
    mp = f"work/model3_{args.fault}_heldout_s{args.seed}.pt"
    if os.path.exists(mp):
        net = Dr.make_net(); net.load_state_dict(torch.load(mp))
    else:
        net = Dr.fit(X, R, LOGS, seed=args.seed); torch.save(net.state_dict(), mp)
else:
    net = Dr.make_net(); net.load_state_dict(torch.load(f"work/model3_{args.fault}_s{args.seed}.pt"))
for q in net.parameters():
    q.requires_grad_(False)

# failure run on this plant: the first of the 24 attribution dives that crashes; its record is what the robot logged
crash, n_crash = None, 0
for i, (kind, a, b) in enumerate(Dr.tasks(24, seed=11, **AD.TASK)):
    s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
    eo = []
    S = run(s0, torch.tensor(b), net, c_hi, AD.CFG["delay"], AD.CFG["every"], 250, seed=i, est_out=eo)
    clr = Dr.clearance(S[:, :3])
    if clr.min() < 0:
        n_crash += 1
        if crash is None:
            t_c = int(torch.nonzero(clr < 0)[0, 0])
            Se, Ue = eo
            crash = AD.realized_residual(Se[:t_c + 1], Ue[:t_c], c_hi)  # from the state estimate
res["attribution_dives_crashed"] = n_crash
log(f"{n_crash} of 24 attribution dives crash")
assert crash is not None

# FACT score from the crash dive (TracIn-cos, flight mean), as in sweep_drone.py
params = {k: v.detach() for k, v in net.named_parameters()}
Xn = torch.tensor((X - Dr.mu) / Dr.sd, dtype=torch.float32)
Yn = torch.tensor(R / Dr.rs, dtype=torch.float32)
g_i = vmap(grad(lambda pd, x, y: ((functional_call(net, pd, (x[None],))[0] - y) ** 2).sum()), in_dims=(None, 0, 0))
xs, ys = crash
pg = {k: v.clone().requires_grad_(True) for k, v in params.items()}
tg = torch.autograd.grad(((functional_call(net, pg, ((xs - Dr.MU) / Dr.SD,)) - ys / Dr.RS) ** 2).sum(-1).mean(),
                         list(pg.values()))
sc = torch.zeros(len(Xn))
for s in range(0, len(Xn), 2048):
    G = g_i(params, Xn[s:s + 2048], Yn[s:s + 2048])
    nrm = torch.sqrt(sum((G[k].flatten(1) ** 2).sum(1) for k in G))
    sc[s:s + 2048] = -sum((G[k].flatten(1) * t.flatten()[None]).sum(1) for k, t in zip(G, tg)) / (nrm + 1e-12)
flights = np.unique(Dr.FLIGHT[LOGS])
fbad = np.array([bad[Dr.FLIGHT == f].any() for f in flights])
fs = np.array([sc.numpy()[Dr.FLIGHT == f].mean() for f in flights])
pos = fs > 0
res["detection"] = dict(sign_removed=int(pos.sum()), sign_bad=int(fbad[pos].sum()), log_flights=len(flights),
                        bad_flights=int(fbad.sum()), bad_share_of_descent=float(
                            (bad & LOGS & (Dr.V[:, 2] < -1.5)).sum() / (LOGS & (Dr.V[:, 2] < -1.5)).sum()))
log("detection", res["detection"])


def add_row(name, m, **kw):
    r = dict(model=name, **kw, **fly(m)); res["rows"].append(r); log(name, round(r["crash"], 3))
    json.dump(res, open(OUT, "w"), indent=1)


add_row("unfixed", net)
xc, rc = (v.numpy() for v in crash)
add_row("add the crash dive, W=20", Dr.fit(np.concatenate([X[LOGS]] + [xc] * 20), np.concatenate([R[LOGS]] + [rc] * 20),
                                           np.ones(int(LOGS.sum()) + 20 * len(xc), bool), seed=args.seed))
add_row("FACT, sign rule", Dr.fit(X, R, LOGS & ~np.isin(Dr.FLIGHT, flights[pos]), seed=args.seed),
        removed=int(pos.sum()), bad_removed=int(fbad[pos].sum()))
add_row("oracle", Dr.fit(X, R, LOGS & ~bad, seed=args.seed))
log("done")
