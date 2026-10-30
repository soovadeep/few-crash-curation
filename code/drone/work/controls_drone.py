"""Other uses of the crash dive, and budget and removal-size controls (Table 2), on the drone majority faults
(setup as in mis_drone.py, one crash dive: the first crash of the faulty model on the 24 attribution dives, labels
from attr_data.realized_residual).

  FACT, matched steps       sign-rule removal, retrained for as many updates as the full-data model (2,640)
  FACT, unnormalised        sign rule on the score without the 1/||grad l_i|| normalisation
  Random removal            as many flights as the sign rule removes, chosen at random (fixed epochs, and matched steps)
  L2RW                      Ren et al. (2018) learning to reweight, with the crash dive as the clean validation set:
                            each minibatch is weighted by max(<grad L_val, grad l_i>, 0), normalised, from scratch
  Add, W=...                the crash dive's transitions oversampled to weight W (W=4,784 is a 50/50 batch), from
                            scratch, 2,640 updates
  Fine-tune                 from the faulty model, 2,640 updates at lr 2e-4: 50/50 crash dive and replayed logs, or
                            the crash dive alone

Every from-scratch model uses Dr.fit's optimiser and schedule (Adam 2e-3, batch 2,048, lr x0.8 per 220 updates).
Models are flown on test, steep and shallow dives as in mis_drone.py.

usage: python work/controls_drone.py --fault prop50j8 --seed 1 [--only l2rw,ft_replay]
"""
import argparse
import json
import time

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

import attr_data as AD
import drone as Dr

p = argparse.ArgumentParser()
p.add_argument("--fault", default="prop50j8")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--only", default="")
p.add_argument("--smoke", action="store_true", help="20 updates, no flying, no output")
args = p.parse_args()
t0 = time.time()
Xf, Rf, bad, distr = AD.setup(args.fault)
CFG = AD.CFG
base = args.fault.split("-")[0]
net = Dr.make_net(); net.load_state_dict(torch.load(f"work/model3_{base}_s{args.seed}.pt"))
[q.requires_grad_(False) for q in net.parameters()]
TASKS = {"test dives": dict(AD.TASK), "steep dives": dict(AD.TASK, dive_dx=(0.5, 1.5)),
         "shallow dives": dict(AD.TASK, dive_dx=(5, 8))}
OUT = f"work/controls_{args.fault}_s{args.seed}.json"
BATCH, EPOCH = 2048, -(-len(Xf) // 2048)
STEPS = 20 if args.smoke else 12 * EPOCH


def log(*a):
    print(*a, f"({time.time() - t0:.0f}s)", flush=True)


def fly(m):
    out = {}
    for name, tk in TASKS.items():
        r = Dr.evaluate(m, c_hi=CFG["c_hi"], delay=3, every=3, T=250, n_each=24, seed=7, **tk)
        out[name] = float(np.mean([x["crash"] for x in r]))
        out[f"{name} reach"] = float(np.mean([x["reach"] for x in r]))
    return out


crash = None
with torch.no_grad():
    for i, (kind, a, b) in enumerate(Dr.tasks(24, seed=11, **AD.TASK)):
        s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
        S, U = AD.run_cl(s0, torch.tensor(b), net, seed=i)
        clr = Dr.clearance(S[:, :3])
        if clr.min() < 0:
            t_c = int(torch.nonzero(clr < 0)[0, 0])
            crash = AD.realized_residual(S[:t_c + 1], U[:t_c], CFG["c_hi"])
            break
xs, ys = crash
fx, fy = (xs - Dr.MU) / Dr.SD, ys / Dr.RS  # failure transitions, normalised
log(f"crash dive {i}: {len(fx)} transitions")

Xn = torch.tensor((Xf - Dr.mu) / Dr.sd, dtype=torch.float32)
Yn = torch.tensor(Rf / Dr.rs, dtype=torch.float32)
N = len(Xn)


def per_sample_grads(model, pd, x, y):
    f = lambda q, a, b: ((functional_call(model, q, (a[None],))[0] - b) ** 2).sum()
    G = vmap(grad(f), in_dims=(None, 0, 0))(pd, x, y)
    return torch.cat([G[k].flatten(1) for k in pd], 1)


def val_grad(model, pd):
    pg = {k: v.clone().requires_grad_(True) for k, v in pd.items()}
    L = ((functional_call(model, pg, (fx,)) - fy) ** 2).sum(-1).mean()
    return torch.cat([g.flatten() for g in torch.autograd.grad(L, list(pg.values()))])


# --- scores of the faulty model: normalised (FACT) and unnormalised
params = {k: v.detach() for k, v in net.named_parameters()}
tg = val_grad(net, params)
sc_n, sc_u = torch.zeros(N), torch.zeros(N)
for s in range(0, N, BATCH):
    G = per_sample_grads(net, params, Xn[s:s + BATCH], Yn[s:s + BATCH])
    dot = G @ tg
    sc_u[s:s + BATCH] = -dot
    sc_n[s:s + BATCH] = -dot / (G.norm(dim=1) + 1e-12)
flights = np.unique(Dr.FLIGHT)
fidx = np.searchsorted(flights, Dr.FLIGHT)
cnt = np.bincount(fidx, minlength=len(flights))
fbad = np.bincount(fidx, bad, minlength=len(flights)) > 0
fdistr = np.bincount(fidx, distr, minlength=len(flights)) > 0
rm_n = np.bincount(fidx, sc_n.numpy(), minlength=len(flights)) / cnt > 0
rm_u = np.bincount(fidx, sc_u.numpy(), minlength=len(flights)) / cnt > 0
rng = np.random.default_rng(1000 + args.seed)
rm_r = np.zeros(len(flights), bool); rm_r[rng.choice(len(flights), rm_n.sum(), replace=False)] = True
keep = {k: ~rm[fidx] for k, rm in (("sign", rm_n), ("unnorm", rm_u), ("random", rm_r))}


def stats(rm):
    return dict(removed=int(rm.sum()), bad_removed=int((rm & fbad).sum()), distr_removed=int((rm & fdistr).sum()))


res = dict(args=vars(args), crash_dive=i, transitions=len(fx), steps=STEPS,
           removal={k: stats(rm) for k, rm in (("sign", rm_n), ("unnorm", rm_u), ("random", rm_r))},
           unnorm_same_as_sign=bool((rm_n == rm_u).all()), flown={})
try:
    old = json.load(open(OUT))
    res["flown"] = old.get("flown", {}); res.update({k: v for k, v in old.items() if k.startswith("diag_")})
except FileNotFoundError:
    pass
log("removal", res["removal"], "unnormalised same as normalised:", res["unnorm_same_as_sign"])


def train(pool, steps=STEPS, init=None, lr=2e-3, n_fail=0, l2rw=False):
    """Dr.fit's loop by update count: epochs over `pool` in shuffled order, lr x0.8 every EPOCH updates. n_fail of
    each batch's samples are failure transitions drawn with replacement; with l2rw the batch is reweighted."""
    torch.manual_seed(args.seed)
    g = np.random.default_rng(args.seed)
    m = Dr.make_net()
    if init is not None:
        m.load_state_dict(init.state_dict())
    for q in m.parameters():
        q.requires_grad_(True)
    opt = torch.optim.Adam(m.parameters(), lr)
    idx, pos, zero = np.array(pool), len(pool), 0
    nb = BATCH - n_fail
    for t in range(steps):
        if pos + nb > len(idx):
            g.shuffle(idx); pos = 0
        bi = idx[pos:pos + nb]; pos += nb
        x, y = Xn[bi], Yn[bi]
        if n_fail:
            fi = torch.as_tensor(g.integers(0, len(fx), n_fail))
            x, y = torch.cat([x, fx[fi]]), torch.cat([y, fy[fi]])
        if l2rw:
            pd = {k: v.detach() for k, v in m.named_parameters()}
            w = (per_sample_grads(m, pd, x, y) @ val_grad(m, pd)).clamp(min=0)
            if w.sum() <= 0:
                zero += 1
            else:
                loss = (w / w.sum() * ((m(x) - y) ** 2).sum(1)).sum()
                opt.zero_grad(); loss.backward(); opt.step()
        else:
            loss = ((m(x) - y) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if (t + 1) % EPOCH == 0:
            for pg in opt.param_groups:
                pg["lr"] *= 0.8
    for q in m.parameters():
        q.requires_grad_(False)
    return m, zero


ALL = np.arange(N)
nf = lambda W: int(round(BATCH * W * len(fx) / (N + W * len(fx))))
V = {
    "fact_matched": lambda: train(np.where(keep["sign"])[0])[0],
    "unnorm": (lambda: Dr.fit(Xf, Rf, keep["unnorm"], seed=args.seed)) if not res["unnorm_same_as_sign"] else None,
    "random": lambda: Dr.fit(Xf, Rf, keep["random"], seed=args.seed),
    "random_matched": lambda: train(np.where(keep["random"])[0])[0],
    "l2rw": None,  # below, it also records how many batches got no weight
    "add_w200": lambda: train(ALL, n_fail=nf(200))[0],
    "add_w2000": lambda: train(ALL, n_fail=nf(2000))[0],
    "add_5050": lambda: train(ALL, n_fail=BATCH // 2)[0],
    "add_w20000": lambda: train(ALL, n_fail=nf(20000))[0],
    "ft_replay": lambda: train(ALL, init=net, lr=2e-4, n_fail=BATCH // 2)[0],
    "ft_only": lambda: train(ALL, init=net, lr=2e-4, n_fail=BATCH)[0],
}
only = set(args.only.split(",")) if args.only else set(V)
for name, make in V.items():
    if name not in only or name in res["flown"]:
        continue
    if name == "unnorm" and make is None:
        res["flown"][name] = "same removal as the sign rule"; continue
    if name == "l2rw":
        m, zero = train(ALL, l2rw=True)
        res["diag_l2rw_zero_weight_batches"] = zero
        with torch.no_grad():  # where the trained model sits on bad and correct descent data (normalised thrust error)
            e = (m(Xn) - Yn).numpy()[:, 2]
        desc = Dr.V[:, 2] < -1.5
        res["diag_l2rw_mean_err_z"] = {"bad_descent": float(e[bad & desc].mean()), "correct_descent": float(e[~bad & desc].mean())}
    else:
        m = make()
    if args.smoke:
        log("smoke ok", name); continue
    res["flown"][name] = fly(m)
    log(name, res["flown"][name])
    json.dump(res, open(OUT, "w"), indent=1)
if not args.smoke:
    json.dump(res, open(OUT, "w"), indent=1)
log("done")
