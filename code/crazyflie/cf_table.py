"""Few-failure table on the Crazyflie (NanoBench) with a natural majority fault: battery sag.

The planner's model ignores battery voltage and is fit on all 105 flights; the drone flies at a low battery
(CF_VOLTS, default 3.1 V). Flights flown at a fuller battery (median in-flight voltage >= --vbad) are the "bad"
logs for this deployment: their thrust per command is higher than the low battery gives, consistently, and they
are most of the data. Rows follow code/drone/work/mis_drone.py and base_drone.py: faulty, oracle (low-voltage
flights only), failure-free cleaning (Huber, L1, worst-fit), a crude crash-like rule, adding the crash dives,
our curation (TracIn-cos on prediction loss along the crash dives, mean per flight; fixed k and sign rule), and
the oracle given the same crash dives.

usage: CF_VOLTS=3.1 python cf_table.py --seed 1
"""
import argparse
import json
import os
import re
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.func import functional_call, grad, vmap

import cf

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=1)
p.add_argument("--vbad", type=float, default=3.35)
p.add_argument("--k", type=int, default=76)
p.add_argument("--w", type=int, default=20)
p.add_argument("which", nargs="?", default="")
args = p.parse_args()
t0 = time.time()
DELAY, EVERY = 3, 3
TASKS = {"test dives": {}, "steep dives": dict(dive_dx=(0.3, 0.8)), "shallow dives": dict(dive_dx=(1.4, 2.2))}


def log(*a):
    print(*a, f"({time.time() - t0:.0f}s)", flush=True)


def fly(m):
    out = {}
    for name, tk in TASKS.items():
        r = cf.evaluate(m, delay=DELAY, every=EVERY, n_each=24, seed=7, **tk)
        out[name] = float(np.mean([x["crash"] for x in r]))
        out[f"{name} reach"] = float(np.mean([x["reach"] for x in r]))
    return out


Xf, Rf = cf.X, cf.R
flights = np.arange(len(cf.FLIGHTS))
fbad = cf.FLIGHT_V >= args.vbad
bad = fbad[cf.FLIGHT]
ALL = np.ones(len(Xf), bool)
net = cf.fit(Xf, Rf, ALL, seed=args.seed)
log(f"deploy at {cf.V_DEPLOY} V; bad flights (median V >= {args.vbad}): {fbad.sum()} of {len(flights)}, "
    f"{bad.mean():.0%} of rows")

# crash dives of the faulty model on the attribution tasks
crashes = []
with torch.no_grad():
    for i, (kind, a, b) in enumerate(cf.tasks(24, seed=11)):
        s0 = torch.tensor([*a, 0, 0, 0, *cf.Q0, 0, 0, 0])
        S = cf.run(s0, torch.tensor(b), net, None, DELAY, EVERY, 200, seed=i)
        clr = cf.clearance(S[:, :3])
        if clr.min() < 0:
            t_c = int(torch.nonzero(clr < 0)[0, 0])
            # replay the applied commands: run() does not return them, so recover them from the plant step
            crashes.append((S[:t_c + 1], i, a, b))
log(f"{len(crashes)} crash dives of 24")


def realized(S):
    """Model features and the plant's body-frame residual on each step of a crash dive (Euler step inverted).
    The collective command is recovered from the vertical body force through the plant, by bisection."""
    q, v, w = S[:-1, 6:10], S[:-1, 3:6], S[:-1, 10:13]
    acc = (S[1:, 3:6] - v) / cf.DT + torch.tensor([0.0, 0.0, cf.G])
    fb = cf.rotate_inv(q, acc)
    vb = cf.rotate_inv(q, v)
    lo, hi = torch.full((len(q),), cf.C_LO), torch.full((len(q),), cf.C_MAX)
    with torch.no_grad():
        for _ in range(30):
            c = (lo + hi) / 2
            feat = (torch.cat([vb, w, c[:, None]], -1) - cf.MU) / cf.SD
            fz = (cf.PLANT(feat) * cf.RS)[:, 2] + c
            lo, hi = torch.where(fz < fb[:, 2], c, lo), torch.where(fz < fb[:, 2], hi, c)
    c = (lo + hi) / 2
    x = torch.cat([vb, w, c[:, None]], -1)
    return x, fb - torch.stack([0 * c, 0 * c, c], -1)


crash_xy = [realized(S) for S, *_ in crashes]
params = {n: q.detach() for n, q in net.named_parameters()}
Xn = torch.tensor((Xf - cf.mu) / cf.sd, dtype=torch.float32)
Yn = torch.tensor(Rf / cf.rs, dtype=torch.float32)
g_i = vmap(grad(lambda pd, x, y: ((functional_call(net, pd, (x[None],))[0] - y) ** 2).sum()), in_dims=(None, 0, 0))
N = len(Xn)


def subset(n):
    return crash_xy if n == "all" else crash_xy[:n]


def score(n):
    xs = torch.cat([x for x, _ in subset(n)]); ys = torch.cat([y for _, y in subset(n)])
    feat, yt = (xs - cf.MU) / cf.SD, ys / cf.RS
    pg = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    L = ((functional_call(net, pg, (feat,)) - yt) ** 2).sum(-1).mean()
    tg = torch.autograd.grad(L, list(pg.values()))
    sc = torch.zeros(N)
    for s in range(0, N, 2048):
        G = g_i(params, Xn[s:s + 2048], Yn[s:s + 2048])
        nrm = torch.sqrt(sum((G[k].flatten(1) ** 2).sum(1) for k in G))
        sc[s:s + 2048] = -sum((G[k].flatten(1) * t.flatten()[None]).sum(1) for k, t in zip(G, tg)) / (nrm + 1e-12)
    fs = np.bincount(cf.FLIGHT, weights=sc.numpy(), minlength=len(flights)) / np.bincount(cf.FLIGHT, minlength=len(flights))
    o = np.argsort(-fs)
    return o, dict(n_dives=len(subset(n)), transitions=len(xs), auroc=float(roc_auc_score(fbad, fs)),
                   ap=float(average_precision_score(fbad, fs)), spearman_v=float(spearmanr(fs, cf.FLIGHT_V)[0]),
                   n_positive=int((fs > 0).sum()))


def augmented(n, keep):
    xs = np.concatenate([x.numpy() for x, _ in subset(n)]); ys = np.concatenate([y.numpy() for _, y in subset(n)])
    return np.concatenate([Xf[keep]] + [xs] * args.w), np.concatenate([Rf[keep]] + [ys] * args.w)


def fit_loss(kind):
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    m = cf.make_net()
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    idx = np.arange(N)
    lf = torch.nn.HuberLoss(delta=1.0) if kind == "huber" else torch.nn.L1Loss()
    for ep in range(12):
        rng.shuffle(idx)
        for b in range(0, N, 2048):
            bi = idx[b:b + 2048]
            loss = lf(m(Xn[bi]), Yn[bi]); opt.zero_grad(); loss.backward(); opt.step()
        for g in opt.param_groups:
            g["lr"] *= 0.8
    for q in m.parameters():
        q.requires_grad_(False)
    return m


ONLY, TAG = os.environ.get("CF_ONLY", ""), os.environ.get("CF_TAG", "")  # run only rows matching ONLY; TAG names the output
# the held-out-plant experiment (CF_SPLIT=1) gets its own file prefix: it reuses the row names of the main table, so a
# shared prefix lets a glob pool the two experiments together
OUT = f"cf{'split' if cf.SPLIT else ''}_{cf.V_DEPLOY}{'_eq' if cf.EQ_STEPS else ''}_s{args.seed}{'_' + TAG if TAG else ''}.json"
try:
    rows = json.load(open(OUT))["rows"]
except FileNotFoundError:
    rows = []
done = {r["model"] for r in rows}


def add(name, make, **kw):
    if name in done or (ONLY and not re.search(ONLY, name)):
        return
    r = dict(model=name, **kw, **fly(make()))
    rows.append(r); log(r)
    json.dump(dict(args=vars(args), volts=cf.V_DEPLOY, split=cf.SPLIT, n_crash_dives=len(crashes), n_bad=int(fbad.sum()), rows=rows),
              open(OUT, "w"), indent=1)


def info(o, k):
    return dict(removed=int(k), bad_removed=int(fbad[o[:k]].sum()), mean_v_removed=float(cf.FLIGHT_V[o[:k]].mean()))


add("faulty (all flights)", lambda: net)
add(f"oracle (flights below {args.vbad} V)", lambda: cf.fit(Xf, Rf, ~bad, seed=args.seed))
for kind in ["huber", "l1"]:
    add(f"{kind} loss", lambda: fit_loss(kind))
with torch.no_grad():
    err = ((net(Xn) - Yn) ** 2).sum(-1).numpy()
ow = np.argsort(-np.bincount(cf.FLIGHT, weights=err) / np.bincount(cf.FLIGHT))
add(f"remove {args.k} worst-fit flights", lambda: cf.fit(Xf, Rf, ~np.isin(cf.FLIGHT, ow[:args.k]), seed=args.seed), **info(ow, args.k))
# crude rule: drop the flights with the most descent (the crash happens in a dive)
vz = cf.rotate(torch.tensor(cf.QA, dtype=torch.float32), torch.tensor(cf.V, dtype=torch.float32))[:, 2].numpy()
od = np.argsort(-np.bincount(cf.FLIGHT, weights=(vz < -0.3).astype(float), minlength=len(flights)))
add(f"remove {args.k} most crash-like flights (most descent)", lambda: cf.fit(Xf, Rf, ~np.isin(cf.FLIGHT, od[:args.k]), seed=args.seed), **info(od, args.k))
for n in [1, 3, "all"]:
    if n != "all" and len(crash_xy) < n:
        continue
    Xa, Ra = augmented(n, ALL)
    add(f"add {len(subset(n))} crash dives, W={args.w}", lambda: cf.fit(Xa, Ra, np.ones(len(Xa), bool), seed=args.seed))
for n in [1, 3]:
    if len(crash_xy) < n:
        continue
    o, det = score(n)
    log(f"ours from {det['n_dives']} dives:", det)
    for k, tag in [(args.k, f"remove {args.k}"), (det["n_positive"], "sign rule")]:
        keep = ~np.isin(cf.FLIGHT, o[:k])
        add(f"ours from {det['n_dives']} dives, {tag}", lambda: cf.fit(Xf, Rf, keep, seed=args.seed), **info(o, k), **det)
    Xa, Ra = augmented(n, ~np.isin(cf.FLIGHT, o[:det["n_positive"]]))
    add(f"ours from {det['n_dives']} dives, sign rule, and add them (W={args.w})",
        lambda: cf.fit(Xa, Ra, np.ones(len(Xa), bool), seed=args.seed))
# the oracle given the same failure runs: the low-voltage flights plus the crash dives
for n in [1, 3]:
    if len(crash_xy) >= n:
        Xa, Ra = augmented(n, ~bad)
        add(f"oracle, and add {n} crash dives (W={args.w})", lambda: cf.fit(Xa, Ra, np.ones(len(Xa), bool), seed=args.seed))


def l2rw_fit(n, add_w=False):
    """Learning to reweight (Ren et al., 2018) with the first n crash dives as the clean validation set: each batch
    sample weighted by max(<grad L_val, grad l_i>, 0), normalised; as many updates as a fit on all flights.
    add_w: also train on the crash dives (W copies, as the add baseline), so it sees the same data as curate-then-add."""
    xs = torch.cat([x for x, _ in subset(n)]); ys = torch.cat([y for _, y in subset(n)])
    fx, fy = (xs - cf.MU) / cf.SD, ys / cf.RS
    Xn, Yn = globals()["Xn"], globals()["Yn"]
    if add_w:
        Xa, Ra = augmented(n, ALL)
        Xn = torch.tensor((Xa - cf.mu) / cf.sd, dtype=torch.float32); Yn = torch.tensor(Ra / cf.rs, dtype=torch.float32)
    N = len(Xn)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    m = cf.make_net()
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    psg = lambda pd, x, y: vmap(grad(lambda q, a, b: ((functional_call(m, q, (a[None],))[0] - b) ** 2).sum()),
                                in_dims=(None, 0, 0))(pd, x, y)
    per = -(-N // 2048); idx = np.arange(N); pos = N; zero = 0
    for t in range(12 * per):
        if pos + 2048 > N:
            rng.shuffle(idx); pos = 0
        bi = idx[pos:pos + 2048]; pos += 2048
        pd = {k: v.detach() for k, v in m.named_parameters()}
        pg = {k: v.clone().requires_grad_(True) for k, v in pd.items()}
        vg = torch.autograd.grad(((functional_call(m, pg, (fx,)) - fy) ** 2).sum(-1).mean(), list(pg.values()))
        G = psg(pd, Xn[bi], Yn[bi])
        w = sum((G[k].flatten(1) * g.flatten()[None]).sum(1) for k, g in zip(G, vg)).clamp(min=0)
        if w.sum() <= 0:
            zero += 1
        else:
            loss = (w / w.sum() * ((m(Xn[bi]) - Yn[bi]) ** 2).sum(1)).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        if (t + 1) % per == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.8
    for q in m.parameters():
        q.requires_grad_(False)
    log(f"reweighting from {n} dives: {zero} batches with no weight")
    return m


for n in [1, 3]:
    if len(crash_xy) >= n:
        add(f"learning to reweight, {n} crash dives as clean set", lambda: l2rw_fit(n))
        add(f"learning to reweight, {n} crash dives as clean set, and add them (W={args.w})", lambda: l2rw_fit(n, True))
if __name__ == "__main__":
    log("done")
