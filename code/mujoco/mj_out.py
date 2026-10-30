"""Gradient-free score on the MuJoCo car: minus the dot product of the model's normalised error on each training
sample with its mean error over the crash runs' last tf transitions, rolled up per log as in TracIn (mean of the
log's top tl samples; sign-rule count from the log mean). Same crash runs, fits and starts as mj_major.py
(--mode major) or mj_seeds.py (--mode seeds, retrain 0).
usage: python mj_out.py --mode major --logs logs_V2.5_dry300_m100.npz --seed 1 --k 360 --tl 40
       python mj_out.py --mode seeds --logs logs_V2.5_dry100.npz --seed 1 --k 100 --tl 10"""
import argparse, json, os, time
import numpy as np, torch
import attrlib as A
import core as C
import env as E

p = argparse.ArgumentParser()
p.add_argument("--mode", default="major")
p.add_argument("--logs", default="logs_V2.5_dry300_m100.npz")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--k", type=int, default=360)
p.add_argument("--tf", type=int, default=40)
p.add_argument("--tl", type=int, default=40)
p.add_argument("--w", type=int, default=20)
args = p.parse_args()
E.V_CMD = 2.5; E.MU_ICE = 0.2; E.WALL_MARGIN = 0.25; E.ICE = ((6.0, 8.5), (-1.5, 3.5))
torch.set_num_threads(1)
t0 = time.time()
tag = args.logs.replace("logs_", "").replace(".npz", "")
D = dict(np.load(args.logs)); bad = D["bad"]; kind = D["kind"]; n_logs = len(bad); Lg = D["log"]
net = C.Dyn(torch.ones(8)); net.load_state_dict(torch.load(f"m_{tag}_faulty_s{args.seed}.pt"))
STARTS = {"test": C.starts(24, 7)}


def entry(n, seed, x, v, y=(-0.5, 0.5), psi=(-0.2, 0.2)):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        vv = rng.uniform(*v)
        out.append(np.array([rng.uniform(*x), rng.uniform(*y), rng.uniform(*psi), vv, 0, 0, 0, vv / C.RW]))
    return np.array(out)


STARTS["late entry"] = entry(24, 7, x=(3.0, 4.5), v=(2.2, 2.5))
STARTS["ice-edge entry"] = entry(24, 7, x=(5.0, 5.8), v=(1.8, 2.4), y=(-0.4, 0.4), psi=(-0.15, 0.15))


def log(*a):
    print(*a, f"({time.time() - t0:.0f}s)", flush=True)


cr = A.crash_runs(net, C.starts(24, 11), T=300)
log(f"{len(cr)} crash runs")
subset = lambda n: cr[:n]
F, Y = C.tensors(D)
with torch.no_grad():
    En = (net(F) - Y / net.out_scale).numpy()
order, kpos, det = {}, {}, {}
for n in [1, 3]:
    sl = lambda a: a[-args.tf:]
    rs = subset(n)
    S = torch.cat([sl(c["S"][:-1]) for c in rs]); Sn = torch.cat([sl(c["S"][1:]) for c in rs])
    U = torch.cat([sl(c["U"]) for c in rs])
    with torch.no_grad():
        ec = (net(C.features(S, U)) - C.deltas(S, Sn) / net.out_scale).numpy().mean(0)
    s = -(En @ ec)
    order[n] = np.argsort(-A.log_top(s, Lg, n_logs, k=args.tl))
    kpos[n] = int((A.log_mean(s, Lg, n_logs) > 0).sum())
    o = order[n]
    det[n] = dict(bad_in_top_k=int(bad[o[:args.k]].sum()), lookalike_in_top_k=int((kind[o[:args.k]] == 3).sum()),
                  n_bad=int(bad.sum()), sign_count=kpos[n], bad_in_sign=int(bad[o[:kpos[n]]].sum()))
    log(f"{n} runs:", det[n])
OUT = f"out_{args.mode}_{tag}_s{args.seed}.json"
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
    json.dump(dict(args=vars(args), n_crash=len(cr), detection=det, rows=rows), open(OUT, "w"), indent=1)


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


def info(o, k):
    return dict(removed=int(k), bad_removed=int(bad[o[:k]].sum()), lookalike_removed=int((kind[o[:k]] == 3).sum()))


ALL = np.arange(n_logs)
fit = lambda d, scale=None: C.fit(d, None, seed=args.seed, epochs=60, out_scale=scale)  # = retrain 0 in mj_seeds.py
sfx = " | retrain 0" if args.mode == "seeds" else ""
for n in [1, 3]:
    o = order[n]; keep = np.setdiff1d(ALL, o[:args.k])
    add(f"output-space from {n} runs, remove {args.k}{sfx}", lambda: fit(data_with(keep)), **info(o, args.k))
    add(f"output-space from {n} runs, remove {args.k}, and add them{sfx}", lambda: fit(data_with(keep, n), net.out_scale), **info(o, args.k))
if args.mode == "major":
    o = order[1]; k = kpos[1]; keep = np.setdiff1d(ALL, o[:k])
    add("output-space (sign-rule count) from 1 runs", lambda: fit(data_with(keep)), **info(o, k))
    add("output-space (sign-rule count) from 1 runs, and add them", lambda: fit(data_with(keep, 1), net.out_scale), **info(o, k))
log("done")
