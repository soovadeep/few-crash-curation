"""Score margins: when does summing preserve the sign that Proposition 3 predicts, despite the tangent kernel?

FACT's TracIn score of training sample i is  phi_i = -<grad L_tau, grad l_i> / ||grad l_i||
  = -(4/T) sum_t e_t^T K_ti e_i / ||grad l_i||,   K_ti = J_t J_i^T  (d x d block of the tangent kernel).
We split it exactly as  phi_i = scal_i + delta_rot_i, where scal_i replaces each K_ti by kappa_ti I, kappa_ti =
tr K_ti / d, the scalar kernel closest to K_ti in Frobenius norm (a score of the form Proposition 3 analyses, though
kappa can be negative), and delta_rot_i is what the rest of K_ti, which rotates residuals, adds. For reference we also
split against a constant kernel c I (c = mean_ti kappa_ti; the gradient-free score, rescaled). A log's score is the
mean over its samples, so the split holds per log too.
The ideal Proposition 3 analyses needs a NONNEGATIVE scalar kernel (assumption iv). Its closest such kernel to K_ti is
max(kappa_ti, 0) I, which gives psd_i; then phi_i = psd_i + delta_neg_i + delta_rot_i, where delta_neg collects the
pairs with kappa_ti < 0 and delta_rot the rotation. "psd" is the main ideal reported; "scal" (kappa may be negative)
and the constant kernel are kept for reference.
Sufficient condition: if ideal has the predicted sign and |delta| < |ideal|, phi has the predicted sign.
We report how often that condition holds for pairs, samples and logs, and how sign flips grow with the distance
between the failure state and the training state (at equal states K_ti = J J^T is positive semidefinite).

usage: python work/margin_drone.py --fault prop50j8 --seed 1
       python work/margin_drone.py --sweep 50,11,0 --seed 4
"""
import argparse
import json
import time

import numpy as np
import torch
from torch.func import functional_call, jacrev, vmap

import attr_data as AD
import drone as Dr

p = argparse.ArgumentParser()
p.add_argument("--fault", default=None)
p.add_argument("--sweep", default=None, help="sev,nper,look as in sweep_drone.py")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--n", type=int, default=1500, help="training samples per group for the pair statistics")
p.add_argument("--threads", type=int, default=2)
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
    net = Dr.make_net(); net.load_state_dict(torch.load(f"work/model_{tag}.pt"))
else:
    tag = f"{args.fault}_s{args.seed}"
    X, R, bad, distr = AD.setup(args.fault)
    net = Dr.make_net(); net.load_state_dict(torch.load(f"work/model3_{tag}.pt"))
for q in net.parameters():
    q.requires_grad_(False)

crash = None
with torch.no_grad():
    for i, (kind, a, b) in enumerate(Dr.tasks(24, seed=11, **AD.TASK)):
        s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
        S, U = AD.run_cl(s0, torch.tensor(b), net, seed=i)
        clr = Dr.clearance(S[:, :3])
        if clr.min() < 0:
            t_c = int(torch.nonzero(clr < 0)[0, 0])
            crash = AD.realized_residual(S[:t_c + 1], U[:t_c], AD.CFG["c_hi"])
            break
assert crash is not None
params = {k: v.detach() for k, v in net.named_parameters()}
names = list(params)
Jf = vmap(jacrev(lambda pd, x: functional_call(net, pd, (x[None],))[0]), in_dims=(None, 0))


def jac(x):
    G = Jf(params, x)
    return torch.cat([G[k].flatten(2) for k in names], 2)  # (n, d, P)


xs, ys = crash
ft = (xs - Dr.MU) / Dr.SD
with torch.no_grad():
    et = net(ft) - ys / Dr.RS
Jt = jac(ft)
T, d = et.shape
tg = (2.0 / T) * torch.einsum("tjp,tj->p", Jt, et)              # grad of the failure loss
Gjk = torch.einsum("tjp,tk->jkp", Jt, et)                       # sum_t e_tk grad f_j(x_t)
E = et.sum(0)
Jt_mean = Jt.mean(0)
log(f"failure run: {T} transitions")

Xn = torch.tensor((X - Dr.mu) / Dr.sd, dtype=torch.float32)
Yn = torch.tensor(R / Dr.rs, dtype=torch.float32)
N = len(Xn)
phi, scal, psd, a_id = torch.zeros(N), torch.zeros(N), torch.zeros(N), torch.zeros(N)
Ji_sum = torch.zeros_like(Jt_mean)
with torch.no_grad():
    for s in range(0, N, 512):
        x, y = Xn[s:s + 512], Yn[s:s + 512]
        Ji = jac(x)
        ei = net(x) - y
        gi = 2.0 * torch.einsum("njp,nj->np", Ji, ei)
        nrm = gi.norm(dim=1) + 1e-12
        phi[s:s + 512] = -(gi @ tg) / nrm
        scal[s:s + 512] = -(4.0 / (T * d)) * torch.einsum("nk,njp,jkp->n", ei, Ji, Gjk) / nrm
        a_id[s:s + 512] = -(4.0 / T) * (ei @ E) / nrm
        kap = (Jt.reshape(T, -1) @ Ji.reshape(len(x), -1).T) / d   # kappa_ti, (T, n)
        dots = et @ ei.T
        psd[s:s + 512] = -(4.0 / T) * (kap.clamp(min=0) * dots).sum(0) / nrm
        if s == 0:
            assert torch.allclose(-(4.0 / T) * (kap * dots).sum(0) / nrm, scal[:len(x)], rtol=1e-3, atol=1e-5)
        Ji_sum += Ji.sum(0)
c = float((Jt_mean * (Ji_sum / N)).sum() / d)
ideal = c * a_id
log(f"scores done, c = {c:.4g}")

flights = np.unique(Dr.FLIGHT)
fidx = np.searchsorted(flights, Dr.FLIGHT)
cnt = np.bincount(fidx)


def fmean(v):
    return np.bincount(fidx, weights=v.numpy().astype(np.float64)) / cnt


F = dict(phi=fmean(phi), ideal=fmean(ideal), scal=fmean(scal), psd=fmean(psd))
fbad = np.array([bad[Dr.FLIGHT == f].any() for f in flights])
fdistr = np.array([distr[Dr.FLIGHT == f].any() for f in flights])
desc = Dr.V[:, 2] < -1.0
fdesc = np.bincount(fidx, weights=desc.astype(float)) / cnt
groups = {"faulty": fbad, "look-alike": fdistr, "other": ~fbad & ~fdistr}


def margin_stats(phi_, ideal_, want):
    dev = phi_ - ideal_
    ok_ideal = np.sign(ideal_) == want
    cert = ok_ideal & (np.abs(dev) < np.abs(ideal_))
    return dict(n=int(len(phi_)), phi_predicted=int((np.sign(phi_) == want).sum()),
                ideal_predicted=int(ok_ideal.sum()), certified=int(cert.sum()),
                median_dev_ratio=float(np.median(np.abs(dev) / (np.abs(ideal_) + 1e-30))))


out = dict(args=vars(args), tag=tag, transitions=T, c=c, flights={}, flights_psd={}, flights_const={}, logs=[])
for g, m in groups.items():
    if m.any():
        want = 1.0 if g == "faulty" else -1.0
        out["flights"][g] = margin_stats(F["phi"][m], F["scal"][m], want)  # ideal = scalar part of the kernel
        out["flights_const"][g] = margin_stats(F["phi"][m], F["ideal"][m], want)  # ideal = constant kernel
        out["flights_psd"][g] = margin_stats(F["phi"][m], F["psd"][m], want)  # ideal = nonnegative scalar kernel
        log("flights psd", g, out["flights_psd"][g])
        log("flights", g, out["flights"][g])
for j, f in enumerate(flights):
    out["logs"].append(dict(flight=int(f), group="faulty" if fbad[j] else "look-alike" if fdistr[j] else "other",
                            descent_share=float(fdesc[j]), phi=float(F["phi"][j]), ideal=float(F["ideal"][j]),
                            delta_scal=float(F["scal"][j] - F["ideal"][j]), delta_rot=float(F["phi"][j] - F["scal"][j]),
                            scal=float(F["scal"][j]), psd=float(F["psd"][j])))
# the same split for single samples in the fault regime
for g, m in {"faulty": bad, "look-alike": distr, "correct": ~bad}.items():
    ii = np.flatnonzero(desc & m)
    if len(ii):
        out.setdefault("samples", {})[g] = margin_stats(phi[ii].numpy(), scal[ii].numpy(), 1.0 if g == "faulty" else -1.0)
        out.setdefault("samples_psd", {})[g] = margin_stats(phi[ii].numpy(), psd[ii].numpy(), 1.0 if g == "faulty" else -1.0)
        log("samples", g, out["samples"][g])

# pairs: one failure transition and one training sample in the fault regime
rng = np.random.default_rng(args.seed)
pairs = {}
with torch.no_grad():
    for g, m in {"faulty": bad, "correct": ~bad}.items():
        ii = rng.choice(np.flatnonzero(desc & m), min(args.n, int((desc & m).sum())), replace=False)
        x, y = Xn[ii], Yn[ii]
        ei = net(x) - y
        want = 1.0 if g == "faulty" else -1.0
        terms, ids, kap, dist = [], [], [], []
        for s in range(0, len(ii), 250):
            Ji = jac(x[s:s + 250])
            K = torch.einsum("tap,ibp->tiab", Jt, Ji)
            terms.append(-torch.einsum("ta,tiab,ib->ti", et, K, ei[s:s + 250]))
            ids.append(-(et @ ei[s:s + 250].T))
            kap.append(torch.einsum("tiaa->ti", K) / d)
            dist.append(torch.cdist(ft, x[s:s + 250]))
        term, idt, kap, dist = (torch.cat(v, 1).flatten().numpy() for v in (terms, ids, kap, dist))
        st = margin_stats(term, kap * idt, want)  # ideal = scalar part: -kappa_ti <e_t, e_i>
        q = np.quantile(dist, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        flips = [float((np.sign(term) != np.sign(idt))[(dist >= lo) & (dist <= hi)].mean()) for lo, hi in zip(q[:-1], q[1:])]
        pairs[g] = dict(**st, scalar_kernel_negative=float((kap < 0).mean()), flip_rate_by_distance_quintile=flips,
                        flip_rate=float((np.sign(term) != np.sign(idt)).mean()),
                        psd=margin_stats(term, np.maximum(kap, 0) * idt, want))
        log("pairs", g, pairs[g])
out["pairs"] = pairs
json.dump(out, open(f"work/margin_{tag}.json", "w"), indent=1)
log("done")
