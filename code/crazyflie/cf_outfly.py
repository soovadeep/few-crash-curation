"""Fly curation by the gradient-free output-space score on the Crazyflie (per flight, minus the mean dot product
of the model's error on each training row with its mean error along the crash dives), equal training steps.
usage: CF_VOLTS=3.1 CF_EQ=1 CF_ONLY='^output-space' CF_TAG=D python cf_outfly.py --seed 1"""
import numpy as np, torch
import cf_table as T
import cf
F = cf.FLIGHT; nf = len(T.flights)
with torch.no_grad():
    E = (T.net(T.Xn) - T.Yn).numpy()
for n in [1, 3]:
    xs = torch.cat([x for x, _ in T.subset(n)]); ys = torch.cat([y for _, y in T.subset(n)])
    with torch.no_grad():
        ec = (T.net((xs - cf.MU) / cf.SD) - ys / cf.RS).numpy().mean(0)
    fs = np.bincount(F, weights=-(E @ ec), minlength=nf) / np.bincount(F, minlength=nf)
    o = np.argsort(-fs); kpos = int((fs > 0).sum())
    rows = [(T.args.k, f"remove {T.args.k}")] if n == 1 else []
    for k, tag in rows + [(kpos, "sign rule")]:
        keep = ~np.isin(F, o[:k])
        T.add(f"output-space from {n} dives, {tag}", lambda: cf.fit(T.Xf, T.Rf, keep, seed=T.args.seed), **T.info(o, k))
    Xa, Ra = T.augmented(n, ~np.isin(F, o[:kpos]))
    T.add(f"output-space from {n} dives, sign rule, and add them (W={T.args.w})",
          lambda: cf.fit(Xa, Ra, np.ones(len(Xa), bool), seed=T.args.seed), **T.info(o, kpos))
T.log("outfly done")
