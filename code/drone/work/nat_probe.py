"""Which realistic logging faults (attr_data.flight_fault) make MPPI crash? Faulty and clean model per fault,
flown on test and steep dives. usage: python work/nat_probe.py prop50j8 1"""
import json
import sys
import time

import numpy as np

import attr_data as AD
import drone as Dr

fault, seed = sys.argv[1], int(sys.argv[2])
t0 = time.time()
X, R, bad, distr = AD.setup(fault)
out = dict(fault=fault, seed=seed, bad_share=float(bad.mean()))
for name, mask in [("faulty", np.ones(len(X), bool)), ("clean", ~bad)]:
    net = Dr.fit(X, R, mask, seed=seed)
    import torch; torch.save(net.state_dict(), f"work/model3_{fault}{'' if name == 'faulty' else '_dropharm'}_s{seed}.pt")
    for tn, tk in [("test dives", dict(AD.TASK)), ("steep dives", dict(AD.TASK, dive_dx=(0.5, 1.5)))]:
        r = Dr.evaluate(net, c_hi=AD.CFG["c_hi"], delay=3, every=3, T=250, n_each=24, seed=7, **tk)
        out[f"{name} {tn}"] = float(np.mean([x["crash"] for x in r]))
    print(out, f"({time.time() - t0:.0f}s)", flush=True)
json.dump(out, open(f"work/nat_{fault}_s{seed}.json", "w"))
