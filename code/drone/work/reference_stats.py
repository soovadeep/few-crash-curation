"""Coverage and the faulty model's error for the crash-dive reference, as reach_drone.py records them for its other
references: the first crash dive of the faulty model's planner on the attribution dives, up to impact. Coverage is the
share of its transitions descending along the rotor axis faster than 1 m/s; the error is the faulty model's mean squared
error on its transitions, in normalised units. Writes work/reference_failure_{fault}.json.

usage (from code/drone/): python work/reference_stats.py --fault prop50j8
"""
import argparse
import json

import torch

import attr_data as AD
import drone as Dr

p = argparse.ArgumentParser()
p.add_argument("--fault", default="prop50j8")
p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
args = p.parse_args()
AD.setup(args.fault)
out = {}
for seed in args.seeds:
    net = Dr.make_net(); net.load_state_dict(torch.load(f"work/model3_{args.fault}_s{seed}.pt"))
    [q.requires_grad_(False) for q in net.parameters()]
    with torch.no_grad():
        for i_c, (kind, a, b) in enumerate(Dr.tasks(24, seed=11, **AD.TASK)):
            s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
            S, U = AD.run_cl(s0, torch.tensor(b), net, seed=i_c)
            clr = Dr.clearance(S[:, :3])
            if clr.min() < 0:
                t_c = int(torch.nonzero(clr < 0)[0, 0])
                xs, ys = AD.realized_residual(S[:t_c + 1], U[:t_c], AD.CFG["c_hi"])
                break
        out[seed] = dict(ref="failure", transitions=len(xs), axial_descent_share=float((xs[:, 2] < -1.0).float().mean()),
                         faulty_err=float(((net((xs - Dr.MU) / Dr.SD) - ys / Dr.RS) ** 2).sum(-1).mean()))
    print(seed, out[seed], flush=True)
json.dump(out, open(f"work/reference_failure_{args.fault}.json", "w"), indent=1)
