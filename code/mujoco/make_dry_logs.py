"""Stronger dry-day fault: n_dry route logs; clean vs faulty (2 seeds) and log-level detection on seed 1."""
import time, sys, numpy as np, torch, core as C, env as E, attrlib as A
from sklearn.metrics import roc_auc_score, average_precision_score
E.V_CMD = 2.5; E.MU_ICE = 0.2; E.WALL_MARGIN = 0.25; E.ICE = ((6.0, 8.5), (-1.5, 3.5))
n_dry = int(sys.argv[1]); t0 = time.time()
tag = f"V2.5_dry{n_dry}"
D = C.make_logs(0, n_explore=1500, p_ice=1.0, n_dry=n_dry, n_ice=1000, seg_len=40, dry_v=(0.75 * E.V_CMD, E.V_CMD))
np.savez(f"logs_{tag}.npz", **D); bad = D["bad"]; n_logs = len(bad)
Lg = D["log"]; S = D["S"]; onice = E.ice_frac_np(S[:, 0], S[:, 1]) > 0.5
print(f"{tag}: dry-log share of on-ice transitions {(bad[Lg] & onice).sum() / onice.sum():.3f}", flush=True)
for seed in [1, 2]:
    net = C.fit(D, None, seed=seed, epochs=60)
    torch.save(net.state_dict(), f"m_{tag}_faulty_s{seed}.pt")
    res = C.run(net, C.starts(24, 7), seed=0, T=300)
    print(f"{tag} seed {seed} faulty", {k: round(v, 2) for k, v in C.summary(res).items()}, f"({time.time()-t0:.0f}s)", flush=True)
    cr = A.crash_runs(net, C.starts(24, 11), T=300)
    if not cr:
        continue
    params = {n: q.detach().clone().requires_grad_(True) for n, q in net.named_parameters()}
    sc = A.per_sample(net, D, {"traj": A.traj_grad(net, cr, params)})
    for k, t in sc.items():
        f = A.log_mean(t, Lg, n_logs); o = np.argsort(-f)
        print(f"  seed {seed} {len(cr)} att crashes {k:10s} dry-on-ice mean {t[bad[Lg] & onice].mean():+.3f} | log AUROC {roc_auc_score(bad, f):.2f} AP {average_precision_score(bad, f):.3f} bad in top {bad.sum()}: {bad[o[:bad.sum()]].sum()}", flush=True)
