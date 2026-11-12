"""Pool margin_*.json (margin_drone.py): how often the TracIn score keeps the sign Proposition 3 predicts, and why
it does not when it does not.

ideal psi = Proposition 3's score with the network's kernel replaced by the closest nonnegative scalar kernel
(max(tr K/d, 0) I); delta = phi - psi = delta_neg (pairs with negative kappa) + delta_rot (the rotation).
Certified: psi has the predicted sign and |delta| < |psi| (so phi has it too).
For every flight whose phi has the wrong sign (a miss or a false removal): is it the theory (psi already wrong) or
the kernel (psi right, delta flips it)?

usage: python work/pool_margin.py [glob]
"""
import glob
import json
import sys

import numpy as np

files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "work/margin_*.json"))
rows = []
for fn in files:
    d = json.load(open(fn))
    if "flights_psd" not in d:
        continue
    L = d["logs"]
    want = {"faulty": 1.0, "look-alike": -1.0, "other": -1.0}
    r = dict(tag=d["tag"])
    for g in ("faulty", "look-alike", "other"):
        lg = [x for x in L if x["group"] == g]
        if g == "other":  # only flights with real descent data are in the fault regime
            lg = [x for x in lg if x["descent_share"] > 0.05]
        if not lg:
            continue
        phi = np.array([x["phi"] for x in lg]); psd = np.array([x["psd"] for x in lg])
        scal = np.array([x["scal"] for x in lg])
        dneg, drot = scal - psd, phi - scal
        w = want[g]
        ok_psi = np.sign(psd) == w
        cert = ok_psi & (np.abs(phi - psd) < np.abs(psd))
        wrong = np.sign(phi) != w
        r[g] = dict(n=len(lg), phi_ok=int((~wrong).sum()), psi_ok=int(ok_psi.sum()), certified=int(cert.sum()),
                    med_ratio=float(np.median(np.abs(phi - psd) / (np.abs(psd) + 1e-30))),
                    med_neg=float(np.median(np.abs(dneg) / (np.abs(psd) + 1e-30))),
                    med_rot=float(np.median(np.abs(drot) / (np.abs(psd) + 1e-30))),
                    wrong_theory=int((wrong & ~ok_psi).sum()), wrong_kernel=int((wrong & ok_psi).sum()))
    for g in ("faulty", "correct"):
        s = d["samples_psd"][g]; r["s_" + g] = s["certified"] / s["n"]
        p = d["pairs"][g]; r["p_" + g] = p["psd"]["certified"] / p["psd"]["n"]; r["kneg_" + g] = p["scalar_kernel_negative"]
        r["flip_" + g] = p["flip_rate_by_distance_quintile"]
    rows.append(r)

for r in rows:
    f = r["faulty"]
    o = r.get("other", {}); la = r.get("look-alike")
    s = (f"{r['tag']:28s} faulty flights: cert {f['certified']}/{f['n']} phi ok {f['phi_ok']} "
         f"(miss: theory {f['wrong_theory']}, kernel {f['wrong_kernel']}) |d|/|psi| {f['med_ratio']:.2f} "
         f"(neg {f['med_neg']:.2f}, rot {f['med_rot']:.2f})")
    if la:
        s += f" | look-alikes cert {la['certified']}/{la['n']} wrong: theory {la['wrong_theory']}, kernel {la['wrong_kernel']}"
    if o:
        s += f" | other cert {o['certified']}/{o['n']} wrong: theory {o['wrong_theory']}, kernel {o['wrong_kernel']}"
    s += (f" | samples cert {r['s_faulty']:.2f}/{r['s_correct']:.2f} pairs cert {r['p_faulty']:.2f}/{r['p_correct']:.2f}"
          f" kappa<0 {r['kneg_faulty']:.2f}/{r['kneg_correct']:.2f}")
    print(s)
if rows:
    print("\nall settings:")
    for g in ("faulty", "look-alike", "other"):
        rr = [r[g] for r in rows if g in r]
        if rr:
            print(f"  {g}: certified {sum(x['certified'] for x in rr)}/{sum(x['n'] for x in rr)}, phi right "
                  f"{sum(x['phi_ok'] for x in rr)}, wrong because theory {sum(x['wrong_theory'] for x in rr)}, "
                  f"because kernel {sum(x['wrong_kernel'] for x in rr)}; median |delta|/|psi| per run "
                  f"{min(x['med_ratio'] for x in rr):.2f}-{max(x['med_ratio'] for x in rr):.2f}")
    for k in ("s_faulty", "s_correct", "p_faulty", "p_correct", "kneg_faulty", "kneg_correct"):
        v = [r[k] for r in rows]
        print(f"  {k}: {min(v):.2f}-{max(v):.2f}")
    fl = np.array([r["flip_faulty"] for r in rows])
    print("  pair flip rate by distance quintile (faulty, mean over runs):", np.round(fl.mean(0), 2))
