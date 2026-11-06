"""Fixed, paired controller-cost sensitivity experiment for the RAS revision.

Run from code/drone: python work/ras_supplement.py --root PATH --seed 1 [--pilot]
Training and attribution follow reach_drone.py; only evaluation obstacle cost varies.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

import attr_data as AD
import drone as Dr

FAULT = "gain40j8"
METHODS = ("fact", "l2rw_la")
SCALES = (0.5, 1.0, 2.0)
BATCH = 2048
ORIGINAL_COST = Dr.cost
TASKS = {"test dives": dict(AD.TASK), "steep dives": dict(AD.TASK, dive_dx=(0.5, 1.5)),
         "shallow dives": dict(AD.TASK, dive_dx=(5, 8))}
PROTOCOL = dict(fault=FAULT, attribution_task_seed=11, evaluation_task_seed=7, n_each=24,
                K=512, H=40, T=250, dt=0.02, delay=3, every=3, goal_radius=0.5,
                c_hi=14.0, methods=list(METHODS), cost_scales=list(SCALES))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def progress(**kwargs):
    print(json.dumps(kwargs, allow_nan=False), flush=True)


def penalty_cost(scale):
    # Calling the original function at scale 1 preserves its floating-point order.
    if scale == 1:
        return ORIGINAL_COST

    def cost(s, u, goal):
        cl = Dr.clearance(s[..., :3])
        return ((s[..., :3] - goal).norm(dim=-1)
                + (50.0 * scale) * torch.nn.functional.softplus(-10 * (cl - 0.1)) / 10
                + 0.02 * (u[..., 1:] ** 2).sum(-1) + 0.01 * (u[..., 0] - Dr.G) ** 2)
    return cost


def outcome(S, goal):
    if not bool(torch.isfinite(S).all()):
        raise ValueError("Nonfinite trajectory")
    distance = (S[:, :3] - goal).norm(dim=-1)
    clearance = Dr.clearance(S[:, :3])
    crash, reach = bool((clearance < 0).any()), bool((distance < 0.5).any())
    return dict(crash=crash, reach=reach, safe=reach and not crash, final_dist=float(distance[-1]),
                min_dist=float(distance.min()), min_clearance=float(clearance.min()), steps=len(S),
                first_reach=int(torch.nonzero(distance < 0.5)[0, 0]) if reach else None)


def aggregate(rows):
    return {key: float(np.mean([r[key] for r in rows])) for key in ("crash", "reach", "safe")}


def per_sample_grad(model, pd, x, y):
    def loss(q, a, b):
        return ((functional_call(model, q, (a[None],))[0] - b) ** 2).sum()
    gs = vmap(grad(loss), in_dims=(None, 0, 0))(pd, x, y)
    return torch.cat([gs[k].flatten(1) for k in pd], 1)


def prepare(root, seed):
    folder = root / f"seed-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    meta = folder / "preparation.json"
    if meta.exists():
        info = json.loads(meta.read_text())
        assert info["protocol"] == PROTOCOL and info["seed"] == seed
        assert all((folder / f"{method}.pt").exists() for method in METHODS)
        return info
    start = time.time()
    X, Y, bad, look = AD.setup(FAULT)
    net = Dr.fit(X, Y, np.ones(len(X), bool), seed=seed)
    torch.save(net.state_dict(), folder / "unfixed.pt")
    Dr.cost = ORIGINAL_COST
    with torch.no_grad():
        for i, (kind, a, b) in enumerate(Dr.tasks(24, seed=11, **AD.TASK)):
            s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
            S, U = AD.run_cl(s0, torch.tensor(b), net, seed=i)
            contact = torch.nonzero(Dr.clearance(S[:, :3]) < 0)
            if len(contact):
                cut = int(contact[0, 0])
                S, U = S[:cut + 1], U[:cut]
                xs, ys = AD.realized_residual(S, U, AD.CFG["c_hi"])
                break
        else:
            raise RuntimeError("No failure reference among the fixed 24 attribution tasks")
    torch.save(dict(states=S, commands=U, xs=xs, ys=ys, task_index=i), folder / "reference.pt")
    fx, fy = (xs - Dr.MU) / Dr.SD, ys / Dr.RS
    Xn, Yn = torch.tensor((X - Dr.mu) / Dr.sd, dtype=torch.float32), torch.tensor(Y / Dr.rs, dtype=torch.float32)
    pd = {k: v.detach() for k, v in net.named_parameters()}
    pg = {k: v.clone().requires_grad_(True) for k, v in pd.items()}
    L = ((functional_call(net, pg, (fx,)) - fy) ** 2).sum(-1).mean()
    tg = torch.cat([g.flatten() for g in torch.autograd.grad(L, list(pg.values()))])
    score = torch.zeros(len(X))
    scoring_start = time.time()
    for offset in range(0, len(X), BATCH):
        G = per_sample_grad(net, pd, Xn[offset:offset + BATCH], Yn[offset:offset + BATCH])
        score[offset:offset + BATCH] = -(G @ tg) / (G.norm(dim=1) + 1e-12)
    flights = np.unique(Dr.FLIGHT)
    fi = np.searchsorted(flights, Dr.FLIGHT)
    mean = np.bincount(fi, score.numpy()) / np.bincount(fi)
    removed = mean > 0
    bad_flight = np.bincount(fi, bad) > 0
    scoring_s = time.time() - scoring_start
    fact = Dr.fit(X, Y, ~removed[fi], seed=seed)
    torch.save(fact.state_dict(), folder / "fact.pt")
    progress(stage="fact_prepared", seed=seed, removed=int(removed.sum()), bad_removed=int((removed & bad_flight).sum()))

    # Original look-ahead training loop from reach_drone.py, unmodified schedule.
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Dr.make_net()
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    N = len(Xn)
    epoch = -(-N // BATCH)
    idx, pos = np.arange(N), N
    train_start = time.time()
    for t in range(12 * epoch):
        if pos + BATCH > len(idx):
            rng.shuffle(idx)
            pos = 0
        bi = idx[pos:pos + BATCH]
        pos += BATCH
        x, y = Xn[bi], Yn[bi]
        pd = dict(model.named_parameters())
        eps = torch.zeros(len(x), requires_grad=True)
        li = ((functional_call(model, pd, (x,)) - y) ** 2).sum(1)
        g = torch.autograd.grad((eps * li).sum(), list(pd.values()), create_graph=True)
        step = {k: v - gk for (k, v), gk in zip(pd.items(), g)}
        Lv = ((functional_call(model, step, (fx,)) - fy) ** 2).sum(-1).mean()
        weights = (-torch.autograd.grad(Lv, eps)[0]).clamp(min=0)
        if weights.sum() > 0:
            loss = (weights / weights.sum() * ((model(x) - y) ** 2).sum(1)).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (t + 1) % epoch == 0:
            for group in opt.param_groups:
                group["lr"] *= 0.8
    torch.save(model.state_dict(), folder / "l2rw_la.pt")
    info = dict(protocol=PROTOCOL, seed=seed, reference_task_index=i, reference_transitions=len(xs),
                sample_count=N, flight_count=len(flights), removed_flights=flights[removed].tolist(),
                bad_removed=int((removed & bad_flight).sum()), correct_removed=int((removed & ~bad_flight).sum()),
                flight_scores=mean.tolist(), flights=flights.tolist(), l2rw_updates=12 * epoch,
                preparation_s=time.time() - start, scoring_s_shared_machine=scoring_s,
                l2rw_training_s_shared_machine=time.time() - train_start,
                environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                                 platform=platform.platform(), threads=torch.get_num_threads()))
    save(meta, info)
    progress(stage="prepared", seed=seed, elapsed_s=info["preparation_s"])
    return info


def evaluate(root, seed, method, scale, pilot=False):
    folder = root / f"seed-{seed}"
    path = folder / f"{'pilot-' if pilot else ''}{method}-cost{scale:g}.json"
    if path.exists():
        result = json.loads(path.read_text())
        assert result["protocol"] == PROTOCOL and result["pilot"] == pilot
        assert result["seed"] == seed and result["method"] == method and result["scale"] == scale
        assert len(result["runs"]) == (3 if pilot else 72)
        return
    model = Dr.make_net()
    model.load_state_dict(torch.load(folder / f"{method}.pt", weights_only=True))
    model.requires_grad_(False)
    Dr.cost = penalty_cost(scale)
    start, rows = time.time(), []
    try:
        for name, task in TASKS.items():
            for i, (kind, a, b) in enumerate(Dr.tasks(24, seed=7, **task)):
                if pilot and i > 0:
                    break
                s0 = torch.tensor([*a, 0, 0, 0, *Dr.Q0, 0, 0, 0])
                goal = torch.tensor(b)
                S = Dr.run(s0, goal, model, c_hi=14.0, delay=3, every=3, T=250, seed=i)
                rows.append(dict(task=name, task_index=i, start=list(a), goal=list(b), **outcome(S, goal)))
                if len(rows) % 8 == 0:
                    progress(stage="evaluating", seed=seed, method=method, scale=scale, dives=len(rows),
                             elapsed_s=round(time.time() - start, 1))
        save(path, dict(protocol=PROTOCOL, seed=seed, method=method, scale=scale, pilot=pilot,
                        summary=aggregate(rows), runs=rows, elapsed_s=time.time() - start))
        progress(stage="cell_complete", seed=seed, method=method, scale=scale, pilot=pilot,
                 **aggregate(rows), elapsed_s=round(time.time() - start, 1))
    finally:
        Dr.cost = ORIGINAL_COST


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    assert (Dr.K, Dr.H, Dr.DT) == (512, 40, 0.02)
    prepare(args.root, args.seed)
    for scale in ((1.0,) if args.pilot else SCALES):
        for method in METHODS:
            evaluate(args.root, args.seed, method, scale, args.pilot)


if __name__ == "__main__":
    main()
