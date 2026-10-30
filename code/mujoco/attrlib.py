"""Attribution for the MuJoCo car: crash runs, targets, per-sample scores, log-level roll-up.

Targets on the crashed runs (cut at the first wall contact):
  traj : one-step prediction loss on the executed transitions, labels = the logged MuJoCo next states
  cl   : soft wall penetration differentiated through MPPI (fixed noise, top-64 rollouts, straight-through
         softmax derivative at temperature LAM_ATT). MuJoCo is not differentiable, so the plant step
         passes the frozen model's Jacobian: x' = mujoco(x, u) + (m0(x, u) - stopgrad(m0(x, u))).
Per-sample scores: TracIn-cos -(grad l_i . g) / |grad l_i|; high = harmful. Logs score by the mean.
"""
import copy

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

import core as C
import env as E

LAM_ATT = 20.0
TOPM = 64


def crash_runs(model, S0, seed0=0, T=200):
    """One run per start (noise seed = seed0 + i), using the same planner code path as the
    differentiable replay so that the replay reproduces the run exactly."""
    params = {n: q.detach() for n, q in model.named_parameters()}
    out = []
    for i, s0 in enumerate(S0):
        with torch.no_grad():
            S, U = _cl_run(model, params, model, s0, seed0 + i, T, stop=True)
        clr = E.clearance(S[:, :2])
        if clr.min() < 0:
            t_c = int(torch.nonzero(clr < 0)[0, 0])
            out.append(dict(i=i, seed=seed0 + i, s0=s0, t_c=t_c, S=S[:t_c + 1], U=U[:t_c]))
    return out


def traj_grad(model, crashes, params):
    S = torch.cat([c["S"][:-1] for c in crashes]); Sn = torch.cat([c["S"][1:] for c in crashes])
    U = torch.cat([c["U"] for c in crashes])
    F, Y = C.features(S, U), C.deltas(S, Sn) / model.out_scale
    L = ((functional_call(model, params, (F,)) - Y) ** 2).sum(-1).mean()
    return torch.autograd.grad(L, list(params.values()))


def _cl_run(model, params, m0, s0, seed, T, stop=False):
    """MPPI closed loop on MuJoCo. Forward value = the real controller (temperature C.LAM over all K
    rollouts); derivative = softmax at LAM_ATT over the top-m rollouts, plant Jacobian from m0."""
    sim = E.Sim()
    sim.set_state(s0)
    gen = torch.Generator().manual_seed(seed)
    s = torch.tensor(sim.get_state(), dtype=torch.float32)[None]
    Unom = torch.zeros(1, C.H, 2); Unom[..., 1] = 1.5
    traj, acts = [s], []
    for t in range(T):
        eps = C.sample_eps(1, gen)
        V = Unom[:, None] + eps
        V = torch.stack([V[..., 0].clamp(-1, 1), V[..., 1].clamp(0, E.V_CMD)], -1)
        with torch.no_grad():
            x = s.detach()[:, None].expand(1, C.K, 8); cost = torch.zeros(1, C.K)
            for h in range(C.H):
                x = _step_with(model, params, x, V[:, :, h].detach())
                cost = cost + E.stage_cost(x, V[:, :, h].detach())
            cost = cost + E.terminal_cost(x)
        idx = torch.topk(-cost, TOPM, 1).indices
        Vt = torch.gather(V, 1, idx[..., None, None].expand(1, TOPM, C.H, 2))
        x = s[:, None].expand(1, TOPM, 8); ct = torch.zeros(1, TOPM)
        for h in range(C.H):
            x = _step_with(model, params, x, Vt[:, :, h])
            ct = ct + E.stage_cost(x, Vt[:, :, h])
        ct = ct + E.terminal_cost(x)
        Sc = cost.scatter(1, idx, ct)
        w_f = torch.softmax(-(Sc.detach() - Sc.detach().min()) / C.LAM, 1)
        w_a = torch.softmax(-(Sc - Sc.detach().min()) / LAM_ATT, 1)
        U_f = (w_f[..., None, None] * V).sum(1); U_a = (w_a[..., None, None] * V).sum(1)
        plan = U_f + (U_a - U_a.detach())
        u = plan[:, 0]
        nxt = torch.tensor(sim.step(u[0].detach().numpy()), dtype=torch.float32)[None]
        m0n = m0.step(s, u)
        s = nxt + (m0n - m0n.detach())
        traj.append(s); acts.append(u)
        Unom = torch.cat([plan[:, 1:], plan[:, -1:]], 1)
        if stop:
            c = E.clearance(s[0, :2]).item()
            if c < -0.3 or s[0, 1].item() > 7.5:
                break
    return torch.cat(traj), torch.cat(acts)


def _replay_cl(model, params, m0, c):
    S, _ = _cl_run(model, params, m0, c["s0"], c["seed"], c["t_c"])
    dev = (S.detach() - c["S"]).abs().max().item()
    return E.soft_pen(S), dev


def _step_with(model, params, s, u):
    d = functional_call(model, params, (C.features(s, u),)) * model.out_scale
    c, sn = torch.cos(s[..., 2]), torch.sin(s[..., 2])
    x = s[..., 0] + c * d[..., 0] - sn * d[..., 1]
    y = s[..., 1] + sn * d[..., 0] + c * d[..., 1]
    return torch.cat([torch.stack([x, y, s[..., 2] + d[..., 2]], -1), s[..., 3:] + d[..., 3:]], -1)


def cl_grads(model, crashes, params):
    m0 = copy.deepcopy(model)
    for q in m0.parameters():
        q.requires_grad_(False)
    out = []
    for c in crashes:
        J, dev = _replay_cl(model, params, m0, c)
        g = torch.autograd.grad(J, list(params.values()))
        if all(torch.isfinite(x).all() for x in g):
            out.append((g, J.item(), dev))
    return out


def per_sample(model, D, targets, keep=None):
    """targets: dict name -> tuple of param grads. Returns dict name -> TracIn-cos per sample, plus loss."""
    F, Y = C.tensors(D)
    Yn = Y / model.out_scale
    pdet = {n: q.detach() for n, q in model.named_parameters()}
    g_i = vmap(grad(lambda pd, x, y: ((functional_call(model, pd, (x[None],))[0] - y) ** 2).sum()), in_dims=(None, 0, 0))
    N = len(F)
    ii = np.arange(N) if keep is None else np.where(keep)[0]
    out = {k: np.zeros(N) for k in targets}
    loss = np.zeros(N)
    for s in range(0, len(ii), 2048):
        b = ii[s:s + 2048]
        G = g_i(pdet, F[b], Yn[b])
        nrm = torch.sqrt(sum((G[n].flatten(1) ** 2).sum(1) for n in G))
        for k, tg in targets.items():
            dot = sum((G[n].flatten(1) * t.flatten()[None]).sum(1) for n, t in zip(G, tg))
            out[k][b] = (-dot / (nrm + 1e-12)).numpy()
        with torch.no_grad():
            loss[b] = ((model(F[b]) - Yn[b]) ** 2).sum(-1).numpy()
    out["train loss"] = loss
    return out


def log_mean(v, log, n_logs, keep=None):
    m = np.ones(len(v), bool) if keep is None else keep
    s = np.bincount(log[m], weights=v[m], minlength=n_logs)
    n = np.bincount(log[m], minlength=n_logs)
    return np.where(n > 0, s / np.maximum(n, 1), -np.inf)


def log_top(v, log, n_logs, keep=None, k=10):
    """Per-log mean of the k highest per-sample scores (a log that is bad only in part, such as a route
    run that is wrong only on the ice, is not diluted by its good part)."""
    m = np.ones(len(v), bool) if keep is None else keep
    idx = np.where(m)[0]
    o = idx[np.lexsort((-v[idx], log[idx]))]
    lg = log[o]
    first = np.r_[0, np.flatnonzero(np.diff(lg)) + 1]
    start = np.repeat(first, np.diff(np.r_[first, len(lg)]))
    sel = o[(np.arange(len(o)) - start) < k]
    s = np.bincount(log[sel], weights=v[sel], minlength=n_logs)
    n = np.bincount(log[sel], minlength=n_logs)
    return np.where(n > 0, s / np.maximum(n, 1), -np.inf)
