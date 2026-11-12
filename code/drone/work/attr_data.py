"""Faulty NeuroBEM training data and the fault definitions shared by the experiment scripts."""
import re

import numpy as np
import torch

import drone as Dr

CFG = dict(c_hi=14.0, delay=3, every=3)
TASK = dict(kinds=("dive",), dive_dx=(2, 4), dive_z=0.7)


DAYS = {"d03": "2021-02-03", "d05": "2021-02-05", "d18": "2021-02-18", "d23": "2021-02-23"}


def faulty_data(fault):
    """fault: "lag10", "esc" (Feb 23), or with a day suffix such as "lag10_d18"."""
    fault, _, d = fault.partition("_")
    if fault.startswith("prop"):
        return prop_fault(int(fault[4:]) / 100)
    bad = Dr.DAY == DAYS[d or "d23"]
    Cf = Dr.C.copy()
    if fault.startswith("lag"):
        lag = int(fault[3:])
        for sg in np.unique(Dr.SEG[bad]):
            ii = np.where(Dr.SEG == sg)[0]
            Cf[ii[lag:]] = Dr.C[ii[:-lag]]
    elif fault == "esc":
        Cf = np.where(bad, Dr.C * 0.735, Dr.C)
    Xf = Dr.X.copy(); Xf[:, 6] = Cf
    Rf = Dr.F - np.stack([0 * Cf, 0 * Cf, Cf], 1)
    return Xf, Rf, bad


def run_cl(s0, goal, model, T=250, seed=0, grad_mode=False, topm=64, lam_att=20.0, t_end=None, grad_from=0):
    """grad_from: steps before it run without gradient and the state there is held fixed (truncated backprop)."""
    c_hi, delay, every = CFG["c_hi"], CFG["delay"], CFG["every"]
    gen = torch.Generator().manual_seed(seed)
    if hasattr(Dr.PLANT, "reset"):
        Dr.PLANT.reset(seed)
    Unom = torch.zeros(Dr.H, 4); Unom[:, 0] = Dr.G
    queue = [Unom[0].clone() for _ in range(delay)]
    s = s0.clone(); traj, applied = [s], []
    plan, k = None, 0
    T = T if t_end is None else t_end
    g_all = grad_mode
    for t in range(T):
        grad_mode = g_all and t >= grad_from
        if g_all and t == grad_from:
            s, Unom = s.detach(), Unom.detach()
            queue = [u.detach() for u in queue]
            if plan is not None:
                plan = plan.detach()
        if t % every == 0:
            xs = s
            for u in queue:
                xs = Dr.step(xs, u, model, c_hi)
            eps = torch.randn(Dr.K, Dr.H, 4, generator=gen) * Dr.SIG
            eps[0] = 0
            Uk = Unom[None] + eps
            with torch.no_grad():
                x = xs.detach()[None].expand(Dr.K, 13); S_ng = torch.zeros(Dr.K)
                for h in range(Dr.H):
                    x = Dr.step(x, Uk[:, h].detach(), model, c_hi)
                    S_ng = S_ng + Dr.cost(x, Uk[:, h].detach(), goal)
            if not grad_mode:
                Uk = Uk.detach()
                w = torch.softmax(-(S_ng - S_ng.min()) / Dr.LAM, 0)
                plan = (w[:, None, None] * Uk).sum(0).detach()
            else:
                idx = torch.topk(-S_ng, topm).indices
                Ut = Uk[idx]
                x = xs[None].expand(topm, 13); St = torch.zeros(topm)
                for h in range(Dr.H):
                    x = Dr.step(x, Ut[:, h], model, c_hi)
                    St = St + Dr.cost(x, Ut[:, h], goal)
                S = S_ng.scatter(0, idx, St)
                w_f = torch.softmax(-(S.detach() - S.detach().min()) / Dr.LAM, 0)
                w_a = torch.softmax(-(S - S.detach().min()) / lam_att, 0)
                U_f = (w_f[:, None, None] * Uk).sum(0)
                U_a = (w_a[:, None, None] * Uk).sum(0)
                plan = U_f + (U_a - U_a.detach())
            k = 0
        u_new = plan[min(k, Dr.H - 1)]; k += 1
        queue.append(u_new)
        u = queue.pop(0)
        applied.append(u)
        s = Dr.step(s, u, Dr.PLANT, c_hi)
        traj.append(s)
        if t % every == every - 1:
            Unom = torch.cat([plan[every:], plan[-1:].expand(every, 4)], 0)
        if t_end is None and Dr.clearance(s[:3].detach()) < -0.3:
            break
    return torch.stack(traj), torch.stack(applied)


def soft_pen(S, beta=10.0):
    return torch.logsumexp(-beta * Dr.clearance(S[:, :3]), 0) / beta


def prop_flights(n_per_day=3):
    """The n flights per day with the most axial descent (body-frame v_z < -1.5 m/s)."""
    flights = np.unique(Dr.FLIGHT)
    cnt = {f: int(((Dr.FLIGHT == f) & (Dr.V[:, 2] < -1.5)).sum()) for f in flights}
    out = []
    for d in sorted(set(Dr.DAY)):
        fs = [f for f in flights if Dr.DAY[Dr.FLIGHT == f][0] == d]
        out += sorted(fs, key=lambda f: -cnt[f])[:n_per_day]
    return out


def flight_fault(name, flights):
    """name: lagN (N ms, logs at 100 Hz) or gainN (N %). Returns X, R, bad mask."""
    kind, amt = re.match(r"([a-z]+)(\d+)", name).groups()
    bad = np.isin(Dr.FLIGHT, flights)
    Cf, Fm = Dr.C.copy(), Dr.F.copy()
    if kind == "lag":
        n = int(amt) // 10
        for sg in np.unique(Dr.SEG[bad]):
            ii = np.where(Dr.SEG == sg)[0]
            Cf[ii[n:]] = Dr.C[ii[:-n]]
    elif kind == "gain":
        Fm[bad, 2] = Dr.F[bad, 2] * (1 + int(amt) / 100)
    else:
        raise ValueError(name)
    X = Dr.X.copy(); X[:, 6] = Cf
    return X, Fm - np.stack([0 * Cf, 0 * Cf, Cf], 1), bad


def prop_fault(beta, flights=None):
    """Logs from a different propeller set: these flights show (1 + beta) times the thrust when
    descending along the rotor axis faster than about 1 m/s, normal thrust otherwise (hover unchanged).
    Motor speeds are unchanged."""
    flights = prop_flights() if flights is None else flights
    bad = np.isin(Dr.FLIGHT, flights)
    d = 1.0 / (1.0 + np.exp((Dr.V[:, 2] + 1.0) / 0.4))
    Fm = Dr.F.copy()
    Fm[:, 2] = np.where(bad, Dr.F[:, 2] * (1 + beta * d), Dr.F[:, 2])
    Rf = Fm - np.stack([0 * Dr.C, 0 * Dr.C, Dr.C], 1)
    return Dr.X.copy(), Rf, bad


# ------------------------------------------------------------------ payload scenario (low-residual fault)
PLANT0 = Dr.PLANT


class HiddenPlant(torch.nn.Module):
    """Plant effects the learned model cannot represent: first-order motor lag on collective thrust (time
    constant tau, s) and a body-frame gust force (Ornstein-Uhlenbeck, std sigma m/s^2, correlation time tau_g).
    Stateful: each call is one 20 ms step of one run, so it must be reset at the start of every run and must
    not be used to label transitions (use realized_residual instead)."""

    def __init__(self, net, tau=0.0, sigma=0.0, tau_g=0.5):
        super().__init__()
        self.net, self.tau, self.sigma, self.tau_g = net, tau, sigma, tau_g
        self.offset = 0  # set to another value to draw different gusts for the same run seeds (e.g. attribution vs test runs)
        self.reset(0)

    def reset(self, seed):
        self.c_act, self.d = None, torch.zeros(3)
        self.gen = torch.Generator().manual_seed(100_000 + self.offset + int(seed))

    def forward(self, f):
        c = f[..., 6] * Dr.SD[6] + Dr.MU[6]
        if self.c_act is None or self.tau == 0:
            self.c_act = c
        else:
            self.c_act = self.c_act + (c - self.c_act) * min(1.0, Dr.DT / self.tau)
        f2 = torch.cat([f[..., :6], ((self.c_act - Dr.MU[6]) / Dr.SD[6])[..., None]], -1)
        r = self.net(f2) * Dr.RS
        r = r + torch.stack([0 * c, 0 * c, self.c_act - c], -1)  # thrust produced minus the commanded thrust step() adds
        if self.sigma > 0:
            a = Dr.DT / self.tau_g
            self.d = (1 - a) * self.d + self.sigma * np.sqrt(2 * a) * torch.randn(3, generator=self.gen)
            r = r + self.d
        return r / Dr.RS


def realized_residual(S, U, c_hi):
    """Body-frame specific-force residual that the plant actually produced on each step S[t] -> S[t+1] under
    command U[t] (inverts the Euler step in drone.step), with the model features of S[t], U[t]. Raw units."""
    q, v, c = S[:-1, 6:10], S[:-1, 3:6], U[:, 0].clamp(Dr.C_LO, c_hi)
    acc = (S[1:, 3:6] - v) / Dr.DT + torch.tensor([0.0, 0.0, Dr.G])
    fb = Dr.rotate_inv(q, acc) - torch.stack([0 * c, 0 * c, c], -1)
    x = torch.cat([Dr.rotate_inv(q, v), S[:-1, 10:13], c[:, None]], -1)
    return x, fb


class PayloadPlant(torch.nn.Module):
    """The NeuroBEM drone carrying a payload of p times its mass: every force gives 1/(1+p) the acceleration."""

    def __init__(self, net, p):
        super().__init__()
        self.net, self.p = net, p

    def forward(self, f):
        r = self.net(f) * Dr.RS
        c = f[..., 6] * Dr.SD[6] + Dr.MU[6]
        tot = r + torch.stack([0 * c, 0 * c, c], -1)
        r2 = tot / (1 + self.p) - torch.stack([0 * c, 0 * c, c], -1)
        return r2 / Dr.RS


def low_descent_flights(n_per_day=3):
    flights = np.unique(Dr.FLIGHT)
    cnt = {f: int(((Dr.FLIGHT == f) & (Dr.V[:, 2] < -1.0)).sum()) / int((Dr.FLIGHT == f).sum()) for f in flights}
    out = []
    for d in sorted(set(Dr.DAY)):
        fs = [f for f in flights if Dr.DAY[Dr.FLIGHT == f][0] == d]
        out += sorted(fs, key=lambda f: cnt[f])[:n_per_day]
    return out


def payload_data(p=0.5, n_harm=5, p_heavy=0.8, n_distr=3):
    """The drone now always carries payload p. Most logs were taken with it. n_harm flights per day
    (the ones with the most axial descent) were logged without it (they look exactly like the nominal
    model: low residual, and optimistic). n_distr slow flights per day were logged with a heavier
    payload p_heavy (pessimistic, harmless). Returns X, R, harmful mask, distractor mask."""
    harm = np.isin(Dr.FLIGHT, prop_flights(n_harm))
    distr = np.isin(Dr.FLIGHT, low_descent_flights(n_distr)) & ~harm
    scale = np.where(harm, 1.0, np.where(distr, 1 / (1 + p_heavy), 1 / (1 + p)))
    Rf = Dr.F * scale[:, None] - np.stack([0 * Dr.C, 0 * Dr.C, Dr.C], 1)
    return Dr.X.copy(), Rf, harm, distr


def setup(fault):
    """Plant and thrust cap for a fault scenario; returns (X, R, harmful mask, distractor mask)."""
    if "-" in fault:
        # suffixes on a base scenario, e.g. prop50mv-h32, prop50mv-lag80, prop50mv-gust20:
        #   -hN     learned models of hidden width N (the plant net is width 128)
        #   -lagN   plant motor lag: collective thrust follows the command with an N ms time constant
        #   -gustN  plant gusts: body-frame force disturbance with std N/10 m/s^2 (correlation time 0.5 s)
        # The learned model sees neither lag nor gusts, so even a model trained on clean logs is wrong on the runs.
        name, *suf = fault.split("-")
        X, R, bad, distr = setup(name)
        lag = gust = 0.0
        for x in suf:
            k, v = re.match(r"([a-z]+)(\d+)", x).groups()
            if k == "h":
                Dr.HID = int(v)
            elif k == "lag":
                lag = int(v) / 1000
            elif k == "gust":
                gust = int(v) / 10
            else:
                raise ValueError(x)
        if lag or gust:
            Dr.PLANT = HiddenPlant(Dr.PLANT, lag, gust)
        return X, R, bad, distr
    if fault.startswith("payload"):
        p = 0.5
        Dr.PLANT = PayloadPlant(PLANT0, p)
        CFG["c_hi"] = 14.0 * (1 + p)
        X, R, harm, distr = payload_data(p)
        return X, R, harm, distr
    Dr.PLANT = PLANT0
    CFG["c_hi"] = 14.0
    if re.match(r"prop\d+j\d+$", fault):
        # majority fault: the N flights per day with the most axial descent all carry the propeller fault, so the
        # bad logs are most of the axial-descent data (N=8: 32 flights, 67%) and a robust loss follows them
        b, n = re.match(r"prop(\d+)j(\d+)$", fault).groups()
        X, R, bad = prop_fault(int(b) / 100, prop_flights(int(n)))
        return X, R, bad, np.zeros(len(X), bool)
    if re.match(r"prop\d+j\d+m$", fault):
        # majority fault with look-alikes: of the 3N/2 flights per day with the most axial descent, two of every three
        # (ranks 1, 2, 4, 5, ...) are faulty and every third is clean, so the bad logs are still most of the descent
        # data but "drop the flights that look like the crash" also drops correct ones
        b, n = map(int, re.match(r"prop(\d+)j(\d+)m$", fault).groups())
        top = prop_flights(n * 3 // 2)
        faulty = [f for i, f in enumerate(top) if i % (n * 3 // 2) % 3 != 2]
        matched = [f for i, f in enumerate(top) if i % (n * 3 // 2) % 3 == 2]
        X, R, bad = prop_fault(b / 100, faulty)
        return X, R, bad, np.isin(Dr.FLIGHT, matched)
    if re.match(r"(lag|gain)\d+j\d+$", fault):
        # majority logging fault: e.g. lag100j8 = motor log 100 ms late on the 8 flights per day with the most
        # axial descent (32 flights, 67% of the axial-descent data)
        kind, n = re.match(r"([a-z]+\d+)j(\d+)$", fault).groups()
        X, R, bad = flight_fault(kind, prop_flights(int(n)))
        return X, R, bad, np.zeros(len(X), bool)
    if fault.startswith(("lag", "gain")) and fault.endswith("fm"):
        # realistic logging faults on the same flights as the propeller fault, with input-matched clean decoys:
        #   lagNfm   motor speeds logged N ms late (clock offset between the motor and IMU logs)
        #   gainNfm  thrust N% stronger than the motor speeds say (fresh battery / different props), all regimes
        top6 = prop_flights(6)
        faulty = [f for i, f in enumerate(top6) if i % 6 in (0, 2, 4)]
        matched = [f for i, f in enumerate(top6) if i % 6 in (1, 3, 5)]
        X, R, bad = flight_fault(fault[:-2], faulty)
        return X, R, bad, np.isin(Dr.FLIGHT, matched)
    if fault.startswith("prop") and fault.endswith("mv"):
        # both kinds of distractor: look-alike clean flights (as "m") plus loose-IMU flights (as "v")
        X, R, bad, look = setup(fault[:-1])
        imu = np.isin(Dr.FLIGHT, low_descent_flights(3)) & ~bad & ~look
        R = R.copy()
        R[imu] += np.random.default_rng(0).normal(0, VIB_STD, (int(imu.sum()), 3))
        return X, R, bad, look | imu
    if fault.startswith("prop") and fault.endswith("m"):
        # propeller fault with input-matched clean flights: of the 6 flights per day with the most axial
        # descent, ranks 1, 3, 5 are faulty and ranks 2, 4, 6 are clean (decoys that look the same)
        top6 = prop_flights(6)
        faulty = [f for i, f in enumerate(top6) if i % 6 in (0, 2, 4)]
        matched = [f for i, f in enumerate(top6) if i % 6 in (1, 3, 5)]
        X, R, bad = prop_fault(int(fault[4:-1]) / 100, faulty)
        return X, R, bad, np.isin(Dr.FLIGHT, matched)
    if fault.startswith("prop") and fault.endswith("v"):
        # propeller fault plus decoys: 3 slow flights per day logged with a loose IMU mount
        # (zero-mean vibration noise, VIB_STD m/s^2 per axis, on the measured force); harmless
        X, R, bad = faulty_data(fault[:-1])
        distr = np.isin(Dr.FLIGHT, low_descent_flights(3)) & ~bad
        R = R.copy()
        R[distr] += np.random.default_rng(0).normal(0, VIB_STD, (int(distr.sum()), 3))
        return X, R, bad, distr
    X, R, bad = faulty_data(fault)
    return X, R, bad, np.zeros(len(X), bool)


VIB_STD = 8.0
