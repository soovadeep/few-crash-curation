"""MuSHR 1:10 car (stock MuJoCo model from prl-mushr/mushr_mujoco_ros, BSD-3) in an L-shaped corridor
whose corner is icy.

The plant is MuJoCo (timestep 2 ms, control every 50 ms). Crashes are geometric: the car centre comes
within CAR_R of a corridor wall. The planner sees the state
    s = (x, y, psi, vx, vy, r, delta, omega)
(planar pose, body-frame velocity, yaw rate, steering angle, mean wheel speed) and commands
    u = (steer in [-1, 1] -> +-0.38 rad, target speed in [0, V_CMD]).
"""
import os

import mujoco
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TS = 0.002
DT = 0.05
NSUB = int(round(DT / TS))
STEER_MAX = 0.38
V_CMD = 4.0
MU_DRY = 1.0
MU_ICE = 0.12

BOX_A = ((-1.0, 8.0), (-1.0, 1.0))   # horizontal leg
BOX_B = ((6.0, 8.0), (-1.0, 9.0))    # vertical leg
ICE = ((4.5, 8.5), (-1.5, 3.5))
CAR_R = 0.25
WALL_MARGIN = 0.15   # wall cost starts this far from the wall
GOAL = (7.0, 8.0)
ARENA_X = (-1.0, 8.0)
ARENA_Y = (-1.0, 9.0)


def _xml():
    car = open(os.path.join(HERE, "mushr/cars/base_car/buddy.xml")).read()
    car = car.replace("<mujoco>", "").replace("</mujoco>", "").replace('<compiler meshdir="cars/meshes" />', "")
    return f'''<mujoco model="mushr_ice">
  <compiler angle="radian" meshdir="{HERE}/mushr/cars/meshes"/>
  <option timestep="{TS}"/>
  {car}
  <worldbody>
    <geom name="floor" type="plane" size="0 0 .25" friction="{MU_DRY} 0.005 0.0001" condim="3"/>
  </worldbody>
</mujoco>'''


def ice_frac_np(x, y, edge=0.05):
    (x0, x1), (y0, y1) = ICE
    s = lambda z: 1.0 / (1.0 + np.exp(-z / edge))
    return s(x - x0) * s(x1 - x) * s(y - y0) * s(y1 - y)


class Sim:
    """One MuJoCo car. mu_ice=MU_DRY gives the 'dry day'.

    Surface friction is per wheel: the four wheel collision geoms get contact priority over the floor,
    and before every physics step each wheel's friction is set from where that wheel is (ice patch with
    a 5 cm soft edge). This avoids seams between floor tiles, which make ellipsoid wheels jump."""

    def __init__(self, mu_ice=None):
        self.mu_ice = MU_ICE if mu_ice is None else mu_ice
        self.m = mujoco.MjModel.from_xml_string(_xml())
        self.d = mujoco.MjData(self.m)
        self.wheels = [g for g in range(self.m.ngeom) if self.m.geom_contype[g] == 1 and
                       self.m.body(self.m.geom_bodyid[g]).name.startswith("buddy_wheel_")]
        assert len(self.wheels) == 4
        self.m.geom_priority[self.wheels] = 3
        j = lambda n: self.m.joint(n)
        self.q_st = [j(n).qposadr[0] for n in ("buddy_steering_wheel", "buddy_wheel_fl_steering", "buddy_wheel_fr_steering")]
        self.v_th = [j(n).dofadr[0] for n in ("buddy_wheel_fl_throttle", "buddy_wheel_fr_throttle",
                                               "buddy_wheel_bl_throttle", "buddy_wheel_br_throttle")]
        self.v_st = [j(n).dofadr[0] for n in ("buddy_steering_wheel", "buddy_wheel_fl_steering", "buddy_wheel_fr_steering")]
        mujoco.mj_resetData(self.m, self.d)
        for _ in range(200):
            self._surface()
            mujoco.mj_step(self.m, self.d)
        self.z0 = float(self.d.qpos[2])

    def set_state(self, s):
        x, y, psi, vx, vy, r, delta, omega = [float(a) for a in s]
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[:3] = [x, y, self.z0]
        d.qpos[3:7] = [np.cos(psi / 2), 0, 0, np.sin(psi / 2)]
        for a in self.q_st:
            d.qpos[a] = delta
        c, sn = np.cos(psi), np.sin(psi)
        d.qvel[:3] = [c * vx - sn * vy, sn * vx + c * vy, 0]
        d.qvel[3:6] = [0, 0, r]
        for a in self.v_th:
            d.qvel[a] = omega
        d.ctrl[:] = [delta, 0.0]
        mujoco.mj_forward(m, d)

    def get_state(self):
        d = self.d
        w, qx, qy, qz = d.qpos[3:7]
        psi = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        c, sn = np.cos(psi), np.sin(psi)
        vw = d.qvel[:3]
        vx, vy = c * vw[0] + sn * vw[1], -sn * vw[0] + c * vw[1]
        # yaw rate: world-frame angular velocity z (free-joint angular velocity is in the body frame)
        R = d.xmat[self.m.body("buddy").id].reshape(3, 3)
        r = (R @ d.qvel[3:6])[2]
        delta = d.qpos[self.q_st[0]]
        omega = np.mean(d.qvel[self.v_th])
        return np.array([d.qpos[0], d.qpos[1], psi, vx, vy, r, delta, omega])

    def step(self, u):
        self.d.ctrl[:] = [STEER_MAX * np.clip(u[0], -1, 1), np.clip(u[1], 0, V_CMD)]
        for _ in range(NSUB):
            self._surface()
            mujoco.mj_step(self.m, self.d)
        return self.get_state()

    def _surface(self):
        xy = self.d.geom_xpos[self.wheels, :2]
        f = ice_frac_np(xy[:, 0], xy[:, 1])
        self.m.geom_friction[self.wheels, 0] = MU_DRY - (MU_DRY - self.mu_ice) * f


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------ geometry and cost (torch, batched)
def _inside(px, py, box):
    (x0, x1), (y0, y1) = box
    return torch.minimum(torch.minimum(px - x0, x1 - px), torch.minimum(py - y0, y1 - py))


def clearance(p):
    px, py = p[..., 0], p[..., 1]
    return torch.maximum(_inside(px, py, BOX_A), _inside(px, py, BOX_B)) - CAR_R


def in_ice(px, py):
    (x0, x1), (y0, y1) = ICE
    return (px > x0) & (px < x1) & (py > y0) & (py < y1)


def remaining(p):
    """Path length to the goal along the corridor (smooth)."""
    return (GOAL[1] - p[..., 1]) + torch.nn.functional.softplus(3.0 * (GOAL[0] - p[..., 0])) / 3.0


def stage_cost(s, u):
    c = clearance(s[..., :2])
    c_wall = 300.0 * torch.nn.functional.softplus(-10.0 * (c - WALL_MARGIN)) / 10.0
    return remaining(s[..., :2]) + c_wall + 0.05 * (u[..., 0] ** 2)


def terminal_cost(s):
    return 5.0 * remaining(s[..., :2])


def soft_pen(S, beta=10.0):
    return torch.logsumexp(-beta * clearance(S[..., :2]), -1) / beta
