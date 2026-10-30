"""Build cf100.npz from NanoBench (Crazyflie 2.1, github.com/syediu/nanobench, BSD-3): airborne rows of every
flight at 100 Hz with body-frame velocity, body rates, attitude, motor PWM, body specific force from the IMU
(5-sample moving average, in m/s^2) and battery voltage.

usage: python build_cf.py <nanobench clone>/datasets/dataset
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

SRC = sys.argv[1]  # <nanobench clone>/datasets/dataset
G = 9.81


def rot_inv(q, v):
    """Rotate world vectors v into the body frame of unit quaternions q = (x, y, z, w)."""
    x, y, z, w = q.T
    R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                  np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                  np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)
    return np.einsum("nji,nj->ni", R, v)


rows, names = [], []
for f in sorted(glob.glob(os.path.join(SRC, "*.csv"))):
    d = pd.read_csv(f)
    q = d[["qx", "qy", "qz", "qw"]].to_numpy()
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    acc = d[["imu_acc_x", "imu_acc_y", "imu_acc_z"]].rolling(5, center=True, min_periods=1).mean().to_numpy() * G
    vb = rot_inv(q, d[["vx", "vy", "vz"]].to_numpy())
    w = d[["wx_vicon", "wy_vicon", "wz_vicon"]].to_numpy()
    pwm = d[[f"motor_motor_m{i}" for i in range(1, 5)]].to_numpy() / 65535.0
    keep = ((d.pz > 0.25) & (np.abs(acc).max(1) < 3 * G) & (acc[:, 2] > 0.3 * G) & (pwm.min(1) > 0.05) & (pwm.max(1) <= 1.0)).to_numpy()  # a few rows carry corrupt motor values
    if keep.sum() < 200:
        continue
    seg = np.cumsum(np.r_[1, np.diff(keep.astype(int)) != 0])  # contiguous airborne runs
    fl = len(names)
    names.append(os.path.basename(f)[:-4])
    rows.append(np.c_[vb, w, q, pwm, acc, d.pwr_pm_vbat.to_numpy(), d[["px", "py", "pz"]].to_numpy(),
                      np.full(len(d), fl), fl * 1000 + seg][keep])
D = np.vstack(rows)
cols = ["vel x", "vel y", "vel z", "ang vel x", "ang vel y", "ang vel z", "quat x", "quat y", "quat z", "quat w",
        "pwm 1", "pwm 2", "pwm 3", "pwm 4", "acc x", "acc y", "acc z", "vbat", "pos x", "pos y", "pos z", "flight", "seg"]
flight = D[:, -2].astype(int)
rng = np.random.default_rng(0)
test_flights = rng.choice(len(names), size=max(1, len(names) // 10), replace=False)
np.savez("cf100.npz", data=D[:, :-2], cols=np.array(cols[:-2]), flight=flight, seg=D[:, -1].astype(int),
         flights=np.array(names), test=np.isin(flight, test_flights))
vf = np.array([np.median(D[flight == i, 17]) for i in range(len(names))])
print(f"{len(names)} flights, {len(D)} rows ({len(D) / 6000:.1f} min); median in-flight V per flight "
      f"{np.percentile(vf, [0, 25, 50, 75, 100]).round(2)}")
