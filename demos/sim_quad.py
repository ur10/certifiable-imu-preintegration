"""Simulated quadrotor mission, written in EuRoC format so every EuRoC tool runs on it unchanged.

Differentially flat trajectory: position p(t) and yaw psi(t) are designed; attitude follows from
the thrust direction a + g e_z. The gyro sample at step k is the exact discrete increment
log(R_k^T R_{k+1}) / dt, so integrating the clean IMU reproduces the trajectory. Noise densities
and bias random walks are EuRoC's (ADIS16448).

Mission (s):  0-16 take-off and hover | 16-64 aggressive figure-8 | 64-96 figure-8 under GNSS
jamming | 96-112 hover with slow yaw | 112-144 zig-zag dashes | 144-176 return and land.

    python demos/sim_quad.py        # -> data/sim/quad_mission/mav0/{imu0,state_groundtruth_estimate0}
"""
import argparse, os
import numpy as np
from _common import ROOT

G = 9.81
DT = 0.005
DUR = 176.0
ND_G, ND_A = 1.6968e-4, 2.0e-3            # EuRoC sensor.yaml densities
RW_G, RW_A = 1.9393e-5, 3.0e-3
BG0 = np.array([-0.0022, 0.0211, 0.0767])  # EuRoC-like initial biases
BA0 = np.array([-0.025, 0.136, 0.065])
PHASES = [(0, 16, 'take-off + hover'), (16, 64, 'figure-8'), (64, 96, 'figure-8, GNSS jammed'),
          (96, 112, 'hover + slow yaw'), (112, 144, 'zig-zag dashes'), (144, 176, 'return + land')]
JAM = (64.0, 96.0)
OUT = os.path.join(ROOT, 'data', 'sim', 'quad_mission', 'mav0')


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * x * (10 - 15 * x + 6 * x * x)          # C2 quintic


def blend(t, a, b):
    return smoothstep((t - a) / (b - a))


def position(t):
    """World position (m), z up. Built from smoothly blended segments."""
    h = 3.0
    z = 0.15 + (h - 0.15) * blend(t, 5.0, 11.0) - (h - 0.15) * blend(t, 166.0, 174.0)
    # figure-8 (lemniscate of Gerono), 16 s period, ramped in and out
    w8 = 2 * np.pi / 16.0
    amp = blend(t, 16.0, 22.0) * (1 - blend(t, 92.0, 98.0))
    x8, y8 = 6.0 * np.sin(w8 * (t - 16)), 3.5 * np.sin(2 * w8 * (t - 16))
    # zig-zag dashes along x, return home at the end
    zz = blend(t, 112, 120) * 8.0 - blend(t, 120, 128) * 8.0 + blend(t, 128, 136) * 8.0 - blend(t, 136, 144) * 8.0
    zy = (blend(t, 116, 124) - blend(t, 132, 140)) * 4.0
    return np.array([amp * x8 + zz, amp * y8 + zy, z])


def yaw(t):
    # follow the figure-8 tangent loosely (via a smooth heading program), slow spin while hovering
    base = 0.6 * np.sin(2 * np.pi * (t - 16) / 16.0) * blend(t, 16, 22) * (1 - blend(t, 92, 98))
    spin = 2 * np.pi * 0.25 * blend(t, 98, 110)            # quarter turn over the hover
    dash = 0.5 * (blend(t, 112, 120) - blend(t, 136, 144))
    return 0.3 + base + spin + dash


def simulate(seed=0):
    t = np.arange(0.0, DUR + DT / 2, DT)
    P = np.array([position(x) for x in t])
    V = np.gradient(P, DT, axis=0); A = np.gradient(V, DT, axis=0)
    psi = np.array([yaw(x) for x in t])
    R = np.empty((len(t), 3, 3))
    for k in range(len(t)):
        zb = A[k] + [0, 0, G]; zb /= np.linalg.norm(zb)
        xc = np.array([np.cos(psi[k]), np.sin(psi[k]), 0.0])
        yb = np.cross(zb, xc); yb /= np.linalg.norm(yb); xb = np.cross(yb, zb)
        R[k] = np.c_[xb, yb, zb]
    rng = np.random.default_rng(seed)
    bg = BG0 + np.cumsum(rng.normal(0, RW_G * np.sqrt(DT), (len(t), 3)), axis=0)
    ba = BA0 + np.cumsum(rng.normal(0, RW_A * np.sqrt(DT), (len(t), 3)), axis=0)
    gyro = np.zeros((len(t), 3)); acc = np.zeros((len(t), 3))
    for k in range(len(t) - 1):
        D = R[k].T @ R[k + 1]
        c = np.clip((np.trace(D) - 1) / 2, -1, 1); th = np.arccos(c)
        w = np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]])
        gyro[k] = (w * (th / (2 * np.sin(th))) if th > 1e-12 else w / 2) / DT
        acc[k] = R[k].T @ (A[k] + [0, 0, G])
    gyro[-1], acc[-1] = gyro[-2], acc[-2]
    gyro_m = gyro + bg + rng.normal(0, ND_G / np.sqrt(DT), gyro.shape)
    acc_m = acc + ba + rng.normal(0, ND_A / np.sqrt(DT), acc.shape)
    return t, P, V, R, bg, ba, gyro_m, acc_m


def R_to_wxyz(R):
    tr = np.trace(R)
    if tr > 0:
        s = 2 * np.sqrt(1 + tr); return np.array([s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]); return np.array([(R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]); return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s])
    s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]); return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4])


def write(out=OUT, seed=0):
    t, P, V, R, bg, ba, gyro, acc = simulate(seed)
    ts = (1_000_000_000_000 + np.round(t * 1e9)).astype(np.int64)
    os.makedirs(os.path.join(out, 'imu0'), exist_ok=True)
    os.makedirs(os.path.join(out, 'state_groundtruth_estimate0'), exist_ok=True)
    with open(os.path.join(out, 'imu0', 'data.csv'), 'w') as f:
        f.write('#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n')
        for k in range(len(t)):
            f.write('%d,%s\n' % (ts[k], ','.join('%.9f' % x for x in np.r_[gyro[k], acc[k]])))
    with open(os.path.join(out, 'state_groundtruth_estimate0', 'data.csv'), 'w') as f:
        f.write('#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,b_w_x,b_w_y,b_w_z,b_a_x,b_a_y,b_a_z\n')
        for k in range(len(t)):
            q = R_to_wxyz(R[k]); q = q if q[0] >= 0 else -q
            f.write('%d,%s\n' % (ts[k], ','.join('%.9f' % x for x in np.r_[P[k], q, V[k], bg[k], ba[k]])))
    sp = np.linalg.norm(V, axis=1); lat = np.linalg.norm(np.c_[np.gradient(V, DT, axis=0)[:, :2]], axis=1)
    tilt = np.degrees(np.arccos(np.clip(R[:, 2, 2], -1, 1)))
    print('wrote %s: %.0f s at %.0f Hz; max speed %.1f m/s, max lateral accel %.1f m/s^2, max tilt %.0f deg'
          % (out, DUR, 1 / DT, sp.max(), lat.max(), tilt.max()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    write(seed=ap.parse_args().seed)
