"""UrbanNav (Hong Kong) data loading for the Cayley GNSS-IMU SDP relaxation.

This module does everything that is *specific to UrbanNav* so that the solver module stays
dataset-agnostic; `sdp_preintegration_cayley.run_urbannav` consumes what is produced here and is
otherwise a line-for-line twin of `run_euroc`.

Five things differ from EuRoC and each one motivates a function below.

1. The IMU stream is not uniformly sampled.  Median dt is 2.490 ms (401.6 Hz) but its standard
   deviation is 0.376 ms, and the timestamps drift 3.04 s over the 785 s medium-urban sequence
   (0.4%).  `cayley_preintegrate` takes a *scalar* dt, so a raw median-dt read would spread that
   0.4% over every edge.  `resample_imu` puts the stream on an exact uniform grid instead, which
   leaves the scalar-dt code path in the solver untouched.

2. Ground truth is 1 Hz, not 200 Hz.  Keyframe intervals must therefore be integer multiples of
   1.0 s; `nearest_gt` refuses anything else rather than silently interpolating attitude.

3. Ground truth carries no IMU biases.  EuRoC hands `run_euroc` the true b_g/b_a to linearise
   about; here `static_bias_init` estimates them from the stationary window at the start of the
   sequence.  These are only a linearisation point -- they do not enter the rank or the gap.

4. Attitude is (roll, pitch, heading) in degrees and velocity is expressed in the *body* frame.
   `attitude_to_R` and the ENU conversion in `load_ground_truth` handle both.  The convention was
   established empirically, not from the documentation -- see `attitude_to_R`.

5. Local gravity in Hong Kong is 9.7877 m/s^2, not the 9.81 hard-coded in the solver module.  The
   0.24% difference is the same order as the accelerometer bias itself, so `override_gravity`
   below swaps it in for the duration of a solve.
"""
import contextlib
import os

import numpy as np

# WGS-84
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)

# Local gravity at the Hong Kong sequences (lat 22.30 deg, h ~ 3 m), from the international
# gravity formula.  EuRoC's 9.81 would bias every accelerometer edge by 0.022 m/s^2.
GRAVITY_HK = 9.7877

SEQUENCES = {
    'medium': dict(imu='xsense_imu_medium_urban1.csv', gt='UrbanNav_TST_GT_raw.txt',
                   label='UrbanNav-HK-Medium-Urban-1 (TST)'),
    'deep': dict(imu='xsense_imu_deep_urban1.csv', gt='UrbanNav_whampoa_raw.txt',
                 label='UrbanNav-HK-Deep-Urban-1 (Whampoa)'),
}

# Per-axis noise densities measured from the stationary window at the start of the medium-urban
# sequence (see `measure_noise_densities`).  The distributed xsens_imu_param.yaml quotes
# gyr_n = 1.027e-2 rad/s/sqrt(Hz), which is 10-20x larger than the data supports; its
# acc_n = 1.120e-2 m/s^2/sqrt(Hz) is the right order.  Both sets are exposed so an experiment can
# report the conservative datasheet numbers as a stress case.
NOISE_EMPIRICAL = dict(gyro_nd=8.0e-4, accel_nd=1.1e-2, gyro_rw=9.1355e-5, accel_rw=1.1752e-4)
NOISE_DATASHEET = dict(gyro_nd=1.0271e-2, accel_nd=1.1197e-2, gyro_rw=9.1355e-5, accel_rw=1.1752e-4)


# --------------------------------------------------------------------------- #
# IMU
# --------------------------------------------------------------------------- #
def load_imu(path, cache=True):
    """Read the UrbanNav `sensor_msgs/Imu` CSV dump -> (t[s], gyro[N,3], accel[N,3]).

    The file is ~110 MB of text with 41 columns, of which we need 7; parsing it takes far longer
    than the solve it feeds, so the result is cached next to the CSV as .npz.
    """
    npz = os.path.splitext(path)[0] + '.npz'
    if cache and os.path.exists(npz) and os.path.getmtime(npz) >= os.path.getmtime(path):
        d = np.load(npz)
        return d['t'], d['gyro'], d['accel']
    # cols: 0 %time(ns); 17:20 angular_velocity xyz; 29:32 linear_acceleration xyz
    raw = np.loadtxt(path, delimiter=',', skiprows=1, usecols=(0, 17, 18, 19, 29, 30, 31))
    t = raw[:, 0] * 1e-9
    gyro = np.ascontiguousarray(raw[:, 1:4])
    accel = np.ascontiguousarray(raw[:, 4:7])
    order = np.argsort(t)  # the dumps are ordered, but a single swapped pair would corrupt every edge
    t, gyro, accel = t[order], gyro[order], accel[order]
    if cache:
        np.savez_compressed(npz, t=t, gyro=gyro, accel=accel)
    return t, gyro, accel


def resample_imu(t, gyro, accel, rate=None):
    """Linearly resample onto an exact uniform grid -> (t_u, gyro_u, accel_u, dt).

    Motivation: the solver preintegrates with a single scalar dt and forms edge durations as
    (b - a) * dt.  On UrbanNav the true sample spacing wanders by ~15% (1-sigma) and accumulates
    0.4% of drift over a sequence, so (b - a) * median_dt disagrees with the wall-clock span of the
    edge.  Resampling makes the scalar-dt assumption exactly true by construction, at the cost of a
    linear interpolation whose error is far below the 5e-4 rad/s gyro noise floor.

    Linear (rather than zero-order-hold) interpolation is used because the accelerometer signal is
    dominated by vehicle/engine vibration at tens of Hz, well below the 400 Hz sample rate, so the
    signal is heavily oversampled and interpolation error is second order in dt.
    """
    if rate is None:
        rate = 1.0 / float(np.median(np.diff(t)))
    dt = 1.0 / rate
    n = int(np.floor((t[-1] - t[0]) / dt)) + 1
    tu = t[0] + dt * np.arange(n)
    gu = np.column_stack([np.interp(tu, t, gyro[:, c]) for c in range(3)])
    au = np.column_stack([np.interp(tu, t, accel[:, c]) for c in range(3)])
    return tu, gu, au, dt


def imu_timing_report(t):
    """Diagnostics justifying `resample_imu`; returned as a dict so experiments can print it."""
    d = np.diff(t)
    med = float(np.median(d))
    return dict(n=len(t), span=float(t[-1] - t[0]), dt_median=med, dt_std=float(d.std()),
                dt_min=float(d.min()), dt_max=float(d.max()), rate=1.0 / med,
                drift=float((t[-1] - t[0]) - (len(t) - 1) * med),
                n_gaps=int((d > 3 * med).sum()))


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
def _dms(D, M, S):
    """Degrees-minutes-seconds -> decimal degrees.  The sign lives on the degrees field only."""
    s = np.sign(D)
    s[s == 0] = 1.0
    return D + s * (M / 60.0 + S / 3600.0)


def geodetic_to_enu(lat, lon, h, lat0, lon0, h0):
    """Local east-north-up tangent plane about (lat0, lon0, h0), degrees in / metres out.

    The sequences span <1 km, so the curvature error of this linearisation is millimetres --
    negligible against the 5 cm ground-truth RMSE.
    """
    N = _A / np.sqrt(1.0 - _E2 * np.sin(np.deg2rad(lat)) ** 2)
    e = (N + h) * np.cos(np.deg2rad(lat)) * (np.deg2rad(lon) - np.deg2rad(lon0))
    n = (N * (1.0 - _E2) + h) * (np.deg2rad(lat) - np.deg2rad(lat0))
    return np.column_stack([e, n, h - h0])


def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def attitude_to_R(roll_deg, pitch_deg, heading_deg):
    """(roll, pitch, heading) in degrees -> R_enu_body, the world-from-body rotation.

    Convention: body is x-right / y-forward / z-up (as stated in UrbanNav's extrinsic.yaml) and
    heading is the NovAtel azimuth, measured *clockwise* from north -- hence Rz(-azimuth).

    This was established empirically rather than taken from the documentation.  Against
    finite-differenced ENU position over the 475 moving epochs of the medium-urban sequence, this
    composition reproduces R @ v_body with a median residual of 0.080 m/s -- which is the 1 Hz
    differencing error, not model error.  Three other plausible orderings/sign choices
    (Rz(pi/2-a)Ry(-p)Rx(r), Rz(a)Rx(p)Ry(r), ...) gave 7.7-11.0 m/s, so the choice is unambiguous.
    """
    r, p, a = np.deg2rad(roll_deg), np.deg2rad(pitch_deg), np.deg2rad(heading_deg)
    return _Rz(-a) @ _Rx(p) @ _Ry(r)


def load_ground_truth(path):
    """Parse a SPAN-CPT/Inertial-Explorer export -> dict of 1 Hz world-frame ground truth.

    Columns are: UTCTime Week GPSTime Lat(D M S) Lon(D M S) H-Ell VelBdyX VelBdyY VelBdyZ
    AccBdyX AccBdyY AccBdyZ Roll Pitch Heading Q.  Note there is no bias ground truth -- see
    `static_bias_init` -- and that velocity is body-frame, so it is rotated to ENU here.

    Q is Inertial Explorer's per-epoch quality flag (1 best).  It is carried through so that
    experiments can exclude or stratify by ground-truth quality; in deep urban 115 of 1539 epochs
    are Q>=4.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            tok = line.split()
            if len(tok) < 20:
                continue
            try:
                rows.append([float(x) for x in tok[:20]])
            except ValueError:
                continue  # header lines
    g = np.asarray(rows)
    if g.size == 0:
        raise ValueError("no numeric rows parsed from %s" % path)
    t = g[:, 0]
    lat = _dms(g[:, 3], g[:, 4], g[:, 5])
    lon = _dms(g[:, 6], g[:, 7], g[:, 8])
    h = g[:, 9]
    v_body = g[:, 10:13]
    roll, pitch, head = g[:, 16], g[:, 17], g[:, 18]
    Q = g[:, 19].astype(int)

    pos = geodetic_to_enu(lat, lon, h, lat[0], lon[0], h[0])
    R = np.stack([attitude_to_R(roll[k], pitch[k], head[k]) for k in range(len(t))])
    vel = np.einsum('kij,kj->ki', R, v_body)  # body -> ENU
    return dict(t=t, pos=pos, vel=vel, R=R, Q=Q, lat=lat, lon=lon, h=h,
                roll=roll, pitch=pitch, heading=head, v_body=v_body)


def nearest_gt(gt_t, ts, tol=0.05):
    """Index of the ground-truth epoch at time `ts`, refusing to interpolate.

    Ground truth is 1 Hz and includes attitude; interpolating it would fabricate rotations and
    quietly corrupt the very quantity the experiment measures.  Keyframes must therefore land on
    ground-truth epochs, which `tol` enforces.
    """
    k = int(np.argmin(np.abs(gt_t - ts)))
    if abs(gt_t[k] - ts) > tol:
        raise ValueError("keyframe at t=%.3f is %.3f s from the nearest 1 Hz ground-truth epoch; "
                         "kf_interval must be an integer multiple of 1.0 s" % (ts, gt_t[k] - ts))
    return k


# --------------------------------------------------------------------------- #
# Bias initialisation
# --------------------------------------------------------------------------- #
def static_bias_init(t, gyro, accel, R0, t0=None, window=5.0, gravity=GRAVITY_HK):
    """Estimate (b_g, b_a) from the stationary window at the start of the sequence.

    UrbanNav ships no bias ground truth, but the vehicle is parked for the first several seconds of
    every sequence.  While stationary the true angular rate is zero and the true specific force is
    R0^T @ (0 - g), so the sample means of the gyro and accelerometer are direct bias estimates.

    These values are only the point about which the preintegration is linearised -- exactly what
    EuRoC's ground-truth biases are used for in `run_euroc`.  They do not enter the objective, the
    constraints, the rank, or the optimality gap; a poor estimate widens the linearisation error of
    the bias Jacobians and nothing else.
    """
    t0 = float(t[0]) if t0 is None else float(t0)
    m = (t >= t0) & (t < t0 + window)
    if m.sum() < 10:
        raise ValueError("static window [%.1f, %.1f) contains only %d samples" % (t0, t0 + window, m.sum()))
    bg = gyro[m].mean(axis=0)
    ba = accel[m].mean(axis=0) - R0.T @ np.array([0.0, 0.0, gravity])
    return bg, ba


def measure_noise_densities(t, gyro, accel, t0=None, window=5.0):
    """Per-axis white-noise densities from the stationary window, in rad/s/sqrt(Hz), m/s^2/sqrt(Hz).

    Reported per axis because they are not isotropic: on the medium-urban sequence the body-x
    accelerometer reads 21.9e-3 against 4.0e-3 on body-z.  That anisotropy is idling engine
    vibration coupling into the sensor, not electrical noise, so `NOISE_EMPIRICAL` deliberately
    does not simply take the worst axis.
    """
    t0 = float(t[0]) if t0 is None else float(t0)
    m = (t >= t0) & (t < t0 + window)
    fs = 1.0 / float(np.median(np.diff(t)))
    return dict(gyro_nd=gyro[m].std(axis=0) / np.sqrt(fs),
                accel_nd=accel[m].std(axis=0) / np.sqrt(fs), fs=fs, n=int(m.sum()))


# --------------------------------------------------------------------------- #
# Segment selection
# --------------------------------------------------------------------------- #
def score_segments(gt, T=8.0, stride=1.0, min_speed=1.0):
    """Rank candidate windows by how much they actually excite the estimation problem.

    A third of the epochs in these sequences are a parked car at a red light.  Such a window has
    near-zero angular rate and near-zero specific force beyond gravity, so the rotation and
    velocity edges carry almost no information and the solve says nothing interesting about
    tightness.  Ranking by (total heading change) x (fraction of the window in motion) puts the
    turns and accelerations first.

    Returns a list of dicts sorted best-first, each with the window start time (as an offset from
    the first ground-truth epoch), the total heading change in degrees, mean speed, and the
    fraction of epochs above `min_speed`.
    """
    t, R, vel = gt['t'], gt['R'], gt['vel']
    head = np.unwrap(np.deg2rad(gt['heading']))
    speed = np.linalg.norm(vel, axis=1)
    out = []
    k = 0
    while t[k] + T <= t[-1]:
        m = (t >= t[k]) & (t <= t[k] + T)
        turn = float(np.abs(np.diff(head[m])).sum()) * 180.0 / np.pi
        moving = float((speed[m] > min_speed).mean())
        out.append(dict(t_start=float(t[k] - t[0]), turn_deg=turn, mean_speed=float(speed[m].mean()),
                        moving_frac=moving, score=turn * moving))
        k += max(1, int(round(stride / np.median(np.diff(t)))))
    out.sort(key=lambda d: -d['score'])
    return out


# --------------------------------------------------------------------------- #
# Gravity override
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def override_gravity(module, g=GRAVITY_HK):
    """Temporarily point `module.GRAVITY` at local gravity for the duration of a solve.

    `build_cost_matrix` and `_chain_ground_truth` read the module-level GRAVITY at call time, so
    swapping it here is enough and needs no change to the solver's numerics.  It is restored on
    exit so that an interleaved EuRoC or synthetic run in the same process is unaffected.
    """
    old = module.GRAVITY
    module.GRAVITY = np.array([0.0, 0.0, -abs(float(g))])
    try:
        yield
    finally:
        module.GRAVITY = old


def resolve(sequence, root):
    """('medium'|'deep', data root) -> (imu path, gt path, human-readable label)."""
    if sequence not in SEQUENCES:
        raise ValueError("unknown sequence %r; expected one of %s" % (sequence, sorted(SEQUENCES)))
    s = SEQUENCES[sequence]
    return os.path.join(root, s['imu']), os.path.join(root, s['gt']), s['label']


_CACHE = {}


def load_sequence(root, sequence='medium', rate=None):
    """Load, resample and cache a sequence -> dict(t, gyro, accel, dt, gt, label, timing).

    Memoised in-process: a noise sweep calls this once per solve, and even from the .npz cache the
    400 Hz stream costs more to materialise than the SDP costs to solve.
    """
    key = (os.path.abspath(root), sequence, rate)
    if key in _CACHE:
        return _CACHE[key]
    imu_path, gt_path, label = resolve(sequence, root)
    t_raw, gyro_raw, accel_raw = load_imu(imu_path)
    timing = imu_timing_report(t_raw)
    t, gyro, accel, dt = resample_imu(t_raw, gyro_raw, accel_raw, rate=rate)
    gt = load_ground_truth(gt_path)
    _CACHE[key] = dict(t=t, gyro=gyro, accel=accel, dt=dt, gt=gt, label=label, timing=timing)
    return _CACHE[key]
