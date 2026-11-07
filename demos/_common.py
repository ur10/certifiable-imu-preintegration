"""Shared set-up for the EuRoC demos.

Every demo poses the same problem to both estimators:

  * GNSS-like velocity and position fixes on every keyframe, simulated as ground truth plus
    N(0, 1 cm/s) and N(0, 1 cm) noise (`simulate_gnss`), with the SAME draws handed to both;
  * the free gauge: neither estimator is told the attitude of any keyframe. Roll and pitch are
    observable through gravity, heading only through horizontal acceleration;
  * the paper's nominal weights (`dataset_cov`, per-edge propagated covariance) and two bias
    blocks per 16 s window.

The Cayley SDP needs no initial guess. Levenberg-Marquardt (GTSAM, standard exp-map
pre-integration) needs one; the demos give it the guesses a real system would have.
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import sdp_preintegration_cayley as C        # noqa: E402
import compare_gtsam_cayley as X             # noqa: E402
import datasets                              # noqa: E402
from gtsam_baseline import solve_gtsam, _navstate_init   # noqa: E402

OUT = os.path.join(ROOT, 'demos', 'out')
FIG = os.path.join(ROOT, 'demos', 'figures')
PRIOR_SIGMAS = (1e-2, 1e-2, 1e-3, 1e-3)      # sigma_v [m/s], sigma_p [m], (unused bias slots)
T, KFI, KFPB = 16.0, 1.0, 8                  # 17 keyframes, 2 bias blocks
ETA_TOL = 1e-6                               # certified = rank 1 AND dual gap <= this

# figure colours: categorical slots 1-2 (blue, orange) of a CVD-validated palette, grey for context
BLUE, ORANGE, GREY, INK, INK2 = '#2a78d6', '#eb6834', '#a3a29c', '#0b0b0b', '#52514e'


def certified(r):
    """Rank one, and the rounded estimate's cost within ETA_TOL (relative) of the dual bound.

    The dual gap `eta_dual` = (f(x_hat) - lb) / (1 + |f| + |lb|) compares the cost of the
    returned estimate with a lower bound on the optimum of the original problem, so a small value
    is the certificate itself. Rank alone is not enough: a loose relaxation can still report one
    dominant eigenvalue. The primal `eta` (against the interior point's own objective) is
    recorded too, but it overstates the gap when that iterate is slightly infeasible.
    """
    ed = r.get('eta_dual')
    return r.get('rank') == 1 and ed is not None and ed <= ETA_TOL


def solve_window(seq, t_start, gnss_seed, bias_init='gt', prior_sigmas=PRIOR_SIGMAS):
    """One SDP solve plus the matched window for the LM baseline."""
    base = datasets.euroc_sequence(seq)
    r = C.run_euroc(base, T=T, kf_interval=KFI, t_start=t_start, weight_mode='dataset_cov',
                    iso_reduce='mean', prior_sigmas=prior_sigmas, gauge_fix=False,
                    add_redundant=True, kf_per_bias=KFPB, accept_any=True,
                    gnss_seed=gnss_seed, bias_init=bias_init,
                    gyro_nd=X.ND_G, accel_nd=X.ND_A, gyro_rw=X.GYRO_RW, accel_rw=X.ACCEL_RW)
    W = X.load_euroc_window(base, T, KFI, t_start, KFPB)
    if r.get('rank') is not None:
        W['v_t'], W['p_t'] = r['v_meas'], r['p_meas']     # LM sees the same GNSS fixes
        W['bgbar'], W['babar'] = r['bgbar'], r['babar']   # ... and the same bias prior
    return r, W


def run_lm(W, R_true, init, seed=0, prior_sigmas=PRIOR_SIGMAS):
    return solve_gtsam(W['gyro'], W['acc'], W['dt'], W['kf_idx'], W['bgbar'], W['babar'],
                       R_true, W['v_t'], W['p_t'], W['blk'], X.ND_G, X.ND_A, prior_sigmas,
                       X.GYRO_RW, X.ACCEL_RW, kf_interval=KFI, initialisation=init, seed=seed)


def initial_attitude(W, R_true, init, seed=0):
    """The initial R_0 that `run_lm(..., init, seed)` starts from (same generator, same draw)."""
    a_mean = np.mean(W['acc'][W['kf_idx'][0]:W['kf_idx'][1]], axis=0) - W['babar']
    Rs, _, _ = _navstate_init(init, R_true, W['v_t'], W['p_t'], np.random.default_rng(seed),
                              a_mean=a_mean)
    return Rs[0]


def rot_angle(Ra, Rb):
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def heading_tilt(R_est, R_true):
    """Split each attitude error into heading (about world z) and tilt (of the body z axis)."""
    hd, tl = [], []
    for Re, Rt in zip(R_est, R_true):
        E = np.asarray(Re) @ np.asarray(Rt).T
        hd.append(abs(np.degrees(np.arctan2(E[1, 0] - E[0, 1], E[0, 0] + E[1, 1]))))
        zt, ze = np.asarray(Rt)[2], np.asarray(Re)[2]     # world z in each body frame
        tl.append(np.degrees(np.arccos(np.clip(zt @ ze, -1.0, 1.0))))
    return float(np.mean(hd)), float(np.mean(tl))


def excitation(r):
    """log10(sigma_2 / sigma_1) of the Wahba alignment matrix.

    sigma_1 is the gravity direction; sigma_2 comes from horizontal acceleration and is what makes
    heading observable. Very negative = the window barely accelerated sideways.
    """
    sv = r.get('wahba_sv')
    if sv is None:
        return None
    sv = np.asarray(sv, float)
    return float(np.log10(max(sv[1], 1e-300) / sv[0]))
