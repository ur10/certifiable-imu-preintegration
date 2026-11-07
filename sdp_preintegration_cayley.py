"""Full Cayley GNSS-IMU smoothing SDP relaxation (paper: 'IMU Preintegration via the Cayley Map').

Aligned with the full GNSS-IMU factor graph:
  * states  X_k = (R_k, p_k) in SE(3),  V_k = v_k in R^3,
  * a SINGLE shared bias  B_0 = (b^g, b^a)  (constant-bias-over-window),
  * phi^I  IMU preintegration factor coupling (X_i, V_i, X_j, V_j, B_0),
  * phi^X  absolute pose = initial pose prior (node 0) + GNSS position (all nodes),
  * phi^V  absolute velocity (GNSS Doppler),
  * phi^B  one broad Gaussian prior on B_0.

Cayley preintegration (eq 4-5) and the exact bias Jacobians (eq 12-14) are finite-difference
verified (check_preintegration_jacobians). Redundant catalog A1,A2, B1(a,b),B1c, B2/B2', B3/B3',
C1/C2 makes the relaxation tight. Runs on synthetic (sick/loop) and EuRoC trajectories.
"""
import argparse
import sys
from dataclasses import dataclass

import numpy as np
import gtsam
from mosek.fusion import Model, Domain, Expr, ObjectiveSense, AccSolutionStatus

import urbannav_data
from sdp_preintegration import build_scenarios, create_preintegration, project_to_so3

I3 = np.eye(3)
GRAVITY = np.array([0.0, 0.0, -9.81])


# --------------------------------------------------------------------------- #
# Cayley map (half-angle convention, eq 1-2)
# --------------------------------------------------------------------------- #
def skew(v):
    v = np.asarray(v).reshape(3)
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def cayley(phi):
    K = 0.5 * skew(phi)
    return np.linalg.solve(I3 - K, I3 + K)


def cayley_inv(R):
    return vee(2.0 * (R - I3) @ np.linalg.inv(R + I3))


def cayley_jac(phi):
    phi = np.asarray(phi).reshape(3)
    return (I3 - 0.5 * skew(phi)) / (1.0 + 0.25 * phi @ phi)


# --------------------------------------------------------------------------- #
# Cayley preintegration with exact bias Jacobians (eq 4-5, 12-14)
# --------------------------------------------------------------------------- #
def cayley_preintegrate(gyro, acc, dt, bgbar, babar):
    gyro = np.asarray(gyro, float).reshape(-1, 3)
    acc = np.asarray(acc, float).reshape(-1, 3)
    n = len(gyro)
    dts = np.full(n, float(dt)) if np.isscalar(dt) else np.asarray(dt, float).reshape(n)

    dR = np.eye(3); dv = np.zeros(3); dp = np.zeros(3)
    JR = np.zeros((3, 3)); Avg = np.zeros((3, 3)); Ava = np.zeros((3, 3))
    Apg = np.zeros((3, 3)); Apa = np.zeros((3, 3))
    for k in range(n):
        h = dts[k]
        acorr = acc[k] - babar
        theta = (gyro[k] - bgbar) * h
        Ck = cayley(theta); Jk = cayley_jac(theta); acx = skew(acorr)
        Apa += Ava * h - 0.5 * dR * h * h                       # eq 14 (ba)
        Apg += Avg * h - 0.5 * (dR @ acx @ JR) * h * h          # eq 14 (bg)
        dp += dv * h + 0.5 * (dR @ acorr) * h * h               # eq 5
        Ava += -dR * h                                          # eq 13 (ba)
        Avg += -(dR @ acx @ JR) * h                             # eq 13 (bg)
        dv += (dR @ acorr) * h                                  # eq 4
        JR = Ck.T @ JR - Jk * h                                 # eq 12
        dR = dR @ Ck                                            # eq 4
    return dict(dR=dR, dv=dv, dp=dp, JR=JR, Avg=Avg, Ava=Ava, Apg=Apg, Apa=Apa)


def iso_project(Sig):
    """Nearest (Frobenius) isotropic-block matrix: every 3x3 block B_ab -> tr(B_ab)/3 * I_3.

    Equivalently S kron I_3 with S = Tr_axis(Sig)/3, a partial trace over the spatial factor, so it
    maps PSD to PSD. Cross-covariance blocks are kept, as scalar multiples of I_3.
    """
    S = np.einsum('aibi->ab', np.asarray(Sig).reshape(3, 3, 3, 3)) / 3.0
    return np.kron(S, I3)


def preintegrate_covariance(gyro, acc, dt, bgbar, babar, sig_g, sig_a, iso_input=False,
                            iso_propagate=False):
    """Standard preintegration covariance recursion (Forster et al., Cayley variant):

        Sigma_ij = A_{j-1} Sigma_{i,j-1} A_{j-1}^T + B_{j-1} Sigma_eta B_{j-1}^T

    with error state [dphi, dv, dp] and Sigma_eta the DISCRETE input noise (density^2 / h). The
    right-Jacobian of Exp is replaced by cayley_jac, and dR_{k,k+1}^T by C_k^T, so this is the exact
    linearised propagation for the Cayley preintegration in cayley_preintegrate.

    sig_g, sig_a are continuous noise DENSITIES (rad/s/sqrt(Hz), m/s^2/sqrt(Hz)).
    iso_input=True replaces both by max(sig_g, sig_a), i.e. it makes the gyroscope and accelerometer
    share ONE density. Note this is channel equalisation, not isotropisation: Sigma_eta below is a
    multiple of I_3 on each 3x3 block either way, so iso_input=False is not anisotropic. On EuRoC it
    inflates the gyro density 11.8x and cuts tau_R/tau_v from 85 to 16 -- the same collapse that
    weight_mode='mle' performs. It is statistically worse but compresses the weight dynamic range,
    which matters when obj_scale=1.0 leaves max|Q| ~ 1e8.
    iso_propagate=True keeps the covariance isotropic THROUGHOUT the interval: after every step
    Sigma is projected onto the isotropic-block subspace {S kron I_3} (iso_project), so the next
    step propagates an isotropic covariance. The default (False) propagates the full anisotropic
    Sigma and leaves the isotropic approximation to isotropic_block_sigmas at the end (paper eq. 23).
    Returns the 9x9 Sigma_ij, blocks ordered [dphi(0:3), dv(3:6), dp(6:9)].
    """
    gyro = np.asarray(gyro, float).reshape(-1, 3)
    acc = np.asarray(acc, float).reshape(-1, 3)
    n = len(gyro)
    dts = np.full(n, float(dt)) if np.isscalar(dt) else np.asarray(dt, float).reshape(n)
    if iso_input:
        sig_g = sig_a = max(sig_g, sig_a)

    dR = np.eye(3)
    Sig = np.zeros((9, 9))
    for k in range(n):
        h = dts[k]
        acorr = acc[k] - babar
        theta = (gyro[k] - bgbar) * h
        Ck = cayley(theta); Jk = cayley_jac(theta); acx = skew(acorr)

        A = np.eye(9)
        A[0:3, 0:3] = Ck.T
        A[3:6, 0:3] = -dR @ acx * h
        A[6:9, 0:3] = -0.5 * dR @ acx * h * h
        A[6:9, 3:6] = I3 * h

        B = np.zeros((9, 6))
        B[0:3, 0:3] = Jk * h
        B[3:6, 3:6] = dR * h
        B[6:9, 3:6] = 0.5 * dR * h * h

        # discrete input noise: continuous density^2 / h
        Seta = np.zeros((6, 6))
        Seta[0:3, 0:3] = I3 * (sig_g ** 2 / h)
        Seta[3:6, 3:6] = I3 * (sig_a ** 2 / h)

        Sig = A @ Sig @ A.T + B @ Seta @ B.T
        if iso_propagate:
            Sig = iso_project(Sig)
        dR = dR @ Ck
    return 0.5 * (Sig + Sig.T)


DEFAULT_MAX_BIAS_PRECISION = 1e7   # matches IMU_Certifiable_Factor/imu_rotation_bias_sdp.py
                                   # NB: defined for reference only -- run_* default to no cap,
                                   # because capping the bias precision changes the optimum (~1.3%
                                   # on EuRoC dataset_cov) rather than merely reconditioning it.

# Loosening MOSEK's interior-point relative duality gap from its 1e-8 default to 1e-7 is the remedy
# recorded in the paper draft for ProblemStatus.Unknown. Measured against 'dataset_cov' it is the
# wrong lever and is deliberately NOT enabled: on the three failing instances tested it did not help
# at all, and on an instance that already converged cleanly it inflated the numerical rank (EuRoC
# dataset/full/anchor, rank 1 -> 2, and 1 -> 11 at a different segment) because stopping earlier
# leaves Z less converged. That trades the rank certificate -- the primary one here -- for the
# solver's own. obj_scale='auto' is the remedy that worked: it fixed all three and was a no-op on
# clean instances (identical rank/gap/error). It is NOT the default -- pass it explicitly.
RELAXED_SOLVER_PARAMS = {'intpntCoTolRelGap': 1e-7}


def isotropic_block_sigmas(Sig, reduce='max'):
    """Collapse the 9x9 propagated covariance to three isotropic scalars, one per block. Blocks
    carry different units (rad, m/s, m), so the reduction is WITHIN each block; a single reduction
    across all nine would mix units and is not meaningful.

    reduce='max'  -- max of the block diagonal (conservative: weakest axis sets the weight).
    reduce='mean' -- tr(Sigma_block)/3, i.e. scalar_precision_from_covariance_block in
                     IMU_Certifiable_Factor. Always <= the max, so weights come out larger.
    """
    if reduce not in ('max', 'mean'):
        raise ValueError("reduce must be 'max' or 'mean', got %r" % reduce)
    f = np.max if reduce == 'max' else np.mean
    d = np.diag(Sig)
    return (float(f(d[0:3])), float(f(d[3:6])), float(f(d[6:9])))


def _deltas(gyro, acc, dt, bg, ba):
    r = cayley_preintegrate(gyro, acc, dt, bg, ba)
    return r['dR'], r['dv'], r['dp']


def check_preintegration_jacobians(seed=0, n=40, eps=1e-6):
    rng = np.random.default_rng(seed)
    gyro = rng.standard_normal((n, 3)) * 0.5; acc = rng.standard_normal((n, 3)) * 2.0
    dt = 0.005; bg = rng.standard_normal(3) * 0.02; ba = rng.standard_normal(3) * 0.02
    r = cayley_preintegrate(gyro, acc, dt, bg, ba)
    JR = np.zeros((3, 3)); Avg = np.zeros((3, 3)); Ava = np.zeros((3, 3)); Apg = np.zeros((3, 3)); Apa = np.zeros((3, 3))
    for c in range(3):
        e = np.zeros(3); e[c] = eps
        dRp, dvp, dpp = _deltas(gyro, acc, dt, bg + e, ba); dRm, dvm, dpm = _deltas(gyro, acc, dt, bg - e, ba)
        JR[:, c] = cayley_inv(r['dR'].T @ dRp) / eps
        Avg[:, c] = (dvp - dvm) / (2 * eps); Apg[:, c] = (dpp - dpm) / (2 * eps)
        dvp2 = _deltas(gyro, acc, dt, bg, ba + e); dvm2 = _deltas(gyro, acc, dt, bg, ba - e)
        Ava[:, c] = (dvp2[1] - dvm2[1]) / (2 * eps); Apa[:, c] = (dvp2[2] - dvm2[2]) / (2 * eps)
    return {'J_R': np.abs(r['JR'] - JR).max(), 'Avg': np.abs(r['Avg'] - Avg).max(),
            'Ava': np.abs(r['Ava'] - Ava).max(), 'Apg': np.abs(r['Apg'] - Apg).max(),
            'Apa': np.abs(r['Apa'] - Apa).max()}


# --------------------------------------------------------------------------- #
# Edge + layout (single shared bias B_0)
# --------------------------------------------------------------------------- #
@dataclass
class CayleyEdge:
    i: int
    j: int
    dRt: np.ndarray; dvt: np.ndarray; dpt: np.ndarray
    JR: np.ndarray; Avg: np.ndarray; Ava: np.ndarray; Apg: np.ndarray; Apa: np.ndarray
    dt_ij: float
    tauR: float = 1.0; tauv: float = 1.0; taup: float = 1.0
    Rbr: np.ndarray = None     # R_breve = dRt (I - 1/2 Gamma), folds in linearisation bias
    Gam: np.ndarray = None     # Gamma = skew(J^R bgbar)
    vbr: np.ndarray = None     # v_breve = dvt - Avg bgbar - Ava babar
    pbr: np.ndarray = None     # p_breve = dpt - Apg bgbar - Apa babar
    blk: int = 0               # index of the bias block B_m this edge's phi^I is tied to


class CayleyLayout:
    """x = [vec(R_k) (9N), {b^g_m}(3M), {b^a_m}(3M), vec(Y_e), vec(Y'_e), Z^v_e, Z^p_e, s].

    M is the number of piecewise-shared bias blocks. M=1 is the single-shared-bias model; M>1 cuts
    the window into blocks of `kf_per_bias` edges, each with its own B_m, consecutive blocks tied by
    a phi^RW random-walk factor (see _build_bias_rw_pairs). n = 15N + 6M + 24E + 1.
    """
    def __init__(s, N, E, M=1):
        s.N = N; s.E = E; s.M = M
        s.R0 = 0; s.v0 = 9 * N; s.p0 = s.v0 + 3 * N
        s.bg0 = s.p0 + 3 * N; s.ba0 = s.bg0 + 3 * M
        s.Y0 = s.ba0 + 3 * M; s.Yp0 = s.Y0 + 9 * E; s.Zv0 = s.Yp0 + 9 * E; s.Zp0 = s.Zv0 + 3 * E
        s.si = s.Zp0 + 3 * E; s.n = s.si + 1
    def R(s, k, r, c): return s.R0 + 9 * k + 3 * c + r
    def v(s, k, c): return s.v0 + 3 * k + c
    def p(s, k, c): return s.p0 + 3 * k + c
    def bg(s, m, c): return s.bg0 + 3 * m + c
    def ba(s, m, c): return s.ba0 + 3 * m + c
    def Y(s, e, r, c): return s.Y0 + 9 * e + 3 * c + r
    def Yp(s, e, r, c): return s.Yp0 + 9 * e + 3 * c + r
    def Zv(s, e, c): return s.Zv0 + 3 * e + c
    def Zp(s, e, c): return s.Zp0 + 3 * e + c


def bias_blocks(E, kf_per_bias=None):
    """Assign each preintegration edge to a bias block -> (block index per edge, block count).

    kf_per_bias=None keeps the single shared bias of the original formulation (every edge in block
    0). Otherwise edge e belongs to block e // kf_per_bias, so a window of E edges carries
    ceil(E / kf_per_bias) blocks -- 7 edges with kf_per_bias=8 give 1 block, 15 give 2, 23 give 3.
    """
    if kf_per_bias is None:
        return [0] * E, 1
    kpb = int(kf_per_bias)
    if kpb < 1:
        raise ValueError("kf_per_bias must be >= 1, got %r" % (kf_per_bias,))
    blk = [e // kpb for e in range(E)]
    return blk, (blk[-1] + 1 if E else 1)


def _build_bias_rw_pairs(M, kf_per_bias, kf_interval, gyro_rw, accel_rw):
    """phi^RW between-factors tying consecutive bias blocks -> [(m, m+1, tau_g, tau_a)].

    The bias is a random walk, so over the dt = kf_per_bias * kf_interval seconds separating two
    block midpoints it drifts with stddev rw * sqrt(dt); the whitened between-factor therefore
    carries precision 1 / (rw^2 * kf_per_bias * kf_interval). With M = 1 there is no pair and the
    model reduces exactly to the single shared bias.
    """
    if M < 2:
        return []
    dt = float(kf_per_bias) * float(kf_interval)
    return [(m, m + 1, 1.0 / (gyro_rw ** 2 * dt), 1.0 / (accel_rw ** 2 * dt)) for m in range(M - 1)]


def _eps(i, j, k):
    return 0 if len({i, j, k}) < 3 else (1 if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1)


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #
# The rotation residual is chordal (Frobenius on 3x3), not geodesic:
#   ||R_a - R_b||_F^2 = 8 sin^2(theta/2) = 2 theta^2 - theta^4/12 + ...
# so tau_R ||.||_F^2 carries an implied precision of 2*tau_R on the angle. Dividing tau_R by this
# factor makes the rotation block statistically commensurate with the (genuinely 3-vector) Delta v
# and Delta p blocks. Verified numerically: ||.||_F^2 / theta^2 -> 2.0000000 for theta <= 1e-1.
CHORDAL_ROT_FACTOR = 2.0


def _raw_mle_weights(gyro_sigma, accel_sigma, T_seg, prior_sigmas, rot_chordal=True):
    """tau = 1/sigma^2 exactly: the inverse noise covariance, used directly as the weight.

    Preintegration error stddevs over a segment of length T, for noise densities sigma_g, sigma_a:
        sigma_theta = sigma_g sqrt(T),  sigma_dv = sigma_a sqrt(T),  sigma_dp = sigma_a sqrt(T^3/3)
    so tau_R = 1/(sigma_g^2 T), tau_v = 1/(sigma_a^2 T), tau_p = 3/(sigma_a^2 T^3). No floors and
    no normalisation -- the absolute scale is the statistically meaningful one (chi-square units).

    rot_chordal=True divides tau_R by CHORDAL_ROT_FACTOR (see above). False reproduces the
    un-corrected weighting that the legacy 'mle' mode and the published figures were run with.
    """
    sig = dict(zip(('gyro', 'accel', 'v', 'p', 'bg', 'ba'),
                   (gyro_sigma, accel_sigma) + tuple(prior_sigmas)))
    bad = [k for k, s in sig.items() if not s > 0]
    if bad:
        raise ValueError("raw MLE weighting needs sigma > 0 for every term; got sigma=0 for %s. "
                         "A zero sigma is an exact constraint, not a weight." % ', '.join(bad))
    cR = CHORDAL_ROT_FACTOR if rot_chordal else 1.0
    return dict(R=1.0 / (cR * gyro_sigma ** 2 * T_seg),
                v=1.0 / (accel_sigma ** 2 * T_seg),
                p=3.0 / (accel_sigma ** 2 * T_seg ** 3),
                pv=1.0 / prior_sigmas[0] ** 2, pp=1.0 / prior_sigmas[1] ** 2,
                pbg=1.0 / prior_sigmas[2] ** 2, pba=1.0 / prior_sigmas[3] ** 2)


def compute_weights(gyro_sigma, accel_sigma, T_seg, mode='unit',
                    prior_sigmas=(0.1, 0.05, 1e-2, 1e-2), rot_chordal=True):
    """Returns dict R,v,p (IMU residuals) and pv,pp,pbg,pba (GNSS v/p + phi^B bias priors).
    prior_sigmas = (sigma_v, sigma_p, sigma_bg, sigma_ba).

    'mle_raw' -- tau = 1/sigma^2 used directly (no floors, no normalisation). Preferred.
    'mle'     -- legacy: sigma floored at 1e-4 (sensor) / 1e-8 (priors), then all weights divided
                 by their max. The division is a uniform rescale and so leaves the argmin
                 unchanged; the *floors* are what actually distort the weighting.
    'unit'    -- all ones.

    rot_chordal applies the chordal-vs-geodesic correction to tau_R; it is ignored by 'unit' and by
    'mle' (legacy 'mle' stays un-corrected so the published figures remain reproducible).
    """
    if mode == 'unit':
        return dict(R=1., v=1., p=1., pv=1., pp=1., pbg=1., pba=1.)
    if mode == 'mle_raw':
        return _raw_mle_weights(gyro_sigma, accel_sigma, T_seg, prior_sigmas, rot_chordal)
    # sensor-noise floor: MLE weight tau=1/sigma^2 is undefined at sigma=0 (infinite precision is
    # unphysical); floor keeps the noiseless point well-defined and the SDP conditioned.
    sg = max(gyro_sigma, 1e-4); sa = max(accel_sigma, 1e-4)
    sv, sp, sbg, sba = [max(s, 1e-8) for s in prior_sigmas]
    w = _raw_mle_weights(sg, sa, T_seg, (sv, sp, sbg, sba), rot_chordal=False)
    m = max(w.values())
    return {k: max(v / m, 1e-8) for k, v in w.items()}


# --------------------------------------------------------------------------- #
# Cost (eq 23-27, single bias) + scalar priors
# --------------------------------------------------------------------------- #
def residual_matrix(layout, edges, scalar_priors=None, rw_pairs=None):
    """B such that the cost is ||B x||^2 = x^T Q x: every whitened residual row, unsquared.

    Rows: per edge the 9 rotation, 3 velocity and 3 position residuals (eq 23-27); one row per
    bias component per phi^RW between-factor; one row per scalar prior (idx, ref, tau), which
    adds sqrt(tau) (x_idx - ref s). Keeping B (rather than only Q = B^T B) is what lets the
    solver centre the problem (see solve_cayley_sdp) and lets rank_and_gap evaluate the rounded
    cost as ||B x||^2 without the cancellation of x^T Q x.

    NB: the rotation residual uses the noiseless-form R_j - R_i dRt - (Y+Y')/2 with the bias
    correction folded into Y, Y' via the lifting; b^g enters only through the constraints.
    """
    n = layout.n; rows = []
    for e, ed in enumerate(edges):
        i, j = ed.i, ed.j; dt = ed.dt_ij
        tR, tv, tp = np.sqrt(ed.tauR), np.sqrt(ed.tauv), np.sqrt(ed.taup)
        B = np.zeros((15, n))
        # rotation (eq 23/26): R_j(I+1/2 Gamma) - R_i R_breve - (Y+Y')/2 ; bias folded into Gamma,R_breve
        B[0:9, layout.R0 + 9 * j:layout.R0 + 9 * (j + 1)] = tR * np.kron((I3 + 0.5 * ed.Gam).T, I3)
        B[0:9, layout.R0 + 9 * i:layout.R0 + 9 * (i + 1)] = -tR * np.kron(ed.Rbr.T, I3)
        B[0:9, layout.Y0 + 9 * e:layout.Y0 + 9 * (e + 1)] = -0.5 * tR * np.eye(9)
        B[0:9, layout.Yp0 + 9 * e:layout.Yp0 + 9 * (e + 1)] = -0.5 * tR * np.eye(9)
        # velocity (eq 24)
        B[9:12, layout.v(j, 0):layout.v(j, 0) + 3] = tv * I3
        B[9:12, layout.v(i, 0):layout.v(i, 0) + 3] = -tv * I3
        B[9:12, layout.si] = -tv * GRAVITY * dt
        B[9:12, layout.R0 + 9 * i:layout.R0 + 9 * (i + 1)] = -tv * np.kron(ed.vbr.reshape(1, 3), I3)
        B[9:12, layout.Zv0 + 3 * e:layout.Zv0 + 3 * e + 3] = -tv * I3
        # position (eq 25)
        B[12:15, layout.p(j, 0):layout.p(j, 0) + 3] = tp * I3
        B[12:15, layout.p(i, 0):layout.p(i, 0) + 3] = -tp * I3
        B[12:15, layout.v(i, 0):layout.v(i, 0) + 3] = -tp * dt * I3
        B[12:15, layout.si] = -tp * 0.5 * GRAVITY * dt * dt
        B[12:15, layout.R0 + 9 * i:layout.R0 + 9 * (i + 1)] = -tp * np.kron(ed.pbr.reshape(1, 3), I3)
        B[12:15, layout.Zp0 + 3 * e:layout.Zp0 + 3 * e + 3] = -tp * I3
        rows.append(B)
    for m0, m1, tau_g, tau_a in (rw_pairs or []):
        for idx_f, tau in ((layout.bg, tau_g), (layout.ba, tau_a)):
            if tau <= 0:
                continue
            for c in range(3):
                r = np.zeros((1, n)); r[0, idx_f(m1, c)] = np.sqrt(tau); r[0, idx_f(m0, c)] = -np.sqrt(tau)
                rows.append(r)
    for idx, ref, tau in (scalar_priors or []):
        if tau <= 0:
            continue
        r = np.zeros((1, n)); r[0, idx] = np.sqrt(tau); r[0, layout.si] = -np.sqrt(tau) * float(ref)
        rows.append(r)
    return np.vstack(rows) if rows else np.zeros((0, n))


def build_cost_matrix(layout, edges, scalar_priors=None, rw_pairs=None):
    """Q = B^T B (see residual_matrix).

    scalar_priors: list of (idx, ref, tau) adding tau (x_idx - ref s)^2.
    rw_pairs: list of (m, m+1, tau_g, tau_a) adding the phi^RW between-factors
    tau_g ||b^g_{m+1} - b^g_m||^2 + tau_a ||b^a_{m+1} - b^a_m||^2. These are the only terms coupling
    distinct bias blocks; with one block the list is empty and the cost is unchanged.
    """
    B = residual_matrix(layout, edges, scalar_priors=scalar_priors, rw_pairs=rw_pairs)
    return B.T @ B


def lift_solution(layout, edges, R, v, p, bg, ba):
    """Homogenised x = [vec R, v, p, b, vec Y, vec Y', Z^v, Z^p, 1] from (R, v, p, b). The lifting
    variables are recomputed from R and b, so x is exactly QCQP-feasible when every R is in SO(3)."""
    x = np.zeros(layout.n)
    for k in range(layout.N):
        x[layout.R0 + 9 * k:layout.R0 + 9 * (k + 1)] = np.asarray(R[k]).flatten('F')
        x[layout.v(k, 0):layout.v(k, 0) + 3] = v[k]; x[layout.p(k, 0):layout.p(k, 0) + 3] = p[k]
    bg = np.atleast_2d(bg); ba = np.atleast_2d(ba)
    for m in range(layout.M):
        x[layout.bg(m, 0):layout.bg(m, 0) + 3] = bg[m]
        x[layout.ba(m, 0):layout.ba(m, 0) + 3] = ba[m]
    for e, ed in enumerate(edges):
        vv = ed.JR @ bg[ed.blk]
        x[layout.Y0 + 9 * e:layout.Y0 + 9 * (e + 1)] = (R[ed.i] @ ed.dRt @ skew(vv)).flatten('F')
        x[layout.Yp0 + 9 * e:layout.Yp0 + 9 * (e + 1)] = (R[ed.j] @ skew(vv)).flatten('F')
        x[layout.Zv(e, 0):layout.Zv(e, 0) + 3] = R[ed.i] @ (ed.Avg @ bg[ed.blk] + ed.Ava @ ba[ed.blk])
        x[layout.Zp(e, 0):layout.Zp(e, 0) + 3] = R[ed.i] @ (ed.Apg @ bg[ed.blk] + ed.Apa @ ba[ed.blk])
    x[layout.si] = 1.0
    return x


def wahba_R0(layout, edges, scalar_priors):
    """R_0 from the GNSS priors and the IMU alone (no anchor), by Wahba's problem -> (R_0, sv).

    Every inertial edge predicts the world-frame velocity and position increments from body-frame
    pre-integrated ones, rotated by R_i = R_0 C_i with C_i the chained Delta R (body 0 -> i):

        a_e = vbar_j - vbar_i - g dt                     ~  R_0 (C_i dv_e)  = R_0 b_e
        c_e = pbar_j - pbar_i - vbar_i dt - g dt^2 / 2   ~  R_0 (C_i dp_e)  = R_0 d_e

    with vbar, pbar the GNSS velocity/position priors. R_0 = argmin_R sum tau_v ||a - R b||^2 +
    tau_p ||c - R d||^2 over SO(3) is Wahba's problem; its closed-form solution is the SVD of
    H = sum tau_v a b^T + tau_p c d^T: H = U S V^T, R_0 = U diag(1, 1, det(U V^T)) V^T.

    sv (the singular values of H) is an observability read-out: gravity alone makes H rank one
    (tilt only), so sv[1]/sv[0] ~ 0 means yaw is unobservable on this window and R_0's yaw is
    arbitrary -- harmless for centring, since the cost is then flat in yaw too.
    Returns (None, None) unless every node carries both a velocity and a position prior.
    """
    ref = {idx: r for idx, r, tau in (scalar_priors or []) if tau > 0}
    try:
        vb = [np.array([ref[layout.v(k, c)] for c in range(3)]) for k in range(layout.N)]
        pb = [np.array([ref[layout.p(k, c)] for c in range(3)]) for k in range(layout.N)]
    except KeyError:
        return None, None
    H = np.zeros((3, 3)); Ci = np.eye(3)
    for ed in edges:
        i, j, dt = ed.i, ed.j, ed.dt_ij
        a = vb[j] - vb[i] - GRAVITY * dt
        c = pb[j] - pb[i] - vb[i] * dt - 0.5 * GRAVITY * dt * dt
        H += ed.tauv * np.outer(a, Ci @ ed.dvt) + ed.taup * np.outer(c, Ci @ ed.dpt)
        Ci = Ci @ ed.dRt
    U, S, Vt = np.linalg.svd(H)
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt, S


def reference_point(layout, edges, anchors, scalar_priors, info=None):
    """A-priori centring point xbar for solve_cayley_sdp, or None if there is no rotation reference.

    R_0 from the hard anchor (gauge 'R' / True), else its soft prior ('R_prior'), else -- free
    gauge -- from wahba_R0 (GNSS priors + IMU); R_k by chaining the pre-integrated Delta R; v_k,
    p_k from the GNSS priors (node-0 anchors where present); every bias block at the phi^B
    reference. Nothing here uses the SDP solution or the ground truth beyond what the problem
    already contains as measurements/anchors. The certificate does not depend on xbar at all
    (the congruence is exact for any xbar); xbar only sets how well conditioned the solve is.
    info (dict), if given, receives the source of R_0 and, for Wahba, its singular values.
    """
    ref = {idx: r for idx, r, tau in (scalar_priors or []) if tau > 0}
    info = {} if info is None else info
    if anchors and 'R0' in anchors:
        R0 = np.asarray(anchors['R0'], float); info['R0_source'] = 'anchor'
    elif all(layout.R(0, r, c) in ref for r in range(3) for c in range(3)):
        R0 = project_to_so3(np.array([[ref[layout.R(0, r, c)] for c in range(3)] for r in range(3)]))
        info['R0_source'] = 'R_prior'
    else:
        R0, sv = wahba_R0(layout, edges, scalar_priors)
        if R0 is None:
            return None
        info.update(R0_source='wahba', R0_ref=R0, wahba_sv=[float(x) for x in sv])
    R = [R0]
    for ed in edges:
        R.append(R[-1] @ ed.dRt)

    def vec(f, k, key):
        if anchors and key in anchors and k == 0:
            return np.asarray(anchors[key], float)
        return np.array([ref.get(f(k, c), 0.0) for c in range(3)])
    v = [vec(layout.v, k, 'v0') for k in range(layout.N)]
    p = [vec(layout.p, k, 'p0') for k in range(layout.N)]
    bg = np.array([[ref.get(layout.bg(0, c), 0.0) for c in range(3)]] * layout.M)
    ba = np.array([[ref.get(layout.ba(0, c), 0.0) for c in range(3)]] * layout.M)
    return lift_solution(layout, edges, R, v, p, bg, ba)


# --------------------------------------------------------------------------- #
# Constraints (single bias)
# --------------------------------------------------------------------------- #
ALL_FAMILIES = ('A1', 'A2', 'B2', 'B3', 'B1c', 'C2')


def _families(redundant):
    """redundant: True (all), False (none), or an iterable of family names -> set of enabled families."""
    if redundant is True:
        return set(ALL_FAMILIES)
    if redundant is False or redundant is None:
        return set()
    return set(redundant)


def _constraints(layout, edges, anchors, add, redundant=True):
    N = layout.N
    fams = _families(redundant)
    add([(1.0, layout.si, layout.si)], 1.0)
    # anchors=None removes gauge fixing entirely: no hard equality pins node 0, and the window is
    # held instead by the GNSS v/p priors that _build_scalar_priors then places on *every* node
    # (first_node=0). This is the formulation the paper describes -- "the GNSS factors are present
    # at every keyframe, so the window carries no separate gauge-fixing prior". It is well posed:
    # the translation gauge is fixed by the position priors, and the rotation gauge by the
    # combination of fixed gravity g^W (which pins roll/pitch) and the world-frame velocity priors
    # (which pin yaw), so no global rotation leaves the cost invariant.
    # `anchors` may hold any subset of {'R0','v0','p0'}; absent keys impose no hard equality, and
    # anchors=None removes gauge fixing entirely. Whatever is not anchored must instead be covered
    # by the GNSS priors that _build_scalar_priors places (use first_node=0 to reach node 0).
    if anchors:
        if 'R0' in anchors:
            R0 = anchors['R0']
            for r in range(3):
                for c in range(3): add([(1.0, layout.R(0, r, c), layout.si)], R0[r, c])
        for key, idx in (('v0', layout.v), ('p0', layout.p)):
            if key in anchors:
                a0 = anchors[key]
                for c in range(3): add([(1.0, idx(0, c), layout.si)], a0[c])
    for k in range(N):
        for a in range(3):
            for b in range(a, 3):
                add([(1.0, layout.R(k, r, a), layout.R(k, r, b)) for r in range(3)], 1.0 if a == b else 0.0)
    for e, ed in enumerate(edges):
        _B1(layout, ed, e, add, prime=False); _B1(layout, ed, e, add, prime=True)
        _C1(layout, ed, e, add, pos=False); _C1(layout, ed, e, add, pos=True)
    if 'A1' in fams or 'A2' in fams:
        for k in range(N):
            if 'A1' in fams: _A1(layout, k, add)
            if 'A2' in fams: _A2(layout, k, add)
    for e, ed in enumerate(edges):
        if 'B2' in fams: _B2(layout, ed, e, add, False); _B2(layout, ed, e, add, True)
        if 'B3' in fams: _B3(layout, ed, e, add, False); _B3(layout, ed, e, add, True)
        if 'B1c' in fams: _B1c(layout, ed, e, add)
        if 'C2' in fams: _C2(layout, ed, e, add, False); _C2(layout, ed, e, add, True)


def _A1(lay, k, add):
    for a in range(3):
        for b in range(a, 3):
            add([(1.0, lay.R(k, a, c), lay.R(k, b, c)) for c in range(3)], 1.0 if a == b else 0.0)


def _A2(lay, k, add):
    for (a, b, c) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        for i in range(3):
            t = [(_eps(i, j, l), lay.R(k, j, a), lay.R(k, l, b)) for j in range(3) for l in range(3) if _eps(i, j, l)]
            t.append((-1.0, lay.R(k, i, c), lay.si)); add(t, 0.0)


def _B1(lay, ed, e, add, prime):
    JR = ed.JR; i = ed.i; j = ed.j
    Yf = (lambda r, c: lay.Yp(e, r, c)) if prime else (lambda r, c: lay.Y(e, r, c))
    Rf = (lambda r, c: lay.R(j, r, c)) if prime else (lambda r, c: lay.R(i, r, c))
    W = I3 if prime else ed.dRt
    def MTY(p, q): return [(W[nn, p], Rf(m, nn), Yf(m, q)) for m in range(3) for nn in range(3)]
    for p in range(3):
        for q in range(p, 3): add(MTY(p, q) + MTY(q, p), 0.0)
    for (a, b, c) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        t = MTY(a, b) + [(-x, ii, jj) for (x, ii, jj) in MTY(b, a)]
        t += [(2 * _eps(a, b, c) * JR[c, d], lay.bg(ed.blk, d), lay.si) for d in range(3)]
        add(t, 0.0)


def _B1c(lay, ed, e, add):
    for p in range(3):
        for q in range(3):
            t = [(ed.dRt[nn, p], lay.R(ed.i, m, nn), lay.Y(e, m, q)) for m in range(3) for nn in range(3)]
            t += [(-1.0, lay.R(ed.j, m, p), lay.Yp(e, m, q)) for m in range(3)]
            add(t, 0.0)


def _B2(lay, ed, e, add, prime):
    JR = ed.JR
    Yf = (lambda r, c: lay.Yp(e, r, c)) if prime else (lambda r, c: lay.Y(e, r, c))
    for a in range(3):
        add([(JR[b, c], Yf(a, b), lay.bg(ed.blk, c)) for b in range(3) for c in range(3)], 0.0)


def _B3(lay, ed, e, add, prime):
    JR = ed.JR; Gm = JR.T @ JR
    Yf = (lambda r, c: lay.Yp(e, r, c)) if prime else (lambda r, c: lay.Y(e, r, c))
    for a in range(3):
        for b in range(a, 3):
            t = [(1.0, Yf(k, a), Yf(k, b)) for k in range(3)]
            if a == b:
                for c in range(3):
                    for d in range(3): t.append((-Gm[c, d], lay.bg(ed.blk, c), lay.bg(ed.blk, d)))
            for c in range(3):
                for d in range(3): t.append((JR[a, c] * JR[b, d], lay.bg(ed.blk, c), lay.bg(ed.blk, d)))
            add(t, 0.0)


def _C1(lay, ed, e, add, pos):
    Ag = ed.Apg if pos else ed.Avg; Aa = ed.Apa if pos else ed.Ava
    Zf = (lambda c: lay.Zp(e, c)) if pos else (lambda c: lay.Zv(e, c))
    for a in range(3):
        t = [(1.0, lay.R(ed.i, m, a), Zf(m)) for m in range(3)]
        t += [(-Ag[a, c], lay.bg(ed.blk, c), lay.si) for c in range(3)]
        t += [(-Aa[a, c], lay.ba(ed.blk, c), lay.si) for c in range(3)]
        add(t, 0.0)


def _C2(lay, ed, e, add, pos):
    Ag = ed.Apg if pos else ed.Avg; Aa = ed.Apa if pos else ed.Ava
    Zf = (lambda c: lay.Zp(e, c)) if pos else (lambda c: lay.Zv(e, c))
    GG = Ag.T @ Ag; GA = Ag.T @ Aa; AA = Aa.T @ Aa
    t = [(1.0, Zf(m), Zf(m)) for m in range(3)]
    for c in range(3):
        for d in range(3):
            t.append((-GG[c, d], lay.bg(ed.blk, c), lay.bg(ed.blk, d)))
            t.append((-2 * GA[c, d], lay.bg(ed.blk, c), lay.ba(ed.blk, d)))
            t.append((-AA[c, d], lay.ba(ed.blk, c), lay.ba(ed.blk, d)))
    add(t, 0.0)


# --------------------------------------------------------------------------- #
# Solve / extract
# --------------------------------------------------------------------------- #
CONDITIONINGS = ('centre_whiten', None)


def _condition(layout, edges, anchors, scalar_priors, B, info=None):
    """Centre-and-whiten change of variables x = T x_t -> (d, xbar), or None if not applicable.

        x_i = d_i x_t_i + xbar_i s   (i != s),     x_s = x_t_s

    xbar is reference_point(); d_i = 1/sqrt(Q_c[i,i]) whitens every coordinate of the centred
    cost Q_c = (B T_c)^T (B T_c). Centring is the part that matters: it turns the s-column of B
    into the residual at xbar, so Q_t[s,s] = f(xbar) instead of a sum of tau * ref^2 terms that
    cancel to f* (4.7e9 -> 4.5e2 on synthetic sick at sigma = 1e-3). Whitening then brings the
    remaining curvature (tau_R ~ 1e7, tau_bg ~ 1e8) to O(1). Whitening WITHOUT centring is
    harmful (it shrinks everything around the uncancelled s-s entry) and is not offered.
    """
    xbar = reference_point(layout, edges, anchors, scalar_priors, info=info)
    if xbar is None:
        return None
    s = layout.si; xbar = xbar.copy(); xbar[s] = 0.0
    col = B @ xbar + B[:, s]
    dq = np.sqrt(np.einsum('ij,ij->j', B, B)); dq[s] = 0.0
    d = np.ones(layout.n); ok = dq > 0; d[ok] = 1.0 / dq[ok]
    Bt = B * d[None, :]; Bt[:, s] = col
    return d, xbar, Bt


def solve_cayley_sdp(layout, edges, anchors, scalar_priors=None, add_redundant=True,
                     obj_scale=1.0, solver_params=None, accept_any=False, diag=None,
                     rw_pairs=None, conditioning='centre_whiten'):
    """Shor relaxation min <Q,Z> s.t. <A_i,Z> = b_i, Z >= 0, solved by MOSEK (interior point).

    conditioning='centre_whiten' (default) solves the SAME relaxation after the exact congruence
    Z = T Z_t T^T of _condition: feasible sets are in bijection and rank and optimum are
    unchanged, but the data MOSEK sees is O(1)-O(1e4) instead of ~1e9 with a cancelling objective.
    Each transformed constraint row has its Z_t[s,s] coefficient folded into the right-hand side
    (Z_t[s,s] = 1 is itself a constraint) and is normalised to unit max coefficient. On the K = 8
    A/B (cayley_conditioning.py) this certifies every window at eta <= 1.3e-8 with no objective
    scaling, against eta up to 3e-3 for the scalar obj_scale='auto'. The rotation reference comes
    from the anchor (gauge 'R' / True), the R_0 prior ('R_prior'), or, under the free gauge, from
    wahba_R0; it falls back to None only if there is no anchor and no full set of GNSS priors.

    conditioning=None is the legacy formulation: Q as built, with obj_scale only for the solver.
    'auto' then targets max|Q| ~ 1e4 -- raw inverse-covariance weights push max|Q| to ~1e8 (MOSEK
    reports ProblemStatus.Unknown), while dividing by the max weight drives the objective below
    MOSEK's absolute tolerance. Never scales up. obj_scale also multiplies Q_t when conditioned,
    but 1.0 is the intended setting there.

    Returns (Z in the original coordinates, Q, p*). diag (if given) receives solsta, prosta,
    conditioning, obj_scale, qmax (raw Q) and qmax_solver, the dual bound dual_lb with
    lmin_S = lambda_min(S), S = Q_t - sum y_m A_m (de-scaled), and '_cond' for rank_and_gap.
    dual_lb = b^T y + min(0, lambda_min(S)) tr(Z_t): a lower bound on p* that does not depend on
    the primal value <Q,Z>, which an interior-point method can return slightly too high.
    """
    if conditioning not in CONDITIONINGS:
        raise ValueError("conditioning must be one of %r, got %r" % (CONDITIONINGS, conditioning))
    n = layout.n; s = layout.si
    B = residual_matrix(layout, edges, scalar_priors=scalar_priors, rw_pairs=rw_pairs)
    Q = B.T @ B
    rinfo = {}
    tr = _condition(layout, edges, anchors, scalar_priors, B, info=rinfo) if conditioning else None
    used = conditioning if tr is not None else None
    if tr is not None:
        d, xbar, Bt = tr; Qt = Bt.T @ Bt
    else:
        d, xbar, Qt = np.ones(n), np.zeros(n), Q
    qmax_t = float(np.abs(Qt).max())
    if isinstance(obj_scale, str):
        if obj_scale != 'auto':
            raise ValueError("obj_scale must be a positive float or 'auto', got %r" % obj_scale)
        obj_scale = min(1.0, 1e4 / qmax_t) if qmax_t > 0 else 1.0
    sc = float(obj_scale)
    # row a of T: (a, d_a) and (s, xbar_a); row s: (s, 1)
    Trow = ([((a, d[a]), (s, xbar[a])) if a != s else ((s, 1.0),) for a in range(n)]
            if used else None)
    trip_r, trip_c, trip_v, trip_k, rhs_l, cons = [], [], [], [], [], []
    with Model('cayley_sdp') as M:
        Z = M.variable('Z', Domain.inPSDCone(n))
        M.objective(ObjectiveSense.Minimize, Expr.dot((Qt * sc).tolist(), Z))

        def put(dct, rhs):
            k = len(cons)
            ex = [Expr.mul(v, Z.index(a, b)) for (a, b), v in dct.items()]
            cons.append(M.constraint(Expr.add(ex) if len(ex) > 1 else ex[0],
                                     Domain.equalsTo(float(rhs))))
            for (a, b), v in dct.items():
                trip_r.append(a); trip_c.append(b); trip_v.append(v); trip_k.append(k)
            rhs_l.append(float(rhs))

        def add(terms, rhs=0.0):
            dct = {}
            if used is None or (len(terms) == 1 and terms[0][1] == s and terms[0][2] == s):
                for c, a, b in terms:          # untransformed (also the s^2 = 1 row)
                    a, b = (a, b) if a <= b else (b, a)
                    dct[(a, b)] = dct.get((a, b), 0.0) + float(c)
                dct = {k: v for k, v in dct.items() if abs(v) > 1e-14}
                if dct:
                    put(dct, rhs)
                return
            for c, a, b in terms:
                for k, ta in Trow[a]:
                    for l, tb in Trow[b]:
                        co = float(c) * ta * tb
                        if co != 0.0:
                            key = (k, l) if k <= l else (l, k)
                            dct[key] = dct.get(key, 0.0) + co
            rhs = float(rhs) - dct.pop((s, s), 0.0)          # fold: Z_t[s,s] = 1
            mx = max([abs(v) for v in dct.values()] + [0.0])
            if mx == 0.0:
                return
            put({k: v / mx for k, v in dct.items() if abs(v) > 1e-13 * mx}, rhs / mx)

        _constraints(layout, edges, anchors, add, redundant=add_redundant)
        for k, v in (solver_params or {}).items():
            M.setSolverParam(k, v)
        if accept_any:
            M.acceptedSolutionStatus(AccSolutionStatus.Anything)
        M.solve()
        solsta = str(M.getPrimalSolutionStatus()); prosta = str(M.getProblemStatus())
        if diag is not None:
            diag.update(solsta=solsta, prosta=prosta, conditioning=used, obj_scale=sc,
                        qmax=float(np.abs(Q).max()), qmax_solver=qmax_t * sc,
                        R0_source=rinfo.get('R0_source'), wahba_sv=rinfo.get('wahba_sv'))
        if 'Optimal' not in solsta and not accept_any:
            return None, Q, prosta
        try:
            Zt = np.array(Z.level()).reshape(n, n)
        except Exception as exc:
            return None, Q, '%s / no level (%s)' % (prosta, type(exc).__name__)
        try:
            y = np.array([float(np.asarray(c.dual()).ravel()[0]) for c in cons])
        except Exception:
            y = None
    pstar = float(np.sum(Qt * Zt))
    if diag is not None:
        dual_lb = lmin = None
        if y is not None:
            # sum_m y_m A_m with A_m symmetric and <A_m, Z> = sum v Z[a,b] over a <= b
            r_, c_, v_ = np.array(trip_r), np.array(trip_c), np.array(trip_v) * y[np.array(trip_k)]
            YA = np.zeros((n, n))
            dg = r_ == c_
            np.add.at(YA, (r_[dg], c_[dg]), v_[dg])
            np.add.at(YA, (r_[~dg], c_[~dg]), 0.5 * v_[~dg]); np.add.at(YA, (c_[~dg], r_[~dg]), 0.5 * v_[~dg])
            by = float(np.dot(y, rhs_l)); trZ = float(np.trace(Zt))
            for sgn in (1.0, -1.0):        # MOSEK's dual sign convention, picked by lambda_min
                lm = float(np.linalg.eigvalsh(sc * Qt - sgn * YA)[0])
                if lmin is None or lm > lmin:
                    lmin, dual_lb = lm, sgn * by + min(0.0, lm) * trZ
            lmin, dual_lb = lmin / sc, dual_lb / sc
        diag.update(dual_lb=dual_lb, lmin_S=lmin, pstar=pstar,
                    _cond=dict(B=B, pstar=pstar) if used else None)
    if used:
        T = np.diag(d); T[:, s] += xbar
        return T @ Zt @ T.T, Q, pstar
    return Zt, Q, pstar


def extract_solution(Z, layout):
    R = []; v = []; p = []
    for k in range(layout.N):
        Rk = np.array([[Z[layout.R(k, r, c), layout.si] for c in range(3)] for r in range(3)])
        R.append(project_to_so3(Rk))
        v.append(np.array([Z[layout.v(k, c), layout.si] for c in range(3)]))
        p.append(np.array([Z[layout.p(k, c), layout.si] for c in range(3)]))
    bg = np.array([[Z[layout.bg(m, c), layout.si] for c in range(3)] for m in range(layout.M)])
    ba = np.array([[Z[layout.ba(m, c), layout.si] for c in range(3)] for m in range(layout.M)])
    return R, v, p, bg, ba


def rank_and_gap(Q, Z, layout, edges, R, v, p, bg, ba, cond=None):
    """cond (diag['_cond'] from a conditioned solve) supplies B and the solver-coordinate p*, so
    that f_hat = ||B x_hat||^2 and p* = <Q_t, Z_t> are evaluated without the cancellation that
    x^T Q x and <Q, Z> suffer when max|Q| ~ 1e9 and f* ~ 1."""
    w = np.sort(np.linalg.eigvalsh((Z + Z.T) / 2))[::-1]
    rank = int(np.sum(w > 1e-6 * max(1.0, w.max())))
    x = lift_solution(layout, edges, R, v, p, bg, ba)
    if cond is not None:
        rounded = float(np.sum((cond['B'] @ x) ** 2)); sdp_opt = float(cond['pstar'])
    else:
        rounded = float(x @ Q @ x); sdp_opt = float(np.sum(Q * Z))
    # NB: the (1 + |rounded|) denominator is NOT scale invariant -- rescaling all weights by c
    # shrinks it. rel_gap divides by |rounded| alone, so it is comparable across weightings.
    gap = (rounded - sdp_opt) / (1.0 + abs(rounded))
    rel_gap = (rounded - sdp_opt) / abs(rounded) if abs(rounded) > 0 else gap
    # eta: the standard relative suboptimality of the certifiable-estimation literature,
    #   eta = |f_hat - p*| / (1 + |f_hat| + |p*|).
    # Symmetric and absolute-valued, so unlike rel_gap it is well defined when the solver
    # returns p* slightly above f_hat, and the +1 keeps it finite as the optimum -> 0.
    eta = abs(rounded - sdp_opt) / (1.0 + abs(rounded) + abs(sdp_opt))
    # log10 of the leading eigenvalue ratio lam_1/lam_2: a continuous stand-in for the integer
    # rank, large when one direction dominates and Z is effectively rank one. Reported in logs
    # because the ratio spans many orders; clipped so an exactly-degenerate lam_2 stays finite.
    lam2 = max(float(w[1]), 1e-300) if w.size > 1 else 1e-300
    log_svr = float(np.log10(max(float(w[0]), 1e-300) / lam2))
    return rank, sdp_opt, rounded, gap, rel_gap, eta, log_svr


def _dual_fields(dg, rounded):
    """Dual-bound certificate fields for a runner's result dict."""
    lb = dg.get('dual_lb')
    return dict(dual_lb=lb, lmin_S=dg.get('lmin_S'), conditioning=dg.get('conditioning'),
                solsta=dg.get('solsta'), prosta=dg.get('prosta'), R0_source=dg.get('R0_source'),
                wahba_sv=dg.get('wahba_sv'),
                eta_dual=None if lb is None else (rounded - lb) / (1.0 + abs(rounded) + abs(lb)))


def _rot_err(R_est, R_true):
    return [np.degrees(np.arccos(np.clip((np.trace(Re.T @ Rt) - 1) / 2, -1, 1))) for Re, Rt in zip(R_est, R_true)]


# --------------------------------------------------------------------------- #
# Edge builders + priors
# --------------------------------------------------------------------------- #
def _chain_ground_truth(edges, R0, v0, p0):
    R = [R0]; v = [v0]; p = [p0]
    for ed in edges:
        dt = ed.dt_ij
        p.append(p[-1] + v[-1] * dt + 0.5 * GRAVITY * dt * dt + R[-1] @ ed.dpt)
        v.append(v[-1] + GRAVITY * dt + R[-1] @ ed.dvt)
        R.append(R[-1] @ ed.dRt)
    return R, v, p


def build_edges(acc, gyro, dt, kf_interval, bgbar, babar, gyro_sigma, accel_sigma,
                weight_mode='unit', prior_sigmas=(0.1, 0.05, 1e-2, 1e-2)):
    spe = int(round(kf_interval / dt)); E = len(acc) // spe
    w = compute_weights(gyro_sigma, accel_sigma, spe * dt, mode=weight_mode, prior_sigmas=prior_sigmas)
    edges = []
    for e in range(E):
        sl = slice(e * spe, (e + 1) * spe)
        pre = cayley_preintegrate(gyro[sl], acc[sl], dt, bgbar, babar)
        Gam = skew(pre['JR'] @ bgbar); Rbr = pre['dR'] @ (I3 - 0.5 * Gam)
        vbr = pre['dv'] - pre['Avg'] @ bgbar - pre['Ava'] @ babar
        pbr = pre['dp'] - pre['Apg'] @ bgbar - pre['Apa'] @ babar
        edges.append(CayleyEdge(i=e, j=e + 1, dRt=pre['dR'], dvt=pre['dv'], dpt=pre['dp'],
                                JR=pre['JR'], Avg=pre['Avg'], Ava=pre['Ava'], Apg=pre['Apg'], Apa=pre['Apa'],
                                dt_ij=spe * dt, tauR=w['R'], tauv=w['v'], taup=w['p'],
                                Rbr=Rbr, Gam=Gam, vbr=vbr, pbr=pbr))
    return edges, w


def _recenter(p_t):
    """Shift a window's position references so the first keyframe sits at the origin.

    The problem is exactly translation-invariant: every inertial residual uses only p_j - p_i (and
    v_i dt), and each GNSS prior moves with its own reference, so subtracting a constant from all
    positions AND all references leaves the cost, the constraints and the argmin unchanged up to
    that shift. Numerically it is not a no-op. build_cost_matrix accumulates tau * ref^2 into
    Q[s,s], so max|Q| grows with the square of the distance from whatever origin the dataset loader
    happened to pick. On UrbanNav, whose ENU frame is anchored at the first fix of the sequence,
    that inflates max|Q| from 8.3e7 at t=13 s to 2.8e10 at t=600 s -- a 340x swing driven purely by
    how far along the drive the window sits, which then wrecks the optimality gap (9.5e-2 instead
    of 1.5e-6). Re-centring removes the dependence. Returns (shifted positions, offset).
    """
    off = np.asarray(p_t[0], float).copy()
    return [np.asarray(p, float) - off for p in p_t], off


def _gauge(gauge_fix, R_t, v_t, p_t, R0_prior_sigma=None):
    """gauge_fix -> (anchors dict or None, first prior node, R_0 soft prior or None).

    Default everywhere is 'R'. With False, yaw is unobservable whenever the window is near
    stationary (the velocity priors are ~0 and there is no horizontal acceleration): UrbanNav
    medium@40 s and deep@90 s then certify at rank 1 with 72-82 deg of pure yaw error, and moving
    windows still carry 1-3 deg. Note 'R' pins tilt as well, which gravity already observes, so
    it acts as an infinitely precise attitude measurement rather than a pure gauge fix.

    False     -- no hard anchor at all; GNSS v/p priors on every node (first_node=0). The window is
                 held only by those priors, so the rotation gauge rests on gravity plus the
                 world-frame velocity priors.
    'R'       -- hard-anchor the initial *rotation* only, GNSS v/p priors on every node. Fixes the
                 rotation gauge, which experiment shows is the component tightness actually depends
                 on, while leaving translation entirely to the priors.
    'R_prior' -- the same rotation gauge imposed *softly*: a Gaussian prior on the nine entries of
                 R_0 with per-entry stddev R0_prior_sigma, instead of a hard equality. This enters
                 the objective, as tau (R_0[r,c] - Rbar_0[r,c] s)^2, rather than the constraint set,
                 so unlike 'R' it does not shrink the SDP's feasible set at all. 'R' is the
                 sigma -> 0 limit.
    True      -- legacy: hard-anchor R_0, v_0 and p_0, GNSS priors on nodes 1..N-1.
    """
    if gauge_fix is False or gauge_fix is None:
        return None, 0, None
    if gauge_fix is True:
        return dict(R0=R_t[0], v0=v_t[0], p0=p_t[0]), 1, None
    key = gauge_fix.upper() if isinstance(gauge_fix, str) else gauge_fix
    if key == 'R':
        return dict(R0=R_t[0]), 0, None
    if key == 'R_PRIOR':
        if not R0_prior_sigma or float(R0_prior_sigma) <= 0:
            raise ValueError("gauge_fix='R_prior' needs R0_prior_sigma > 0; got %r" % (R0_prior_sigma,))
        return None, 0, (R_t[0], 1.0 / float(R0_prior_sigma) ** 2)
    raise ValueError("gauge_fix must be False, True, 'R' or 'R_prior'; got %r" % (gauge_fix,))


def _build_scalar_priors(layout, v_ref, p_ref, bgbar, babar, w, first_node=1, R0_prior=None):
    """GNSS v/p priors on nodes `first_node`..N-1 + one broad phi^B on B_0.

    first_node=1 leaves node 0 to the hard anchor in `_constraints`; first_node=0 places a GNSS
    prior on every node and is what to use when gauge fixing is disabled (anchors=None).

    R0_prior=(Rbar_0, tau) additionally penalises tau ||R_0 - Rbar_0 s||_F^2, fixing the rotation
    gauge through the objective instead of through a hard equality.
    """
    pri = []
    if R0_prior is not None:
        R0ref, tauR0 = R0_prior
        for r in range(3):
            for c in range(3):
                pri.append((layout.R(0, r, c), R0ref[r, c], tauR0))
    for k in range(first_node, layout.N):
        for c in range(3):
            pri.append((layout.v(k, c), v_ref[k][c], w['pv']))
            pri.append((layout.p(k, c), p_ref[k][c], w['pp']))
    # One broad absolute phi^B on B_0 only. Blocks 1..M-1 are informed by their own phi^I factors
    # plus the phi^RW chain back to B_0, which is what makes the multi-block model a genuine test
    # rather than M independent copies of the single-bias problem.
    for c in range(3):
        pri.append((layout.bg(0, c), bgbar[c], w['pbg']))
        pri.append((layout.ba(0, c), babar[c], w['pba']))
    return pri


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
def run_synthetic(scenario_name='sick', T=6.0, dt=0.01, kf_interval=1.0, accel_sigma=1e-3,
                  gyro_sigma=1e-3, add_redundant=True, seed=0, weight_mode='unit',
                  prior_sigmas=(0.1, 0.05, 1e-2, 1e-2), obj_scale=1.0, noise_sigma=None,
                  iso_input=False, iso_reduce='max', bias_prior_rw=True, iso_propagate=False,
                  max_bias_precision=None, rot_chordal=True, gt_from_clean=True,
                  gauge_fix='R', R0_prior_sigma=None, kf_per_bias=None,
                  gyro_rw=1.9393e-5, accel_rw=3.0e-3, conditioning='centre_whiten',
                  solver_params=None, accept_any=False, diag=None):
    ang, lin = build_scenarios()[scenario_name]
    scenario = gtsam.ConstantTwistScenario(ang, lin)
    actual_bias, pim_params, _ = create_preintegration(gyro_sigma=gyro_sigma, accel_sigma=accel_sigma)
    runner = gtsam.ScenarioRunner(scenario, pim_params, dt, actual_bias)
    rng = np.random.default_rng(seed)
    bgbar = np.asarray(actual_bias.gyroscope()); babar = np.asarray(actual_bias.accelerometer())
    # ScenarioRunner samples its own noise inside measured*(), at density/sqrt(dt), which swamps
    # any injected noise below ~1e-2 and leaves the low end of a sweep uncontrolled. We therefore
    # build the noise-free stream from actual*() plus the true bias and add exactly the noise we
    # intend, so noise_sigma is the only noise source.
    acc = []; gyro = []; acc0 = []; gyro0 = []
    for t in np.arange(0, T, dt):
        a0 = np.asarray(runner.actualSpecificForce(t)) + babar
        w0 = np.asarray(runner.actualAngularVelocity(t)) + bgbar
        acc0.append(a0); gyro0.append(w0)
        ns = accel_sigma if noise_sigma is None else noise_sigma
        gs = gyro_sigma if noise_sigma is None else noise_sigma
        acc.append(a0 + rng.normal(0, ns, 3) if ns > 0 else a0.copy())
        gyro.append(w0 + rng.normal(0, gs, 3) if gs > 0 else w0.copy())
    wm = 'mle_raw' if weight_mode == 'dataset_cov' else weight_mode
    edges, w = build_edges(acc, gyro, dt, kf_interval, bgbar, babar, gyro_sigma, accel_sigma,
                           weight_mode=wm, prior_sigmas=prior_sigmas)
    if weight_mode == 'dataset_cov':
        # Same three steps as run_euroc: (1) input densities, combining the modelled density
        # with any injected noise; (2) full Jacobian covariance propagation per edge, collapsed
        # to one scalar per block; (3) bias prior from the random walk over the window.
        nd_g = float(np.hypot(gyro_sigma * np.sqrt(dt), (noise_sigma or 0.0) * np.sqrt(dt)))
        nd_a = float(np.hypot(accel_sigma * np.sqrt(dt), (noise_sigma or 0.0) * np.sqrt(dt)))
        spe = int(round(kf_interval / dt))
        cR = CHORDAL_ROT_FACTOR if rot_chordal else 1.0
        for e, ed in enumerate(edges):
            sl = slice(e * spe, (e + 1) * spe)
            Sig = preintegrate_covariance(gyro[sl], acc[sl], dt, bgbar, babar,
                                          nd_g, nd_a, iso_input=iso_input,
                                          iso_propagate=iso_propagate)
            sR, sv, sp = isotropic_block_sigmas(Sig, reduce=iso_reduce)
            ed.tauR, ed.tauv, ed.taup = 1.0 / (cR * sR), 1.0 / sv, 1.0 / sp
        bdt = float(T)
        ps = ((prior_sigmas[0], prior_sigmas[1], gyro_rw * np.sqrt(bdt), accel_rw * np.sqrt(bdt))
              if bias_prior_rw else tuple(prior_sigmas))
        cap = np.inf if max_bias_precision is None else float(max_bias_precision)
        w = dict(w, pv=1.0 / ps[0] ** 2, pp=1.0 / ps[1] ** 2,
                 pbg=min(1.0 / ps[2] ** 2, cap), pba=min(1.0 / ps[3] ** 2, cap))
    N = len(edges) + 1
    R0 = np.asarray(scenario.pose(0.0).rotation().matrix())
    ns0 = scenario.navState(0.0); v0 = np.asarray(ns0.velocity()); p0 = np.asarray(ns0.pose().translation())
    # Chain the truth from the CLEAN increments so the residuals do not vanish at the truth and
    # the optimum scales with the injected noise. gt_from_clean=False restores the old behaviour.
    if gt_from_clean:
        edges0, _ = build_edges(acc0, gyro0, dt, kf_interval, bgbar, babar, gyro_sigma,
                                accel_sigma, weight_mode=wm, prior_sigmas=prior_sigmas)
        R_t, v_t, p_t = _chain_ground_truth(edges0, R0, v0, p0)
    else:
        R_t, v_t, p_t = _chain_ground_truth(edges, R0, v0, p0)
    blk, M = bias_blocks(len(edges), kf_per_bias)
    lay = CayleyLayout(N, len(edges), M)
    rw_pairs = _build_bias_rw_pairs(M, kf_per_bias, kf_interval, gyro_rw, accel_rw)
    anchors, first_node, R0_prior = _gauge(gauge_fix, R_t, v_t, p_t, R0_prior_sigma)
    priors = _build_scalar_priors(lay, v_t, p_t, bgbar, babar, w, first_node=first_node,
                                  R0_prior=R0_prior)
    dg = {} if diag is None else diag
    Z, Q, info = solve_cayley_sdp(lay, edges, anchors, scalar_priors=priors,
                                  add_redundant=add_redundant, obj_scale=obj_scale,
                                  solver_params=solver_params, accept_any=accept_any, diag=dg,
                                  rw_pairs=rw_pairs, conditioning=conditioning)
    if Z is None:
        return dict(status=info, rank=None)
    R, v, p, bg, ba = extract_solution(Z, lay)
    rank, sdp_opt, rounded, gap, rel_gap, eta, log_svr = rank_and_gap(Q, Z, lay, edges, R, v, p, bg, ba,
                                                                      cond=dg.pop('_cond', None))
    return dict(status='ok', **_dual_fields(dg, rounded), rank=rank, sdp_opt=sdp_opt, gap=gap, rel_gap=rel_gap, eta=eta, log_svr=log_svr, rounded=rounded, rot_err=_rot_err(R, R_t),
                bg=bg, ba=ba, num_bias=M, kf_per_bias=kf_per_bias, n_var=lay.n,
                bias_err_per_block=[float(np.linalg.norm(np.r_[bg[m] - bgbar, ba[m] - babar]))
                                    for m in range(M)],
                bias_err=float(np.mean([np.linalg.norm(np.r_[bg[m] - bgbar, ba[m] - babar])
                                        for m in range(M)])), num_nodes=N,
                R_est=R, R_true=R_t, pos=p_t, pos_est=p, vel_true=v_t, vel_est=v)


def run_euroc(base, T=12.0, kf_interval=1.0, t_start=0.5, add_redundant=True,
              gyro_nd=1.6968e-4, accel_nd=2.0e-3, noise_sigma=0.0, seed=0,
              weight_mode='dataset', prior_sigmas=(0.1, 0.05, 1e-2, 1e-2), gauge_fix='R',
              R0_prior_sigma=None, recenter=True, kf_per_bias=None,
              gyro_rw=1.9393e-5, accel_rw=3.0e-3, obj_scale=1.0, rot_chordal=True,
              iso_input=False, bias_prior_rw=True, bias_interval=None,
              max_bias_precision=None, iso_reduce='max', iso_propagate=False,
              conditioning='centre_whiten', solver_params=None, accept_any=False, diag=None,
              gnss_seed=None, bias_init='gt'):
    """GNSS-IMU smoothing over one EuRoC window.

    By default the GNSS priors are the ground-truth velocity/position and the bias linearisation
    point is the ground-truth bias at the window start (the paper's setting). For a run that uses
    no ground truth at all beyond what a GNSS receiver would report:
      gnss_seed -- corrupt the v/p references with N(0, prior_sigmas[0/1]) noise, drawn by
                   `simulate_gnss` (so a baseline can reproduce the exact same measurements);
      bias_init -- linearisation point and prior mean for the bias: 'gt' (default), 'zero', or
                   an explicit (bg, ba), e.g. a calibration from another sequence. Pair a non-'gt'
                   value with bias_prior_rw=False and prior_sigmas[2:] set to its uncertainty.
                   Note the relaxation is only tight with an informative bias prior: broad ones
                   (0.05 rad/s, 0.3 m/s^2) give rank > 100 even at the true bias.
    """
    imu = np.loadtxt(f"{base}/imu0/data.csv", delimiter=",", skiprows=1)
    gt = np.loadtxt(f"{base}/state_groundtruth_estimate0/data.csv", delimiter=",", skiprows=1)
    it = imu[:, 0].astype(np.int64); gyro_all = imu[:, 1:4].copy(); acc_all = imu[:, 4:7].copy()
    gt_t = gt[:, 0].astype(np.int64); quat = gt[:, 4:8]; pos = gt[:, 1:4]; vel = gt[:, 8:11]
    bw = gt[:, 11:14]; ba_ = gt[:, 14:17]
    rng = np.random.default_rng(seed)
    if noise_sigma > 0:
        gyro_all += rng.normal(0, noise_sigma, gyro_all.shape); acc_all += rng.normal(0, noise_sigma, acc_all.shape)
    dt = float(np.median(np.diff(it)) * 1e-9)
    t0 = gt_t[0] + int(t_start * 1e9)
    K = int(round(T / kf_interval)); kf = [t0 + int(k * kf_interval * 1e9) for k in range(K + 1)]

    def gnearest(ts): return int(np.argmin(np.abs(gt_t - ts)))
    g0 = gnearest(t0); bgbar = bw[g0].copy(); babar = ba_[g0].copy()
    bg_true, ba_true = bgbar.copy(), babar.copy()
    if isinstance(bias_init, str) and bias_init == 'zero':
        bgbar = np.zeros(3); babar = np.zeros(3)
    elif not isinstance(bias_init, str):
        bgbar = np.asarray(bias_init[0], float).copy(); babar = np.asarray(bias_init[1], float).copy()
    elif bias_init != 'gt':
        raise ValueError("bias_init must be 'gt', 'zero' or (bg, ba); got %r" % (bias_init,))

    def Rgt(ts):
        q = quat[gnearest(ts)]; return project_to_so3(np.asarray(gtsam.Rot3.Quaternion(q[0], q[1], q[2], q[3]).matrix()))
    if weight_mode == 'dataset_cov':
        # (1) isotropic input noise from the datasheet; (2) full Jacobian covariance propagation per
        # edge, then collapse each block to a scalar by max-of-diagonal; (3) bias prior from
        # Sigma_bgd = dt * Cov(eta_bg). tau_R/tau_v/tau_p are set PER EDGE inside the loop below.
        nd_inj = noise_sigma * np.sqrt(dt)
        nd_g = float(np.hypot(gyro_nd, nd_inj)); nd_a = float(np.hypot(accel_nd, nd_inj))
        bdt = float(T if bias_interval is None else bias_interval)
        ps = ((prior_sigmas[0], prior_sigmas[1], gyro_rw * np.sqrt(bdt), accel_rw * np.sqrt(bdt))
              if bias_prior_rw else tuple(prior_sigmas))
        # cap the bias precision (IMU_Certifiable_Factor convention): the random walk is so small
        # that 1/sigma_bg^2 ~ 3e8 dominates max|Q| and wrecks the solver conditioning.
        cap = np.inf if max_bias_precision is None else float(max_bias_precision)
        w = dict(R=None, v=None, p=None, pv=1.0 / ps[0] ** 2, pp=1.0 / ps[1] ** 2,
                 pbg=min(1.0 / ps[2] ** 2, cap), pba=min(1.0 / ps[3] ** 2, cap))
    elif weight_mode in ('dataset', 'dataset_raw'):
        # sensor.yaml noise densities are already in the units the MLE formulas want:
        # sigma_theta = nd_g sqrt(T), sigma_dv = nd_a sqrt(T), sigma_dp = nd_a sqrt(T^3/3), so feeding
        # the densities (not the per-sample rate sigmas) makes tau_. = 1/sigma_.^2 exactly. Injected
        # noise is a per-sample rate sigma, so convert it to a density via *sqrt(dt) before combining.
        nd_inj = noise_sigma * np.sqrt(dt)
        nd_g = float(np.hypot(gyro_nd, nd_inj)); nd_a = float(np.hypot(accel_nd, nd_inj))
        # bias priors from the dataset random walks, drifting over the whole window (single shared B_0)
        ps = (prior_sigmas[0], prior_sigmas[1], gyro_rw * np.sqrt(T), accel_rw * np.sqrt(T))
        w = compute_weights(nd_g, nd_a, kf_interval, prior_sigmas=ps, rot_chordal=rot_chordal,
                            mode='mle_raw' if weight_mode == 'dataset_raw' else 'mle')
    else:
        sig_eff = float(np.hypot(max(gyro_nd, accel_nd) / np.sqrt(dt), noise_sigma))
        w = compute_weights(sig_eff, sig_eff, kf_interval, mode=weight_mode, prior_sigmas=prior_sigmas)
    edges = []; R_t = [Rgt(kf[0])]; v_t = [vel[gnearest(kf[0])].copy()]; p_t = [pos[gnearest(kf[0])].copy()]
    for k in range(K):
        a = int(np.searchsorted(it, kf[k])); b = int(np.searchsorted(it, kf[k + 1]))
        pre = cayley_preintegrate(gyro_all[a:b], acc_all[a:b], dt, bgbar, babar)
        Gam = skew(pre['JR'] @ bgbar); Rbr = pre['dR'] @ (I3 - 0.5 * Gam)
        vbr = pre['dv'] - pre['Avg'] @ bgbar - pre['Ava'] @ babar
        pbr = pre['dp'] - pre['Apg'] @ bgbar - pre['Apa'] @ babar
        if weight_mode == 'dataset_cov':
            Sig = preintegrate_covariance(gyro_all[a:b], acc_all[a:b], dt, bgbar, babar,
                                          nd_g, nd_a, iso_input=iso_input,
                                          iso_propagate=iso_propagate)
            sR, sv, sp = isotropic_block_sigmas(Sig, reduce=iso_reduce)
            cR = CHORDAL_ROT_FACTOR if rot_chordal else 1.0
            tR_e, tv_e, tp_e = 1.0 / (cR * sR), 1.0 / sv, 1.0 / sp
        else:
            tR_e, tv_e, tp_e = w['R'], w['v'], w['p']
        edges.append(CayleyEdge(i=k, j=k + 1, dRt=pre['dR'], dvt=pre['dv'], dpt=pre['dp'],
                                JR=pre['JR'], Avg=pre['Avg'], Ava=pre['Ava'], Apg=pre['Apg'], Apa=pre['Apa'],
                                dt_ij=(b - a) * dt, tauR=tR_e, tauv=tv_e, taup=tp_e,
                                Rbr=Rbr, Gam=Gam, vbr=vbr, pbr=pbr))
        R_t.append(Rgt(kf[k + 1])); v_t.append(vel[gnearest(kf[k + 1])].copy()); p_t.append(pos[gnearest(kf[k + 1])].copy())
    N = len(edges) + 1
    blk, M = bias_blocks(len(edges), kf_per_bias)
    for e, ed in enumerate(edges):
        ed.blk = blk[e]
    lay = CayleyLayout(N, len(edges), M)
    rw_pairs = _build_bias_rw_pairs(M, kf_per_bias, kf_interval, gyro_rw, accel_rw)
    p_off = np.zeros(3)
    if recenter:
        p_t, p_off = _recenter(p_t)
    v_m, p_m = v_t, p_t
    if gnss_seed is not None:
        v_m, p_m = simulate_gnss(v_t, p_t, prior_sigmas, gnss_seed)
    anchors, first_node, R0_prior = _gauge(gauge_fix, R_t, v_t, p_t, R0_prior_sigma)
    priors = _build_scalar_priors(lay, v_m, p_m, bgbar, babar, w, first_node=first_node,
                                  R0_prior=R0_prior)
    dg = {} if diag is None else diag
    Z, Q, info = solve_cayley_sdp(lay, edges, anchors, scalar_priors=priors,
                                  add_redundant=add_redundant, obj_scale=obj_scale,
                                  solver_params=solver_params, accept_any=accept_any, diag=dg,
                                  rw_pairs=rw_pairs, conditioning=conditioning)
    if Z is None:
        return dict(status=info, rank=None)
    R, v, p, bg, ba = extract_solution(Z, lay)
    rank, sdp_opt, rounded, gap, rel_gap, eta, log_svr = rank_and_gap(Q, Z, lay, edges, R, v, p, bg, ba,
                                                                      cond=dg.pop('_cond', None))
    return dict(status='ok', **_dual_fields(dg, rounded), rank=rank, sdp_opt=sdp_opt, gap=gap, rel_gap=rel_gap, eta=eta, log_svr=log_svr, rounded=rounded, rot_err=_rot_err(R, R_t),
                bg=bg, ba=ba, num_bias=M, kf_per_bias=kf_per_bias, n_var=lay.n,
                bias_err_per_block=[float(np.linalg.norm(np.r_[bg[m] - bg_true, ba[m] - ba_true]))
                                    for m in range(M)],
                bg_true=bg_true, ba_true=ba_true, bgbar=bgbar, babar=babar,
                num_nodes=N, R_est=R, R_true=R_t, v_est=v, p_est=p, v_true=v_t, pos=p_t, kf=kf, gauge_fix=gauge_fix,
                v_meas=v_m, p_meas=p_m, pos_offset=p_off)


def simulate_gnss(v_t, p_t, prior_sigmas, seed):
    """GNSS velocity/position fixes: the true values plus N(0, sigma_v), N(0, sigma_p) noise.

    One generator per seed, drawn in a fixed order (all velocities, then all positions), so any
    baseline handed the same seed sees exactly the same measurements.
    """
    rng = np.random.default_rng(seed)
    sv, sp = float(prior_sigmas[0]), float(prior_sigmas[1])
    v_m = [np.asarray(v, float) + rng.normal(0.0, sv, 3) for v in v_t]
    p_m = [np.asarray(p, float) + rng.normal(0.0, sp, 3) for p in p_t]
    return v_m, p_m


def run_urbannav(root, sequence='medium', T=8.0, kf_interval=1.0, t_start=6.0, add_redundant=True,
                 noise_model='empirical', noise_sigma=0.0, seed=0,
                 weight_mode='mle', prior_sigmas=(0.1, 0.05, 1e-2, 1e-2),
                 gyro_nd=None, accel_nd=None, gyro_rw=None, accel_rw=None,
                 obj_scale=1.0, rot_chordal=True, iso_input=False, bias_prior_rw=True,
                 bias_interval=None, max_bias_precision=None, iso_reduce='max', iso_propagate=False,
                 solver_params=None, accept_any=False, diag=None, conditioning='centre_whiten',
                 static_window=5.0, gravity=None, rate=None, gauge_fix='R',
                 R0_prior_sigma=None, recenter=True, kf_per_bias=None):
    """UrbanNav (Hong Kong) twin of `run_euroc`.

    Structurally identical to `run_euroc` -- same edges, same weights, same constraints, same
    relaxation -- so results are directly comparable.  Everything dataset-specific lives in
    `urbannav_data`; the five differences from EuRoC are documented there.  In particular the
    "GNSS" priors are still ground-truth-derived here (velocity and position at every keyframe,
    at `prior_sigmas`), exactly as in `run_euroc`; substituting real receiver fixes is a separate
    change.

    `t_start` is an offset in seconds from the first ground-truth epoch, and must land on a 1 Hz
    ground-truth epoch, as must every keyframe -- hence `kf_interval` being an integer multiple of
    1.0 s.  `noise_model` selects the accelerometer/gyro densities used by the MLE weights:
    'empirical' (measured from the stationary window) or 'datasheet' (the distributed
    xsens_imu_param.yaml, whose gyro density is 10-20x conservative).
    """
    seq = urbannav_data.load_sequence(root, sequence, rate=rate)
    t_all, gt, dt = seq['t'], seq['gt'], seq['dt']
    gyro_all, acc_all = seq['gyro'].copy(), seq['accel'].copy()
    g_local = urbannav_data.GRAVITY_HK if gravity is None else float(gravity)

    nd = dict(urbannav_data.NOISE_EMPIRICAL if noise_model == 'empirical' else urbannav_data.NOISE_DATASHEET)
    if gyro_nd is not None: nd['gyro_nd'] = gyro_nd
    if accel_nd is not None: nd['accel_nd'] = accel_nd
    if gyro_rw is not None: nd['gyro_rw'] = gyro_rw
    if accel_rw is not None: nd['accel_rw'] = accel_rw
    gyro_nd, accel_nd = float(nd['gyro_nd']), float(nd['accel_nd'])
    gyro_rw, accel_rw = float(nd['gyro_rw']), float(nd['accel_rw'])

    rng = np.random.default_rng(seed)
    if noise_sigma > 0:
        gyro_all += rng.normal(0, noise_sigma, gyro_all.shape); acc_all += rng.normal(0, noise_sigma, acc_all.shape)

    gt_t = gt['t']
    t0 = gt_t[0] + float(t_start)
    K = int(round(T / kf_interval))
    kf = [t0 + k * kf_interval for k in range(K + 1)]
    kidx = [urbannav_data.nearest_gt(gt_t, ts) for ts in kf]

    # UrbanNav has no bias ground truth; linearise about the stationary window at the start of the
    # sequence instead (see urbannav_data.static_bias_init -- this is a linearisation point only).
    # Deliberately estimated from the *clean* stream rather than the noise-injected one: holding the
    # linearisation point fixed across a noise sweep isolates the effect of measurement noise on the
    # relaxation, instead of confounding it with a drifting bias estimate.  This mirrors run_euroc,
    # where the ground-truth biases are likewise unaffected by `noise_sigma`.
    bgbar, babar = urbannav_data.static_bias_init(t_all, seq['gyro'], seq['accel'], gt['R'][0],
                                                  window=static_window, gravity=g_local)

    if weight_mode == 'dataset_cov':
        nd_inj = noise_sigma * np.sqrt(dt)
        nd_g = float(np.hypot(gyro_nd, nd_inj)); nd_a = float(np.hypot(accel_nd, nd_inj))
        bdt = float(T if bias_interval is None else bias_interval)
        ps = ((prior_sigmas[0], prior_sigmas[1], gyro_rw * np.sqrt(bdt), accel_rw * np.sqrt(bdt))
              if bias_prior_rw else tuple(prior_sigmas))
        cap = np.inf if max_bias_precision is None else float(max_bias_precision)
        w = dict(R=None, v=None, p=None, pv=1.0 / ps[0] ** 2, pp=1.0 / ps[1] ** 2,
                 pbg=min(1.0 / ps[2] ** 2, cap), pba=min(1.0 / ps[3] ** 2, cap))
    elif weight_mode in ('dataset', 'dataset_raw'):
        nd_inj = noise_sigma * np.sqrt(dt)
        nd_g = float(np.hypot(gyro_nd, nd_inj)); nd_a = float(np.hypot(accel_nd, nd_inj))
        ps = (prior_sigmas[0], prior_sigmas[1], gyro_rw * np.sqrt(T), accel_rw * np.sqrt(T))
        w = compute_weights(nd_g, nd_a, kf_interval, prior_sigmas=ps, rot_chordal=rot_chordal,
                            mode='mle_raw' if weight_mode == 'dataset_raw' else 'mle')
    else:
        sig_eff = float(np.hypot(max(gyro_nd, accel_nd) / np.sqrt(dt), noise_sigma))
        w = compute_weights(sig_eff, sig_eff, kf_interval, mode=weight_mode, prior_sigmas=prior_sigmas)

    edges = []
    R_t = [gt['R'][kidx[0]]]; v_t = [gt['vel'][kidx[0]].copy()]; p_t = [gt['pos'][kidx[0]].copy()]
    for k in range(K):
        a = int(np.searchsorted(t_all, kf[k])); b = int(np.searchsorted(t_all, kf[k + 1]))
        pre = cayley_preintegrate(gyro_all[a:b], acc_all[a:b], dt, bgbar, babar)
        Gam = skew(pre['JR'] @ bgbar); Rbr = pre['dR'] @ (I3 - 0.5 * Gam)
        vbr = pre['dv'] - pre['Avg'] @ bgbar - pre['Ava'] @ babar
        pbr = pre['dp'] - pre['Apg'] @ bgbar - pre['Apa'] @ babar
        if weight_mode == 'dataset_cov':
            Sig = preintegrate_covariance(gyro_all[a:b], acc_all[a:b], dt, bgbar, babar,
                                          nd_g, nd_a, iso_input=iso_input,
                                          iso_propagate=iso_propagate)
            sR, sv, sp = isotropic_block_sigmas(Sig, reduce=iso_reduce)
            cR = CHORDAL_ROT_FACTOR if rot_chordal else 1.0
            tR_e, tv_e, tp_e = 1.0 / (cR * sR), 1.0 / sv, 1.0 / sp
        else:
            tR_e, tv_e, tp_e = w['R'], w['v'], w['p']
        edges.append(CayleyEdge(i=k, j=k + 1, dRt=pre['dR'], dvt=pre['dv'], dpt=pre['dp'],
                                JR=pre['JR'], Avg=pre['Avg'], Ava=pre['Ava'], Apg=pre['Apg'], Apa=pre['Apa'],
                                dt_ij=(b - a) * dt, tauR=tR_e, tauv=tv_e, taup=tp_e,
                                Rbr=Rbr, Gam=Gam, vbr=vbr, pbr=pbr))
        R_t.append(gt['R'][kidx[k + 1]]); v_t.append(gt['vel'][kidx[k + 1]].copy()); p_t.append(gt['pos'][kidx[k + 1]].copy())
    N = len(edges) + 1
    blk, M = bias_blocks(len(edges), kf_per_bias)
    for e, ed in enumerate(edges):
        ed.blk = blk[e]
    lay = CayleyLayout(N, len(edges), M)
    rw_pairs = _build_bias_rw_pairs(M, kf_per_bias, kf_interval, gyro_rw, accel_rw)
    p_off = np.zeros(3)
    if recenter:
        p_t, p_off = _recenter(p_t)
    anchors, first_node, R0_prior = _gauge(gauge_fix, R_t, v_t, p_t, R0_prior_sigma)
    priors = _build_scalar_priors(lay, v_t, p_t, bgbar, babar, w, first_node=first_node,
                                  R0_prior=R0_prior)
    # Local gravity (9.7877 in Hong Kong) rather than the module default 9.81; the 0.24% difference
    # is the same order as the accelerometer bias, so it is not negligible for the velocity edges.
    with urbannav_data.override_gravity(sys.modules[__name__], g_local):
        dg = {} if diag is None else diag
        Z, Q, info = solve_cayley_sdp(lay, edges, anchors, scalar_priors=priors,
                                      add_redundant=add_redundant, obj_scale=obj_scale,
                                      solver_params=solver_params, accept_any=accept_any, diag=dg,
                                      rw_pairs=rw_pairs, conditioning=conditioning)
        if Z is None:
            return dict(status=info, rank=None)
        R, v, p, bg, ba = extract_solution(Z, lay)
        rank, sdp_opt, rounded, gap, rel_gap, eta, log_svr = rank_and_gap(Q, Z, lay, edges, R, v, p, bg, ba,
                                                                      cond=dg.pop('_cond', None))
    speed = np.linalg.norm(np.asarray(v_t), axis=1)
    return dict(status='ok', **_dual_fields(dg, rounded), rank=rank, sdp_opt=sdp_opt, gap=gap, rel_gap=rel_gap, eta=eta, log_svr=log_svr, rounded=rounded,
                rot_err=_rot_err(R, R_t), num_nodes=N, R_est=R, R_true=R_t, v_est=v, p_est=p, v_true=v_t, pos=p_t, kf=kf,
                pos_offset=p_off,
                bg=bg, ba=ba, bgbar=bgbar, babar=babar, label=seq['label'],
                num_bias=M, kf_per_bias=kf_per_bias, n_var=lay.n,
                bias_err_per_block=[float(np.linalg.norm(np.r_[bg[m] - bgbar, ba[m] - babar]))
                                    for m in range(M)],
                Q_flag=gt['Q'][kidx], mean_speed=float(speed.mean()),
                turn_deg=float(np.abs(np.diff(np.unwrap(np.deg2rad(gt['heading'][kidx])))).sum() * 180 / np.pi))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Cayley GNSS-IMU SDP relaxation (single shared bias)")
    parser.add_argument("--euroc", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="sick")
    parser.add_argument("--T", type=float, default=6.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--kf", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--weight-mode", type=str, default="dataset",
                        choices=["unit", "mle", "dataset", "dataset_raw", "dataset_cov"])
    parser.add_argument("--gauge-fix", default="R", choices=["R", "free", "full"],
                        help="R: hard-anchor R_0 (default); free: GNSS priors only; full: anchor R_0, v_0, p_0")
    parser.add_argument("--no-redundant", action="store_true")
    args = parser.parse_args()

    print("Jacobian finite-difference check:")
    for k, v in check_preintegration_jacobians().items():
        print("  %-6s max|analytic-FD| = %.2e" % (k, v))
    red = not args.no_redundant
    gfix = {'R': 'R', 'free': False, 'full': True}[args.gauge_fix]
    if args.euroc:
        r = run_euroc(args.euroc, T=args.T, kf_interval=args.kf, add_redundant=red,
                      weight_mode=args.weight_mode, gauge_fix=gfix)
        tag = "EuRoC"
    else:
        r = run_synthetic(args.scenario, T=args.T, dt=args.dt, kf_interval=args.kf,
                          accel_sigma=args.sigma, gyro_sigma=args.sigma, add_redundant=red, weight_mode=args.weight_mode,
                          gauge_fix=gfix)
        tag = "synthetic:%s" % args.scenario
    print("\n%s  redundant=%s  weights=%s" % (tag, red, args.weight_mode))
    if r['rank'] is None:
        print("  SOLVER FAILED:", r['status'])
    else:
        print("  nodes=%d  rank(Z*)=%d  sdp_opt=%.2e  rel_gap=%.2e  mean_rot_err=%.3f deg" % (
            r['num_nodes'], r['rank'], r['sdp_opt'], r['gap'], float(np.mean(r['rot_err']))))
