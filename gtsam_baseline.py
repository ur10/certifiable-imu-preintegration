"""Standard exponential-map IMU pre-integration baseline, on the *same* GNSS-IMU smoothing
problem the Cayley SDP solves, so the two can be compared apples to apples.

Matched between the two pipelines
---------------------------------
  * the same raw IMU samples over the same keyframe grid,
  * the same continuous noise densities sigma_g, sigma_a, so the MLE weights agree
    (verified: the propagated 9x9 covariances match to 4e-4 relative),
  * the same GNSS position/velocity factors on every node with the same precisions,
  * the same bias linearisation point bbar, the same B_0 prior sigma, the same number of
    bias blocks M and the same random-walk factors between them,
  * the same gravity magnitude and the same recentred world frame.

Deliberately NOT matched -- this is the comparison
--------------------------------------------------
  * propagation: GTSAM integrates rotations with exp and propagates the full 9x9 Sigma with
    exp right-Jacobians; the Cayley pipeline integrates with Cay and collapses each 3x3
    block of Sigma to a scalar (the isotropic surrogate the QCQP needs).
  * solver: GTSAM runs Levenberg-Marquardt from an initial guess; the Cayley pipeline solves
    one SDP with no initial guess and returns a rank certificate.

Because the baseline is a local solver, `initialisation` selects the initial guess, which is
the variable that actually decides whether it finds the global optimum.
"""
import time
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, V, B

I3 = np.eye(3)


def _gravity_align(a_body, gravity):
    """Coarse alignment: roll/pitch from the measured specific force, heading left at zero.

    With g^W = (0,0,-g) the specific force at low acceleration is atilde ~= R^T (0,0,+g), so the
    initial R is the minimal rotation carrying atilde/|atilde| onto the world z axis.
    """
    u = np.asarray(a_body, float)
    n = np.linalg.norm(u)
    if n < 1e-9:
        return np.eye(3)
    u = u / n
    ez = np.array([0.0, 0.0, 1.0])
    v = np.cross(u, ez); c = float(u @ ez)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _navstate_init(mode, R_t, v_t, p_t, rng, rot_sigma_deg=0.0, pos_sigma=0.0,
                   a_mean=None, gravity=9.81, heading_deg=0.0):
    """Initial guess for every node. Returns lists (R, v, p)."""
    N = len(R_t)
    if mode == 'gravity':
        # what a real system actually has: GNSS gives p and v, the accelerometer gives
        # roll/pitch, heading is unknown and starts at zero.
        R0 = _gravity_align(a_mean, gravity)
        return [R0.copy() for _ in range(N)], [v.copy() for v in v_t], [p.copy() for p in p_t]
    if mode == 'gt':
        return [R.copy() for R in R_t], [v.copy() for v in v_t], [p.copy() for p in p_t]
    if mode == 'gnss':
        # what a real system has before optimising: GNSS gives p and v, attitude is unknown
        # beyond gravity, so start every rotation at identity.
        return [np.eye(3) for _ in range(N)], [v.copy() for v in v_t], [p.copy() for p in p_t]
    if mode == 'perturbed':
        Rs, vs, ps = [], [], []
        for k in range(N):
            w = rng.normal(0.0, np.deg2rad(rot_sigma_deg), 3)
            Rs.append(R_t[k] @ gtsam.Rot3.Expmap(w).matrix())
            vs.append(v_t[k] + rng.normal(0.0, pos_sigma, 3))
            ps.append(p_t[k] + rng.normal(0.0, pos_sigma, 3))
        return Rs, vs, ps
    if mode == 'angle':
        # attitude perturbed by EXACTLY `rot_sigma_deg` about a uniformly random axis, so the
        # initial attitude error equals that angle. One draw per node would average out; use a
        # single common draw so the whole window starts equally wrong.
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        dR = gtsam.Rot3.Expmap(u * np.deg2rad(rot_sigma_deg)).matrix()
        return [dR @ R for R in R_t], [v.copy() for v in v_t], [p.copy() for p in p_t]
    if mode == 'tilt_heading':
        # decomposed error: `heading_deg` about world z (what gravity cannot fix) and
        # `rot_sigma_deg` about world x (what gravity does fix).
        ps_, th = np.deg2rad(rot_sigma_deg), np.deg2rad(heading_deg)
        Rx = np.array([[1, 0, 0], [0, np.cos(ps_), -np.sin(ps_)], [0, np.sin(ps_), np.cos(ps_)]])
        Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        dR = Rz @ Rx
        return [dR @ R for R in R_t], [v.copy() for v in v_t], [p.copy() for p in p_t]
    if mode == 'heading':
        # correct roll/pitch from gravity, heading deliberately wrong by `heading_deg`
        R0 = _gravity_align(a_mean, gravity)
        th = np.deg2rad(heading_deg)
        Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        return [Rz @ R0 for _ in range(N)], [v.copy() for v in v_t], [p.copy() for p in p_t]
    if mode in ('haar', 'haar_shared'):
        # attitude genuinely unknown: draw from the Haar (uniform) measure on SO(3) via a
        # unit quaternion with a normally distributed 4-vector. 'haar_shared' uses one draw
        # for the whole window -- the direct analogue of the identity row, which likewise
        # puts the same (wrong) rotation on every node. 'haar' redraws per node, which is
        # strictly harder because the initial guess is not even self-consistent.
        def _haar():
            q = rng.normal(size=4); q /= np.linalg.norm(q)
            return np.asarray(gtsam.Rot3.Quaternion(q[0], q[1], q[2], q[3]).matrix())
        if mode == 'haar_shared':
            R0 = _haar()
            Rs = [R0.copy() for _ in range(N)]
        else:
            Rs = [_haar() for _ in range(N)]
        return Rs, [v.copy() for v in v_t], [p.copy() for p in p_t]
    if mode == 'haar_pv':
        # attitude unknown AND the GNSS-measured states corrupted: one Haar draw on every
        # keyframe, plus an isotropic perturbation of sigma `pos_sigma` on every velocity and
        # position. pos_sigma is absolute (m and m/s), so it is read against the GNSS prior
        # sigmas rather than against the trajectory scale.
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        R0 = np.asarray(gtsam.Rot3.Quaternion(q[0], q[1], q[2], q[3]).matrix())
        Rs = [R0.copy() for _ in range(N)]
        vs = [v + rng.normal(0.0, pos_sigma, 3) for v in v_t]
        ps = [p + rng.normal(0.0, pos_sigma, 3) for p in p_t]
        return Rs, vs, ps
    if mode == 'random_rot':
        # the realistic hard case: GNSS pins position and velocity, attitude is unknown.
        Rs = [gtsam.Rot3.Expmap(rng.normal(0, 1, 3) * np.pi).matrix() for _ in range(N)]
        return Rs, [v.copy() for v in v_t], [p.copy() for p in p_t]
    if mode == 'random':
        Rs = [gtsam.Rot3.Expmap(rng.normal(0, 1, 3) * np.pi).matrix() for _ in range(N)]
        return Rs, [rng.normal(0, 1, 3) for _ in range(N)], [rng.normal(0, 1, 3) for _ in range(N)]
    raise ValueError('unknown initialisation %r' % (mode,))


def solve_gtsam(gyro, acc, dt, kf_idx, bgbar, babar, R_t, v_t, p_t, blk,
                nd_g, nd_a, prior_sigmas, gyro_rw, accel_rw, kf_interval,
                gravity=9.81, initialisation='gnss', seed=0,
                rot_sigma_deg=0.0, pos_sigma=0.0, heading_deg=0.0, integration_sigma=1e-6,
                bias_init_k=0.0, max_iter=200, lambda_upper=1e12):
    """Build and solve the exp-map factor graph. kf_idx[k] indexes the IMU stream at node k."""
    rng = np.random.default_rng(seed)
    N = len(kf_idx)
    M = int(max(blk)) + 1
    sig_pv, sig_pp = float(prior_sigmas[0]), float(prior_sigmas[1])

    params = gtsam.PreintegrationParams.MakeSharedU(gravity)     # n_gravity = (0,0,-g)
    params.setGyroscopeCovariance(nd_g ** 2 * I3)
    params.setAccelerometerCovariance(nd_a ** 2 * I3)
    params.setIntegrationCovariance(integration_sigma ** 2 * I3)
    bias_lin = gtsam.imuBias.ConstantBias(babar, bgbar)

    graph = gtsam.NonlinearFactorGraph()

    # --- inertial factors, one per edge, against that edge's bias block -------------------
    pims = []
    for k in range(N - 1):
        pim = gtsam.PreintegratedImuMeasurements(params, bias_lin)
        for s in range(kf_idx[k], kf_idx[k + 1]):
            pim.integrateMeasurement(acc[s], gyro[s], dt)
        graph.push_back(gtsam.ImuFactor(X(k), V(k), X(k + 1), V(k + 1), B(blk[k]), pim))
        pims.append(pim)

    # --- GNSS position + velocity on every node, matching _build_scalar_priors ------------
    pos_noise = gtsam.noiseModel.Isotropic.Sigma(3, sig_pp)
    vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, sig_pv)
    for k in range(N):
        graph.push_back(gtsam.GPSFactor(X(k), p_t[k], pos_noise))
        graph.push_back(gtsam.PriorFactorVector(V(k), v_t[k], vel_noise))

    # --- bias prior on B_0 and random walk between blocks --------------------------------
    bdt = kf_interval * (N - 1)
    sig_bg, sig_ba = gyro_rw * np.sqrt(bdt), accel_rw * np.sqrt(bdt)
    graph.push_back(gtsam.PriorFactorConstantBias(
        B(0), bias_lin, gtsam.noiseModel.Diagonal.Sigmas(np.r_[np.full(3, sig_ba), np.full(3, sig_bg)])))
    if M > 1:
        n_edge_per_blk = max(1, (N - 1) // M)
        rwdt = kf_interval * n_edge_per_blk
        rw = gtsam.noiseModel.Diagonal.Sigmas(
            np.r_[np.full(3, accel_rw * np.sqrt(rwdt)), np.full(3, gyro_rw * np.sqrt(rwdt))])
        for m in range(M - 1):
            graph.push_back(gtsam.BetweenFactorConstantBias(
                B(m), B(m + 1), gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)), rw))

    # --- initial values -------------------------------------------------------------------
    a_mean = np.mean(acc[kf_idx[0]:kf_idx[min(1, len(kf_idx) - 1)]], axis=0) - babar \
        if kf_idx[-1] > kf_idx[0] else np.array([0.0, 0.0, gravity])
    R0, v0, p0 = _navstate_init(initialisation, R_t, v_t, p_t, rng, rot_sigma_deg, pos_sigma,
                                a_mean=a_mean, gravity=gravity, heading_deg=heading_deg)
    vals = gtsam.Values()
    for k in range(N):
        vals.insert(X(k), gtsam.Pose3(gtsam.Rot3(R0[k]), p0[k]))
        vals.insert(V(k), v0[k])
    for m in range(M):
        if bias_init_k > 0.0:
            # `bias_init_k` sigmas off the B_0 prior. Only the initial value moves; the prior
            # factor and the pre-integration linearisation point stay at bbar.
            vals.insert(B(m), gtsam.imuBias.ConstantBias(
                babar + rng.normal(0.0, bias_init_k * sig_ba, 3),
                bgbar + rng.normal(0.0, bias_init_k * sig_bg, 3)))
        else:
            vals.insert(B(m), bias_lin)

    lm = gtsam.LevenbergMarquardtParams()
    lm.setMaxIterations(max_iter)
    lm.setlambdaUpperBound(lambda_upper)   # default 1e5 aborts at iteration 0 from a poor guess
    opt = gtsam.LevenbergMarquardtOptimizer(graph, vals, lm)
    t0 = time.time()
    res = opt.optimize()
    secs = time.time() - t0

    R_e = [res.atPose3(X(k)).rotation().matrix() for k in range(N)]
    p_e = [res.atPose3(X(k)).translation() for k in range(N)]
    v_e = [res.atVector(V(k)) for k in range(N)]
    bias = [res.atConstantBias(B(m)) for m in range(M)]
    return dict(R_est=R_e, v_est=v_e, p_est=p_e,
                bg=[b.gyroscope() for b in bias], ba=[b.accelerometer() for b in bias],
                cost0=float(graph.error(vals)), cost=float(graph.error(res)),
                iters=int(opt.iterations()), secs=secs, num_bias=M,
                initialisation=initialisation)
