"""Apples-to-apples: standard exp-map pre-integration (GTSAM/LM) vs the Cayley SDP factor,
on the same GNSS-IMU smoothing problem the paper evaluates.

Matched between the two pipelines
    same raw IMU over the same keyframe grid; the same continuous noise densities, so the MLE
    weights agree (propagated 9x9 covariances match to 4e-4 relative); the same GNSS
    position/velocity factors on every node at the same precisions; the same bias
    linearisation point, B_0 prior, bias-block count M and random-walk factors; the same
    gravity and recentred world frame.

Differing, by design
    propagation -- exp with the full 9x9 Sigma, versus Cay with each 3x3 block collapsed to a
    scalar (the isotropic surrogate the QCQP needs);
    solution    -- Levenberg-Marquardt from an initial guess, versus one SDP with no initial
                   guess and a rank certificate.

    python3 compare_gtsam_cayley.py matched      -> matched.json
    python3 compare_gtsam_cayley.py divergence   -> divergence.json
    python3 compare_gtsam_cayley.py plot
"""
import argparse, json, sys, time
import numpy as np
import gtsam

import sdp_preintegration_cayley as C
import urbannav_data
from gtsam_baseline import solve_gtsam

import datasets
EUROC = datasets.EUROC          # override with $EUROC_MH01, see datasets.py
URBANNAV = datasets.URBANNAV    # override with $URBANNAV_ROOT
ND_G, ND_A = 1.6968e-4, 2.0e-3            # EuRoC sensor.yaml densities
GYRO_RW, ACCEL_RW = 1.9393e-5, 3.0e-3
PRIOR_SIGMAS = (1e-2, 1e-2, 1e-3, 1e-3)   # the paper's nominal GNSS precisions


# --------------------------------------------------------------------------- #
# window loaders -- replicate run_euroc / run_urbannav preparation exactly
# --------------------------------------------------------------------------- #
def load_euroc_window(base, T, kf_interval, t_start, kf_per_bias=None, recenter=True):
    imu = np.loadtxt(f"{base}/imu0/data.csv", delimiter=",", skiprows=1)
    gt = np.loadtxt(f"{base}/state_groundtruth_estimate0/data.csv", delimiter=",", skiprows=1)
    it = imu[:, 0].astype(np.int64); gyro = imu[:, 1:4].copy(); acc = imu[:, 4:7].copy()
    gt_t = gt[:, 0].astype(np.int64); quat = gt[:, 4:8]; pos = gt[:, 1:4]; vel = gt[:, 8:11]
    bw = gt[:, 11:14]; ba_ = gt[:, 14:17]
    dt = float(np.median(np.diff(it)) * 1e-9)
    t0 = gt_t[0] + int(t_start * 1e9)
    K = int(round(T / kf_interval))
    kf = [t0 + int(k * kf_interval * 1e9) for k in range(K + 1)]

    def gnear(ts): return int(np.argmin(np.abs(gt_t - ts)))
    g0 = gnear(t0)

    def Rgt(ts):
        q = quat[gnear(ts)]
        return C.project_to_so3(np.asarray(gtsam.Rot3.Quaternion(q[0], q[1], q[2], q[3]).matrix()))

    p_t = [pos[gnear(t)].copy() for t in kf]
    p_off = np.zeros(3)
    if recenter:
        p_t, p_off = C._recenter(p_t)
    blk, M = C.bias_blocks(K, kf_per_bias)
    return dict(gyro=gyro, acc=acc, dt=dt, kf_idx=[int(np.searchsorted(it, t)) for t in kf],
                bgbar=bw[g0].copy(), babar=ba_[g0].copy(),
                R_t=[Rgt(t) for t in kf], v_t=[vel[gnear(t)].copy() for t in kf], p_t=p_t,
                blk=blk, M=M, N=K + 1, gravity=9.81, nd_g=ND_G, nd_a=ND_A,
                gyro_rw=GYRO_RW, accel_rw=ACCEL_RW)


def load_urbannav_window(root, sequence, T, kf_interval, t_start, kf_per_bias=None, recenter=True):
    seq = urbannav_data.load_sequence(root, sequence)
    t_all, gt, dt = seq['t'], seq['gt'], seq['dt']
    gyro, acc = seq['gyro'].copy(), seq['accel'].copy()
    g_local = urbannav_data.GRAVITY_HK
    nd = dict(urbannav_data.NOISE_EMPIRICAL)
    gt_t = gt['t']
    t0 = gt_t[0] + float(t_start)
    K = int(round(T / kf_interval))
    kf = [t0 + k * kf_interval for k in range(K + 1)]
    kidx = [urbannav_data.nearest_gt(gt_t, ts) for ts in kf]
    bgbar, babar = urbannav_data.static_bias_init(t_all, seq['gyro'], seq['accel'], gt['R'][0],
                                                  window=5.0, gravity=g_local)
    p_t = [gt['pos'][k].copy() for k in kidx]
    p_off = np.zeros(3)
    if recenter:
        p_t, p_off = C._recenter(p_t)
    blk, M = C.bias_blocks(K, kf_per_bias)
    return dict(gyro=gyro, acc=acc, dt=dt, kf_idx=[int(np.searchsorted(t_all, t)) for t in kf],
                bgbar=bgbar, babar=babar,
                R_t=[gt['R'][k] for k in kidx], v_t=[gt['vel'][k].copy() for k in kidx], p_t=p_t,
                blk=blk, M=M, N=K + 1, gravity=g_local,
                nd_g=float(nd['gyro_nd']), nd_a=float(nd['accel_nd']),
                gyro_rw=float(nd['gyro_rw']), accel_rw=float(nd['accel_rw']))


def load_window(ds, T, kfi, off, kfpb):
    if ds == 'euroc':
        return load_euroc_window(EUROC, T, kfi, off, kfpb)
    return load_urbannav_window(URBANNAV, ds, T, kfi, off, kfpb)


def solve_ours(ds, T, kfi, off, kfpb, prior_sigmas):
    kw = dict(T=T, kf_interval=kfi, t_start=off, weight_mode='dataset_cov', iso_reduce='mean',
              obj_scale='auto', prior_sigmas=prior_sigmas, gauge_fix=False, accept_any=True,
              add_redundant=True, kf_per_bias=kfpb)
    if ds == 'euroc':
        return C.run_euroc(EUROC, gyro_nd=ND_G, accel_nd=ND_A,
                           gyro_rw=GYRO_RW, accel_rw=ACCEL_RW, **kw)
    return C.run_urbannav(URBANNAV, sequence=ds, **kw)


# --------------------------------------------------------------------------- #
def rot_errors(R_est, R_true):
    out = []
    for Re, Rt in zip(R_est, R_true):
        c = (np.trace(np.asarray(Re).T @ np.asarray(Rt)) - 1.0) / 2.0
        out.append(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    return np.array(out)


def vec_errors(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b), axis=1)


def summarize(tag, R_e, v_e, p_e, W, extra=None):
    re_ = rot_errors(R_e, W['R_t']); ve_ = vec_errors(v_e, W['v_t']); pe_ = vec_errors(p_e, W['p_t'])
    d = dict(method=tag, rot_mean=float(re_.mean()), rot_max=float(re_.max()),
             vel_mean=float(ve_.mean()), pos_mean=float(pe_.mean()), pos_max=float(pe_.max()),
             rot_per_node=re_.tolist(), pos_per_node=pe_.tolist())
    if extra: d.update(extra)
    return d


def run_gtsam(W, prior_sigmas, init, seed=0, **kw):
    return solve_gtsam(W['gyro'], W['acc'], W['dt'], W['kf_idx'], W['bgbar'], W['babar'],
                       W['R_t'], W['v_t'], W['p_t'], W['blk'],
                       W['nd_g'], W['nd_a'], prior_sigmas, W['gyro_rw'], W['accel_rw'],
                       kf_interval=1.0, gravity=W['gravity'], initialisation=init, seed=seed, **kw)


# --------------------------------------------------------------------------- #
# Experiment 1: matched configuration, the paper's nominal setting
# --------------------------------------------------------------------------- #
MATCHED = [('euroc', 16.0, 8, [6.0, 40.0, 90.0, 132.0]),
           ('medium', 16.0, 8, [6.0, 40.0]),
           ('deep', 16.0, 8, [6.0, 40.0])]


def exp_matched(out='matched.json'):
    rows = []
    for ds, T, kfpb, offs in MATCHED:
        for off in offs:
            W = load_window(ds, T, 1.0, off, kfpb)
            t0 = time.time(); r = solve_ours(ds, T, 1.0, off, kfpb, PRIOR_SIGMAS); secs = time.time() - t0
            if r.get('rank') is None:
                rows.append(dict(ds=ds, off=off, method='cayley-sdp', failed=str(r.get('status'))))
            else:
                rows.append(dict(ds=ds, off=off, **summarize(
                    'cayley-sdp', r['R_est'], r['v_est'], r['p_est'], W,
                    dict(secs=secs, rank=int(r['rank']), eta=float(r['eta']),
                         certified=bool(r['rank'] == 1), n_var=int(r['n_var'])))))
            for init in ('gt', 'gnss'):
                g = run_gtsam(W, PRIOR_SIGMAS, init)
                rows.append(dict(ds=ds, off=off, **summarize(
                    'gtsam-' + init, g['R_est'], g['v_est'], g['p_est'], W,
                    dict(secs=g['secs'], iters=g['iters'], cost=g['cost'], certified=False))))
            for d in rows[-3:]:
                print("  %-7s off=%-6.0f %-12s rot %7.3f/%8.3f deg  pos %8.5f m  vel %8.5f  %6.2fs%s"
                      % (d['ds'], d['off'], d['method'], d.get('rot_mean', np.nan),
                         d.get('rot_max', np.nan), d.get('pos_mean', np.nan),
                         d.get('vel_mean', np.nan), d.get('secs', np.nan),
                         '  rank=%d' % d['rank'] if 'rank' in d else ''), flush=True)
    json.dump(rows, open(out, 'w'), indent=1); print('wrote', out)
    return rows


# --------------------------------------------------------------------------- #
# Experiment 2: weaken the GNSS until the local solver stops finding the optimum
# --------------------------------------------------------------------------- #
SIGMAS = [1e-2, 1e-1, 3e-1, 1e0, 3e0, 1e1, 3e1]
SEEDS = range(8)


def exp_divergence(out='divergence.json', ds='euroc', T=16.0, off=6.0, kfpb=8):
    W = load_window(ds, T, 1.0, off, kfpb)
    rows = []
    for sp in SIGMAS:
        ps = (sp, sp, 1e-3, 1e-3)
        ref = run_gtsam(W, ps, 'gt')
        rows.append(dict(sigma=sp, **summarize('gtsam-gt', ref['R_est'], ref['v_est'], ref['p_est'],
                                               W, dict(cost=ref['cost'], secs=ref['secs']))))
        for sd in SEEDS:
            g = run_gtsam(W, ps, 'random_rot', seed=sd)
            rows.append(dict(sigma=sp, seed=sd, **summarize(
                'gtsam-random_rot', g['R_est'], g['v_est'], g['p_est'], W,
                dict(cost=g['cost'], secs=g['secs'],
                     missed=bool(g['cost'] > ref['cost'] * 1.01 + 1e-12)))))
        t0 = time.time(); r = solve_ours(ds, T, 1.0, off, kfpb, ps); secs = time.time() - t0
        if r.get('rank') is None:
            rows.append(dict(sigma=sp, method='cayley-sdp', failed=str(r.get('status'))))
        else:
            rows.append(dict(sigma=sp, **summarize(
                'cayley-sdp', r['R_est'], r['v_est'], r['p_est'], W,
                dict(secs=secs, rank=int(r['rank']), eta=float(r['eta']),
                     log_svr=float(r['log_svr']), certified=bool(r['rank'] == 1)))))
        at = [d for d in rows if d['sigma'] == sp]
        ref_ = [d for d in at if d['method'] == 'gtsam-gt'][0]
        rnd = [d for d in at if d['method'] == 'gtsam-random_rot']
        ours = [d for d in at if d['method'] == 'cayley-sdp'][0]
        print("  sigma=%-6.2g  ref %7.3f deg | rand-init med %8.3f deg, missed %d/%d | "
              "ours %7.3f deg rank=%s"
              % (sp, ref_['rot_mean'], np.median([d['rot_mean'] for d in rnd]),
                 sum(d['missed'] for d in rnd), len(rnd),
                 ours.get('rot_mean', float('nan')), ours.get('rank', 'FAIL')), flush=True)
    json.dump(rows, open(out, 'w'), indent=1); print('wrote', out)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', nargs='?', default='matched', choices=['matched', 'divergence', 'both'])
    a = ap.parse_args()
    if a.what in ('matched', 'both'): exp_matched()
    if a.what in ('divergence', 'both'): exp_divergence()
