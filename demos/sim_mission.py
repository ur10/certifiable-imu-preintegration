"""Demo 3 -- a certificate as a safety monitor, on the simulated quadrotor mission (sim_quad.py).

Sliding 16 s windows every 4 s along the mission.

  stage 1 (slow, cached)  the Cayley SDP on every window, bias carried from the previous window,
                          GNSS degraded to 2 m / 0.5 m/s while jammed (64-96 s);
  stage 2 (fast)          LM run the way a sliding-window smoother runs it: each window warm-started
                          from LM's own previous estimate (new keyframes chained with the gyro), cold
                          started from a random attitude at a scripted estimator reset (--reset,
                          default 44 s; the draw is fixed by the window index, not chosen). Every
                          window is also cold-started from N_DRAWS random attitudes, to report how
                          often a reset at that moment would trap LM.

Attitude prior: like the EuRoC sweep, both estimators get a weak prior on the first keyframe's
attitude (`--r0-sigma`, default 0.05 per rotation-matrix entry, about 2 deg; gauge_fix='R_prior'
for the SDP, the matching rotation prior for LM). Take-off and the landing hover have no
horizontal acceleration, so without it their heading is undefined. `--r0-sigma 0` reproduces the
unanchored run (-> sim_mission_free.json).

    python demos/sim_quad.py && python demos/sim_mission.py     # ~25 min -> demos/out/sim_mission.json
"""
import argparse, json, os, time
import numpy as np
from _common import (OUT, ROOT, T, KFI, KFPB, certified, heading_tilt, excitation, rot_angle,
                     so3_exp)
import sdp_preintegration_cayley as C
import compare_gtsam_cayley as X
from gtsam_baseline import solve_gtsam
import sim_quad as S

BASE = S.OUT
STEP = 4.0
SIG_OK, SIG_JAM = (1e-2, 1e-2), (0.5, 2.0)           # (sigma_v, sigma_p)
N_DRAWS = 20
R0_SIGMA = 0.05                                      # weak attitude prior on keyframe 0 (0 = none)


def cache_path(r0_sigma):
    return os.path.join(OUT, 'sim_sdp_cache.json' if not r0_sigma else 'sim_sdp_cache_Rp%g.json' % r0_sigma)


def sigmas(t0):
    lo, hi = max(t0, S.JAM[0]), min(t0 + T, S.JAM[1])
    sv, sp = SIG_JAM if hi - lo >= T / 2 else SIG_OK
    return (sv, sp, 1e-3, 1e-3)


def mean_err(Ra, Rb):
    return float(np.mean([rot_angle(a, b) for a, b in zip(Ra, Rb)]))


def stage_sdp(starts, r0_sigma=R0_SIGMA):
    CACHE = cache_path(r0_sigma)
    rows = json.load(open(CACHE)) if os.path.exists(CACHE) else []
    done = {r['t0'] for r in rows}
    carried = (S.BG0, S.BA0)                         # calibrated once, on the ground before take-off
    if rows:
        carried = (np.asarray(rows[-1]['bg0']), np.asarray(rows[-1]['ba0']))
    for wi, t0 in enumerate(starts):
        if t0 in done:
            continue
        ps = sigmas(t0)
        tic = time.time()
        r = C.run_euroc(BASE, T=T, kf_interval=KFI, t_start=t0, weight_mode='dataset_cov',
                        iso_reduce='mean', prior_sigmas=ps, add_redundant=True,
                        gauge_fix='R_prior' if r0_sigma else False, R0_prior_sigma=r0_sigma or None,
                        kf_per_bias=KFPB, accept_any=True, gnss_seed=7000 + wi, bias_init=carried,
                        gyro_nd=S.ND_G, accel_nd=S.ND_A, gyro_rw=S.RW_G, accel_rw=S.RW_A,
                        solver_params={'numThreads': 8})
        secs = time.time() - tic
        if r.get('rank') is None:
            print('t=%5.0f SDP failed: %s' % (t0, r.get('status')), flush=True)
            continue
        hd, tl = heading_tilt(r['R_est'], r['R_true'])
        row = dict(t0=float(t0), sigma_v=ps[0], sigma_p=ps[1], rank=int(r['rank']), eta=float(r['eta']),
                   eta_dual=r['eta_dual'], solsta=r.get('solsta'), secs=secs, excitation=excitation(r),
                   rot_err=float(np.mean(r['rot_err'])), heading_err=hd, tilt_err=tl,
                   R_true=[np.asarray(R).tolist() for R in r['R_true']],
                   R_sdp=[np.asarray(R).tolist() for R in r['R_est']],
                   v_meas=np.asarray(r['v_meas']).tolist(), p_meas_local=np.asarray(r['p_meas']).tolist(),
                   p_meas=(np.asarray(r['p_meas']) + r['pos_offset']).tolist(),
                   p_true=(np.asarray(r['pos']) + r['pos_offset']).tolist(),
                   bgbar=np.asarray(r['bgbar']).tolist(), babar=np.asarray(r['babar']).tolist(),
                   bg0=np.asarray(r['bg'][0]).tolist(), ba0=np.asarray(r['ba'][0]).tolist(),
                   bias_err=[float(x) for x in r['bias_err_per_block']])
        row['certified'] = certified(row)
        # the next window starts STEP later, inside this window's first bias block
        carried = (np.asarray(r['bg'][0]), np.asarray(r['ba'][0]))
        rows.append(row)
        json.dump(rows, open(CACHE, 'w'))
        print('t=%5.0f sigma_p %-5g rank %d gap %.0e %-4s rot %5.2f head %5.2f exc %5.1f %4.0fs'
              % (t0, ps[1], row['rank'], row['eta_dual'], 'CERT' if row['certified'] else '-',
                 row['rot_err'], hd, row['excitation'] or np.nan, secs), flush=True)
    return rows


def lm_window(W, row, R_init=None, init='gt', seed=0, r0_sigma=R0_SIGMA):
    ps = (row['sigma_v'], row['sigma_p'], 1e-3, 1e-3)
    kw = dict(anchor_R0=np.asarray(row['R_true'][0]), anchor_sigma=r0_sigma / np.sqrt(2.0)) if r0_sigma else {}
    return solve_gtsam(W['gyro'], W['acc'], W['dt'], W['kf_idx'], W['bgbar'], W['babar'],
                       row['R_true'], W['v_t'], W['p_t'], W['blk'], S.ND_G, S.ND_A, ps, S.RW_G, S.RW_A,
                       kf_interval=KFI, initialisation=init, R_init=R_init, seed=seed, **kw)


def chain(W, R_start, k_from):
    """Extend attitudes past keyframe k_from by integrating the bias-corrected gyro."""
    R = list(R_start)
    for k in range(k_from, len(W['kf_idx']) - 1):
        Rk = R[k]
        for s in range(W['kf_idx'][k], W['kf_idx'][k + 1]):
            Rk = Rk @ so3_exp((W['gyro'][s] - W['bgbar']) * W['dt'])
        R.append(Rk)
    return R


def stage_lm(rows, reset, r0_sigma=R0_SIGMA):
    prev = None                                     # (t0, R_est) of LM's previous window
    for wi, row in enumerate(rows):
        row['certified'] = certified(row)
        W = X.load_euroc_window(BASE, T, KFI, row['t0'], KFPB)
        W['v_t'] = [np.asarray(v) for v in row['v_meas']]
        W['p_t'] = [np.asarray(p) for p in row['p_meas_local']]     # same recentring as the SDP
        W['bgbar'], W['babar'] = np.asarray(row['bgbar']), np.asarray(row['babar'])
        R_t = [np.asarray(R) for R in row['R_true']]
        lm = lambda **k: lm_window(W, row, r0_sigma=r0_sigma, **k)
        ref = lm(init='gt')
        draws = [lm(init='haar_shared', seed=1000 * wi + k) for k in range(N_DRAWS)]
        cold = draws[0]
        if prev is None:
            g = lm(init='gravity'); mode = 'gravity'
        elif prev[0] < reset <= row['t0']:
            g = cold; mode = 'reset'
        else:
            shift = int(round((row['t0'] - prev[0]) / KFI))
            keep = [np.asarray(R) for R in prev[1][shift:]]
            g = lm(R_init=chain(W, keep, len(keep) - 1)); mode = 'warm'
        prev = (row['t0'], g['R_est'])
        R_sdp = [np.asarray(R) for R in row['R_sdp']]
        hd, tl = heading_tilt(g['R_est'], R_t)
        row['lm'] = dict(mode=mode, rot_err=mean_err(g['R_est'], R_t), heading_err=hd, tilt_err=tl,
                         vs_sdp=mean_err(g['R_est'], R_sdp), global_=bool(g['cost'] <= ref['cost'] * 1.01 + 1e-9),
                         R=[np.asarray(R).tolist() for R in g['R_est']])
        trapped = [not (q['cost'] <= ref['cost'] * 1.01 + 1e-9) for q in draws]
        row['lm_cold'] = dict(rot_err=mean_err(cold['R_est'], R_t), global_=not trapped[0],
                              trapped_frac=float(np.mean(trapped)))
        print('t=%5.0f LM %-7s rot %7.2f vs SDP %7.2f %s | random-attitude restarts trapped: %2.0f%%'
              % (row['t0'], mode, row['lm']['rot_err'], row['lm']['vs_sdp'],
                 'global' if row['lm']['global_'] else 'LOCAL MIN', 100 * row['lm_cold']['trapped_frac']), flush=True)
    return rows


def main(reset=44.0, r0_sigma=R0_SIGMA):
    starts = [float(t) for t in np.arange(0.0, S.DUR - T + 1e-6, STEP)]
    rows = stage_sdp(starts, r0_sigma)
    rows = stage_lm(rows, reset, r0_sigma)
    out = dict(reset=reset, step=STEP, T=T, jam=S.JAM, phases=S.PHASES, r0_sigma=r0_sigma, windows=rows)
    json.dump(out, open(os.path.join(OUT, 'sim_mission.json' if r0_sigma else 'sim_mission_free.json'), 'w'))
    n = len(rows)
    print('certified %d/%d; random-attitude restarts trapped %.0f%% overall; '
          'LM sliding window (reset at %g s) at a local minimum in %d/%d windows'
          % (sum(r['certified'] for r in rows), n, 100 * np.mean([r['lm_cold']['trapped_frac'] for r in rows]),
             reset, sum(not r['lm']['global_'] for r in rows), n))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--reset', type=float, default=44.0)
    ap.add_argument('--r0-sigma', type=float, default=R0_SIGMA, help='attitude prior on keyframe 0; 0 = none')
    a = ap.parse_args()
    main(a.reset, a.r0_sigma)
