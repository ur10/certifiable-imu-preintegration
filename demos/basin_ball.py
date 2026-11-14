"""Data for the 3D demos: the attitude basin of attraction, and one window replayed at 200 Hz.

Same EuRoC MH_04_difficult window as demos/basins.py. Every attitude error the whole window can
start with is a rotation vector w with |w| <= pi, i.e. a point in a ball of radius pi. We fill that
ball with points drawn uniformly (by volume), start Levenberg-Marquardt from R_k = Exp(w) R_k^true at every keyframe,
record every LM iterate, and note where each run ends. The Cayley SDP is solved once.

    python demos/basin_ball.py            # ~1 min -> demos/out/basin_ball.json, replay_window.json
"""
import argparse, json, os, time
import numpy as np
from _common import (OUT, solve_window, run_lm, rot_angle, certified, so3_log, so3_exp,
                     heading_tilt, datasets, T)


def dense_window(seq, t_start):
    """200 Hz ground truth and IMU for the window (positions recentred like the solver's)."""
    base = datasets.euroc_sequence(seq)
    gt = np.loadtxt(os.path.join(base, 'state_groundtruth_estimate0', 'data.csv'), delimiter=',',
                    skiprows=1)
    imu = np.loadtxt(os.path.join(base, 'imu0', 'data.csv'), delimiter=',', skiprows=1)
    t0 = gt[0, 0] + t_start * 1e9
    m = (gt[:, 0] >= t0) & (gt[:, 0] <= t0 + T * 1e9)
    mi = (imu[:, 0] >= t0) & (imu[:, 0] <= t0 + T * 1e9)
    return dict(t=((gt[m, 0] - t0) * 1e-9).tolist(), p=gt[m, 1:4], q_wxyz=gt[m, 4:8].tolist(),
                imu_t=((imu[mi, 0] - t0) * 1e-9).tolist(), gyro=imu[mi, 1:4].tolist())


def main(seq='MH_04_difficult', t_start=50.0, n=2000, gnss_seed=0, seed=0):
    tic = time.time()
    r, W = solve_window(seq, t_start, gnss_seed)
    print('SDP rank %d dual gap %.1e certified %s (%.0f s)'
          % (r['rank'], r['eta_dual'], certified(r), time.time() - tic))
    R_t = r['R_true']
    ref = run_lm(W, R_t, 'gt')
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    grid = u * (np.pi * rng.random(n) ** (1.0 / 3.0))[:, None]      # uniform in the ball
    runs = []
    tic = time.time()
    for w in grid:
        E = so3_exp(w)
        res = run_lm(W, R_t, 'gt', R_init=[E @ R for R in R_t], record=True)
        path = [so3_log(Rs[0] @ R_t[0].T).tolist() for Rs in res['R_path']]
        runs.append(dict(w0=w.tolist(), path=path, cost=res['cost'], iters=len(path) - 1,
                         rot_err=float(np.mean([rot_angle(a, b) for a, b in zip(res['R_est'], R_t)])),
                         global_=bool(res['cost'] <= ref['cost'] * 1.01 + 1e-9),
                         R_est=[np.asarray(R).tolist() for R in res['R_est']]))
    bad = [q for q in runs if not q['global_']]
    print('%d LM runs in %.0f s: %d global, %d local (median %.1f deg)'
          % (len(runs), time.time() - tic, len(runs) - len(bad), len(bad),
             np.median([q['rot_err'] for q in bad]) if bad else np.nan))
    # distinct end points among the failures (cluster by final cost)
    if bad:
        cs = np.array([q['cost'] for q in bad])
        print('local-minimum costs / optimum:', np.unique(np.round(cs / ref['cost'], 0)))

    os.makedirs(OUT, exist_ok=True)
    slim = [{k: q[k] for k in ('w0', 'path', 'cost', 'iters', 'rot_err', 'global_')} for q in runs]
    json.dump(dict(seq=seq, t_start=t_start, n=n, ref_cost=ref['cost'],
                   sdp=dict(rank=r['rank'], eta_dual=r['eta_dual'], certified=certified(r),
                            rot_err=float(np.mean(r['rot_err'])),
                            err_vec=so3_log(np.asarray(r['R_est'][0]) @ R_t[0].T).tolist()),
                   runs=slim),
              open(os.path.join(OUT, 'basin_ball.json'), 'w'))

    # one representative trapped run (median attitude error among the failures) for the replay
    pick = sorted(bad, key=lambda q: q['rot_err'])[len(bad) // 2] if bad else None
    D = dense_window(seq, t_start)
    hd, tl = heading_tilt(pick['R_est'], R_t) if pick else (None, None)
    json.dump(dict(seq=seq, t_start=t_start, T=T, kf_t=[float(k) for k in range(int(T) + 1)],
                   p_true=(np.asarray(r['pos']) + r['pos_offset']).tolist(),
                   p_off=np.asarray(r['pos_offset']).tolist(),
                   p_meas=(np.asarray(r['p_meas']) + r['pos_offset']).tolist(),
                   R_true=[np.asarray(R).tolist() for R in R_t],
                   R_sdp=[np.asarray(R).tolist() for R in r['R_est']],
                   R_lm_bad=pick['R_est'] if pick else None,
                   lm_bad=dict(rot_err=pick['rot_err'], heading_err=hd, tilt_err=tl,
                               cost_ratio=pick['cost'] / ref['cost'], w0=pick['w0']) if pick else None,
                   sdp=dict(rank=r['rank'], eta=r['eta'], eta_dual=r['eta_dual'],
                            rot_err=float(np.mean(r['rot_err']))),
                   bg=[np.asarray(b).tolist() for b in r['bg']], blk_edges=8,
                   dense=dict(t=D['t'], p=D['p'].tolist(), q_wxyz=D['q_wxyz'], imu_t=D['imu_t'],
                              gyro=D['gyro'])),
              open(os.path.join(OUT, 'replay_window.json'), 'w'))
    print('wrote basin_ball.json, replay_window.json; picked trapped run: %.1f deg (heading %.1f, '
          'tilt %.1f)' % (pick['rot_err'], hd, tl) if pick else '')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000, help='LM starts, uniform in the rotation-vector ball')
    a = ap.parse_args()
    main(n=a.n)
