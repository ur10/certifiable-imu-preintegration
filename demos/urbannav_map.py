"""Demo data for the UrbanNav map: certified GNSS-IMU smoothing along two Hong Kong drives.

Back-to-back 16 s windows over UrbanNav medium-urban (TST) and deep-urban (Whampoa). Same setting
as the EuRoC demos: GNSS fixes simulated from ground truth (1 cm, 1 cm/s), attitude withheld
(free gauge), bias linearised at the static estimate from the start of each drive. Each window
records the certificate, the heading excitation, the per-keyframe heading of the truth and of
each estimate, and LM (exp-map, GTSAM) from a gravity-aligned and an identity start.

    python demos/urbannav_map.py --workers 5        # ~40 min; -> demos/out/urbannav_map.json
"""
import argparse, json, os, time
from multiprocessing import Pool
import numpy as np
from _common import OUT, PRIOR_SIGMAS, T, KFI, KFPB, certified, heading_tilt, excitation, rot_angle
import _common  # noqa: F401  (puts the repo root on sys.path)
import sdp_preintegration_cayley as C
import compare_gtsam_cayley as X
import urbannav_data
import datasets
from gtsam_baseline import solve_gtsam


def heading_deg(R):
    """Compass-style heading (deg, 0 = north, clockwise) of the vehicle's forward axis (body +y)."""
    f = np.asarray(R)[:, 1]
    return float(np.degrees(np.arctan2(f[0], f[1])) % 360.0)


def mean_err(Ra, Rb):
    return float(np.mean([rot_angle(a, b) for a, b in zip(Ra, Rb)]))


def window(job):
    seq, wi, t0 = job
    tic = time.time()
    try:
        r = C.run_urbannav(datasets.urbannav(), sequence=seq, T=T, kf_interval=KFI, t_start=t0,
                           weight_mode='dataset_cov', iso_reduce='mean', prior_sigmas=PRIOR_SIGMAS,
                           gauge_fix=False, add_redundant=True, kf_per_bias=KFPB, accept_any=True,
                           gnss_seed=5000 + wi, solver_params={'numThreads': 2})
    except Exception as exc:                                   # noqa: BLE001
        return dict(seq=seq, t0=t0, failed=repr(exc))
    secs = time.time() - tic
    if r.get('rank') is None:
        return dict(seq=seq, t0=t0, failed=str(r.get('status')))
    R_t = r['R_true']
    hd, tl = heading_tilt(r['R_est'], R_t)
    row = dict(seq=seq, t0=t0, rank=int(r['rank']), eta=float(r['eta']), eta_dual=r['eta_dual'],
               solsta=r.get('solsta'), secs=secs, excitation=excitation(r),
               rot_err=float(np.mean(r['rot_err'])), heading_err=hd, tilt_err=tl,
               mean_speed=r['mean_speed'], lat=np.asarray(r['lat']).tolist(),
               lon=np.asarray(r['lon']).tolist(),
               hdg_true=[heading_deg(R) for R in R_t], hdg_sdp=[heading_deg(R) for R in r['R_est']],
               lm={})
    row['certified'] = certified(row)
    W = X.load_urbannav_window(datasets.urbannav(), seq, T, KFI, t0, KFPB)
    W['v_t'], W['p_t'] = r['v_meas'], r['p_meas']
    W['bgbar'], W['babar'] = r['bgbar'], r['babar']

    def lm(init):
        return solve_gtsam(W['gyro'], W['acc'], W['dt'], W['kf_idx'], W['bgbar'], W['babar'],
                           R_t, W['v_t'], W['p_t'], W['blk'], W['nd_g'], W['nd_a'], PRIOR_SIGMAS,
                           W['gyro_rw'], W['accel_rw'], kf_interval=KFI, gravity=W['gravity'],
                           initialisation=init)
    ref = lm('gt')
    for init in ('gt', 'gravity', 'gnss'):
        g = ref if init == 'gt' else lm(init)
        row['lm'][init] = dict(rot_err=mean_err(g['R_est'], R_t), vs_sdp=mean_err(g['R_est'], r['R_est']),
                               cost=g['cost'], global_=bool(g['cost'] <= ref['cost'] * 1.01 + 1e-9),
                               hdg=[heading_deg(R) for R in g['R_est']])
    print('%-6s t=%5.0f rank %d gap %.0e %-4s rot %6.2f head %6.2f exc %5.1f speed %4.1f | LM grav %s ident %s | %3.0fs'
          % (seq, t0, row['rank'], row['eta_dual'], 'CERT' if row['certified'] else '-', row['rot_err'],
             hd, row['excitation'] or np.nan, row['mean_speed'],
             'ok' if row['lm']['gravity']['global_'] else 'LOCAL', 'ok' if row['lm']['gnss']['global_'] else 'LOCAL',
             secs), flush=True)
    return row


def main(workers=5):
    jobs = []
    for seq in ('medium', 'deep'):
        g = urbannav_data.load_sequence(datasets.urbannav(), seq)['gt']
        dur = g['t'][-1] - g['t'][0]
        jobs += [(seq, wi, float(t0)) for wi, t0 in enumerate(np.arange(6.0, dur - T - 1.0, T))]
    print('%d windows' % len(jobs), flush=True)
    with Pool(workers) as pool:
        rows = pool.map(window, jobs, chunksize=1)
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, 'urbannav_map.json')
    json.dump(rows, open(fn, 'w'))
    ok = [q for q in rows if 'failed' not in q]
    print('wrote %s: %d windows, %d certified, %d failed to solve'
          % (fn, len(ok), sum(q['certified'] for q in ok), len(rows) - len(ok)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=5)
    a = ap.parse_args()
    main(a.workers)
