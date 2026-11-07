"""Demo 2 -- certified GNSS-IMU smoothing, window after window, over whole EuRoC flights.

Each sequence is tiled with back-to-back 16 s windows. Every window is solved twice on the same
measurements:

  Cayley SDP   no initial guess; returns a certificate (rank, primal and dual optimality gap)
  LM (GTSAM)   exp-map pre-integration, started where a real system would start it:
               'gravity'  roll/pitch from the accelerometer, heading 0  (coarse alignment)
               'gnss'     identity attitude
               and, as a reference only, 'gt' (started at the true attitude)

Bias is handled as an online system would: the first window is linearised at the bias the
dataset reports at take-off; every later window is linearised at, and given a prior around, the
previous window's SDP estimate (`--bias carry`, the default). `--bias gt` uses the dataset bias
for every window instead, as the paper does.

    python demos/sequence_sweep.py --seq MH_04_difficult      # ~5 min per sequence
    python demos/plot_sweep.py                                # -> demos/figures/sweep_*.png
"""
import argparse, json, os, time
import numpy as np
from _common import (OUT, T, solve_window, run_lm, rot_angle, certified, heading_tilt,
                     excitation, datasets)


def duration(seq):
    g = np.loadtxt(os.path.join(datasets.euroc_sequence(seq), 'state_groundtruth_estimate0',
                                'data.csv'), delimiter=',', skiprows=1, usecols=0)
    return (g[-1] - g[0]) * 1e-9


def mean_err(Ra, Rb):
    return float(np.mean([rot_angle(a, b) for a, b in zip(Ra, Rb)]))


def main(seq, bias='carry', step=T):
    starts = np.arange(0.0, duration(seq) - T - 0.5, step)
    rows, carried = [], None
    for wi, t0 in enumerate(starts):
        bi = 'gt' if (bias == 'gt' or carried is None) else carried
        tic = time.time()
        r, W = solve_window(seq, float(t0), gnss_seed=1000 + wi, bias_init=bi)
        secs = time.time() - tic
        if r.get('rank') is None:
            rows.append(dict(seq=seq, t0=float(t0), failed=str(r.get('status'))))
            print('%s t=%5.0f  SDP FAILED %s' % (seq, t0, r.get('status')), flush=True)
            continue
        R_t = r['R_true']
        hd, tl = heading_tilt(r['R_est'], R_t)
        row = dict(seq=seq, t0=float(t0), bias_mode='gt' if isinstance(bi, str) else 'carry',
                   rank=int(r['rank']), eta=float(r['eta']), eta_dual=r['eta_dual'],
                   log_svr=float(r['log_svr']), solsta=r.get('solsta'), certified=certified(r),
                   secs=secs, rot_err=float(np.mean(r['rot_err'])), heading_err=hd, tilt_err=tl,
                   excitation=excitation(r), R0_source=r.get('R0_source'),
                   bias_err=[float(x) for x in r['bias_err_per_block']],
                   bias_init_err=float(np.linalg.norm(np.r_[r['bgbar'] - r['bg_true'],
                                                            r['babar'] - r['ba_true']])),
                   p_true=(np.asarray(r['pos']) + r['pos_offset']).tolist(),
                   p_est=(np.asarray(r['p_est']) + r['pos_offset']).tolist(), lm={})
        ref = run_lm(W, R_t, 'gt')
        for init in ('gt', 'gravity', 'gnss'):
            g = ref if init == 'gt' else run_lm(W, R_t, init)
            ghd, gtl = heading_tilt(g['R_est'], R_t)
            row['lm'][init] = dict(rot_err=mean_err(g['R_est'], R_t), heading_err=ghd, tilt_err=gtl,
                                   vs_sdp=mean_err(g['R_est'], r['R_est']), cost=g['cost'],
                                   secs=g['secs'], iters=g['iters'],
                                   global_=bool(g['cost'] <= ref['cost'] * 1.01 + 1e-9))
        rows.append(row)
        # carry the last bias block forward as the next window's linearisation point and prior
        carried = (np.asarray(r['bg'][-1]), np.asarray(r['ba'][-1]))
        lm = row['lm']
        print('%s t=%5.0f  SDP rank %d eta %.0e/%.0e %-5s %5.2f deg (head %5.2f)  exc %5.1f | '
              'LM grav %6.2f%s  ident %6.2f%s | %4.0fs'
              % (seq, t0, r['rank'], r['eta'], r['eta_dual'], 'CERT' if row['certified'] else '-',
                 row['rot_err'], hd, row['excitation'] or np.nan,
                 lm['gravity']['rot_err'], '' if lm['gravity']['global_'] else '*',
                 lm['gnss']['rot_err'], '' if lm['gnss']['global_'] else '*', secs), flush=True)
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, 'sweep_%s_%s.json' % (seq, bias))
    json.dump(rows, open(fn, 'w'), indent=1)
    print('wrote', fn)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='MH_04_difficult', choices=datasets.EUROC_SEQUENCES)
    ap.add_argument('--bias', default='carry', choices=['carry', 'gt'])
    a = ap.parse_args()
    main(a.seq, a.bias)
