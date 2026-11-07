"""Isotropic weighting: where does the isotropic approximation enter the covariance propagation?

  A  'post'  -- propagate the full anisotropic Sigma_ij over the keyframe interval, then collapse
               each block to tau^-1 = tr(Sigma_block)/3 (paper eq. 23; iso_reduce='mean').
  A' 'post-max' -- same propagation, collapse by the max of the block diagonal (iso_reduce='max').
  B  'step'  -- keep Sigma isotropic THROUGHOUT: after every IMU step project it onto
               {S kron I_3} (iso_project), so every step propagates an isotropic covariance.

Everything else is the paper's nominal configuration (weight_mode='dataset_cov', no anchor,
obj_scale='auto', prior_sigmas=(1e-2, 1e-2, 1e-3, 1e-3)). Solution quality is reported both as
tightness (rank, eta) and as accuracy against ground truth (rotation, position, velocity, bias).

    python3 experiments/cayley_iso_propagation.py --nkf 8                 # quick check, M = 1
    python3 experiments/cayley_iso_propagation.py --nkf 16 --kpb 8        # Table II size, M = 2
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
import argparse, json, time
import numpy as np
import sdp_preintegration_cayley as C
import datasets

VARIANTS = {'A_post_mean': dict(iso_reduce='mean', iso_propagate=False),
            'B_step_iso':  dict(iso_reduce='mean', iso_propagate=True),
            'A_post_max':  dict(iso_reduce='max', iso_propagate=False)}
NOMINAL = dict(weight_mode='dataset_cov', kf_interval=1.0, prior_sigmas=(1e-2, 1e-2, 1e-3, 1e-3),
               obj_scale=1.0, add_redundant=True, accept_any=True)
# gauge_fix=False (the paper's free gauge) leaves yaw unobservable on near-stationary windows:
# UrbanNav medium@40 s / deep@90 s certify at rank 1 with 72-82 deg of pure yaw error, and moving
# windows still carry 1-3 deg. 'R' hard-anchors R_0, removing that gauge freedom, so the A/B
# comparison measures the noise model rather than the gauge.
GAUGES = {'free': dict(gauge_fix=False), 'R': dict(gauge_fix='R'),
          'R_prior': dict(gauge_fix='R_prior', R0_prior_sigma=1e-2)}
OFFSETS = [6.0, 40.0, 90.0, 132.0]


def _metrics(r, bgbar=None, babar=None):
    if r is None or r.get('rank') is None:
        return dict(rank=None, status=str(None if r is None else r.get('status')))
    p_est = np.asarray(r.get('p_est', r.get('pos_est'))); p_t = np.asarray(r['pos'])
    v_est = np.asarray(r['v_est'] if 'v_est' in r else r['vel_est'])
    v_t = np.asarray(r['v_true'] if 'v_true' in r else r['vel_true'])
    f, p = float(r['rounded']), float(r['sdp_opt'])
    d = dict(rank=int(r['rank']), eta=float(r['eta']), rel=abs(f - p) / abs(f), f_hat=f,
             log_svr=float(r['log_svr']),
             rot_mean=float(np.mean(r['rot_err'])), rot_max=float(np.max(r['rot_err'])),
             pos_rmse=float(np.sqrt(np.mean(np.sum((p_est - p_t) ** 2, 1)))),
             vel_rmse=float(np.sqrt(np.mean(np.sum((v_est - v_t) ** 2, 1)))),
             rot_per_node=[float(x) for x in r['rot_err']])
    if 'bias_err_per_block' in r and bgbar is None:
        d['bias_err'] = float(np.mean(r['bias_err_per_block']))
    return d


def _solve(ds, off, nkf, kpb, seed, sigma, **kw):
    T = float(nkf) + (0.0 if ds in ('sick', 'loop') else 0.5)
    if ds in ('sick', 'loop'):
        return C.run_synthetic(ds, T=T, dt=0.01, gyro_sigma=1e-3, accel_sigma=1e-3,
                               noise_sigma=sigma, seed=seed, kf_per_bias=kpb, **kw)
    if ds == 'euroc':
        return C.run_euroc(datasets.euroc(), T=T, t_start=off, kf_per_bias=kpb, **kw)
    return C.run_urbannav(datasets.urbannav(), ds, T=T, t_start=off, kf_per_bias=kpb, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nkf', type=int, default=8)
    ap.add_argument('--kpb', type=int, default=None)
    ap.add_argument('--ds', nargs='*', default=['euroc', 'medium', 'deep', 'sick', 'loop'])
    ap.add_argument('--offsets', nargs='*', type=float, default=OFFSETS)
    ap.add_argument('--sigmas', nargs='*', type=float, default=[1e-3, 1e-2, 1e-1])
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--gauge', choices=sorted(GAUGES), default='R')
    ap.add_argument('--out', default='iso_propagation.json')
    a = ap.parse_args()
    jobs = []
    for ds in a.ds:
        if ds in ('sick', 'loop'):
            jobs += [(ds, None, s, sd) for s in a.sigmas for sd in range(a.seeds)]
        else:
            jobs += [(ds, off, None, 0) for off in a.offsets]
    rows = []
    for ds, off, sig, seed in jobs:
        for name, var in VARIANTS.items():
            kw = dict(NOMINAL, **GAUGES[a.gauge]); kw.update(var); t0 = time.time()
            try:
                d = _metrics(_solve(ds, off, a.nkf, a.kpb, seed, sig, **kw))
            except Exception as exc:
                d = dict(rank=None, status='EXC %s' % exc)
            d.update(variant=name, gauge=a.gauge, ds=ds, off=off, sigma=sig, seed=seed, nkf=a.nkf, kpb=a.kpb,
                     secs=time.time() - t0)
            rows.append(d)
            print('%-12s %-6s off=%-5s sig=%-6s seed=%d rank=%-3s eta=%-9.2e rot=%.4f pos=%.4f '
                  'vel=%.4f (%.1fs)' % (name, ds, off, sig, seed, d.get('rank'), d.get('eta', np.nan),
                                        d.get('rot_mean', np.nan), d.get('pos_rmse', np.nan),
                                        d.get('vel_rmse', np.nan), d['secs']), flush=True)
            json.dump(rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
