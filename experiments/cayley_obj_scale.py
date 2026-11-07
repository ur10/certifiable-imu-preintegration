"""A/B test of obj_scale: 'auto' (Q rescaled so max|Q| ~ 1e4) vs 1.0 (raw inverse-covariance Q).

The cone is scale invariant, so in exact arithmetic the two give the same Z* and the same estimate;
any difference is numerical. Per window this records MOSEK's statuses, max|Q|, tightness (rank, eta),
accuracy against ground truth, solve time, and how far apart the two estimates are.

Weighting is the paper's (weight_mode='dataset_cov', iso_reduce='mean'); the gauge is the library
default ('R'). Same windows as cayley_iso_propagation.py.

    python3 experiments/cayley_obj_scale.py --nkf 8
    python3 experiments/cayley_obj_scale.py --nkf 16 --kpb 8 --ds euroc medium deep
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
import argparse, json, time
import numpy as np
import cayley_iso_propagation as P

SCALES = {'auto': 'auto', '1.0': 1.0}
BASE = {k: v for k, v in dict(P.NOMINAL, iso_reduce='mean', iso_propagate=False).items()
        if k != 'obj_scale'}


def _angle(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nkf', type=int, default=8)
    ap.add_argument('--kpb', type=int, default=None)
    ap.add_argument('--ds', nargs='*', default=['euroc', 'medium', 'deep', 'sick', 'loop'])
    ap.add_argument('--offsets', nargs='*', type=float, default=P.OFFSETS)
    ap.add_argument('--sigmas', nargs='*', type=float, default=[1e-3, 1e-2, 1e-1])
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--out', default='obj_scale.json')
    a = ap.parse_args()
    jobs = []
    for ds in a.ds:
        jobs += ([(ds, None, s, sd) for s in a.sigmas for sd in range(a.seeds)]
                 if ds in ('sick', 'loop') else [(ds, off, None, 0) for off in a.offsets])
    rows = []
    for ds, off, sig, seed in jobs:
        est = {}
        for name, sc in SCALES.items():
            diag = {}; t0 = time.time()
            try:
                r = P._solve(ds, off, a.nkf, a.kpb, seed, sig, obj_scale=sc, diag=diag, **BASE)
                d = P._metrics(r)
            except Exception as exc:
                r, d = None, dict(rank=None, status='EXC %s' % exc)
            d.update(scale=name, ds=ds, off=off, sigma=sig, seed=seed, nkf=a.nkf, kpb=a.kpb,
                     secs=time.time() - t0, solsta=diag.get('solsta'), prosta=diag.get('prosta'),
                     qmax=diag.get('qmax'), applied_scale=diag.get('obj_scale'))
            if r is not None and r.get('rank') is not None:
                est[name] = (r['R_est'], np.asarray(r.get('p_est', r.get('pos_est'))))
            rows.append(d)
        if len(est) == 2:   # distance between the two estimates, per pair
            (Ra, pa), (Rb, pb) = est['auto'], est['1.0']
            dR = max(_angle(x, y) for x, y in zip(Ra, Rb)); dp = float(np.abs(pa - pb).max())
            rows[-1].update(d_rot_vs_auto=dR, d_pos_vs_auto=dp)
            rows[-2].update(d_rot_vs_auto=0.0, d_pos_vs_auto=0.0)
        for d in rows[-2:]:
            print('%-4s %-6s off=%-5s sig=%-6s seed=%d qmax=%-8.1e prosta=%-22s solsta=%-22s rank=%-3s '
                  'eta=%-9.2e rot=%.4f pos=%.4f |dR|=%s (%.1fs)' % (
                      d['scale'], ds, off, sig, seed, d['qmax'] or np.nan,
                      str(d['prosta']).split('.')[-1], str(d['solsta']).split('.')[-1],
                      d.get('rank'), d.get('eta', np.nan), d.get('rot_mean', np.nan),
                      d.get('pos_rmse', np.nan), '%.1e' % d['d_rot_vs_auto'] if 'd_rot_vs_auto' in d else '-',
                      d['secs']), flush=True)
        json.dump(rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
